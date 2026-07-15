from decimal import Decimal
from typing import Any

from atlas_api.dynamo import (
    DynamoBucketStore,
    DynamoCacheStore,
    DynamoJobStore,
    dynamo_backends,
)
from atlas_core.budget import DailyBudgetGuard
from atlas_core.cache import CacheEntry, ResponseCache
from atlas_core.ratelimit import RateLimiter


class FakeTable:
    """The slice of a boto3 DynamoDB Table these adapters lean on, over an in-memory dict."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        item = self.items.get(kwargs["Key"]["pk"])
        return {"Item": item} if item is not None else {}

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        item = kwargs["Item"]
        self.items[item["pk"]] = item
        return {}

    def scan(self, **kwargs: Any) -> dict[str, Any]:
        return {"Items": list(self.items.values())}

    def update_item(self, **kwargs: Any) -> dict[str, Any]:  # atomic ADD, as the budget guard uses
        pk = kwargs["Key"]["pk"]
        item = self.items.setdefault(pk, {"pk": pk, "spent": Decimal(0)})
        item["spent"] = item.get("spent", Decimal(0)) + kwargs["ExpressionAttributeValues"][":c"]
        return {"Attributes": {"spent": item["spent"]}}


class FakeResource:
    def __init__(self) -> None:
        self.tables: dict[str, FakeTable] = {}

    def Table(self, name: str) -> FakeTable:
        return self.tables.setdefault(name, FakeTable())


def test_bucket_round_trip_and_missing_key() -> None:
    store = DynamoBucketStore(FakeTable())
    assert store.get("ip") is None
    store.set("ip", 4.5, 1000.0)
    assert store.get("ip") == (4.5, 1000.0)


def test_cache_round_trip_reconstructs_entry() -> None:
    table = FakeTable()
    store = DynamoCacheStore(table)
    store.put(CacheEntry(embedding=[0.1, 0.2, 0.3], payload={"brief": "hi"}, expires_at=1234.0))
    (entry,) = store.recent()
    assert entry.embedding == [0.1, 0.2, 0.3]
    assert entry.payload == {"brief": "hi"}
    assert entry.expires_at == 1234.0


def test_backends_are_wired_to_their_tables() -> None:
    resource = FakeResource()
    limiter, cache, budget = dynamo_backends(
        resource, bucket_table="buckets", cache_table="cache", budget_table="budget"
    )
    assert isinstance(limiter, RateLimiter)
    assert isinstance(cache, ResponseCache)
    assert isinstance(budget, DailyBudgetGuard)

    limiter.allow("ip")  # first token spent, so the bucket table now holds this IP's state
    assert resource.tables["buckets"].items  # non-empty: the adapter wrote through

    import numpy as np

    vec = np.array([1.0, 0.0], dtype=np.float32)
    cache.put(vec, {"brief": "cached"})
    assert cache.get(vec) == {"brief": "cached"}  # round-trips through the cache table


def test_job_store_lifecycle() -> None:
    store = DynamoJobStore(FakeTable())
    job_id = store.create("what is X?")

    created = store.get(job_id)
    assert created is not None
    assert created.status == "pending"
    assert created.question == "what is X?"

    store.mark_running(job_id)
    running = store.get(job_id)
    assert running is not None and running.status == "running"

    store.finish(job_id, {"brief": "done"})
    done = store.get(job_id)
    assert done is not None
    assert done.status == "done"
    assert done.result == {"brief": "done"}
    assert done.error is None


def test_job_store_fail_records_error() -> None:
    store = DynamoJobStore(FakeTable())
    job_id = store.create("q")
    store.fail(job_id, "boom")
    failed = store.get(job_id)
    assert failed is not None
    assert failed.status == "error"
    assert failed.error == "boom"


def test_job_store_missing_returns_none() -> None:
    assert DynamoJobStore(FakeTable()).get("nope") is None
