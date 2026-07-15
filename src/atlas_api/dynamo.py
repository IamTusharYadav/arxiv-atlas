import json
from decimal import Decimal
from time import time
from typing import Any, Protocol
from uuid import uuid4

from atlas_api.jobs import Job
from atlas_core.budget import DailyBudgetGuard
from atlas_core.cache import CacheEntry, ResponseCache
from atlas_core.ratelimit import RateLimiter

# An idle bucket is fully refilled after burst/rate seconds, so a reaped one is indistinguishable
# from a fresh one; a short TTL just keeps the table from accumulating dead IPs.
_BUCKET_TTL_S = 3600
_JOB_TTL_S = 24 * 3600


class _Table(Protocol):
    def get_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def put_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def scan(self, **kwargs: Any) -> dict[str, Any]: ...
    def update_item(self, **kwargs: Any) -> dict[str, Any]: ...


class _DynamoResource(Protocol):
    def Table(self, name: str) -> _Table: ...


class DynamoBucketStore:
    def __init__(self, table: _Table) -> None:
        self._table = table

    def get(self, key: str) -> tuple[float, float] | None:
        item = self._table.get_item(Key={"pk": key}).get("Item")
        if item is None:
            return None
        return float(item["tokens"]), float(item["updated_at"])

    def set(self, key: str, tokens: float, updated_at: float) -> None:
        # DynamoDB rejects float; Decimal(str(x)) avoids the binary-float precision blowup.
        self._table.put_item(
            Item={
                "pk": key,
                "tokens": Decimal(str(tokens)),
                "updated_at": Decimal(str(updated_at)),
                "expires_at": Decimal(int(updated_at) + _BUCKET_TTL_S),
            }
        )


class DynamoCacheStore:
    def __init__(self, table: _Table) -> None:
        self._table = table

    def recent(self) -> list[CacheEntry]:
        # ponytail: full scan per lookup, and only the first scan page. Fine at demo volume where
        # TTL keeps live entries to a handful; add a GSI or paginate if it ever grows.
        items = self._table.scan().get("Items", [])
        return [
            CacheEntry(
                embedding=json.loads(i["embedding"]),
                payload=json.loads(i["payload"]),
                expires_at=float(i["expires_at"]),
            )
            for i in items
        ]

    def put(self, entry: CacheEntry) -> None:
        # embedding and payload go in as JSON strings, sidestepping DynamoDB's float-to-Decimal
        # conversion across a 384-float vector and a nested answer dict.
        self._table.put_item(
            Item={
                "pk": uuid4().hex,
                "embedding": json.dumps(entry.embedding),
                "payload": json.dumps(entry.payload),
                "expires_at": Decimal(int(entry.expires_at)),
            }
        )


class DynamoJobStore:
    def __init__(self, table: _Table) -> None:
        self._table = table

    def create(self, question: str) -> str:
        job_id = uuid4().hex
        self._table.put_item(
            Item={
                "pk": job_id,
                "status": "pending",
                "question": question,
                "expires_at": Decimal(int(time()) + _JOB_TTL_S),
            }
        )
        return job_id

    def get(self, job_id: str) -> Job | None:
        item = self._table.get_item(Key={"pk": job_id}).get("Item")
        if item is None:
            return None
        result = item.get("result")
        error = item.get("error")
        progress = item.get("progress")
        return Job(
            id=job_id,
            status=str(item["status"]),
            question=str(item.get("question", "")),
            result=json.loads(result) if result else None,
            error=str(error) if error else None,
            progress=json.loads(progress) if progress else [],
        )

    def mark_running(self, job_id: str) -> None:
        self._patch(job_id, status="running")

    def set_progress(self, job_id: str, progress: list[dict[str, str]]) -> None:
        self._patch(job_id, progress=json.dumps(progress))

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        self._patch(job_id, status="done", result=json.dumps(result))

    def fail(self, job_id: str, error: str) -> None:
        self._patch(job_id, status="error", error=error)

    def _patch(self, job_id: str, **changes: Any) -> None:
        # Read-modify-write, not UpdateItem: put_item keeps the fake table trivial and sidesteps
        # reserved-word aliasing. A job has one writer (the worker), so there is no race.
        item = self._table.get_item(Key={"pk": job_id}).get("Item")
        if item is None:
            return
        item.update(changes)
        self._table.put_item(Item=item)


def dynamo_backends(
    resource: _DynamoResource,
    *,
    bucket_table: str,
    cache_table: str,
    budget_table: str,
) -> tuple[RateLimiter, ResponseCache, DailyBudgetGuard]:
    # The budget guard already speaks the raw Table interface; the bucket and cache get an adapter.
    return (
        RateLimiter(DynamoBucketStore(resource.Table(bucket_table))),
        ResponseCache(DynamoCacheStore(resource.Table(cache_table))),
        DailyBudgetGuard(resource.Table(budget_table)),
    )
