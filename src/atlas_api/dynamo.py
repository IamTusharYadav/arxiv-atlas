import json
from decimal import Decimal
from typing import Any, Protocol
from uuid import uuid4

from atlas_core.budget import DailyBudgetGuard
from atlas_core.cache import CacheEntry, ResponseCache
from atlas_core.ratelimit import RateLimiter

# An idle bucket is fully refilled after burst/rate seconds, so a reaped one is indistinguishable
# from a fresh one; a short TTL just keeps the table from accumulating dead IPs.
_BUCKET_TTL_S = 3600


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


def dynamo_backends(
    resource: _DynamoResource,
    *,
    bucket_table: str,
    cache_table: str,
    budget_table: str,
) -> tuple[RateLimiter, ResponseCache, DailyBudgetGuard]:
    """Wire the three request-path guards to their DynamoDB tables. The budget guard already
    speaks the raw Table interface; the bucket and cache get a translating adapter."""
    return (
        RateLimiter(DynamoBucketStore(resource.Table(bucket_table))),
        ResponseCache(DynamoCacheStore(resource.Table(cache_table))),
        DailyBudgetGuard(resource.Table(budget_table)),
    )
