"""Fail-closed daily spend guard: a global cap across every query for the day.

The per-query cap in the harness bounds one runaway loop; this bounds the day's
aggregate so abuse or a flood of queries cannot drain Bedrock credits (plan risk 1).
The running total lives in a DynamoDB atomic counter keyed per UTC day so concurrent
Lambda invocations share one number no in-process counter could give them. Fail-closed
means any backend error denies the request rather than letting spend proceed unverified.

The table is injected (boto3 Table in the API path, a fake in tests) so atlas_core
stays free of the AWS SDK; the Phase C API constructs the real handle.
"""

import logging
from datetime import UTC, datetime
from decimal import Decimal
from time import time
from typing import Any, Protocol

log = logging.getLogger(__name__)

# Bedrock is the swing cost under the plan's <$30/mo target; cap the day well below it.
DAILY_CAP_USD = 3.00
_TTL_DAYS = 3  # keep a few days of counters for the cost report, then let TTL reap them


class DailyBudgetExceeded(RuntimeError):
    """Today's aggregate spend hit the cap, or the counter could not be read; refuse."""


class _CounterTable(Protocol):
    def update_item(self, **kwargs: Any) -> dict[str, Any]: ...


class DailyBudgetGuard:
    def __init__(self, table: _CounterTable, *, daily_cap_usd: float = DAILY_CAP_USD) -> None:
        self._table = table
        self._cap = daily_cap_usd

    def check(self) -> None:
        """Gate a query before its first Bedrock call. Raises if the day is at or over
        cap, or if the counter is unreadable (fail-closed)."""
        total = self._add(0.0)
        if total >= self._cap:
            raise DailyBudgetExceeded(f"daily spend ${total:.4f} at cap ${self._cap:.2f}")

    def charge(self, cost_usd: float) -> float:
        """Record a finished run's actual spend; returns the day's new total."""
        return self._add(cost_usd)

    def _add(self, cost_usd: float) -> float:
        # ADD is atomic, so the counter is always accurate; two queries that both clear
        # check() at just under cap can each still charge, overshooting by up to one
        # per-query cap apiece before the next check() denies. That bound is acceptable.
        # ponytail: bounded overshoot; switch to a conditional UpdateItem if a hard,
        # never-exceed cap is ever required.
        key = "budget#" + datetime.now(UTC).strftime("%Y-%m-%d")
        try:
            resp = self._table.update_item(
                Key={"pk": key},
                UpdateExpression="ADD spent :c SET expires_at = if_not_exists(expires_at, :ttl)",
                ExpressionAttributeValues={
                    ":c": Decimal(str(cost_usd)),
                    ":ttl": int(time()) + _TTL_DAYS * 86_400,
                },
                ReturnValues="UPDATED_NEW",
            )
        except Exception as err:
            log.error("budget counter unavailable, failing closed: %s", err)
            raise DailyBudgetExceeded("budget counter unavailable") from err
        return float(resp["Attributes"]["spent"])
