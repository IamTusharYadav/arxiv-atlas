from decimal import Decimal
from typing import Any

import pytest

from atlas_core.budget import DailyBudgetExceeded, DailyBudgetGuard


class _FakeTable:
    """Mimics the one DynamoDB behaviour the guard leans on: atomic ADD to `spent`,
    returning the new total. `fail` simulates the backend being unreachable."""

    def __init__(self) -> None:
        self.spent = Decimal(0)
        self.fail = False

    def update_item(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail:
            raise RuntimeError("dynamodb unreachable")
        self.spent += kwargs["ExpressionAttributeValues"][":c"]
        return {"Attributes": {"spent": self.spent}}


def test_charge_accumulates_and_check_passes_under_cap() -> None:
    table = _FakeTable()
    guard = DailyBudgetGuard(table, daily_cap_usd=1.00)
    assert guard.charge(0.30) == pytest.approx(0.30)
    assert guard.charge(0.40) == pytest.approx(0.70)
    guard.check()  # 0.70 < 1.00, no raise


def test_check_denies_at_or_over_cap() -> None:
    table = _FakeTable()
    guard = DailyBudgetGuard(table, daily_cap_usd=1.00)
    guard.charge(1.00)
    with pytest.raises(DailyBudgetExceeded):
        guard.check()


def test_fails_closed_when_backend_unavailable() -> None:
    table = _FakeTable()
    table.fail = True
    guard = DailyBudgetGuard(table)
    with pytest.raises(DailyBudgetExceeded):
        guard.check()


def test_charge_swallows_backend_error() -> None:
    # The money is already spent when charge() runs; a backend blip must not fail the
    # finished answer (or mask the original error on the abort path). check() still gates.
    table = _FakeTable()
    table.fail = True
    guard = DailyBudgetGuard(table)
    assert guard.charge(0.05) == 0.0
