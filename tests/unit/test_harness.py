import pytest

from atlas_agents.bedrock import HAIKU, Completion
from atlas_agents.harness import (
    BudgetExceeded,
    IterationsExhausted,
    RunContext,
    run_loop,
)


def completion(cost_usd: float) -> Completion:
    return Completion(text="x", model=HAIKU, input_tokens=100, output_tokens=50, cost_usd=cost_usd)


def test_loop_stops_when_task_returns_answer() -> None:
    calls = []

    def task(ctx: RunContext) -> str | None:
        calls.append(ctx.iteration)
        return "answer" if ctx.iteration == 2 else None

    answer, ctx = run_loop(task)
    assert answer == "answer"
    assert calls == [0, 1, 2]
    assert ctx.iteration == 2


def test_loop_raises_after_max_iters() -> None:
    calls: list[int] = []

    def task(ctx: RunContext) -> str | None:
        calls.append(ctx.iteration)
        return None

    with pytest.raises(IterationsExhausted):
        run_loop(task, max_iters=6)
    assert len(calls) == 6


def test_budget_accounting_sums_step_costs() -> None:
    def task(ctx: RunContext) -> str | None:
        ctx.record("retrieve", "12 candidates")  # free step, traced but not charged
        ctx.record("rerank", "kept 5", completion(0.01))
        ctx.record("synthesize", "brief written", completion(0.04))
        return "done"

    _, ctx = run_loop(task, budget_usd=0.12)
    assert ctx.spent_usd == pytest.approx(0.05)
    assert [r.step for r in ctx.trace] == ["retrieve", "rerank", "synthesize"]
    assert ctx.trace[0].cost_usd == 0.0
    assert ctx.trace[1].input_tokens == 100


def test_budget_cap_aborts_run() -> None:
    def task(ctx: RunContext) -> str | None:
        ctx.record("rerank", "s", completion(0.10))
        ctx.record("synthesize", "s", completion(0.05))  # crosses the 0.12 cap
        pytest.fail("run should have aborted before this line")

    with pytest.raises(BudgetExceeded, match="synthesize"):
        run_loop(task, budget_usd=0.12)


def test_budget_carries_across_iterations() -> None:
    def task(ctx: RunContext) -> str | None:
        ctx.record("rerank", "s", completion(0.05))
        return None

    with pytest.raises(BudgetExceeded):
        run_loop(task, budget_usd=0.12)  # third iteration crosses the cap
