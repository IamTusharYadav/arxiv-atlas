"""Generic agent loop: iteration cap, per-query budget cap, per-step trace.

No framework (ADR 0002): the loop is a plain function-call cycle. Concrete steps
(planner, retriever, reranker, extractor, synthesizer) arrive in later commits; the
harness owns only termination, budget accounting, and the trace that ships with
every answer.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from atlas_agents.bedrock import Completion

log = logging.getLogger(__name__)

# Plan acceptance: a grounded brief costs <= $0.12 uncached within <= 6 loop iterations.
MAX_ITERS = 6
BUDGET_USD = 0.12

class BudgetExceeded(RuntimeError):
    """The query crossed its per-query spend cap; the run aborts rather than overspend."""


class IterationsExhausted(RuntimeError):
    """The task never produced an answer within the iteration cap (a stalled loop)."""


@dataclass(frozen=True)
class StepRecord:
    """One trace entry; the UI's agent-trace panel renders these with per-step costs."""

    step: str
    summary: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class RunContext:
    """Handed to the task each iteration; steps report spend and trace through it."""

    budget_usd: float = BUDGET_USD
    iteration: int = 0
    spent_usd: float = 0.0
    trace: list[StepRecord] = field(default_factory=list)

    def record(self, step: str, summary: str, completion: Completion | None = None) -> None:
        """Append a trace entry. Steps with model spend pass their Completion; the charge
        aborts the run the moment the cap is crossed, bounding overshoot to one call."""
        if completion is None:
            self.trace.append(StepRecord(step=step, summary=summary))
            return
        self.trace.append(
            StepRecord(
                step=step,
                summary=summary,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=completion.cost_usd,
            )
        )
        self.spent_usd += completion.cost_usd
        if self.spent_usd > self.budget_usd:
            raise BudgetExceeded(
                f"query spend ${self.spent_usd:.4f} exceeds cap ${self.budget_usd:.2f} "
                f"at step {step!r}"
            )


def run_loop[T](
    task: Callable[[RunContext], T | None],
    *,
    max_iters: int = MAX_ITERS,
    budget_usd: float = BUDGET_USD,
) -> tuple[T, RunContext]:
    """Call `task` until it returns an answer; returning None requests another iteration.

    The context comes back alongside the answer so callers can surface the trace and
    total cost with the response.
    """
    ctx = RunContext(budget_usd=budget_usd)
    for iteration in range(max_iters):
        ctx.iteration = iteration
        result = task(ctx)
        if result is not None:
            log.info(
                "loop done: %d iteration(s), %d step(s), $%.4f",
                iteration + 1,
                len(ctx.trace),
                ctx.spent_usd,
            )
            return result, ctx
    raise IterationsExhausted(f"no answer after {max_iters} iterations")
