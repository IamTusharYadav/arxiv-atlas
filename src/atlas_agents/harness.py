"""Generic agent loop: iteration cap, per-query budget cap, per-step trace. Framework-free by
ADR 0002 (a plain function-call cycle); it owns termination and budget accounting, not the
steps themselves."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

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


class StepSink(Protocol):
    """Where each trace entry also goes for external observability (Langfuse). `model` and
    `version` (the prompt registry tag) are attached to the span; steps without a model call
    pass neither. Implementations must never raise: tracing is fire-and-forget."""

    def step(self, record: StepRecord, *, model: str | None, version: str | None) -> None: ...


class _NullSink:
    def step(self, record: StepRecord, *, model: str | None, version: str | None) -> None:
        pass


_NULL_SINK: StepSink = _NullSink()


@dataclass
class RunContext:
    """Handed to the task each iteration; steps report spend and trace through it."""

    budget_usd: float = BUDGET_USD
    iteration: int = 0
    spent_usd: float = 0.0
    trace: list[StepRecord] = field(default_factory=list)
    sink: StepSink = _NULL_SINK

    def record(
        self,
        step: str,
        summary: str,
        completion: Completion | None = None,
        *,
        model: str | None = None,
        version: str | None = None,
    ) -> None:
        """Append a trace entry and mirror it to the sink. Steps with model spend pass their
        Completion; the charge aborts the run the moment the cap is crossed, bounding
        overshoot to one call. The span is emitted before the charge so an aborting step is
        still traced."""
        if completion is None:
            record = StepRecord(step=step, summary=summary)
        else:
            record = StepRecord(
                step=step,
                summary=summary,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=completion.cost_usd,
            )
        self.trace.append(record)
        self.sink.step(record, model=model, version=version)
        if completion is None:
            return
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
    sink: StepSink = _NULL_SINK,
) -> tuple[T, RunContext]:
    """Call `task` until it returns an answer; returning None requests another iteration.

    The context comes back alongside the answer so callers can surface the trace and
    total cost with the response.
    """
    ctx = RunContext(budget_usd=budget_usd, sink=sink)
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
