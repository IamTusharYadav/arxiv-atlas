"""Aggregate judgements into per-dimension means and compare a run against the stored
baseline. The CI gate (commit 29) blocks a merge when a gated dimension regresses by more
than REGRESSION_DROP; comparison is on the aggregate, never per-query, so one noisy answer
cannot fail the gate."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from evals.judge import GATED_DIMS, Judgement

REGRESSION_DROP = 0.3
BASELINE_PATH = Path(__file__).parent / "baseline.json"
HISTORY_PATH = Path(__file__).parent / "history.json"


@dataclass(frozen=True)
class Aggregate:
    relevance: float
    faithfulness: float
    citation_correctness: float
    n: int


@dataclass(frozen=True)
class Comparison:
    current: Aggregate
    baseline: Aggregate | None
    regressions: list[str]  # gated dimensions that dropped more than REGRESSION_DROP

    @property
    def passed(self) -> bool:
        return not self.regressions


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 3)


def aggregate(judgements: list[Judgement]) -> Aggregate:
    if not judgements:
        raise ValueError("no judgements to aggregate")
    scores = [j.scores for j in judgements]
    return Aggregate(
        relevance=_mean([s.relevance for s in scores]),
        faithfulness=_mean([s.faithfulness for s in scores]),
        citation_correctness=_mean([s.citation_correctness for s in scores]),
        n=len(scores),
    )


def compare(current: Aggregate, baseline: Aggregate | None) -> Comparison:
    regressions = [
        dim
        for dim in GATED_DIMS
        if baseline is not None
        and getattr(baseline, dim) - getattr(current, dim) > REGRESSION_DROP
    ]
    return Comparison(current=current, baseline=baseline, regressions=regressions)


def load_baseline(path: Path = BASELINE_PATH) -> Aggregate | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Aggregate(**data)


def save_baseline(aggregate: Aggregate, path: Path = BASELINE_PATH) -> None:
    path.write_text(json.dumps(asdict(aggregate), indent=2) + "\n", encoding="utf-8")


def append_history(aggregate: Aggregate, path: Path = HISTORY_PATH) -> None:
    history = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    history.append({"timestamp": datetime.now(UTC).isoformat(), **asdict(aggregate)})
    path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
