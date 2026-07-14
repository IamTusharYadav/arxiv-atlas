import json
from pathlib import Path

from evals.baseline import (
    Aggregate,
    aggregate,
    append_history,
    compare,
    load_baseline,
    save_baseline,
)
from evals.judge import Judgement, Scores


def judgement(relevance: int, faithfulness: int, citation: int) -> Judgement:
    scores = Scores(
        relevance=relevance,
        faithfulness=faithfulness,
        citation_correctness=citation,
        rationale="x",
    )
    return Judgement(query_id="q", scores=scores, cost_usd=0.0)


def test_aggregate_means_per_dimension() -> None:
    agg = aggregate([judgement(4, 5, 3), judgement(2, 5, 5)])
    assert agg.relevance == 3.0
    assert agg.faithfulness == 5.0
    assert agg.citation_correctness == 4.0
    assert agg.n == 2


def test_no_baseline_passes() -> None:
    result = compare(aggregate([judgement(3, 3, 3)]), None)
    assert result.passed
    assert result.regressions == []


def test_gated_regression_fails() -> None:
    baseline = Aggregate(relevance=4.5, faithfulness=4.5, citation_correctness=4.5, n=10)
    current = aggregate([judgement(4, 4, 4)])  # 0.5 drop on every dimension
    result = compare(current, baseline)
    assert not result.passed
    assert set(result.regressions) == {"relevance", "faithfulness"}


def test_citation_drop_alone_does_not_gate() -> None:
    baseline = Aggregate(relevance=4.0, faithfulness=4.0, citation_correctness=5.0, n=10)
    current = aggregate([judgement(4, 4, 1)])  # only citation regressed
    assert compare(current, baseline).passed


def test_small_dip_within_tolerance_passes() -> None:
    baseline = Aggregate(relevance=4.2, faithfulness=4.2, citation_correctness=4.0, n=10)
    current = aggregate([judgement(4, 4, 4)])  # 0.2 drop, under the 0.3 threshold
    assert compare(current, baseline).passed


def test_baseline_roundtrip_and_history_append(tmp_path: Path) -> None:
    agg = Aggregate(relevance=4.0, faithfulness=4.5, citation_correctness=3.5, n=15)
    baseline_path = tmp_path / "baseline.json"
    save_baseline(agg, baseline_path)
    assert load_baseline(baseline_path) == agg

    history_path = tmp_path / "history.json"
    append_history(agg, history_path)
    append_history(agg, history_path)
    history = json.loads(history_path.read_text())
    assert len(history) == 2
    assert history[0]["relevance"] == 4.0
    assert "timestamp" in history[0]


def test_load_missing_baseline_returns_none(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "nope.json") is None
