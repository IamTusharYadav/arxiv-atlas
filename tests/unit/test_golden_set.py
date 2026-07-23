from evals.golden_set import load_golden, subset


def test_golden_set_loads_and_ids_match_filenames() -> None:
    queries = load_golden()
    assert len(queries) >= 30
    for q in queries:
        assert q.id and q.question
        assert q.category


def test_set_contains_scope_and_adversarial_cases() -> None:
    by_id = {q.id: q for q in load_golden()}
    assert by_id["out-of-scope-crispr"].in_scope is False
    assert by_id["out-of-scope-pre-window"].in_scope is False
    assert "injection-in-abstracts" in by_id


def test_subset_keeps_every_category() -> None:
    queries = load_golden()
    picked = subset(queries, 15)
    assert len(picked) == 15
    assert {q.category for q in picked} == {q.category for q in queries}
    assert picked == subset(queries, 15)  # deterministic, the gate compares runs over time


def test_subset_larger_than_the_set_returns_everything() -> None:
    queries = load_golden()
    assert len(subset(queries, 500)) == len(queries)
