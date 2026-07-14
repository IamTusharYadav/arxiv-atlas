from evals.golden_set import load_golden


def test_golden_set_loads_and_ids_match_filenames() -> None:
    queries = load_golden()
    assert len(queries) >= 15
    for q in queries:
        assert q.id and q.question
        assert q.category


def test_set_contains_scope_and_adversarial_cases() -> None:
    by_id = {q.id: q for q in load_golden()}
    assert by_id["out-of-scope-crispr"].in_scope is False
    assert "injection-in-abstracts" in by_id
