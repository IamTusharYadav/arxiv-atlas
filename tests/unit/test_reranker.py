import json

from atlas_agents.harness import RunContext
from atlas_agents.steps.extractor import _ABSTRACT_CHARS as EXTRACTOR_CHARS
from atlas_agents.steps.reranker import _ABSTRACT_CHARS as RERANK_CHARS
from atlas_agents.steps.reranker import MAX_CANDIDATES, rerank
from atlas_core.vectorstore import ScoredPaper
from tests.conftest import make_bedrock_client, make_message, make_paper


def candidates(*ids: str) -> list[ScoredPaper]:
    return [ScoredPaper(paper=make_paper(arxiv_id=i), score=0.7) for i in ids]


def scores_json(scores: dict[str, int]) -> str:
    return json.dumps({"scores": [{"arxiv_id": k, "score": v} for k, v in scores.items()]})


def test_rerank_keeps_top_scorers_above_floor() -> None:
    client, fake = make_bedrock_client(
        [make_message(scores_json({"2607.00001": 9, "2607.00002": 3, "2607.00003": 7}))]
    )
    ctx = RunContext()

    kept = rerank(client, ctx, "kv cache?", candidates("2607.00001", "2607.00002", "2607.00003"))

    assert [s.paper.arxiv_id for s in kept] == ["2607.00001", "2607.00003"]  # 3 < floor of 5
    assert kept[0].score == 9.0
    assert ctx.trace[0].summary == "kept 2 of 3"
    assert ctx.spent_usd > 0
    # All candidates went out in one batched prompt.
    prompt = fake.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert prompt.count("<paper id=") == 3


def test_rerank_ignores_invented_ids_and_drops_unscored() -> None:
    client, _ = make_bedrock_client(
        [make_message(scores_json({"2607.00001": 8, "9999.99999": 10}))]
    )
    kept = rerank(client, RunContext(), "q", candidates("2607.00001", "2607.00002"))
    assert [s.paper.arxiv_id for s in kept] == ["2607.00001"]


def test_rerank_caps_at_keep() -> None:
    ids = [f"2607.{i:05d}" for i in range(10)]
    client, _ = make_bedrock_client([make_message(scores_json(dict.fromkeys(ids, 8)))])
    kept = rerank(client, RunContext(), "q", candidates(*ids), keep=8)
    assert len(kept) == 8


def test_rerank_sends_only_the_abstract_lead() -> None:
    # Reranking reads a trimmed lead, a strictly smaller window than the extractor; guards
    # against silently restoring the per-candidate token cost.
    assert RERANK_CHARS <= EXTRACTOR_CHARS // 2
    abstract = "LEAD_MARKER " + "x" * 2000 + " TAIL_MARKER"
    client, fake = make_bedrock_client([make_message(scores_json({"2607.00001": 7}))])
    rerank(client, RunContext(), "q", [ScoredPaper(paper=make_paper(abstract=abstract), score=0.7)])
    prompt = fake.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert "LEAD_MARKER" in prompt
    assert "TAIL_MARKER" not in prompt  # tail past the window never reaches the model


def test_rerank_caps_candidates_fed_to_the_model() -> None:
    # Late loop rounds widen retrieval past what the response token ceiling can score;
    # the cap keeps the best-first head so the output always fits max_tokens.
    ids = [f"2607.{i:05d}" for i in range(MAX_CANDIDATES + 15)]
    client, fake = make_bedrock_client([make_message(scores_json({ids[0]: 8}))])
    kept = rerank(client, RunContext(), "q", candidates(*ids))
    prompt = fake.calls[0]["messages"][0]["content"]  # type: ignore[index]
    assert prompt.count("<paper id=") == MAX_CANDIDATES
    assert [s.paper.arxiv_id for s in kept] == [ids[0]]


def test_rerank_empty_candidates_skips_model_call() -> None:
    client, fake = make_bedrock_client([])
    ctx = RunContext()
    assert rerank(client, ctx, "q", []) == []
    assert fake.calls == []
    assert ctx.trace[0].summary == "no candidates to score"
