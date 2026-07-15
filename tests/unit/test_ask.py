import json

import pytest

from atlas_agents.ask import ask
from atlas_agents.bedrock import SONNET
from atlas_agents.harness import BudgetExceeded
from atlas_core.vectorstore import QdrantStore
from tests.conftest import FakeEmbedder, make_bedrock_client, make_message, make_paper

IDS = ["2607.00001", "2607.00002", "2607.00003"]


def seed(store: QdrantStore, embedder: FakeEmbedder) -> None:
    papers = [make_paper(arxiv_id=i) for i in IDS]
    vectors = embedder.embed([p.arxiv_id for p in papers])
    store.upsert([(p, v.tolist()) for p, v in zip(papers, vectors, strict=True)])


def plan_json() -> str:
    return json.dumps(
        {
            "in_scope": True,
            "subqueries": ["kv cache quantization"],
            "stop_criterion": "one paper per method family",
            "scope_note": "",
        }
    )


def rerank_json(scores: dict[str, int]) -> str:
    return json.dumps({"scores": [{"arxiv_id": k, "score": v} for k, v in scores.items()]})


def extract_json(ids: list[str]) -> str:
    return json.dumps({"papers": [{"arxiv_id": i, "claims": [f"claim from {i}"]} for i in ids]})


def check_json(sufficient: bool, refined: list[str] | None = None) -> str:
    return json.dumps({"sufficient": sufficient, "refined_subqueries": refined or []})


def test_ask_single_round_happy_path(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    brief_md = f"Quantization leads [{IDS[0]}]."
    client, fake = make_bedrock_client(
        [
            make_message(plan_json()),
            make_message(rerank_json({IDS[0]: 9, IDS[1]: 7, IDS[2]: 2})),
            make_message(extract_json([IDS[0], IDS[1]])),
            make_message(check_json(sufficient=True)),
            make_message(brief_md, model=SONNET),
        ]
    )

    answer = ask("kv cache?", client=client, store=memory_store, embedder=fake_embedder)

    assert answer.brief == brief_md
    assert {s.paper.arxiv_id for s in answer.papers} == {IDS[0], IDS[1]}
    assert answer.cost_usd > 0
    assert [r.step for r in answer.trace] == [
        "planner",
        "retriever",
        "reranker",
        "extractor",
        "check",
        "synthesizer",
    ]
    # Synthesis ran on Sonnet, everything else on Haiku.
    assert fake.calls[-1]["model"] == SONNET
    assert all("haiku" in str(c["model"]) for c in fake.calls[:-1])


def test_ask_out_of_scope_declines_without_search(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    plan = json.dumps(
        {
            "in_scope": False,
            "subqueries": [],
            "stop_criterion": "",
            "scope_note": "Gene editing lives outside this corpus.",
        }
    )
    client, fake = make_bedrock_client([make_message(plan)])

    answer = ask("CRISPR advances?", client=client, store=memory_store, embedder=fake_embedder)

    assert "cs.AI, cs.LG and cs.CL only" in answer.brief
    assert "Gene editing" in answer.brief
    assert answer.papers == []
    assert len(fake.calls) == 1  # planner only; no retrieval, no Sonnet


def test_ask_insufficient_evidence_runs_second_round(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    brief_md = f"Both families covered [{IDS[0]}] [{IDS[2]}]."
    client, _ = make_bedrock_client(
        [
            make_message(plan_json()),
            make_message(rerank_json({IDS[0]: 9})),
            make_message(extract_json([IDS[0]])),
            make_message(check_json(sufficient=False, refined=["token eviction"])),
            make_message(rerank_json({IDS[2]: 8})),
            make_message(extract_json([IDS[2]])),
            make_message(check_json(sufficient=True)),
            make_message(brief_md, model=SONNET),
        ]
    )

    answer = ask("kv cache?", client=client, store=memory_store, embedder=fake_embedder)

    assert answer.brief == brief_md
    assert {s.paper.arxiv_id for s in answer.papers} == {IDS[0], IDS[2]}
    assert [r.step for r in answer.trace].count("retriever") == 2


def test_ask_returns_gathered_evidence_when_the_cap_fires(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # Extraction lands, then the check call blows the per-query cap. The claims already paid
    # for come back as a partial answer rather than being thrown away.
    client, fake = make_bedrock_client(
        [
            make_message(plan_json()),
            make_message(rerank_json({IDS[0]: 9, IDS[1]: 7})),
            make_message(extract_json([IDS[0], IDS[1]])),
            make_message(check_json(sufficient=False), output_tokens=5_000_000),
        ]
    )

    answer = ask("kv cache?", client=client, store=memory_store, embedder=fake_embedder)

    assert answer.partial is True
    assert "stopped early" in answer.brief
    assert f"claim from {IDS[0]}" in answer.brief
    assert f"[{IDS[0]}]" in answer.brief  # findings stay attributed to their paper
    assert {s.paper.arxiv_id for s in answer.papers} == {IDS[0], IDS[1]}
    assert answer.cost_usd > 0
    assert [r.step for r in answer.trace][:3] == ["planner", "retriever", "reranker"]
    # The budget is what ran out, so assembling the partial must not call the model again.
    assert len(fake.calls) == 4
    assert all("haiku" in str(c["model"]) for c in fake.calls)


def test_ask_cap_with_no_evidence_still_raises(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    # The cap fires before anything was gathered; there is no partial answer to give, so the
    # honest outcome is the error.
    client, _ = make_bedrock_client([make_message(plan_json(), output_tokens=5_000_000)])
    with pytest.raises(BudgetExceeded):
        ask("kv cache?", client=client, store=memory_store, embedder=fake_embedder)


def test_ask_no_relevant_papers_declines_gracefully(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder)
    client, fake = make_bedrock_client(
        [
            make_message(plan_json()),
            make_message(rerank_json({})),  # round 1: nothing survives
            make_message(rerank_json({})),  # round 2 (last): still nothing
        ]
    )

    answer = ask(
        "kv cache?", client=client, store=memory_store, embedder=fake_embedder, max_iters=2
    )

    assert "No papers" in answer.brief
    assert answer.papers == []
    # No Sonnet call was made for the decline.
    assert all("haiku" in str(c["model"]) for c in fake.calls)
