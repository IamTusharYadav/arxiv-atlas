from atlas_agents.harness import RunContext
from atlas_agents.steps.retriever import retrieve
from atlas_core.vectorstore import QdrantStore
from tests.conftest import FakeEmbedder, make_paper


def seed(store: QdrantStore, embedder: FakeEmbedder, n: int) -> None:
    papers = [make_paper(arxiv_id=f"2607.{i:05d}") for i in range(n)]
    vectors = embedder.embed([p.arxiv_id for p in papers])
    store.upsert([(p, v.tolist()) for p, v in zip(papers, vectors, strict=True)])


def test_retrieve_merges_and_dedupes_across_subqueries(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder, 6)
    ctx = RunContext()

    # The same subquery twice returns identical hits; dedupe keeps each paper once.
    candidates = retrieve(memory_store, fake_embedder, ctx, ["kv cache", "kv cache"], per_query=4)

    ids = [c.paper.arxiv_id for c in candidates]
    assert len(ids) == len(set(ids)) == 4
    # Sorted best-first.
    assert [c.score for c in candidates] == sorted((c.score for c in candidates), reverse=True)


def test_retrieve_is_free_and_traced(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    seed(memory_store, fake_embedder, 3)
    ctx = RunContext()
    retrieve(memory_store, fake_embedder, ctx, ["quantization"])
    assert ctx.spent_usd == 0
    assert ctx.trace[0].step == "retriever"
    assert "1 subqueries" in ctx.trace[0].summary
