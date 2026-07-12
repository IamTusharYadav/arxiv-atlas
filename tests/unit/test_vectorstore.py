from datetime import UTC, datetime

from atlas_core.vectorstore import QdrantStore
from tests.conftest import FakeEmbedder, make_paper


def test_upsert_is_idempotent(memory_store: QdrantStore, fake_embedder: FakeEmbedder) -> None:
    paper = make_paper(arxiv_id="2607.00001")
    vector = fake_embedder.embed([paper.abstract])[0].tolist()
    memory_store.upsert([(paper, vector)])
    memory_store.upsert([(paper, vector)])
    assert memory_store.count() == 1


def test_new_version_overwrites_in_place(
    memory_store: QdrantStore, fake_embedder: FakeEmbedder
) -> None:
    v1 = make_paper(arxiv_id="2607.00001", version=1)
    v2 = make_paper(arxiv_id="2607.00001", version=2)
    vectors = fake_embedder.embed([v1.abstract, v2.abstract])
    memory_store.upsert([(v1, vectors[0].tolist())])
    memory_store.upsert([(v2, vectors[1].tolist())])

    assert memory_store.count() == 1
    stored = next(memory_store.iter_papers())
    assert stored.paper.version == 2


def test_latest_updated_at(memory_store: QdrantStore, fake_embedder: FakeEmbedder) -> None:
    assert memory_store.latest_updated_at() is None
    papers = [
        make_paper(arxiv_id="2607.00001", updated_at=datetime(2026, 7, 1, tzinfo=UTC)),
        make_paper(arxiv_id="2607.00002", updated_at=datetime(2026, 7, 5, tzinfo=UTC)),
        make_paper(arxiv_id="2607.00003", updated_at=datetime(2026, 7, 3, tzinfo=UTC)),
    ]
    vectors = fake_embedder.embed([p.arxiv_id for p in papers])
    memory_store.upsert([(paper, vec.tolist()) for paper, vec in zip(papers, vectors, strict=True)])
    assert memory_store.latest_updated_at() == datetime(2026, 7, 5, tzinfo=UTC)
