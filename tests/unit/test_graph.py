import math
from pathlib import Path

from atlas_core.embedding import CONTRACT
from atlas_core.graph import link_paper, read_adjacency, write_adjacency
from atlas_core.models import Edge
from atlas_core.vectorstore import QdrantStore
from tests.conftest import make_paper


def unit_vector(x: float, y: float) -> list[float]:
    vector = [0.0] * CONTRACT.dimension
    norm = math.hypot(x, y)
    vector[0] = x / norm
    vector[1] = y / norm
    return vector


def seed_geometry(store: QdrantStore) -> list[float]:
    """p0 along the x axis; p1 identical, p2 at cosine 0.7, p3 orthogonal."""
    base = unit_vector(1.0, 0.0)
    store.upsert(
        [
            (make_paper(arxiv_id="p0", title="Base Paper on Retrieval Methods"), base),
            (
                make_paper(arxiv_id="p1", title="Identical Twin of the Base Paper"),
                unit_vector(1.0, 0.0),
            ),
            (
                make_paper(arxiv_id="p2", title="A Related Paper About Retrieval"),
                unit_vector(0.7, math.sqrt(1 - 0.49)),
            ),
            (
                make_paper(arxiv_id="p3", title="Something Orthogonal Entirely"),
                unit_vector(0.0, 1.0),
            ),
        ]
    )
    return base


def test_link_paper_applies_threshold_and_excludes_self(memory_store: QdrantStore) -> None:
    base = seed_geometry(memory_store)
    edges = link_paper(memory_store, "p0", base)
    assert [e.target for e in edges] == ["p1", "p2"]
    assert all(e.source == "p0" for e in edges)
    assert edges[0].weight >= 0.99
    assert 0.65 <= edges[1].weight <= 0.75


def test_link_paper_respects_top_k(memory_store: QdrantStore) -> None:
    base = seed_geometry(memory_store)
    edges = link_paper(memory_store, "p0", base, top_k=1)
    assert [e.target for e in edges] == ["p1"]


def test_adjacency_roundtrip(memory_store: QdrantStore, tmp_path: Path) -> None:
    base = seed_geometry(memory_store)
    memory_store.set_edges("p0", link_paper(memory_store, "p0", base))
    path = tmp_path / "adjacency.parquet"

    count = write_adjacency(memory_store, path)
    edges = read_adjacency(path)
    assert count == 2
    assert {(e.source, e.target) for e in edges} == {("p0", "p1"), ("p0", "p2")}
    assert all(isinstance(e, Edge) for e in edges)


def test_empty_corpus_writes_empty_artifact(memory_store: QdrantStore, tmp_path: Path) -> None:
    path = tmp_path / "adjacency.parquet"
    assert write_adjacency(memory_store, path) == 0
    assert read_adjacency(path) == []
