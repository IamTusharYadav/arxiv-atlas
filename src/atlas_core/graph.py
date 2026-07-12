from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atlas_core.models import Edge
from atlas_core.vectorstore import VectorStore

TOP_K = 10
MIN_SIMILARITY = 0.62

_SCHEMA = pa.schema([("source", pa.string()), ("target", pa.string()), ("weight", pa.float32())])


def link_paper(
    store: VectorStore,
    arxiv_id: str,
    vector: list[float],
    top_k: int = TOP_K,
    min_similarity: float = MIN_SIMILARITY,
) -> list[Edge]:
    """Nearest neighbors above the similarity floor, excluding the paper itself.

    Edges are stored one-directionally on the newer paper; the adjacency artifact treats
    them as undirected at read time, so older papers' payloads are never rewritten.
    """
    hits = store.search(vector, limit=top_k + 1)
    return [
        Edge(source=arxiv_id, target=hit.paper.arxiv_id, weight=round(hit.score, 4))
        for hit in hits
        if hit.paper.arxiv_id != arxiv_id and hit.score >= min_similarity
    ][:top_k]


def write_adjacency(store: VectorStore, path: Path) -> int:
    """Rebuild the compact edge artifact from the full corpus. Returns the edge count."""
    sources: list[str] = []
    targets: list[str] = []
    weights: list[float] = []
    for stored in store.iter_papers():
        for edge in stored.edges:
            sources.append(edge.source)
            targets.append(edge.target)
            weights.append(edge.weight)
    table = pa.table(
        {
            "source": pa.array(sources, pa.string()),
            "target": pa.array(targets, pa.string()),
            "weight": pa.array(weights, pa.float32()),
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, path)
    return int(table.num_rows)


def read_adjacency(path: Path) -> list[Edge]:
    table = pq.read_table(path)
    return [
        Edge(source=s, target=t, weight=w)
        for s, t, w in zip(
            table["source"].to_pylist(),
            table["target"].to_pylist(),
            table["weight"].to_pylist(),
            strict=True,
        )
    ]
