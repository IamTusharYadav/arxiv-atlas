"""Retrieval tool: vector search over the corpus for each planned subquery.

No model call; the only cost is a Qdrant round trip per subquery. Candidates from
all subqueries are merged and deduplicated, keeping each paper's best score.
"""

from atlas_agents.harness import RunContext
from atlas_core.embedding import QUERY_PREFIX, Embedder
from atlas_core.vectorstore import ScoredPaper, VectorStore

PER_QUERY = 10


def retrieve(
    store: VectorStore,
    embedder: Embedder,
    ctx: RunContext,
    subqueries: list[str],
    per_query: int = PER_QUERY,
) -> list[ScoredPaper]:
    vectors = embedder.embed([QUERY_PREFIX + q for q in subqueries])
    best: dict[str, ScoredPaper] = {}
    for vector in vectors.tolist():
        for hit in store.search(vector, limit=per_query):
            seen = best.get(hit.paper.arxiv_id)
            if seen is None or hit.score > seen.score:
                best[hit.paper.arxiv_id] = hit
    candidates = sorted(best.values(), key=lambda s: s.score, reverse=True)
    ctx.record("retriever", f"{len(candidates)} candidates from {len(subqueries)} subqueries")
    return candidates
