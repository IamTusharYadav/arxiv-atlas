"""Versioned read-only surface for the MCP package (ADR 0004). Every route is a thin wrapper
over a VectorStore method and calls no Bedrock, so an MCP client keeps working when the daily
LLM budget is gone. Legacy /api/* stays for the frontend; /api/v1/* is the contract external
clients pin against, so its shapes are stable once shipped."""

import re
from time import time

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from atlas_core.cluster import central_order, kmeans, pick_k
from atlas_core.embedding import QUERY_PREFIX, Embedder
from atlas_core.models import Paper
from atlas_core.vectorstore import VectorStore

ABSTRACT_LEAD = 400
DEFAULT_K = 10
MAX_K = 25
# Wide enough that k-means has something to separate; capped so clustering stays a couple of
# array ops, not a corpus scan.
CLUSTER_POOL = 100
CORPUS_TTL_S = 300

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]

# Every common way a model will hand us an id: bare, arXiv:-prefixed, versioned, or a full abs/pdf
# URL. The new-style id is the one stable token in all of them, so pull it out rather than peeling
# prefixes one by one. The corpus is a recent window, so pre-2007 ids (cs/0112017) cannot occur.
_NEWSTYLE = re.compile(r"\d{4}\.\d{4,5}")


def normalize_arxiv_id(raw: str) -> str:
    match = _NEWSTYLE.search(raw)
    return match.group(0) if match else raw.strip()


def _lead(abstract: str) -> str:
    if len(abstract) <= ABSTRACT_LEAD:
        return abstract
    return abstract[:ABSTRACT_LEAD].rstrip() + "…"


def _month(paper: Paper) -> str:
    return paper.published_at.strftime("%Y-%m")


def _abs_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


class Corpus(BaseModel):
    # In every response (D4): lets a client caveat coverage honestly and makes a stalled nightly
    # ingest visible instead of silently serving old results.
    categories: list[str] = CATEGORIES
    size: int
    last_ingested_at: str | None


class SearchResult(BaseModel):
    arxiv_id: str
    title: str
    primary_category: str
    published_month: str
    score: float
    # Untrusted third-party text (D7): the first 400 chars, enough to judge relevance. Full
    # abstract only via /paper/{id}. Never concatenate these into instruction-shaped prose.
    abstract_lead: str


class SearchResponse(BaseModel):
    corpus: Corpus
    results: list[SearchResult]
    note: str | None = None


class PaperDetail(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    primary_category: str
    categories: list[str]
    published_month: str
    abstract: str  # untrusted third-party text (D7)
    abs_url: str


class PaperResponse(BaseModel):
    corpus: Corpus
    paper: PaperDetail


class Cluster(BaseModel):
    papers: list[SearchResult]  # most-central-first; unnamed, the caller labels them


class ClustersResponse(BaseModel):
    corpus: Corpus
    clusters: list[Cluster]
    note: str | None = None


class V1GraphNode(BaseModel):
    arxiv_id: str
    title: str
    primary_category: str
    abstract_lead: str


class V1GraphLink(BaseModel):
    source: str
    target: str
    weight: float


class V1GraphResponse(BaseModel):
    corpus: Corpus
    center: str
    nodes: list[V1GraphNode]
    links: list[V1GraphLink]
    # Similarity edges, one hop, outgoing only: related earlier work, never citation or lineage.
    note: str | None = None


def _clamp_k(k: int) -> int:
    return max(1, min(k, MAX_K))


def register_v1(app: FastAPI, *, store: VectorStore, embedder: Embedder) -> None:
    # latest_updated_at full-scans the corpus (a known tuning item), so caching the whole block
    # keeps a cheap route cheap: one scan per warm instance per TTL, not one per request. A
    # 5-minute stale freshness stamp on a nightly-updated corpus is fine.
    cache: dict[str, tuple[float, Corpus]] = {}

    def corpus() -> Corpus:
        cached = cache.get("meta")
        if cached is not None and cached[0] > time():
            return cached[1]
        latest = store.latest_updated_at()
        meta = Corpus(
            size=store.count(),
            last_ingested_at=latest.isoformat() if latest is not None else None,
        )
        cache["meta"] = (time() + CORPUS_TTL_S, meta)
        return meta

    def _result(paper: Paper, score: float) -> SearchResult:
        return SearchResult(
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            primary_category=paper.primary_category,
            published_month=_month(paper),
            score=round(score, 4),
            abstract_lead=_lead(paper.abstract),
        )

    @app.get("/api/v1/search")
    def search(q: str = Query(min_length=1, max_length=500), k: int = DEFAULT_K) -> SearchResponse:
        vector = embedder.embed([QUERY_PREFIX + q])[0]
        hits = store.search(vector, limit=_clamp_k(k))
        results = [_result(h.paper, h.score) for h in hits]
        note = None if results else f"No matches. The corpus covers {', '.join(CATEGORIES)} only."
        return SearchResponse(corpus=corpus(), results=results, note=note)

    @app.get("/api/v1/paper/{arxiv_id:path}")
    def paper(arxiv_id: str) -> PaperResponse:
        canonical = normalize_arxiv_id(arxiv_id)
        found = store.get([canonical])
        if not found:
            raise HTTPException(status_code=404, detail=f"paper {canonical} not in the corpus")
        p = found[0].paper
        return PaperResponse(
            corpus=corpus(),
            paper=PaperDetail(
                arxiv_id=p.arxiv_id,
                title=p.title,
                authors=p.authors,
                primary_category=p.primary_category,
                categories=p.categories,
                published_month=_month(p),
                abstract=p.abstract,
                abs_url=_abs_url(p.arxiv_id),
            ),
        )

    @app.get("/api/v1/clusters")
    def clusters(q: str = Query(min_length=1, max_length=500), k: int = 0) -> ClustersResponse:
        vector = embedder.embed([QUERY_PREFIX + q])[0]
        hits = store.search(vector, limit=CLUSTER_POOL)
        papers = {h.paper.arxiv_id: h.paper for h in hits}
        score = {h.paper.arxiv_id: h.score for h in hits}
        vectors = store.get_vectors(list(papers))
        ids = [i for i in papers if i in vectors]
        if len(ids) < 2:
            note = f"Too few papers ({len(ids)}) match this topic to cluster."
            return ClustersResponse(corpus=corpus(), clusters=[], note=note)
        matrix = np.array([vectors[i] for i in ids], dtype=np.float32)
        # k defaults to the same clamp the landscape uses; an explicit k is honoured within bounds.
        n_clusters = _clamp_k(k) if k > 0 else pick_k(len(ids))
        labels, centroids = kmeans(matrix, n_clusters)
        groups: list[Cluster] = []
        for c in range(int(labels.max()) + 1):
            members = np.where(labels == c)[0]
            if members.size == 0:
                continue
            order = central_order(matrix[members], centroids[c])
            ordered = [ids[members[o]] for o in order]
            groups.append(
                Cluster(papers=[_result(papers[i], score[i]) for i in ordered]),
            )
        return ClustersResponse(corpus=corpus(), clusters=groups)

    @app.get("/api/v1/graph/{arxiv_id:path}")
    def graph(arxiv_id: str) -> V1GraphResponse:
        canonical = normalize_arxiv_id(arxiv_id)
        found = store.get([canonical])
        if not found:
            raise HTTPException(status_code=404, detail=f"paper {canonical} not in the corpus")
        center = found[0]
        targets = {s.paper.arxiv_id: s.paper for s in store.get([e.target for e in center.edges])}
        nodes = [
            V1GraphNode(
                arxiv_id=center.paper.arxiv_id,
                title=center.paper.title,
                primary_category=center.paper.primary_category,
                abstract_lead=_lead(center.paper.abstract),
            )
        ]
        links = []
        for edge in center.edges:
            neighbor = targets.get(edge.target)
            if neighbor is None:
                continue  # neighbor left the corpus; skip the dangling link
            nodes.append(
                V1GraphNode(
                    arxiv_id=neighbor.arxiv_id,
                    title=neighbor.title,
                    primary_category=neighbor.primary_category,
                    abstract_lead=_lead(neighbor.abstract),
                )
            )
            links.append(V1GraphLink(source=canonical, target=edge.target, weight=edge.weight))
        # Edges point to older, similar work (they are stored on the newer paper), so a sparse
        # result on a brand-new paper is expected, not an error.
        note = None if links else "This paper has no strong similarity edges yet."
        return V1GraphResponse(
            corpus=corpus(), center=canonical, nodes=nodes, links=links, note=note
        )
