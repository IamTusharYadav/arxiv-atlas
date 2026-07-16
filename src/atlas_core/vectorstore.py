import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, cast

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from atlas_core.config import Settings
from atlas_core.embedding import CONTRACT
from atlas_core.models import Edge, Paper

COLLECTION = "papers"

# Qdrant Cloud rejects request bodies over 32 MB; a 5k-point backfill window is ~49 MB.
# 512 points is ~5 MB, leaving headroom for payload growth.
UPSERT_BATCH = 512


@dataclass
class ScoredPaper:
    paper: Paper
    score: float


@dataclass
class StoredPaper:
    paper: Paper
    edges: list[Edge] = field(default_factory=list)


class VectorStore(Protocol):
    def ensure_collection(self) -> None: ...

    def upsert(self, items: Sequence[tuple[Paper, Sequence[float]]]) -> None: ...

    def search(self, vector: Sequence[float], limit: int) -> list[ScoredPaper]: ...

    def get(self, arxiv_ids: Sequence[str]) -> list[StoredPaper]: ...

    def get_vectors(self, arxiv_ids: Sequence[str]) -> dict[str, list[float]]: ...

    def set_edges(self, arxiv_id: str, edges: Sequence[Edge]) -> None: ...

    def iter_papers(self) -> Iterator[StoredPaper]: ...

    def latest_updated_at(self) -> datetime | None: ...

    def count(self) -> int: ...


def point_id(arxiv_id: str) -> str:
    """Deterministic point id so re-ingesting the same paper overwrites in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"arxiv:{arxiv_id}"))


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str = COLLECTION) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_settings(cls, settings: Settings) -> "QdrantStore":
        if settings.qdrant_url == ":memory:":
            return cls(QdrantClient(location=":memory:"))
        return cls(QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key))

    def ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=qm.VectorParams(
                    size=CONTRACT.dimension, distance=qm.Distance.COSINE
                ),
            )

    def upsert(self, items: Sequence[tuple[Paper, Sequence[float]]]) -> None:
        points = [
            qm.PointStruct(
                id=point_id(paper.arxiv_id),
                vector=list(vector),
                payload=paper.model_dump(mode="json"),
            )
            for paper, vector in items
        ]
        for start in range(0, len(points), UPSERT_BATCH):
            self._client.upsert(
                self._collection, points=points[start : start + UPSERT_BATCH], wait=True
            )

    def search(self, vector: Sequence[float], limit: int) -> list[ScoredPaper]:
        result = self._client.query_points(
            self._collection, query=list(vector), limit=limit, with_payload=True
        )
        return [
            ScoredPaper(paper=_paper_from_payload(p.payload), score=p.score)
            for p in result.points
            if p.payload is not None
        ]

    def get(self, arxiv_ids: Sequence[str]) -> list[StoredPaper]:
        """Papers and their edges by id, for the graph route. Missing ids are simply absent
        from the result; order is not guaranteed."""
        points = self._client.retrieve(
            self._collection, ids=[point_id(a) for a in arxiv_ids], with_payload=True
        )
        return [_stored_from_payload(p.payload) for p in points if p.payload is not None]

    def get_vectors(self, arxiv_ids: Sequence[str]) -> dict[str, list[float]]:
        """Stored embeddings by id, for clustering retrieved papers without re-embedding."""
        points = self._client.retrieve(
            self._collection,
            ids=[point_id(a) for a in arxiv_ids],
            with_payload=["arxiv_id"],
            with_vectors=True,
        )
        out: dict[str, list[float]] = {}
        for p in points:
            if p.payload is None or not isinstance(p.vector, list):
                continue
            # unnamed single vectors are flat; the list[list] case is multivector configs
            # this collection never uses
            out[str(p.payload["arxiv_id"])] = [float(x) for x in cast("list[float]", p.vector)]
        return out

    def set_edges(self, arxiv_id: str, edges: Sequence[Edge]) -> None:
        self._client.set_payload(
            self._collection,
            payload={"edges": [{"target": e.target, "weight": e.weight} for e in edges]},
            points=[point_id(arxiv_id)],
            wait=True,
        )

    def iter_papers(self) -> Iterator[StoredPaper]:
        offset = None
        while True:
            points, offset = self._client.scroll(
                self._collection, limit=512, offset=offset, with_payload=True, with_vectors=False
            )
            for point in points:
                if point.payload is not None:
                    yield _stored_from_payload(point.payload)
            if offset is None:
                return

    def latest_updated_at(self) -> datetime | None:
        # Full scan instead of a payload index; fine at nightly cadence and ~35k points.
        latest: datetime | None = None
        for stored in self.iter_papers():
            if latest is None or stored.paper.updated_at > latest:
                latest = stored.paper.updated_at
        return latest

    def count(self) -> int:
        return self._client.count(self._collection).count


def _paper_from_payload(payload: dict[str, Any]) -> Paper:
    return Paper.model_validate(payload)


def _stored_from_payload(payload: dict[str, Any]) -> StoredPaper:
    paper = _paper_from_payload(payload)
    edges = [
        Edge(source=paper.arxiv_id, target=e["target"], weight=e["weight"])
        for e in payload.get("edges", [])
    ]
    return StoredPaper(paper=paper, edges=edges)
