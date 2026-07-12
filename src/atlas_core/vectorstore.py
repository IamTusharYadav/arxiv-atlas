import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from atlas_core.config import Settings
from atlas_core.embedding import CONTRACT
from atlas_core.models import Edge, Paper

COLLECTION = "papers"


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
        if points:
            self._client.upsert(self._collection, points=points, wait=True)

    def search(self, vector: Sequence[float], limit: int) -> list[ScoredPaper]:
        result = self._client.query_points(
            self._collection, query=list(vector), limit=limit, with_payload=True
        )
        return [
            ScoredPaper(paper=_paper_from_payload(p.payload), score=p.score)
            for p in result.points
            if p.payload is not None
        ]

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
                if point.payload is None:
                    continue
                paper = _paper_from_payload(point.payload)
                edges = [
                    Edge(source=paper.arxiv_id, target=e["target"], weight=e["weight"])
                    for e in point.payload.get("edges", [])
                ]
                yield StoredPaper(paper=paper, edges=edges)
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
