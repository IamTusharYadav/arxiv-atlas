import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from qdrant_client import QdrantClient

from atlas_core.embedding import CONTRACT
from atlas_core.models import Paper
from atlas_core.vectorstore import QdrantStore

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministic per-text vectors; no model download. Distinct texts land near-orthogonal
    in 384 dims, so no accidental edges appear in tests."""

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        out = np.empty((len(texts), CONTRACT.dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            vector = np.random.default_rng(seed).standard_normal(CONTRACT.dimension)
            out[i] = (vector / np.linalg.norm(vector)).astype(np.float32)
        return out


def make_paper(
    arxiv_id: str = "2607.00001",
    version: int = 1,
    title: str = "A Perfectly Plausible Paper About Language Models",
    updated_at: datetime | None = None,
    **overrides: object,
) -> Paper:
    fields: dict[str, object] = {
        "arxiv_id": arxiv_id,
        "version": version,
        "title": title,
        "abstract": "We study language models. " * 10,
        "authors": ["Ada Lovelace"],
        "categories": ["cs.LG"],
        "primary_category": "cs.LG",
        "published_at": datetime(2026, 7, 1, tzinfo=UTC),
        "updated_at": updated_at or datetime(2026, 7, 2, tzinfo=UTC),
    }
    fields.update(overrides)
    return Paper.model_validate(fields)


@pytest.fixture
def memory_store() -> QdrantStore:
    store = QdrantStore(QdrantClient(location=":memory:"))
    store.ensure_collection()
    return store


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
