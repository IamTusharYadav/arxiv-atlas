"""Contract tests against the real embedding model. Downloads ~130 MB on first run, so they
are deselected by default; run with `uv run pytest -m model` (needs the ingest extra)."""

import numpy as np
import pytest

from atlas_core.embedding import CONTRACT, QUERY_PREFIX, SentenceTransformerEmbedder

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    pytest.importorskip("sentence_transformers")
    return SentenceTransformerEmbedder()


def test_contract_shape_and_normalization(embedder: SentenceTransformerEmbedder) -> None:
    vectors = embedder.embed(["KV cache compression for transformers", "Medieval castles"])
    assert vectors.shape == (2, CONTRACT.dimension)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)


def test_semantic_rank_order(embedder: SentenceTransformerEmbedder) -> None:
    query, related, unrelated = embedder.embed(
        [
            QUERY_PREFIX + "How can the memory footprint of the KV cache be reduced?",
            "We propose a quantization scheme that compresses the key-value cache of large "
            "language models by 4x with negligible accuracy loss.",
            "We present a survey of crop rotation strategies in medieval European agriculture.",
        ]
    )
    assert float(query @ related) > float(query @ unrelated)
