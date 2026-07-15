"""Deselected by default (downloads ~130 MB on first run); run with
`uv run pytest -m model` (needs the ingest extra). The onnx parity test also needs an export
from scripts/export_onnx.py and the onnx extra; it skips if the artifact is absent."""

import os
from pathlib import Path

import numpy as np
import pytest

from atlas_core.embedding import CONTRACT, QUERY_PREFIX, OnnxEmbedder, SentenceTransformerEmbedder

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    pytest.importorskip("sentence_transformers")
    return SentenceTransformerEmbedder()


@pytest.fixture(scope="module")
def onnx_embedder() -> OnnxEmbedder:
    onnx_dir = Path(os.environ.get("ATLAS_ONNX_DIR", "models"))
    model_path = onnx_dir / "model_quantized.onnx"
    tokenizer_path = onnx_dir / "tokenizer.json"
    if not (model_path.exists() and tokenizer_path.exists()):
        pytest.skip(f"no exported onnx model under {onnx_dir}; run scripts/export_onnx.py")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    return OnnxEmbedder(model_path, tokenizer_path)


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


def test_onnx_backend_matches_torch(
    embedder: SentenceTransformerEmbedder, onnx_embedder: OnnxEmbedder
) -> None:
    texts = [
        QUERY_PREFIX + "How can the memory footprint of the KV cache be reduced?",
        "We propose a quantization scheme that compresses the key-value cache of large "
        "language models by 4x with negligible accuracy loss.",
        "We present a survey of crop rotation strategies in medieval European agriculture.",
    ]
    torch_vecs = embedder.embed(texts)
    onnx_vecs = onnx_embedder.embed(texts)
    assert onnx_vecs.shape == torch_vecs.shape == (3, CONTRACT.dimension)
    # Same text through both backends: INT8 adds only small noise, so cosine stays near 1.
    per_text_cos = np.sum(torch_vecs * onnx_vecs, axis=1)
    assert per_text_cos.min() > 0.98
    # The cross-backend retrieval decision (onnx query vs torch-built passages) is unchanged, so
    # the corpus embedded during ingestion never needs re-embedding for the query backend.
    q = onnx_vecs[0]
    assert float(q @ torch_vecs[1]) > float(q @ torch_vecs[2])
