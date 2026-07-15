from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from atlas_core.features import normalize_for_embedding
from atlas_core.models import Paper


@dataclass(frozen=True)
class EmbeddingContract:
    model_id: str
    dimension: int
    version: int


# Frozen contract: changing model, dimension, or format forces a full re-index.
CONTRACT = EmbeddingContract(model_id="BAAI/bge-small-en-v1.5", dimension=384, version=1)

# bge-small-en-v1.5 expects this prefix on queries and no prefix on passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def passage_text(paper: Paper) -> str:
    return normalize_for_embedding(f"{paper.title}. {paper.abstract}")


class Embedder(Protocol):
    """Returns an (n, CONTRACT.dimension) float32 array of L2-normalized vectors."""

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str = CONTRACT.model_id) -> None:
        # Heavy import (pulls torch); only installed via the "ingest" extra.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_id, device="cpu")

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        vectors = self._model.encode(
            texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)


class OnnxEmbedder:
    def __init__(self, model_path: str | Path, tokenizer_path: str | Path) -> None:
        # Query-time backend for Lambda: onnxruntime + a fast tokenizer, no torch. Runs an INT8
        # export of the same bge-small model, so it stays inside the frozen contract.
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=512)
        self._tokenizer.enable_padding()

    def embed(self, texts: list[str]) -> npt.NDArray[np.float32]:
        encodings = self._tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        # A model exported without token_type_ids would reject the extra feed key.
        feed = {name: value for name, value in feed.items() if name in self._input_names}
        last_hidden = self._session.run(None, feed)[0]
        cls = last_hidden[:, 0]  # bge uses the CLS token as the sentence embedding
        normalized = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        return np.asarray(normalized, dtype=np.float32)
