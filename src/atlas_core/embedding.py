from dataclasses import dataclass
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
