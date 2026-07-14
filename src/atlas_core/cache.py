import logging
from dataclasses import dataclass
from time import time
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

log = logging.getLogger(__name__)

DEFAULT_TTL_S = 24 * 3600
# Embeddings are L2-normalized, so this is a cosine floor; near-duplicate questions only.
SIMILARITY_FLOOR = 0.97


@dataclass(frozen=True)
class CacheEntry:
    embedding: list[float]
    payload: dict[str, Any]  # the caller's serialized answer; opaque to the cache
    expires_at: float


class CacheStore(Protocol):
    def recent(self) -> list[CacheEntry]: ...
    def put(self, entry: CacheEntry) -> None: ...


class ResponseCache:
    def __init__(
        self,
        store: CacheStore,
        *,
        similarity_floor: float = SIMILARITY_FLOOR,
        ttl_s: int = DEFAULT_TTL_S,
    ) -> None:
        self._store = store
        self._floor = similarity_floor
        self._ttl_s = ttl_s

    def get(self, embedding: npt.NDArray[np.float32]) -> dict[str, Any] | None:
        try:
            candidates = self._store.recent()
        except Exception as err:
            log.warning("cache lookup failed, treating as miss: %s", err)
            return None
        # Re-check expiry here: DynamoDB TTL deletion lags up to 48h, so a "recent" row
        # may already be stale. ponytail: linear scan, fine at demo volume; add a vector
        # index if the cache ever grows past a few hundred live entries.
        now = time()
        best: CacheEntry | None = None
        best_sim = self._floor
        for entry in candidates:
            if entry.expires_at <= now:
                continue
            sim = float(np.dot(embedding, np.asarray(entry.embedding, dtype=np.float32)))
            if sim >= best_sim:
                best, best_sim = entry, sim
        return best.payload if best is not None else None

    def put(self, embedding: npt.NDArray[np.float32], payload: dict[str, Any]) -> None:
        entry = CacheEntry(
            embedding=[float(x) for x in embedding],
            payload=payload,
            expires_at=time() + self._ttl_s,
        )
        try:
            self._store.put(entry)
        except Exception as err:
            log.warning("cache write failed, response not cached: %s", err)
