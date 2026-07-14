import logging
from math import ceil
from time import time
from typing import Protocol

log = logging.getLogger(__name__)

# Coarse abuse stop for anonymous traffic; the fail-closed budget guard is the real spend cap.
DEFAULT_RATE_PER_S = 1 / 2
DEFAULT_BURST = 15


class BucketStore(Protocol):
    def get(self, key: str) -> tuple[float, float] | None: ...
    def set(self, key: str, tokens: float, updated_at: float) -> None: ...


class RateLimiter:
    def __init__(
        self,
        store: BucketStore,
        *,
        rate_per_s: float = DEFAULT_RATE_PER_S,
        burst: int = DEFAULT_BURST,
    ) -> None:
        self._store = store
        self._rate = rate_per_s
        self._burst = burst

    def allow(self, key: str) -> int:
        """Consume a token for `key`; return 0 if allowed, else seconds until the next token."""
        now = time()
        try:
            state = self._store.get(key)
            if state is None:
                tokens = float(self._burst)
            else:
                tokens = min(self._burst, state[0] + (now - state[1]) * self._rate)
            if tokens >= 1:
                self._store.set(key, tokens - 1, now)
                return 0
        except Exception as err:
            # Fail open: the budget guard is the hard cap, so a store blip shouldn't deny everyone.
            log.warning("rate-limit store unavailable, allowing request: %s", err)
            return 0
        # Denied, no write: the next call refills from the same timestamp to the same count anyway.
        return ceil((1 - tokens) / self._rate)
