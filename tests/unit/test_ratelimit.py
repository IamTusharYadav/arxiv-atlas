from time import time

from atlas_core.ratelimit import RateLimiter


class MemoryBuckets:
    """In-memory stand-in for the DynamoDB bucket table; `fail` simulates it being unreachable."""

    def __init__(self) -> None:
        self.buckets: dict[str, tuple[float, float]] = {}
        self.fail = False

    def get(self, key: str) -> tuple[float, float] | None:
        if self.fail:
            raise RuntimeError("dynamodb unreachable")
        return self.buckets.get(key)

    def set(self, key: str, tokens: float, updated_at: float) -> None:
        if self.fail:
            raise RuntimeError("dynamodb unreachable")
        self.buckets[key] = (tokens, updated_at)


def test_burst_allows_then_denies() -> None:
    limiter = RateLimiter(MemoryBuckets(), rate_per_s=1 / 6, burst=3)
    assert [limiter.allow("ip") for _ in range(3)] == [0, 0, 0]
    assert limiter.allow("ip") > 0  # fourth request denied with a positive retry-after


def test_refill_after_elapsed_time() -> None:
    store = MemoryBuckets()
    store.buckets["ip"] = (0.0, time() - 12)  # emptied 12s ago; at 1/6/s that is ~2 tokens back
    limiter = RateLimiter(store, rate_per_s=1 / 6, burst=5)
    assert limiter.allow("ip") == 0


def test_separate_ips_have_separate_buckets() -> None:
    limiter = RateLimiter(MemoryBuckets(), rate_per_s=1 / 6, burst=1)
    assert limiter.allow("a") == 0
    assert limiter.allow("a") > 0  # a exhausted
    assert limiter.allow("b") == 0  # b independent


def test_fails_open_when_store_unavailable() -> None:
    store = MemoryBuckets()
    store.fail = True
    limiter = RateLimiter(store, burst=1)
    assert limiter.allow("ip") == 0  # a store error never denies
