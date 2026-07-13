import numpy as np

from atlas_core.cache import CacheEntry, ResponseCache


class _FakeStore:
    def __init__(self) -> None:
        self.entries: list[CacheEntry] = []
        self.fail = False

    def recent(self) -> list[CacheEntry]:
        if self.fail:
            raise RuntimeError("dynamodb unreachable")
        return self.entries

    def put(self, entry: CacheEntry) -> None:
        if self.fail:
            raise RuntimeError("dynamodb unreachable")
        self.entries.append(entry)


def _unit(*values: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_put_then_near_identical_query_hits() -> None:
    cache = ResponseCache(_FakeStore(), similarity_floor=0.97)
    cache.put(_unit(1, 0, 0), {"brief": "cached"})
    # Almost the same direction: cosine ~0.999, above the floor.
    assert cache.get(_unit(1, 0.05, 0)) == {"brief": "cached"}


def test_distant_query_misses() -> None:
    cache = ResponseCache(_FakeStore(), similarity_floor=0.97)
    cache.put(_unit(1, 0, 0), {"brief": "cached"})
    assert cache.get(_unit(0, 1, 0)) is None  # orthogonal, cosine 0


def test_expired_entry_is_ignored() -> None:
    store = _FakeStore()
    store.entries.append(CacheEntry(embedding=[1.0, 0.0, 0.0], payload={"x": 1}, expires_at=0.0))
    cache = ResponseCache(store)
    assert cache.get(_unit(1, 0, 0)) is None


def test_lookup_failure_fails_open_as_miss() -> None:
    store = _FakeStore()
    store.fail = True
    cache = ResponseCache(store)
    assert cache.get(_unit(1, 0, 0)) is None  # no raise
