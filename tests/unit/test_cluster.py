import numpy as np
import pytest

from atlas_core.cluster import central_order, kmeans, pick_k


def blob(center: list[float], n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.asarray(center, dtype=np.float32) + rng.normal(0, 0.05, (n, len(center)))
    return (points / np.linalg.norm(points, axis=1, keepdims=True)).astype(np.float32)


def test_kmeans_separates_obvious_blobs() -> None:
    a = blob([1.0, 0.0, 0.0], 10, seed=1)
    b = blob([0.0, 1.0, 0.0], 10, seed=2)
    labels, _ = kmeans(np.vstack([a, b]), k=2)
    assert len(set(labels[:10].tolist())) == 1
    assert len(set(labels[10:].tolist())) == 1
    assert labels[0] != labels[10]


def test_kmeans_is_deterministic() -> None:
    points = np.vstack([blob([1.0, 0.0, 0.0], 8, seed=3), blob([0.0, 0.0, 1.0], 8, seed=4)])
    first, _ = kmeans(points, k=3)
    second, _ = kmeans(points, k=3)
    assert np.array_equal(first, second)


def test_kmeans_clamps_k_to_n() -> None:
    points = blob([1.0, 0.0, 0.0], 2, seed=5)
    labels, centroids = kmeans(points, k=10)
    assert len(labels) == 2
    assert centroids.shape[0] == 2


def test_kmeans_rejects_empty() -> None:
    with pytest.raises(ValueError):
        kmeans(np.empty((0, 3), dtype=np.float32), k=2)


def test_pick_k_scales_with_corpus() -> None:
    assert pick_k(1) == 1
    assert pick_k(6) == 3  # small retrievals still split into the minimum directions
    assert pick_k(100) == 6
    assert pick_k(45) == 3


def test_central_order_puts_nearest_first() -> None:
    centroid = np.asarray([1.0, 0.0], dtype=np.float32)
    points = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    assert central_order(points, centroid).tolist() == [1, 2, 0]
