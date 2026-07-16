"""Plain k-means for grouping retrieved papers into research directions.

Embeddings are L2-normalized, so Euclidean k-means is monotone in cosine similarity and
scikit-learn would buy nothing but a dependency. Deterministic by construction: fixed-seed
k-means++ initialization, so the same retrieval always yields the same directions.
"""

import numpy as np
import numpy.typing as npt

_MAX_ROUNDS = 50


def pick_k(n: int, *, lo: int = 3, hi: int = 6, per_cluster: int = 15) -> int:
    return max(1, min(hi, max(lo, n // per_cluster), n))


def kmeans(
    vectors: npt.NDArray[np.float32], k: int, seed: int = 0
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]:
    """Cluster rows of `vectors` into k groups; returns (labels, centroids)."""
    n = vectors.shape[0]
    if n == 0 or k < 1:
        raise ValueError("kmeans needs at least one vector and k >= 1")
    k = min(k, n)
    rng = np.random.default_rng(seed)

    # k-means++ seeding: spread the initial centroids, which matters more than iterations do.
    centroids = vectors[rng.integers(n)][np.newaxis, :].astype(np.float32)
    while centroids.shape[0] < k:
        d2 = np.min(((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = float(d2.sum())
        # zero total: all remaining points coincide with a centroid, any pick works
        pick = int(rng.integers(n)) if total == 0.0 else int(rng.choice(n, p=d2 / total))
        centroids = np.vstack([centroids, vectors[pick][np.newaxis, :]])

    labels = np.full(n, -1, dtype=np.intp)
    for _round in range(_MAX_ROUNDS):
        dists = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = vectors[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
    return labels, centroids


def central_order(
    vectors: npt.NDArray[np.float32], centroid: npt.NDArray[np.float32]
) -> npt.NDArray[np.intp]:
    """Row indices sorted most-central-first; used to pick a cluster's representative papers."""
    return np.argsort(((vectors - centroid) ** 2).sum(axis=1))
