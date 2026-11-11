"""§1 Eq. 1: exact squared Euclidean assignment with bounded scratch space."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from e_means.config import positive_int


def matrix(x: ArrayLike, dtype: str | np.dtype = "float64", *, name: str = "X") -> NDArray:
    """Validate real, nonempty finite matrices; no implicit complex truncation."""
    raw = np.asarray(x)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or min(raw.shape) == 0:
        raise ValueError(f"{name} must be a nonempty real numeric matrix")
    out = np.asarray(raw, dtype=dtype)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain finite values representable in {dtype}")
    return out


def assign_nearest(
    x: ArrayLike, centroids: ArrayLike, *, chunk_size: int = 1024,
    centroid_chunk_size: int = 256, dtype: str | np.dtype = "float64",
) -> tuple[NDArray, NDArray]:
    """Eq. 1: return (labels, squared distances), both shape (n,).

    [UNSPECIFIED] Use direct differences, lowest-index ties, and float64.
    Alternatives: norm/dot identity or arbitrary ties. Direct differences avoid
    cancellation for large offsets. Scratch is O(chunk_size*centroid_chunk_size),
    not O(n*k*d). Work is O(n*k*d); no approximate search.
    """
    positive_int("chunk_size", chunk_size)
    positive_int("centroid_chunk_size", centroid_chunk_size)
    if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    x, c = matrix(x, dtype), matrix(centroids, dtype, name="centroids")
    if x.shape[1] != c.shape[1]:
        raise ValueError("X and centroids must have the same feature dimension")
    labels = np.empty(len(x), dtype=np.int64)
    distances = np.empty(len(x), dtype=dtype)
    # §1 Eq. 1, (b,d), (q,d) -> tiled distances (b,q).
    with np.errstate(over="raise", invalid="raise"):
        for start in range(0, len(x), chunk_size):
            rows = x[start:start + chunk_size]
            best = np.full(len(rows), np.inf, dtype=dtype)
            ids = np.zeros(len(rows), dtype=np.int64)
            for first in range(0, len(c), centroid_chunk_size):
                centers = c[first:first + centroid_chunk_size]
                dist = np.zeros((len(rows), len(centers)), dtype=dtype)
                for dim in range(x.shape[1]):
                    diff = rows[:, dim, None] - centers[None, :, dim]  # (b,q)
                    np.square(diff, out=diff)
                    dist += diff
                local = dist.argmin(axis=1)  # (b,q) -> (b,)
                candidate = dist[np.arange(len(rows)), local]
                better = candidate < best  # strict: earlier index wins ties
                best[better], ids[better] = candidate[better], first + local[better]
            labels[start:start + len(rows)] = ids
            distances[start:start + len(rows)] = best
    return labels, distances


def mse(x: ArrayLike, centroids: ArrayLike, **kwargs) -> float:
    """§1 Eq. 1: average squared norm per vector, with FP64 reduction."""
    _, distances = assign_nearest(x, centroids, **kwargs)
    return float(np.mean(distances, dtype=np.float64))


def batch_statistics(x: NDArray, labels: NDArray, k: int) -> tuple[NDArray, NDArray]:
    """§3.1 Eq. 3: (b,d), (b,) -> per-cluster sums (k,d), counts (k,)."""
    sums = np.zeros((k, x.shape[1]), dtype=x.dtype)
    np.add.at(sums, labels, x)
    counts = np.bincount(labels, minlength=k).astype(x.dtype)
    return sums, counts
