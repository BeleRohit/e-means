"""Independent PyTorch CPU/CUDA implementation of §1 and Algorithm 1.

Tensor arithmetic stays on the selected device. NumPy supplies *only* the
Phase 1 RNG protocol (initial row IDs, permutations, weighted donor choices).
Revival copies k masses to the host to preserve that protocol. There are no
custom kernels, autograd, mixed precision or approximate assignments.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor

from e_means.config import EMeansConfig, LloydConfig, finite_range, positive_int
from e_means.estimators import EpochRecord
from e_means.schedules import cosine_alpha, resolve_recipe, slow_alpha


def tensor_matrix(x: Any, dtype: torch.dtype, device: torch.device, name: str = "X") -> Tensor:
    """Engineering validation mirrors NumPy: finite, nonempty real matrices."""
    if isinstance(x, Tensor):
        if x.layout != torch.strided or x.is_complex() or x.dtype == torch.bool:
            raise ValueError(f"{name} must be a dense real numeric matrix")
        raw = x.detach()
    else:
        raw = np.asarray(x)
        if raw.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a real numeric matrix")
        # Negative-stride NumPy views are valid Phase 1 inputs too.
        raw = np.ascontiguousarray(raw)
    out = torch.as_tensor(raw, dtype=dtype, device=device)
    if out.ndim != 2 or min(out.shape) == 0 or not bool(torch.isfinite(out).all()):
        raise ValueError(f"{name} must be a nonempty finite matrix")
    return out


@torch.no_grad()
def assign_nearest(x: Tensor, centroids: Tensor, *, chunk_size: int = 1024,
                   centroid_chunk_size: int = 256) -> tuple[Tensor, Tensor]:
    """§1 Eq. 1: tiled direct differences; first index wins ties.

    Scratch O(chunk_size*centroid_chunk_size), without (n,k,d) expansion or
    norm/dot cancellation. Deliberately matches Phase 1 accumulation order.
    """
    positive_int("chunk_size", chunk_size)
    positive_int("centroid_chunk_size", centroid_chunk_size)
    if not isinstance(x, Tensor) or not isinstance(centroids, Tensor):
        raise ValueError("assignment requires torch tensors")
    if x.dtype not in (torch.float32, torch.float64) or centroids.dtype != x.dtype or centroids.device != x.device:
        raise ValueError("X/centroids must share FP32/FP64 dtype and device")
    x = tensor_matrix(x, x.dtype, x.device)
    c = tensor_matrix(centroids, x.dtype, x.device, "centroids")
    if x.shape[1] != c.shape[1]:
        raise ValueError("X and centroids must have the same feature dimension")
    labels = torch.empty(len(x), dtype=torch.long, device=x.device)
    distances = torch.empty(len(x), dtype=x.dtype, device=x.device)
    for start in range(0, len(x), chunk_size):
        rows = x[start:start + chunk_size]  # (b,d)
        best = torch.full((len(rows),), torch.inf, dtype=x.dtype, device=x.device)
        ids = torch.zeros(len(rows), dtype=torch.long, device=x.device)
        for first in range(0, len(c), centroid_chunk_size):
            centers = c[first:first + centroid_chunk_size]  # (q,d)
            dist = torch.zeros((len(rows), len(centers)), dtype=x.dtype, device=x.device)
            for dim in range(x.shape[1]):
                diff = rows[:, dim, None] - centers[None, :, dim]  # (b,q)
                dist.add_(diff.square())
            if not bool(torch.isfinite(dist).all()):
                raise FloatingPointError("squared-distance arithmetic overflowed")
            candidate, local = dist.min(dim=1)  # (b,q) -> (b,)
            better = candidate < best
            best = torch.where(better, candidate, best)
            ids = torch.where(better, first + local, ids)
        labels[start:start + len(rows)] = ids
        distances[start:start + len(rows)] = best
    return labels, distances


@torch.no_grad()
def batch_statistics(x: Tensor, labels: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """§3.1 Eq. 3: assigned rows -> sums (k,d), counts (k,)."""
    sums = torch.zeros((k, x.shape[1]), dtype=x.dtype, device=x.device)
    sums.index_add_(0, labels, x)
    counts = torch.bincount(labels, minlength=k).to(x.dtype)
    return sums, counts


@torch.no_grad()
def last_epoch_mean(sums: Tensor, counts: Tensor, fallback: Tensor) -> Tensor:
    """§3.2 fixed-code means; empty slots retain last valid EMA position."""
    result = fallback.clone()
    occupied = counts > 0
    result[occupied] = sums[occupied] / counts[occupied, None]  # (occupied,d)
    return result


@dataclass
class TorchEMAState:
    """§3.1 Eqs. 2–4, literal sum/count representation plus zero-mass fallback."""

    sums: Tensor
    counts: Tensor
    fallback: Tensor

    @classmethod
    @torch.no_grad()
    def initialize(cls, centroids: Tensor, mass: float = 1e-10) -> "TorchEMAState":
        c = tensor_matrix(centroids, centroids.dtype, centroids.device).clone()
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be positive and finite")
        counts = torch.full((len(c),), mass, dtype=c.dtype, device=c.device)
        if not bool(((counts > 0) & torch.isfinite(counts)).all()):
            raise ValueError("mass is not representable in state dtype")
        return cls(c * counts[:, None], counts, c.clone())

    @torch.no_grad()
    def centroids(self) -> Tensor:
        return last_epoch_mean(self.sums, self.counts, self.fallback)

    @torch.no_grad()
    def update(self, sums: Tensor, counts: Tensor, alpha: float) -> None:
        """Eq. 2, identical alpha applied to sums and counts; no gradient graph."""
        finite_range("alpha", alpha, 0, 1)
        if sums.shape != self.sums.shape or counts.shape != self.counts.shape:
            raise ValueError("batch statistics must match state shapes")
        if (sums.device != self.sums.device or counts.device != self.counts.device or
                sums.dtype != self.sums.dtype or counts.dtype != self.counts.dtype):
            raise ValueError("batch statistics must match state dtype/device")
        if not bool(torch.isfinite(sums).all() & torch.isfinite(counts).all() & (counts >= 0).all()):
            raise ValueError("batch statistics must be finite; counts nonnegative")
        self.fallback = self.centroids()
        self.sums = (1 - alpha) * self.sums + alpha * sums
        self.counts = (1 - alpha) * self.counts + alpha * counts
        if not bool(torch.isfinite(self.sums).all() & torch.isfinite(self.counts).all()):
            raise FloatingPointError("EMA arithmetic overflowed")

    @torch.no_grad()
    def split(self, donors: Any, targets: Any, epsilon: float) -> None:
        """Appendix H.1 Eq. 8. Shadow state uses same IDs and its own geometry."""
        donors = torch.as_tensor(donors, dtype=torch.long, device=self.sums.device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=self.sums.device)
        centers = self.centroids()
        source = centers[donors].clone()  # (splits,d)
        half = self.counts[donors] / 2
        sign = torch.ones(centers.shape[1], dtype=centers.dtype, device=centers.device)
        sign[1::2] = -1
        centers[donors] = source * (1 - epsilon * sign)
        centers[targets] = source * (1 + epsilon * sign)
        self.counts[donors] = half
        self.counts[targets] = half
        self.sums[donors] = centers[donors] * half[:, None]
        self.sums[targets] = centers[targets] * half[:, None]
        self.fallback = centers

    @torch.no_grad()
    def revive(self, rng: np.random.Generator, threshold: float = 0.1,
               epsilon: float = 2**-7) -> tuple[np.ndarray, np.ndarray]:
        """Appendix H.1: Phase 1 RNG and FP64 threshold/weight arithmetic.

        Host transfer is explicit and bounded by k; no tensor clustering runs
        in NumPy. Alternative torch.multinomial would change seed trajectories.
        """
        finite_range("threshold", threshold, 0, 1)
        finite_range("epsilon", epsilon, 0, 1)
        mass = self.counts.detach().cpu().numpy()
        dead = mass < threshold * np.mean(mass, dtype=np.float64)
        targets = np.flatnonzero(dead)
        live = np.flatnonzero(~dead & (mass > 0))
        size = min(len(targets), len(live))
        targets = targets[:size]
        if size == 0:
            return np.empty(0, dtype=np.int64), targets
        weights = mass[live].astype(np.float64)
        weights /= weights.max()
        weights /= weights.sum()
        donors = rng.choice(live, size=size, replace=False, p=weights)
        self.split(donors, targets, epsilon)
        return donors, targets


class _TorchEstimator:
    """Shared API, input handling and fixed-center metrics for both algorithms."""

    def __init__(self, config: LloydConfig | EMeansConfig, *, device: str | torch.device | None = None):
        self.config = config
        self.device = None if device is None else torch.device(device)
        if self.device is not None and self.device.type not in ("cpu", "cuda"):
            raise ValueError("supported devices are cpu and cuda")

    def _sync(self) -> None:
        if self.device_.type == "cuda":
            torch.cuda.synchronize(self.device_)

    def _inputs(self, x: Any, init: Any, test: Any):
        device = self.device or (x.device if isinstance(x, Tensor) else torch.device("cpu"))
        if device.type not in ("cpu", "cuda"):
            raise ValueError("supported devices are cpu and cuda")
        dtype = getattr(torch, self.config.dtype)
        x = tensor_matrix(x, dtype, device)
        self.device_ = x.device
        k = self.config.n_clusters
        if k > len(x):
            raise ValueError("n_clusters cannot exceed number of rows")
        streams = np.random.SeedSequence(self.config.seed).spawn(3)
        initial_rng, order_rng, revival_rng = map(np.random.default_rng, streams)
        c = (x[torch.as_tensor(initial_rng.choice(len(x), k, replace=False), device=device)].clone()
             if init is None else tensor_matrix(init, dtype, device, "init_centroids").clone())
        if c.shape != (k, x.shape[1]):
            raise ValueError("init_centroids must have shape (n_clusters, n_features)")
        test = None if test is None else tensor_matrix(test, dtype, device, "eval_X")
        if test is not None and test.shape[1] != x.shape[1]:
            raise ValueError("eval_X must have the same feature dimension as X")
        return x, c, test, order_rng, revival_rng

    def _assignment(self, x: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
        return assign_nearest(x, c, chunk_size=self.config.assignment_chunk_size,
                              centroid_chunk_size=self.config.centroid_chunk_size)

    def _record(self, x: Tensor, c: Tensor, test: Tensor | None, epoch: int,
                moving: float | None, train_seconds: float, eval_seconds: float,
                updates: int, revivals: int) -> EpochRecord:
        self._sync()
        start = perf_counter()
        train = self._assignment(x, c)[1].double().mean().item()
        test_mse = None if test is None else self._assignment(test, c)[1].double().mean().item()
        self._sync()
        return EpochRecord(epoch, train, test_mse, moving, train_seconds,
                           eval_seconds + perf_counter() - start, updates, revivals)

    def _publish(self, x: Tensor, c: Tensor, history: list[EpochRecord]) -> None:
        labels, distances = self._assignment(x, c)
        self.cluster_centers_ = c.clone()
        self.labels_ = labels
        self.mse_ = distances.double().mean().item()
        self.inertia_ = distances.double().sum().item()
        self.history_ = history
        self.n_features_in_ = x.shape[1]

    @torch.no_grad()
    def predict(self, x: Any) -> Tensor:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("estimator is not fitted")
        return self._assignment(tensor_matrix(x, self.cluster_centers_.dtype, self.device_), self.cluster_centers_)[0]

    def fit_predict(self, x: Any, *, init_centroids: Any = None) -> Tensor:
        return self.fit(x, init_centroids=init_centroids).labels_.clone()


class TorchLloydKMeans(_TorchEstimator):
    """§1 full-batch Lloyd with Phase 1 empty/stopping/API semantics."""

    @torch.no_grad()
    def fit(self, x: Any, *, init_centroids: Any = None, eval_X: Any = None) -> "TorchLloydKMeans":
        x, c, test, _, _ = self._inputs(x, init_centroids, eval_X)
        history = [self._record(x, c, test, 0, None, 0, 0, 0, 0)]
        train_seconds = 0.0
        for epoch in range(1, self.config.max_iter + 1):
            self._sync()
            start = perf_counter()
            labels, _ = self._assignment(x, c)
            sums, counts = batch_statistics(x, labels, len(c))
            new = last_epoch_mean(sums, counts, c)
            shift = torch.linalg.vector_norm(new - c, dim=1).max().item()
            c = new
            self._sync()
            train_seconds += perf_counter() - start
            history.append(self._record(x, c, test, epoch, None, train_seconds,
                                        history[-1].evaluation_seconds, epoch, 0))
            if shift <= self.config.tol:
                break
        self._publish(x, c, history)
        self.n_iter_ = epoch
        return self


class TorchEMeans(_TorchEstimator):
    """§3 Algorithm 1, with optional §3.2/Appendix I constant shadow EMA."""

    @torch.no_grad()
    def fit(self, x: Any, *, init_centroids: Any = None, eval_X: Any = None) -> "TorchEMeans":
        x, c, test, order_rng, revival_rng = self._inputs(x, init_centroids, eval_X)
        recipe = resolve_recipe(self.config, len(x))
        state = TorchEMAState.initialize(c, self.config.initial_mass)
        slow = None
        if self.config.finalize == "slow_ema":
            slow = TorchEMAState.initialize(c, self.config.initial_mass)
            recipe["slow_alpha"] = slow_alpha(self.config, recipe["alpha0"])
        history = [self._record(x, c, test, 0, None, 0, 0, 0, 0)]
        step, revivals, train_seconds = 0, 0, 0.0
        pass_labels = torch.full((len(x),), -1, dtype=torch.long, device=x.device)
        for epoch in range(self.config.epochs):
            self._sync()
            start = perf_counter()
            order = torch.as_tensor(order_rng.permutation(len(x)), device=x.device)
            rows = (len(x) // recipe["batch_size"] * recipe["batch_size"]
                    if self.config.drop_last else len(x))
            pass_sums = torch.zeros_like(c)
            pass_counts = torch.zeros(len(c), dtype=c.dtype, device=c.device)
            moving_sse = 0.0
            for offset in range(0, rows, recipe["batch_size"]):
                ids = order[offset:min(offset + recipe["batch_size"], rows)]
                batch = x[ids]
                labels, distances = self._assignment(batch, state.centroids())
                sums, counts = batch_statistics(batch, labels, len(c))
                moving_sse += distances.double().sum().item()
                if epoch == self.config.epochs - 1:
                    pass_sums += sums
                    pass_counts += counts
                    pass_labels[ids] = labels
                alpha = cosine_alpha(step, recipe["total_steps"], recipe["alpha0"], self.config.alpha_min_factor)
                state.update(sums, counts, alpha)
                if slow is not None:
                    slow.update(sums, counts, recipe["slow_alpha"])
                step += 1
                if step % self.config.revival_interval == 0:
                    donors, targets = state.revive(revival_rng, self.config.revival_threshold, self.config.split_epsilon)
                    if slow is not None:
                        slow.split(donors, targets, self.config.split_epsilon)
                    revivals += len(targets)
            c = state.centroids()
            self._sync()
            train_seconds += perf_counter() - start
            history.append(self._record(x, c, test, epoch + 1, moving_sse / rows, train_seconds,
                                        history[-1].evaluation_seconds, step, revivals))
        self._sync()
        start = perf_counter()
        final = (last_epoch_mean(pass_sums, pass_counts, c) if self.config.finalize == "last_epoch_mean"
                 else slow.centroids() if slow is not None else c.clone())
        self._sync()
        finalization_seconds = perf_counter() - start
        self._publish(x, final, history)
        self.recipe_, self.ema_state_ = recipe, state
        self.ema_centers_ = c.clone()
        self.last_epoch_labels_ = pass_labels
        self.last_epoch_sums_, self.last_epoch_counts_ = pass_sums, pass_counts
        self.finalization_seconds_ = finalization_seconds
        self.n_steps_, self.n_revivals_, self.n_iter_ = step, revivals, self.config.epochs
        self.slow_state_ = slow
        self.slow_centers_ = None if slow is None else slow.centroids()
        return self
