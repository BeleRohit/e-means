"""§1 Lloyd baseline and §3 Algorithm 1 e-means, using only NumPy."""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
from numpy.typing import ArrayLike, NDArray

from e_means.assignment import assign_nearest, batch_statistics, matrix
from e_means.config import EMeansConfig, LloydConfig
from e_means.schedules import cosine_alpha, resolve_recipe, slow_alpha
from e_means.state import EMAState, last_epoch_mean


@dataclass(frozen=True)
class EpochRecord:
    """Eq. 1 fixed-center metrics; moving_mse is a separate online diagnostic."""

    epoch: int
    train_mse: float
    test_mse: float | None
    moving_mse: float | None
    training_seconds: float
    evaluation_seconds: float
    updates: int
    revivals: int


class _NumpyEstimator:
    """Shared validation and exact assignment, independent of optimizer state."""

    def _assignment(self, x: ArrayLike, c: NDArray) -> tuple[NDArray, NDArray]:
        return assign_nearest(x, c, chunk_size=self.config.assignment_chunk_size,
                              centroid_chunk_size=self.config.centroid_chunk_size,
                              dtype=self.config.dtype)

    def _inputs(self, x: ArrayLike, init: ArrayLike | None, test: ArrayLike | None):
        x = matrix(x, self.config.dtype)
        k = self.config.n_clusters
        if k > len(x):
            raise ValueError("n_clusters cannot exceed number of rows")
        # [UNSPECIFIED] Independent Generator streams; alternative RandomState.
        streams = np.random.SeedSequence(self.config.seed).spawn(3)
        initial_rng, order_rng, revival_rng = map(np.random.default_rng, streams)
        c = (x[initial_rng.choice(len(x), k, replace=False)].copy() if init is None
             else matrix(init, self.config.dtype, name="init_centroids").copy())
        if c.shape != (k, x.shape[1]):
            raise ValueError("init_centroids must have shape (n_clusters, n_features)")
        test = None if test is None else matrix(test, self.config.dtype, name="eval_X")
        if test is not None and test.shape[1] != x.shape[1]:
            raise ValueError("eval_X must have the same feature dimension as X")
        return x, c, test, order_rng, revival_rng

    def _record(self, x, c, test, epoch, moving, train_seconds, eval_seconds, updates, revivals):
        start = perf_counter()
        train_mse = float(np.mean(self._assignment(x, c)[1], dtype=np.float64))
        test_mse = None if test is None else float(np.mean(self._assignment(test, c)[1], dtype=np.float64))
        return EpochRecord(epoch, train_mse, test_mse, moving, train_seconds,
                           eval_seconds + perf_counter() - start, updates, revivals)

    def _publish(self, x, c, history) -> None:
        labels, distances = self._assignment(x, c)
        self.cluster_centers_ = c.copy()
        self.labels_ = labels  # fresh final assignments, not moving-pass codes
        self.mse_ = float(np.mean(distances, dtype=np.float64))
        self.inertia_ = float(np.sum(distances, dtype=np.float64))
        self.history_ = history
        self.n_features_in_ = x.shape[1]

    def predict(self, x: ArrayLike) -> NDArray:
        """§1 Eq. 1 nearest-centroid IDs; fit must finish successfully first."""
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("estimator is not fitted")
        return self._assignment(x, self.cluster_centers_)[0]

    def fit_predict(self, x: ArrayLike, *, init_centroids: ArrayLike | None = None) -> NDArray:
        return self.fit(x, init_centroids=init_centroids).labels_.copy()


class LloydKMeans(_NumpyEstimator):
    """§1 Lloyd: full assignment then exact means; preserve empty centers.

    [UNSPECIFIED] Absolute max centroid displacement stopping and retained
    empty centers. Alternatives: relative loss stopping, donor splitting.
    """

    def __init__(self, config: LloydConfig):
        self.config = config

    def fit(self, x: ArrayLike, *, init_centroids: ArrayLike | None = None,
            eval_X: ArrayLike | None = None) -> "LloydKMeans":
        x, c, test, _, _ = self._inputs(x, init_centroids, eval_X)
        history = [self._record(x, c, test, 0, None, 0.0, 0.0, 0, 0)]
        train_seconds = 0.0
        for epoch in range(1, self.config.max_iter + 1):
            start = perf_counter()
            labels, _ = self._assignment(x, c)
            sums, counts = batch_statistics(x, labels, len(c))
            new = last_epoch_mean(sums, counts, c)
            shift = float(np.max(np.linalg.norm(new - c, axis=1)))
            c = new
            train_seconds += perf_counter() - start
            history.append(self._record(x, c, test, epoch, None, train_seconds,
                                        history[-1].evaluation_seconds, epoch, 0))
            if shift <= self.config.tol:
                break
        self._publish(x, c, history)
        self.n_iter_ = epoch
        return self


class EMeans(_NumpyEstimator):
    """§3 Algorithm 1: minibatch EMA with schedule, revival, train finalizer.

    last_epoch_labels_ captures moving assignments, -1 for dropped tail rows.
    labels_ always contains nearest assignments to returned centers for all X.
    """

    def __init__(self, config: EMeansConfig):
        self.config = config

    def fit(self, x: ArrayLike, *, init_centroids: ArrayLike | None = None,
            eval_X: ArrayLike | None = None) -> "EMeans":
        x, c, test, order_rng, revival_rng = self._inputs(x, init_centroids, eval_X)
        recipe = resolve_recipe(self.config, len(x))
        state = EMAState.initialize(c, self.config.initial_mass)
        slow = None
        if self.config.finalize == "slow_ema":
            slow = EMAState.initialize(c, self.config.initial_mass)
            recipe["slow_alpha"] = slow_alpha(self.config, recipe["alpha0"])
        history = [self._record(x, c, test, 0, None, 0.0, 0.0, 0, 0)]
        step, revivals, train_seconds = 0, 0, 0.0
        pass_labels = np.full(len(x), -1, dtype=np.int64)
        for epoch in range(self.config.epochs):
            start = perf_counter()
            order = order_rng.permutation(len(x))  # (n,)
            rows = (len(x) // recipe["batch_size"] * recipe["batch_size"]
                    if self.config.drop_last else len(x))
            pass_sums = np.zeros_like(c)
            pass_counts = np.zeros(len(c), dtype=c.dtype)
            moving_sse = 0.0
            for offset in range(0, rows, recipe["batch_size"]):
                ids = order[offset:min(offset + recipe["batch_size"], rows)]
                batch = x[ids]  # (B,d), last B may be smaller when keeping tail
                labels, distances = self._assignment(batch, state.centroids())
                sums, counts = batch_statistics(batch, labels, len(c))
                moving_sse += float(np.sum(distances, dtype=np.float64))
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
            train_seconds += perf_counter() - start
            history.append(self._record(x, c, test, epoch + 1, moving_sse / rows,
                                        train_seconds, history[-1].evaluation_seconds, step, revivals))
        start = perf_counter()
        final = (last_epoch_mean(pass_sums, pass_counts, c)
                 if self.config.finalize == "last_epoch_mean" else
                 slow.centroids() if slow is not None else c.copy())
        finalization_seconds = perf_counter() - start
        self._publish(x, final, history)
        self.recipe_ = recipe
        self.ema_state_ = state
        self.ema_centers_ = c.copy()
        self.last_epoch_labels_ = pass_labels
        self.last_epoch_sums_ = pass_sums
        self.last_epoch_counts_ = pass_counts
        self.finalization_seconds_ = finalization_seconds
        self.n_steps_ = step
        self.n_revivals_ = revivals
        self.n_iter_ = self.config.epochs
        # Cleared on refit as well: do not retain a shadow from an earlier run.
        self.slow_state_ = slow
        self.slow_centers_ = None if slow is None else slow.centroids()
        return self
