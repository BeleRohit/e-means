"""§3.1 sum/count state, Eq. 2 updates, and Appendix H.1 donor splitting."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from e_means.assignment import matrix
from e_means.config import finite_range


@dataclass
class EMAState:
    """Eqs. 2–4: explicit sums S (k,d) and counts N (k,).

    fallback only defines readout at zero mass (alpha=1 or underflow).
    [FROM_OFFICIAL_CODE] e_means.py L489–504 preserves empty positions.
    """

    sums: NDArray
    counts: NDArray
    fallback: NDArray

    @classmethod
    def initialize(cls, centroids: NDArray, mass: float = 1e-10) -> "EMAState":
        """§3.1 initialization: S=N*c, N=1e-10; caller owns no state storage."""
        c = matrix(centroids, centroids.dtype).copy()
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass must be positive and finite")
        counts = np.full(len(c), mass, dtype=c.dtype)  # (k,)
        if not (counts > 0).all() or not np.isfinite(counts).all():
            raise ValueError("mass is not representable in state dtype")
        return cls(c * counts[:, None], counts, c.copy())

    def centroids(self) -> NDArray:
        """Eq. 4: ratio where N>0, preserve position where Eq. 4 is undefined."""
        c = self.fallback.copy()  # (k,d)
        np.divide(self.sums, self.counts[:, None], out=c, where=self.counts[:, None] > 0)
        return c

    def update(self, sums: NDArray, counts: NDArray, alpha: float) -> None:
        """Eq. 2: alpha=1-gamma multiplies both new sums and new counts."""
        finite_range("alpha", alpha, 0.0, 1.0)
        if sums.shape != self.sums.shape or counts.shape != self.counts.shape:
            raise ValueError("batch statistics must match state shapes")
        if not np.isfinite(sums).all() or not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("batch statistics must be finite, counts nonnegative")
        self.fallback = self.centroids()
        with np.errstate(over="raise", invalid="raise"):
            self.sums = (1 - alpha) * self.sums + alpha * sums  # (k,d)
            self.counts = (1 - alpha) * self.counts + alpha * counts  # (k,)

    def revive(self, rng: np.random.Generator, threshold: float = 0.1,
               epsilon: float = 2.0**-7) -> tuple[NDArray, NDArray]:
        """Appendix H.1/Eq. 8: weighted live donors, opposite perturbations.

        [FROM_OFFICIAL_CODE] e_means.py L261–282: only live, positive donors;
        cap splits at number of live donors. Excess dead slots wait until later.
        [UNSPECIFIED] RNG: Generator.choice without replacement.
        Alternative: upstream Gumbel top-k. Zero donors remain identical under
        Eq. 8; we do not invent an additive displacement to force revival.
        """
        finite_range("threshold", threshold, 0.0, 1.0)
        finite_range("epsilon", epsilon, 0.0, 1.0)
        dead = self.counts < threshold * np.mean(self.counts, dtype=np.float64)
        targets = np.flatnonzero(dead)
        live = np.flatnonzero(~dead & (self.counts > 0))
        size = min(len(targets), len(live))
        targets = targets[:size]
        if size == 0:
            return np.empty(0, dtype=np.int64), targets
        weights = self.counts[live].astype(np.float64)
        weights /= weights.max()
        weights /= weights.sum()
        donors = rng.choice(live, size=size, replace=False, p=weights)
        self.split(donors, targets, epsilon)
        return donors, targets

    def split(self, donors: NDArray, targets: NDArray, epsilon: float) -> None:
        """Appendix H.1 Eq. 8 on supplied disjoint IDs, also for shadow EMA.

        §3.2 shadow state follows main-state donor/target choices but splits its
        own sums/counts. This interpretation is explicit; no second RNG draw.
        """
        centers = self.centroids()
        source = centers[donors].copy()  # (size,d)
        half = self.counts[donors] / 2  # (size,)
        sign = np.where(np.arange(centers.shape[1]) % 2 == 0, 1, -1).astype(centers.dtype)
        centers[donors] = source * (1 - epsilon * sign)  # (size,d)
        centers[targets] = source * (1 + epsilon * sign)
        self.counts[donors] = half
        self.counts[targets] = half
        self.sums[donors] = centers[donors] * half[:, None]
        self.sums[targets] = centers[targets] * half[:, None]
        self.fallback = centers


def last_epoch_mean(sums: NDArray, counts: NDArray, fallback: NDArray) -> NDArray:
    """§3.2: mean of moving-pass assignments, no fresh assignment sweep.

    [FROM_OFFICIAL_CODE] empty clusters preserve final EMA positions. Closing
    totals are not split/reset at revival. This minimizes fixed-code SSE, which
    does not imply improvement over final EMA's freshly reassigned MSE.
    """
    result = fallback.copy()  # (k,d)
    np.divide(sums, counts[:, None], out=result, where=counts[:, None] > 0)
    return result
