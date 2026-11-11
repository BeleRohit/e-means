"""Validated algorithm settings; paper references use arXiv:2611.09501v1."""

from dataclasses import dataclass
from numbers import Integral
import math


def positive_int(name: str, value: int) -> None:
    """Engineering validation: reject silent truncation and boolean counts."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def finite_range(name: str, value: float, low: float, high: float) -> None:
    """Engineering validation for bounded scalar hyperparameters."""
    if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")


@dataclass(frozen=True)
class LloydConfig:
    """§1 Lloyd baseline; tol is absolute maximum centroid displacement."""

    n_clusters: int
    max_iter: int = 25  # §4 baseline budget
    tol: float = 0.0  # [UNSPECIFIED] exact fixed point; alternative relative MSE
    seed: int = 0  # [UNSPECIFIED] caller-controlled; alternative any fixed seed
    dtype: str = "float64"  # [UNSPECIFIED] correctness first; alternative float32
    assignment_chunk_size: int = 1024  # engineering memory bound
    centroid_chunk_size: int = 256  # engineering memory bound

    def __post_init__(self) -> None:
        for name in ("n_clusters", "max_iter", "assignment_chunk_size", "centroid_chunk_size"):
            positive_int(name, getattr(self, name))
        if not math.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and nonnegative")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class EMeansConfig:
    """§3 Algorithm 1. None batch_size/alpha0 selects paper Table 2 laws."""

    n_clusters: int
    epochs: int = 3
    batch_size: int | None = None
    alpha0: float | None = None
    alpha_min_factor: float = 0.1  # Eq. 5
    initial_mass: float = 1e-10  # §3.1
    revival_interval: int = 5  # Algorithm 1
    revival_threshold: float = 0.1  # §3.1
    split_epsilon: float = 2.0**-7  # Appendix H.1: BF16 machine epsilon
    finalize: str = "last_epoch_mean"  # §3.2
    drop_last: bool = True  # §3.1 M = T floor(n/B); see REPRODUCTION_NOTES
    seed: int = 0
    dtype: str = "float64"
    assignment_chunk_size: int = 1024
    centroid_chunk_size: int = 256
    slow_ema_factor: float | None = None  # Appendix I: multiplier of epsilon*alpha0
    slow_half_life: float | None = None  # explicit convention: logical updates

    def __post_init__(self) -> None:
        LloydConfig(self.n_clusters, seed=self.seed, dtype=self.dtype,
                    assignment_chunk_size=self.assignment_chunk_size,
                    centroid_chunk_size=self.centroid_chunk_size)
        positive_int("epochs", self.epochs)
        positive_int("revival_interval", self.revival_interval)
        if self.batch_size is not None:
            positive_int("batch_size", self.batch_size)
        if self.alpha0 is not None:
            finite_range("alpha0", self.alpha0, 0.0, 1.0)
            if self.alpha0 == 0:
                raise ValueError("alpha0 must be > 0")
        finite_range("alpha_min_factor", self.alpha_min_factor, 0.0, 1.0)
        finite_range("revival_threshold", self.revival_threshold, 0.0, 1.0)
        finite_range("split_epsilon", self.split_epsilon, 0.0, 1.0)
        if not math.isfinite(self.initial_mass) or self.initial_mass <= 0:
            raise ValueError("initial_mass must be positive and finite")
        if self.finalize not in ("last_epoch_mean", "none", "slow_ema"):
            raise ValueError("finalize must be last_epoch_mean, none or slow_ema")
        if self.slow_ema_factor is not None and self.slow_half_life is not None:
            raise ValueError("choose slow_ema_factor or slow_half_life, not both")
        for name in ("slow_ema_factor", "slow_half_life"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
            if value is not None and self.finalize != "slow_ema":
                raise ValueError(f"{name} requires finalize='slow_ema'")
        if not isinstance(self.drop_last, bool):
            raise ValueError("drop_last must be bool")
