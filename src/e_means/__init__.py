"""Independent NumPy implementation of Bessa et al. (2026), Phase 1."""

from e_means.config import EMeansConfig, LloydConfig
from e_means.estimators import EMeans, LloydKMeans
from e_means.assignment import assign_nearest, mse

__all__ = ["EMeans", "EMeansConfig", "LloydKMeans", "LloydConfig", "assign_nearest", "mse"]


def __getattr__(name: str):
    """Optional backend stays lazy: importing NumPy API never imports torch."""
    if name in ("TorchEMeans", "TorchLloydKMeans"):
        from e_means.torch_backend import TorchEMeans, TorchLloydKMeans
        return {"TorchEMeans": TorchEMeans, "TorchLloydKMeans": TorchLloydKMeans}[name]
    raise AttributeError(name)
