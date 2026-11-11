"""§3.1 Eq. 5 and §3.3 Eqs. 6–7, using published Table 2 coefficients."""

import math

from e_means.config import EMeansConfig, finite_range, positive_int


def cosine_alpha(step: int, total_steps: int, alpha0: float, floor: float = 0.1) -> float:
    """Eq. 5 literally uses t/M, t=0,...,M-1; final step is above floor."""
    positive_int("total_steps", total_steps)
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < total_steps:
        raise ValueError("step must be an integer in [0, total_steps)")
    finite_range("alpha0", alpha0, 0.0, 1.0)
    finite_range("floor", floor, 0.0, 1.0)
    return alpha0 * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * step / total_steps)))


def paper_batch_size(n: int, k: int, epochs: int) -> int:
    """Eq. 7/Table 2: nearest power of two, capped at n; /8 above 2M."""
    for name, value in (("n", n), ("k", k), ("epochs", epochs)):
        positive_int(name, value)
    raw = 0.035 * k**0.13 * n**0.88 * epochs**0.81
    if n > 2_000_000:
        raw /= 8  # §3.3 prose, before rounding (0.0044 in Table 2 is rounded)
    # [FROM_OFFICIAL_CODE] base.py L66–73: logarithmic proximity, ties up.
    return min(n, 2 ** max(0, math.floor(math.log2(raw) + 0.5)))


def paper_alpha0(n: int, k: int, epochs: int, batch_size: int) -> float:
    """Eq. 6/Table 2; no undocumented lower clamp or refitted coefficients."""
    for name, value in (("n", n), ("k", k), ("epochs", epochs), ("batch_size", batch_size)):
        positive_int(name, value)
    return min(0.824 * k**-0.46 * batch_size**0.90 * n**-0.45 * epochs**-0.11, 1.0)


def resolve_recipe(config: EMeansConfig, n: int) -> dict:
    """Resolve explicit overrides before computing step law at actual B."""
    batch = config.batch_size if config.batch_size is not None else paper_batch_size(n, config.n_clusters, config.epochs)
    batch = min(batch, n)
    steps = n // batch if config.drop_last else (n + batch - 1) // batch
    alpha = config.alpha0 if config.alpha0 is not None else paper_alpha0(n, config.n_clusters, config.epochs, batch)
    return dict(batch_size=batch, alpha0=alpha, steps_per_epoch=steps,
                total_steps=steps * config.epochs, drop_last=config.drop_last,
                batch_source="explicit" if config.batch_size is not None else "paper_table_2",
                alpha_source="explicit" if config.alpha0 is not None else "paper_table_2")


def slow_alpha(config: EMeansConfig, alpha0: float) -> float:
    """§3.2/Appendix I: constant EMA speed, independent of cosine progress.

    Appendix I specifies a multiplier of epsilon*alpha0. Default multiplier 1
    is an explicit research choice (not a universally optimal paper setting).
    §3.2 mentions h without units; optional half-life here is in logical batch
    updates: gamma**h=1/2. No clipping of invalid speeds is performed.
    """
    value = (-math.expm1(-math.log(2) / config.slow_half_life)
             if config.slow_half_life is not None else
             (1.0 if config.slow_ema_factor is None else config.slow_ema_factor)
             * config.alpha_min_factor * alpha0)
    if not 0 < value <= 1:
        raise ValueError("resolved slow EMA step must be in (0, 1]")
    return value
