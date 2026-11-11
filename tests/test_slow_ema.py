"""Independent slow count/sum oracle for §3.2 and Appendix I."""

from dataclasses import replace
import math
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from e_means import EMeans, EMeansConfig
from e_means.schedules import slow_alpha


def test_half_life_and_speed_conventions():
    cfg = EMeansConfig(2, finalize="slow_ema", slow_half_life=10)
    alpha = slow_alpha(cfg, 0.2)
    assert_allclose((1-alpha)**10, 0.5, atol=1e-15)
    assert_allclose(slow_alpha(replace(cfg, slow_half_life=None, slow_ema_factor=0.25), 0.2), 0.005)
    assert_allclose(slow_alpha(replace(cfg, slow_half_life=None), 0.2), 0.02)


@pytest.mark.parametrize("kwargs", [{"slow_ema_factor": -1}, {"slow_half_life": 0},
    {"slow_half_life": float("nan")}, {"slow_ema_factor": 1, "slow_half_life": 1}])
def test_invalid_slow_configs(kwargs):
    with pytest.raises(ValueError):
        EMeansConfig(2, finalize="slow_ema", **kwargs)
    with pytest.raises(ValueError):
        EMeansConfig(2, slow_ema_factor=1)


def test_reject_step_above_one_without_silent_clipping():
    with pytest.raises(ValueError):
        EMeans(EMeansConfig(1, finalize="slow_ema", alpha0=1, slow_ema_factor=11)).fit([[0], [1]])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_independent_scalar_shadow_oracle_and_assignment_isolation(dtype):
    x = np.array([[0., 1.], [2., 1.], [7., 3.], [9., 2.], [10., 4.]], dtype=dtype)
    init = x[[0, 4]].copy()
    cfg = EMeansConfig(2, epochs=3, batch_size=2, alpha0=0.3, dtype=dtype,
                      drop_last=False, revival_threshold=0, finalize="slow_ema", slow_ema_factor=0.25)
    model = EMeans(cfg).fit(x, init_centroids=init)
    main = EMeans(replace(cfg, finalize="none", slow_ema_factor=None)).fit(x, init_centroids=init)
    # Shadow cannot affect training codes, counts, or main trajectory.
    assert_array_equal(model.last_epoch_labels_, main.last_epoch_labels_)
    assert_array_equal(model.ema_centers_, main.ema_centers_)
    s, mass = init.astype(float)*1e-10, np.full(2, 1e-10)
    slow_s, slow_n = s.copy(), mass.copy()
    rng = np.random.default_rng(np.random.SeedSequence(0).spawn(3)[1])
    t, M = 0, 9
    a_slow = 0.25*0.1*0.3
    for _ in range(3):
        order = rng.permutation(5)
        for offset in range(0, 5, 2):
            groups = [[], []]
            for i in order[offset:offset+2]:
                distances = [sum((float(x[i,j])-s[z,j]/mass[z])**2 for j in range(2)) for z in range(2)]
                groups[int(np.argmin(distances))].append(x[i])
            alpha = 0.3*(0.1+0.9*(1+math.cos(math.pi*t/M))/2)
            for z in range(2):
                for j in range(2):
                    value = sum(float(row[j]) for row in groups[z])
                    s[z,j] = (1-alpha)*s[z,j]+alpha*value
                    slow_s[z,j] = (1-a_slow)*slow_s[z,j]+a_slow*value
                mass[z] = (1-alpha)*mass[z]+alpha*len(groups[z])
                slow_n[z] = (1-a_slow)*slow_n[z]+a_slow*len(groups[z])
            t += 1
    tol = 2e-6 if dtype == "float32" else 1e-13
    assert_allclose(model.cluster_centers_, slow_s/slow_n[:,None], atol=tol, rtol=tol)
    assert_allclose(model.slow_state_.counts, slow_n, atol=tol, rtol=tol)


def test_shadow_split_uses_main_ids_own_mass_and_geometry(monkeypatch):
    from e_means.state import EMAState
    calls = []
    original = EMAState.split
    def capture(self, donors, targets, epsilon):
        before, mass = self.centroids().copy(), self.counts.copy()
        original(self, donors, targets, epsilon)
        if len(donors):
            sign = np.array([1, -1, 1])*epsilon
            assert_allclose(self.centroids()[targets], before[donors]*(1+sign))
            assert_allclose(self.counts[targets], mass[donors]/2)
            calls.append((donors.copy(), targets.copy(), before, mass))
    monkeypatch.setattr(EMAState, "split", capture)
    x = np.tile([[2.,4.,8.],[2.1,4.1,8.1]], (10, 1))
    EMeans(EMeansConfig(2, epochs=1, batch_size=2, alpha0=0.2, finalize="slow_ema")).fit(x, init_centroids=[[2,4,8],[100,100,100]])
    assert len(calls) >= 2
    assert_array_equal(calls[0][0], calls[1][0])
    assert_array_equal(calls[0][1], calls[1][1])
    assert not np.allclose(calls[0][3], calls[1][3])  # shadow has its own mass
