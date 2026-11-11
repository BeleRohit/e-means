"""Independent numerical oracles for paper Eqs. 1–8 and Lloyd's algorithm."""

import math
from dataclasses import replace
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from e_means import EMeans, EMeansConfig, LloydConfig, LloydKMeans, assign_nearest, mse
from e_means.assignment import batch_statistics
from e_means.schedules import cosine_alpha, paper_alpha0, paper_batch_size, resolve_recipe
from e_means.state import EMAState, last_epoch_mean


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("chunks", [(1, 1), (7, 3), (100, 100)])
def test_assignment_matches_broadcast_oracle(dtype, chunks):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(23, 5)).astype(dtype)
    c = rng.normal(size=(9, 5)).astype(dtype)
    distances = ((x[:, None] - c[None]) ** 2).sum(axis=2)
    labels, actual = assign_nearest(x, c, chunk_size=chunks[0], centroid_chunk_size=chunks[1], dtype=dtype)
    assert_array_equal(labels, distances.argmin(axis=1))
    assert_allclose(actual, distances.min(axis=1), rtol=2e-7)


def test_large_offset_and_ties():
    x = np.array([[1e12 + 1, 1e12]])
    c = np.array([[1e12, 1e12], [1e12 + 2, 1e12], [1e12 + 1, 1e12 + 3]])
    labels, dist = assign_nearest(x, c, centroid_chunk_size=1)
    assert_array_equal(labels, [0])
    assert_allclose(dist, [1])
    assert mse([[0, 0]], [[2, 3]]) == 13  # Eq. 1 divides by n, not n*d


def test_statistics_and_hand_computed_ema():
    x = np.array([[1., 2.], [3., 4.], [9., 8.]])
    sums, counts = batch_statistics(x, np.array([0, 0, 1]), 3)
    assert_array_equal(counts, [2, 1, 0])
    assert_array_equal(sums, [[4, 6], [9, 8], [0, 0]])
    state = EMAState.initialize(np.array([[0., 0.], [10., 10.], [20., 20.]]), mass=2)
    state.update(sums, counts, alpha=0.25)
    assert_allclose(state.counts, [2, 1.75, 1.5])
    assert_allclose(state.sums, [[1, 1.5], [17.25, 17], [30, 30]])
    assert_allclose(state.centroids(), state.sums / state.counts[:, None])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_alpha_one_empty_and_underflow_are_finite(dtype):
    state = EMAState.initialize(np.array([[4., 2.], [9., 1.]], dtype=dtype))
    state.update(np.array([[8., 4.], [0., 0.]], dtype=dtype), np.array([2., 0.], dtype=dtype), 1)
    assert_allclose(state.centroids(), [[4, 2], [9, 1]])
    for _ in range(1100):
        state.update(np.zeros((2, 2), dtype=dtype), np.zeros(2, dtype=dtype), 0.5)
    assert np.isfinite(state.centroids()).all()
    assert_allclose(state.centroids(), [[4, 2], [9, 1]])


def test_cosine_literal_equation_not_endpoint_interpolation():
    values = [cosine_alpha(t, 10, 0.8) for t in range(10)]
    assert values[0] == 0.8
    assert_allclose(values[5], 0.8 * 0.55)
    assert values[-1] > 0.08
    assert np.all(np.diff(values) < 0)
    assert cosine_alpha(0, 1, 0.8) == 0.8


def test_published_scaling_laws_and_overrides():
    n, k, t = 1_000_000, 1024, 3
    raw = 0.035 * k**0.13 * n**0.88 * t**0.81
    expected = 2 ** math.floor(math.log2(raw) + 0.5)
    assert paper_batch_size(n, k, t) == expected
    assert_allclose(paper_alpha0(n, k, t, expected), min(0.824*k**-0.46*expected**0.90*n**-0.45*t**-0.11, 1))
    n = 2_000_001
    raw = 0.035*k**0.13*n**0.88*t**0.81 / 8
    assert paper_batch_size(n, k, t) == 2 ** math.floor(math.log2(raw) + 0.5)
    assert paper_batch_size(1, 1, 1) == 1
    cfg = EMeansConfig(2, epochs=3, batch_size=7)
    recipe = resolve_recipe(cfg, 20)
    assert recipe["total_steps"] == 6
    assert recipe["alpha0"] == paper_alpha0(20, 2, 3, 7)
    assert resolve_recipe(replace(cfg, drop_last=False), 20)["total_steps"] == 9
    assert resolve_recipe(replace(cfg, batch_size=100), 20)["batch_size"] == 20


def test_revival_equation_and_donor_cap():
    state = EMAState.initialize(np.array([[2., 4., 8.], [10., 10., 10.], [20., 20., 20.]]))
    state.counts[:] = [10, 0, 0]
    state.sums[:] = state.fallback * state.counts[:, None]
    donors, targets = state.revive(np.random.default_rng(12))
    assert_array_equal(donors, [0])
    assert_array_equal(targets, [1])  # one live donor: third slot waits
    assert_array_equal(state.counts, [5, 5, 0])
    delta = 2**-7 * np.array([1, -1, 1])
    assert_allclose(state.centroids()[0], np.array([2, 4, 8]) * (1-delta))
    assert_allclose(state.centroids()[1], np.array([2, 4, 8]) * (1+delta))
    assert_allclose(state.sums[:2], state.centroids()[:2] * state.counts[:2, None])
    assert_allclose(state.sums.sum(axis=0), [20, 40, 80])


def test_revival_excludes_dead_and_samples_without_replacement():
    state = EMAState.initialize(np.arange(18, dtype=float).reshape(6, 3))
    state.counts[:] = [10, 20, 30, 0, 0, 0]
    state.sums[:] = state.fallback * state.counts[:, None]
    donors, targets = state.revive(np.random.default_rng(0))
    assert set(donors) == {0, 1, 2}
    assert_array_equal(targets, [3, 4, 5])


def test_zero_perturbation_follows_equation_and_disable():
    state = EMAState.initialize(np.zeros((2, 2)))
    state.counts[:] = [2, 0]
    assert len(state.revive(np.random.default_rng(0), threshold=0)[0]) == 0
    state.revive(np.random.default_rng(0))
    assert_allclose(state.centroids(), 0)  # Eq. 8 cannot separate zero vectors


def test_closing_means_minimize_fixed_assignment_sse():
    x = np.array([[0., 1.], [2., 3.], [9., 8.]])
    labels = np.array([0, 0, 1])
    old = np.array([[4., 4.], [6., 6.], [99., 99.]])
    sums, counts = batch_statistics(x, labels, 3)
    new = last_epoch_mean(sums, counts, old)
    assert_allclose(new, [[1, 2], [9, 8], [99, 99]])
    assert ((x-new[labels])**2).sum() <= ((x-old[labels])**2).sum()


def test_lloyd_hand_result_empty_and_monotonicity():
    x = np.array([[0.], [2.], [8.], [10.]])
    model = LloydKMeans(LloydConfig(3, max_iter=10)).fit(x, init_centroids=[[0], [8], [100]])
    assert_allclose(model.cluster_centers_, [[1], [9], [100]])
    assert model.mse_ == 1
    assert model.inertia_ == 4
    assert_array_equal(model.labels_, [0, 0, 1, 1])
    assert np.all(np.diff([row.train_mse for row in model.history_]) <= 1e-12)
    assert model.n_iter_ == 2


def test_lloyd_random_monotonicity():
    x = np.random.default_rng(71).normal(size=(150, 8))
    model = LloydKMeans(LloydConfig(9, max_iter=20)).fit(x)
    assert np.all(np.diff([row.train_mse for row in model.history_]) <= 1e-12)
    assert_array_equal(model.labels_, model.predict(x))


@pytest.mark.parametrize("keep_tail", [True, False])
def test_e_means_matches_independent_scalar_training_oracle(keep_tail):
    # Scalar oracle: no call to implementation's statistics, EMA or schedule.
    x = np.array([[0., 1.], [2., 1.], [7., 3.], [9., 2.], [10., 4.]])
    init = x[[0, 4]].copy()
    cfg = EMeansConfig(2, epochs=3, batch_size=2, alpha0=0.3,
                      revival_threshold=0, drop_last=not keep_tail)
    model = EMeans(cfg).fit(x, init_centroids=init)
    rng = np.random.default_rng(np.random.SeedSequence(0).spawn(3)[1])
    s, mass = init * 1e-10, np.full(2, 1e-10)
    steps = 3 * (3 if keep_tail else 2)
    t = 0
    for epoch in range(3):
        order = rng.permutation(5)
        closing = [[], []]
        for offset in range(0, 5 if keep_tail else 4, 2):
            groups = [[], []]
            for i in order[offset:min(offset+2, 5 if keep_tail else 4)]:
                center = s / mass[:, None]
                distances = [sum((x[i, j]-center[z, j])**2 for j in range(2)) for z in range(2)]
                label = int(np.argmin(distances))
                groups[label].append(x[i])
                closing[label].append(x[i])
            a = 0.3 * (0.1 + 0.9 * (1+math.cos(math.pi*t/steps))/2)
            for z in range(2):
                for j in range(2):
                    s[z, j] = (1-a)*s[z, j] + a*sum(row[j] for row in groups[z])
                mass[z] = (1-a)*mass[z] + a*len(groups[z])
            t += 1
    expected = s / mass[:, None]
    for z in range(2):
        if closing[z]:
            expected[z] = np.mean(closing[z], axis=0)
    assert_allclose(model.cluster_centers_, expected, atol=1e-13)
    assert model.n_steps_ == steps
    assert np.count_nonzero(model.last_epoch_labels_ < 0) == (0 if keep_tail else 1)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_full_batch_one_step_and_finalization(dtype):
    x = np.array([[0., 0.], [2., 2.], [8., 8.], [10., 10.]], dtype=dtype)
    model = EMeans(EMeansConfig(2, epochs=1, batch_size=4, alpha0=1, dtype=dtype)).fit(x, init_centroids=x[[0, 3]])
    assert_allclose(model.cluster_centers_, [[1, 1], [9, 9]])
    assert_array_equal(model.labels_, model.predict(x))
    assert model.cluster_centers_.dtype == np.dtype(dtype)


def test_revival_occurs_every_five_global_updates():
    x = np.tile([[2., 4.], [3., 5.]], (10, 1))
    cfg = EMeansConfig(3, epochs=2, batch_size=2, alpha0=0.2)
    model = EMeans(cfg).fit(x, init_centroids=[[2, 4], [1000, 1000], [2000, 2000]])
    assert model.history_[1].revivals > 0
    assert model.n_revivals_ > 0
    assert np.isfinite(model.cluster_centers_).all()
    for label in range(3):
        selected = x[model.last_epoch_labels_ == label]
        if len(selected):
            assert_allclose(model.cluster_centers_[label], selected.mean(axis=0))


def test_revival_cadence_crosses_epoch_boundaries(monkeypatch):
    calls, steps = [], [0]
    original_update, original_revive = EMAState.update, EMAState.revive
    def update(self, *args):
        original_update(self, *args)
        steps[0] += 1
    def revive(self, *args):
        calls.append(steps[0])
        return original_revive(self, *args)
    monkeypatch.setattr(EMAState, "update", update)
    monkeypatch.setattr(EMAState, "revive", revive)
    EMeans(EMeansConfig(2, epochs=4, batch_size=2)).fit(np.arange(12, dtype=float).reshape(6, 2))
    assert calls == [5, 10]  # 3 updates per epoch, never reset cadence at epoch end


def test_weighted_donor_sampling_favors_larger_live_cluster():
    rng = np.random.default_rng(41)
    chosen = []
    for _ in range(1000):
        state = EMAState.initialize(np.ones((3, 2)))
        state.counts[:] = [9, 1, 0]
        state.sums[:] = state.counts[:, None]
        donor, _ = state.revive(rng)
        chosen.append(donor[0])
    # The exact probability is .9. Broad bound detects uniform selection
    # without demanding a particular RNG implementation/sequence.
    assert 0.85 < np.mean(np.array(chosen) == 0) < 0.95


def test_no_finalization_readout_and_precision_agreement():
    x = np.random.default_rng(9).normal(size=(96, 7))
    cfg = EMeansConfig(3, epochs=2, batch_size=12, alpha0=0.2,
                      revival_threshold=0, finalize="none")
    a = EMeans(cfg).fit(x, init_centroids=x[:3])
    b = EMeans(replace(cfg, dtype="float32")).fit(x, init_centroids=x[:3])
    assert_array_equal(a.cluster_centers_, a.ema_centers_)
    assert_allclose(a.cluster_centers_, b.cluster_centers_, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("model", [LloydKMeans(LloydConfig(4)), EMeans(EMeansConfig(4, batch_size=7))])
def test_reproducibility_input_ownership_and_refit(model):
    x = np.random.default_rng(2).normal(size=(43, 5))
    original = x.copy()
    np.random.seed(55)
    model.fit(x)
    centers, labels = model.cluster_centers_.copy(), model.labels_.copy()
    np.random.seed(88)
    model.fit(x)
    assert_array_equal(model.cluster_centers_, centers)
    assert_array_equal(model.labels_, labels)
    assert_array_equal(x, original)
    assert model.mse_ == mse(x, centers)


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": 0}, {"alpha0": 0}, {"alpha0": float("nan")},
    {"alpha0": 1.1}, {"revival_threshold": -1}, {"split_epsilon": float("inf")},
    {"initial_mass": 0}, {"dtype": "float16"}, {"seed": -1}, {"epochs": 1.5}, {"drop_last": 1}])
def test_bad_configs(kwargs):
    with pytest.raises(ValueError):
        EMeansConfig(2, **kwargs)


@pytest.mark.parametrize("bad", [[], [[1], [float("nan")]], [[float("inf")]], [[1+1j]], [["x"]]])
def test_bad_input(bad):
    with pytest.raises(ValueError):
        EMeans(EMeansConfig(1)).fit(bad)


def test_validation_and_unfitted():
    with pytest.raises(RuntimeError):
        LloydKMeans(LloydConfig(1)).predict([[0]])
    with pytest.raises(ValueError):
        EMeans(EMeansConfig(3)).fit([[0], [1]])
    with pytest.raises(ValueError):
        EMeans(EMeansConfig(1)).fit([[0], [1]], init_centroids=[[0, 1]])
    with pytest.raises(ValueError):
        EMeans(EMeansConfig(1)).fit([[0], [1]], eval_X=[[0, 1]])
    with pytest.raises(ValueError):
        assign_nearest([[1]], [[0]], chunk_size=0)
    with pytest.raises(FloatingPointError):
        assign_nearest([[1e308]], [[-1e308]])


def test_degenerate_identical_data_and_singleton():
    for x, k in [(np.zeros((12, 3)), 4), (np.array([[3., 4.]]), 1)]:
        for model in (LloydKMeans(LloydConfig(k)), EMeans(EMeansConfig(k, batch_size=2, alpha0=1))):
            model.fit(x)
            assert model.mse_ == 0
            assert np.isfinite(model.cluster_centers_).all()
