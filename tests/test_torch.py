"""CPU tensor parity with independent NumPy baseline; optional CUDA correctness."""

from dataclasses import replace
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

torch = pytest.importorskip("torch")

from e_means import EMeans, EMeansConfig, LloydConfig, LloydKMeans
from e_means.assignment import assign_nearest as numpy_assign, batch_statistics as numpy_stats
from e_means.state import EMAState, last_epoch_mean as numpy_closing
from e_means.schedules import cosine_alpha
from e_means.torch_backend import (TorchEMeans, TorchLloydKMeans, TorchEMAState,
                                  assign_nearest, batch_statistics, last_epoch_mean)


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tolerances(dtype):
    # Direct sequential FP64 sums: roundoff ~1e-15. FP32 repeated arithmetic
    # allows up to 2e-5 absolute/relative error over multi-epoch trajectories.
    return dict(atol=2e-5, rtol=2e-5) if dtype == "float32" else dict(atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("chunks", [(1, 1), (7, 3), (128, 128)])
def test_assignment_and_statistics(dtype, chunks):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(31, 7)).astype(dtype)
    c = rng.normal(size=(6, 7)).astype(dtype)
    labels, dist = assign_nearest(torch.from_numpy(x), torch.from_numpy(c), chunk_size=chunks[0], centroid_chunk_size=chunks[1])
    expected_labels, expected_dist = numpy_assign(x, c, dtype=dtype)
    assert_array_equal(labels.numpy(), expected_labels)
    assert_allclose(dist.numpy(), expected_dist, **tolerances(dtype))
    s, n = batch_statistics(torch.from_numpy(x), labels, 6)
    ns, nn = numpy_stats(x, expected_labels, 6)
    assert_allclose(s.numpy(), ns, **tolerances(dtype))
    assert_array_equal(n.numpy(), nn)


def test_large_offset_ties_and_overflow():
    x = torch.tensor([[1e12+1, 1e12]], dtype=torch.float64)
    c = torch.tensor([[1e12, 1e12], [1e12+2, 1e12]], dtype=torch.float64)
    labels, dist = assign_nearest(x, c, centroid_chunk_size=1)
    assert labels.item() == 0 and dist.item() == 1
    with pytest.raises(FloatingPointError):
        assign_nearest(torch.tensor([[1e308]], dtype=torch.float64), torch.tensor([[-1e308]], dtype=torch.float64))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_state_updates_schedule_split_and_finalizer(dtype):
    c = np.array([[2., 4., 8.], [10., 20., 30.], [100., 100., 100.]], dtype=dtype)
    ns = EMAState.initialize(c)
    ts = TorchEMAState.initialize(torch.from_numpy(c))
    rng = np.random.default_rng(3)
    for step in range(10):
        rows = rng.normal(size=(8, 3)).astype(dtype)
        labels = np.arange(8) % 2  # leave one empty
        sums, counts = numpy_stats(rows, labels, 3)
        alpha = cosine_alpha(step, 10, 0.2)
        ns.update(sums, counts, alpha)
        ts.update(torch.from_numpy(sums), torch.from_numpy(counts), alpha)
        assert_allclose(ts.counts.numpy(), ns.counts, **tolerances(dtype))
        assert_allclose(ts.sums.numpy(), ns.sums, **tolerances(dtype))
        assert_allclose(ts.centroids().numpy(), ns.centroids(), **tolerances(dtype))
    nd, nt = ns.revive(np.random.default_rng(21))
    td, tt = ts.revive(np.random.default_rng(21))
    assert_array_equal(td, nd)
    assert_array_equal(tt, nt)
    assert_allclose(ts.centroids().numpy(), ns.centroids(), **tolerances(dtype))
    assert_allclose(last_epoch_mean(torch.from_numpy(sums), torch.from_numpy(counts), ts.centroids()).numpy(),
                    numpy_closing(sums, counts, ns.centroids()), **tolerances(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_alpha_one_empty_state(dtype):
    state = TorchEMAState.initialize(torch.tensor([[1., 2.], [3., 4.]], dtype=getattr(torch, dtype)))
    state.update(torch.tensor([[4., 8.], [0., 0.]], dtype=getattr(torch, dtype)),
                 torch.tensor([2., 0.], dtype=getattr(torch, dtype)), 1)
    assert_allclose(state.centroids().numpy(), [[2, 4], [3, 4]])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("finalizer", ["none", "last_epoch_mean", "slow_ema"])
@pytest.mark.parametrize("drop_last", [False, True])
def test_full_trajectory_with_controlled_random_choices(dtype, finalizer, drop_last, monkeypatch):
    x = np.random.default_rng(27).normal(size=(65, 4)).astype(dtype)
    # Force actual revivals with two initially remote centroids.
    init = np.vstack([x[:2], np.full((2, 4), 100)]).astype(dtype)
    cfg = EMeansConfig(4, epochs=3, batch_size=7, alpha0=0.2, seed=22,
                      dtype=dtype, finalize=finalizer, drop_last=drop_last,
                      assignment_chunk_size=5, centroid_chunk_size=2)
    numpy_states, torch_states, numpy_splits, torch_splits = [], [], [], []
    nupdate, tupdate = EMAState.update, TorchEMAState.update
    nrevive, trevive = EMAState.revive, TorchEMAState.revive
    def nu(self, *args):
        nupdate(self, *args)
        numpy_states.append((self.sums.copy(), self.counts.copy()))
    def tu(self, *args):
        tupdate(self, *args)
        torch_states.append((self.sums.numpy().copy(), self.counts.numpy().copy()))
    def nr(self, *args):
        pair = nrevive(self, *args)
        numpy_splits.append(pair)
        return pair
    def tr(self, *args):
        pair = trevive(self, *args)
        torch_splits.append(pair)
        return pair
    monkeypatch.setattr(EMAState, "update", nu)
    monkeypatch.setattr(TorchEMAState, "update", tu)
    monkeypatch.setattr(EMAState, "revive", nr)
    monkeypatch.setattr(TorchEMAState, "revive", tr)
    a = EMeans(cfg).fit(x, init_centroids=init, eval_X=x[:9])
    b = TorchEMeans(cfg).fit(torch.from_numpy(x), init_centroids=init, eval_X=x[:9])
    assert len(numpy_states) == len(torch_states) > 0
    for ns, ts in zip(numpy_states, torch_states):
        for expected, actual in zip(ns, ts):
            assert_allclose(actual, expected, **tolerances(dtype))
    for ns, ts in zip(numpy_splits, torch_splits):
        for expected, actual in zip(ns, ts):
            assert_array_equal(actual, expected)
    assert a.n_revivals_ == b.n_revivals_ > 0
    for attribute in ("cluster_centers_", "ema_centers_", "last_epoch_sums_", "last_epoch_counts_"):
        assert_allclose(getattr(b, attribute).numpy(), getattr(a, attribute), **tolerances(dtype))
    assert_array_equal(b.labels_.numpy(), a.labels_)
    assert_array_equal(b.last_epoch_labels_.numpy(), a.last_epoch_labels_)
    for na, tb in zip(a.history_, b.history_):
        assert_allclose([tb.train_mse, tb.test_mse], [na.train_mse, na.test_mse], **tolerances(dtype))
        if na.moving_mse is not None:
            assert_allclose(tb.moving_mse, na.moving_mse, **tolerances(dtype))
        assert tb.updates == na.updates
    assert a.recipe_ == b.recipe_


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_lloyd_full_trajectory(dtype):
    x = np.random.default_rng(12).normal(size=(91, 6)).astype(dtype)
    cfg = LloydConfig(5, max_iter=8, dtype=dtype, assignment_chunk_size=7, centroid_chunk_size=2)
    a, b = LloydKMeans(cfg).fit(x), TorchLloydKMeans(cfg).fit(x)
    assert_allclose(b.cluster_centers_.numpy(), a.cluster_centers_, **tolerances(dtype))
    assert_array_equal(b.labels_.numpy(), a.labels_)
    assert_allclose([r.train_mse for r in b.history_], [r.train_mse for r in a.history_], **tolerances(dtype))
    assert np.all(np.diff([r.train_mse for r in b.history_]) <= 1e-6)


@pytest.mark.parametrize("cls,cfg", [(TorchEMeans, EMeansConfig(2)), (TorchLloydKMeans, LloydConfig(2))])
def test_reproducibility_ownership_and_no_grad(cls, cfg):
    x = torch.arange(30, dtype=torch.float64).reshape(15, 2).requires_grad_()
    original = x.detach().clone()
    model = cls(cfg).fit(x)
    centers = model.cluster_centers_.clone()
    torch.manual_seed(100)
    model.fit(x)
    assert torch.equal(centers, model.cluster_centers_)
    assert torch.equal(x.detach(), original)
    assert not model.cluster_centers_.requires_grad and x.grad is None
    assert torch.equal(model.predict(x), model.labels_)


@pytest.mark.parametrize("bad", [[], [[float("nan")]], [[1+1j]], [[True]]])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        TorchEMeans(EMeansConfig(1)).fit(bad)


def test_validation_and_empty_clusters():
    with pytest.raises(RuntimeError):
        TorchEMeans(EMeansConfig(1)).predict([[0]])
    with pytest.raises(ValueError):
        TorchEMeans(EMeansConfig(3)).fit([[0], [1]])
    with pytest.raises(ValueError):
        TorchEMeans(EMeansConfig(1), device="mps")
    with pytest.raises(ValueError):
        TorchEMeans(EMeansConfig(1)).fit([[0]], eval_X=[[0, 1]])
    for cls, cfg in ((TorchEMeans, EMeansConfig(3, alpha0=1)), (TorchLloydKMeans, LloydConfig(3))):
        m = cls(cfg).fit(torch.zeros((12, 3)))
        assert m.mse_ == 0 and bool(torch.isfinite(m.cluster_centers_).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable; no CUDA performance claim")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("algorithm", ["lloyd", "emeans"])
def test_cuda_correctness(dtype, algorithm):
    x = np.random.default_rng(13).normal(size=(80, 5)).astype(dtype)
    cfg = (EMeansConfig(4, epochs=2, batch_size=8, dtype=dtype, finalize="slow_ema") if algorithm == "emeans"
           else LloydConfig(4, max_iter=5, dtype=dtype))
    cls = TorchEMeans if algorithm == "emeans" else TorchLloydKMeans
    expected = cls(cfg).fit(x)
    actual = cls(cfg, device="cuda").fit(torch.from_numpy(x).cuda())
    assert actual.cluster_centers_.device.type == "cuda"
    assert_allclose(actual.cluster_centers_.cpu().numpy(), expected.cluster_centers_.numpy(), **tolerances(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_half_life_backend_parity(dtype):
    x = np.random.default_rng(14).normal(size=(51, 4)).astype(dtype)
    cfg = EMeansConfig(3, epochs=2, batch_size=8, dtype=dtype, finalize="slow_ema", slow_half_life=20)
    a, b = EMeans(cfg).fit(x), TorchEMeans(cfg).fit(x)
    assert_allclose(b.cluster_centers_.numpy(), a.cluster_centers_, **tolerances(dtype))
