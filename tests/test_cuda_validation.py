"""Milestone CUDA matrix plus CPU rehearsals of every assertion.

CPU parameter cases validate the test design locally. Only actual `cuda`
parameter cases can satisfy the NVIDIA execution gate. No emulation is used.
"""

from dataclasses import replace
import math
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

torch = pytest.importorskip("torch")
from e_means import EMeans, EMeansConfig, LloydKMeans, LloydConfig
from e_means.assignment import assign_nearest as np_assign, batch_statistics as np_stats
from e_means.state import EMAState
from e_means.torch_backend import (TorchEMeans, TorchLloydKMeans, TorchEMAState,
                                  assign_nearest, batch_statistics, last_epoch_mean)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def execution_device(request):
    if request.param == "cuda" and (not torch.cuda.is_available() or torch.version.cuda is None):
        pytest.skip("actual NVIDIA CUDA hardware/build required")
    previous = torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    yield request.param
    torch.use_deterministic_algorithms(previous)
    torch.set_num_threads(threads)


def tol(dtype, trajectory=False):
    # Primitives are small sums. The trajectory budget allows accumulated
    # reduction-order error, not arbitrary assignment changes: exact codes and
    # donor IDs are checked on these controlled fixtures. Never auto-relax.
    if dtype == "float32":
        return dict(atol=1e-4 if trajectory else 3e-6, rtol=1e-4 if trajectory else 3e-6)
    return dict(atol=1e-10 if trajectory else 2e-12, rtol=1e-10 if trajectory else 2e-12)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("chunks", [(3, 2), (64, 64)])
def test_assignment_tiling_and_first_tie(execution_device, dtype, chunks):
    x = np.random.default_rng(40).normal(size=(31, 32)).astype(dtype)
    c = x[[0, 7, 19, 28]].copy()
    c[1] = c[0]  # exact duplicate exercises deterministic first-index ties
    expected, distance = np_assign(x, c, dtype=dtype)
    labels, measured = assign_nearest(torch.tensor(x, device=execution_device), torch.tensor(c, device=execution_device),
                                      chunk_size=chunks[0], centroid_chunk_size=chunks[1])
    assert labels.device.type == execution_device
    assert_array_equal(labels.cpu().numpy(), expected)
    assert_allclose(measured.cpu().numpy(), distance, **tol(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ema_counts_sums_and_literal_cosine(execution_device, dtype):
    c = np.array([[2,4,8], [8,12,16], [50,50,50]], dtype=dtype)
    reference = EMAState.initialize(c)
    state = TorchEMAState.initialize(torch.tensor(c, device=execution_device))
    rng = np.random.default_rng(8)
    for step in range(17):
        x = rng.normal(size=(19, 3)).astype(dtype)
        labels = np.arange(19) % 2
        sums, counts = np_stats(x, labels, 3)
        ts, tn = batch_statistics(torch.tensor(x, device=execution_device),
                                  torch.tensor(labels, device=execution_device), 3)
        # Literal Eq. 5, independent of shared schedule implementation.
        alpha = 0.3*(0.1+0.9*(1+math.cos(math.pi*step/17))/2)
        reference.update(sums, counts, alpha)
        state.update(ts, tn, alpha)
        assert_allclose(state.sums.cpu().numpy(), reference.sums, **tol(dtype))
        assert_allclose(state.counts.cpu().numpy(), reference.counts, **tol(dtype))
        assert_allclose(state.centroids().cpu().numpy(), reference.centroids(), **tol(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_weighted_revival_and_shadow_own_split(execution_device, dtype):
    c = np.arange(12, dtype=dtype).reshape(4, 3) + 1
    reference = EMAState.initialize(c)
    reference.counts[:] = [16, 8, 0, 0]
    reference.sums[:] = c*reference.counts[:,None]
    state = TorchEMAState.initialize(torch.tensor(c, device=execution_device))
    state.counts[:] = torch.tensor(reference.counts, device=execution_device)
    state.sums[:] = torch.tensor(reference.sums, device=execution_device)
    donors, targets = reference.revive(np.random.default_rng(12))
    td, tt = state.revive(np.random.default_rng(12))
    assert_array_equal(td, donors)
    assert_array_equal(tt, targets)
    assert len(set(donors)) == 2 and set(donors) == {0,1}
    assert_allclose(state.centroids().cpu().numpy(), reference.centroids(), **tol(dtype))
    assert_allclose(state.counts.cpu().numpy(), reference.counts, **tol(dtype))
    shadow = TorchEMAState.initialize(torch.tensor(c*2, device=execution_device), mass=2)
    shadow.split(donors, targets, 2**-7)
    assert_allclose(shadow.counts.cpu().numpy()[targets], 1)
    sign = np.array([1,-1,1])*2**-7
    assert_allclose(shadow.centroids().cpu().numpy()[targets], c[donors]*2*(1+sign), **tol(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_closing_and_constant_shadow_finalizers(execution_device, dtype):
    c = np.array([[0.,0.],[10.,10.],[99.,99.]], dtype=dtype)
    sums = np.array([[4.,6.],[9.,8.],[0.,0.]], dtype=dtype)
    counts = np.array([2.,1.,0.], dtype=dtype)
    tensor = lambda x: torch.tensor(x, device=execution_device)
    result = last_epoch_mean(tensor(sums), tensor(counts), tensor(c))
    assert_allclose(result.cpu().numpy(), [[2,3],[9,8],[99,99]], **tol(dtype))
    state = TorchEMAState.initialize(tensor(c))
    ns, nn = c.astype(float)*1e-10, np.full(3, 1e-10)
    alpha = 1-2**(-1/10)  # explicit half-life=10 logical updates
    for _ in range(20):
        state.update(tensor(sums), tensor(counts), alpha)
        ns = (1-alpha)*ns+alpha*sums
        nn = (1-alpha)*nn+alpha*counts
    assert_allclose(state.centroids().cpu().numpy(), ns/nn[:,None], **tol(dtype, True))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("algorithm", ["lloyd", "none", "last_epoch_mean", "slow_ema"])
@pytest.mark.parametrize("drop_last", [False, True])
def test_full_training_against_both_cpu_references(execution_device, dtype, algorithm, drop_last, monkeypatch):
    x = np.random.default_rng(27).normal(size=(65, 4)).astype(dtype)
    init = np.vstack([x[:2], np.full((2,4), 100)]).astype(dtype)
    if algorithm == "lloyd":
        cfg = LloydConfig(4, max_iter=6, dtype=dtype, assignment_chunk_size=9, centroid_chunk_size=2)
        numpy_cls, cls = LloydKMeans, TorchLloydKMeans
    else:
        cfg = EMeansConfig(4, epochs=3, batch_size=7, alpha0=0.2, seed=22, dtype=dtype,
                          finalize=algorithm, drop_last=drop_last, assignment_chunk_size=9, centroid_chunk_size=2)
        numpy_cls, cls = EMeans, TorchEMeans
    expected = numpy_cls(cfg).fit(x, init_centroids=init, eval_X=x[:11])
    cpu = cls(cfg, device="cpu").fit(x, init_centroids=init, eval_X=x[:11])
    observed_splits = []
    original = TorchEMAState.revive
    def observe(self, *args):
        result = original(self, *args)
        observed_splits.append(result)
        return result
    monkeypatch.setattr(TorchEMAState, "revive", observe)
    actual = cls(cfg, device=execution_device).fit(torch.tensor(x, device=execution_device), init_centroids=init, eval_X=x[:11])
    for reference in (expected.cluster_centers_, cpu.cluster_centers_.numpy()):
        assert_allclose(actual.cluster_centers_.cpu().numpy(), reference, **tol(dtype, True))
    assert_array_equal(actual.labels_.cpu().numpy(), expected.labels_)
    assert_array_equal(actual.predict(x).cpu().numpy(), expected.labels_)
    for a, b in zip(actual.history_, expected.history_):
        assert_allclose([a.train_mse,a.test_mse], [b.train_mse,b.test_mse], **tol(dtype, True))
        assert a.updates == b.updates and a.revivals == b.revivals
    if algorithm != "lloyd":
        assert actual.n_revivals_ == expected.n_revivals_ > 0
        assert sum(len(pair[0]) for pair in observed_splits) == actual.n_revivals_
        assert_array_equal(actual.last_epoch_labels_.cpu().numpy(), expected.last_epoch_labels_)
        assert_allclose(actual.ema_state_.sums.cpu().numpy(), expected.ema_state_.sums, **tol(dtype, True))
        if algorithm == "slow_ema":
            assert_allclose(actual.slow_state_.sums.cpu().numpy(), expected.slow_state_.sums, **tol(dtype, True))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_singleton_empty_and_no_grad(execution_device, dtype):
    state = TorchEMAState.initialize(torch.tensor([[2.,4.],[8.,16.]], dtype=getattr(torch,dtype), device=execution_device))
    state.update(torch.tensor([[4.,8.],[0.,0.]], dtype=getattr(torch,dtype), device=execution_device),
                 torch.tensor([2.,0.], dtype=getattr(torch,dtype), device=execution_device), 1)
    assert_allclose(state.centroids().cpu().numpy(), [[2,4],[8,16]])
    x = torch.zeros((12,3), dtype=getattr(torch,dtype), device=execution_device, requires_grad=True)
    model = TorchEMeans(EMeansConfig(3, epochs=1, alpha0=1, dtype=dtype), device=execution_device).fit(x)
    assert model.mse_ == 0 and not model.cluster_centers_.requires_grad and x.grad is None
