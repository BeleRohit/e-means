"""Optional offline oracle tests; authors' code is never a library dependency.

Set EMEANS_REFERENCE_DIR to the pinned directory prepared by
scripts/fetch_reference.py and install the reference extra.
"""

import hashlib
import importlib.util
import functools
import os
from pathlib import Path
import sys
import types
import subprocess

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from e_means import EMeans, EMeansConfig
from e_means.assignment import assign_nearest, batch_statistics
from e_means.schedules import cosine_alpha
from e_means.state import EMAState, last_epoch_mean


HASHES = {
    "base.py": "bcffd7829ebfc25e1354308770b49e6f99d8566ba5e474a41a22e9ec414b17cd",
    "e_means.py": "b1710504fdbef21efdd33d8489a56f8e7fc6056182afbed3f355985302c2e77c",
}


@pytest.fixture(scope="module")
def reference():
    location = os.getenv("EMEANS_REFERENCE_DIR")
    if location is None:
        pytest.skip("set EMEANS_REFERENCE_DIR for pinned upstream comparisons")
    if importlib.util.find_spec("faiss") is None:
        pytest.skip("install faiss-cpu for pinned upstream comparisons")
    if os.getenv("EMEANS_ISOLATED_REFERENCE") != "1":
        return None  # actual oracle is loaded in a clean subprocess below
    faiss = pytest.importorskip("faiss")
    faiss.omp_set_num_threads(1)
    root = Path(location)
    for filename, expected in HASHES.items():
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected, "reference source does not match pinned commit"
    name = "_authors_emeans_reference"
    package = types.ModuleType(name)
    package.__path__ = [str(root)]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".e_means", root / "e_means.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pytestmark = pytest.mark.reference


def test_schedules_match_authors(reference):
    for step in range(17):
        assert_allclose(cosine_alpha(step, 17, 0.2), reference._scheduled_alpha(0.2, step, 17), atol=1e-15)


@pytest.mark.parametrize("initial_mass", [1e-10, 2**-16])
def test_batch_updates_and_finalizer_against_authors(reference, initial_mass):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 5)).astype("float32")
    init = x[[0, 20, 50]].copy()
    ref = reference.NumpyEmeans(5, 3, nepochs=2, batch_size=12, alpha0=0.2, unused_threshold=0)
    ref.begin(len(x), init)
    # Explicitly align the paper initialization; default upstream differs.
    ref._usage[:] = initial_mass
    state = EMAState.initialize(init, initial_mass)
    for epoch in range(2):
        closing_sums = np.zeros_like(init)
        closing_counts = np.zeros(3, dtype="float32")
        for offset in range(0, len(x), 12):
            batch = x[offset:offset+12]
            labels, _ = assign_nearest(batch, state.centroids(), dtype="float32")
            stats = ref.accumulate(batch)
            if epoch == 1:
                assert_array_equal(labels, stats["labels"])
            sums, counts = batch_statistics(batch, labels, 3)
            closing_sums += sums
            closing_counts += counts
            state.update(sums, counts, cosine_alpha(epoch*10+offset//12, 20, 0.2))
            ref.update(stats)
            assert_allclose(state.counts, ref._usage, rtol=2e-6, atol=1e-7)
            assert_allclose(state.centroids(), ref.current_centroids(), rtol=1e-4, atol=2e-6)
        ref.end_epoch()
    assert_allclose(last_epoch_mean(closing_sums, closing_counts, state.centroids()), ref.finalize(), rtol=2e-6, atol=2e-6)


def test_end_to_end_controlled_order_against_authors(reference):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(120, 5)).astype("float32")
    init = x[[0, 40, 80]]
    cfg = EMeansConfig(3, epochs=2, batch_size=12, alpha0=0.2, initial_mass=2**-16,
                      revival_threshold=0, dtype="float32", seed=11)
    ours = EMeans(cfg).fit(x, init_centroids=init)
    ref = reference.NumpyEmeans(5, 3, nepochs=2, batch_size=12, alpha0=0.2, unused_threshold=0)
    ref.begin(len(x), init)
    order_rng = np.random.default_rng(np.random.SeedSequence(11).spawn(3)[1])
    for _ in range(2):
        order = order_rng.permutation(len(x))
        for offset in range(0, len(x), 12):
            ref.update(ref.accumulate(x[order[offset:offset+12]]))
        ref.end_epoch()
    assert_allclose(ours.cluster_centers_, ref.finalize(), atol=3e-6, rtol=1e-5)


def test_split_geometry_against_authors_with_aligned_delta(reference, monkeypatch):
    init = np.array([[2., 4., 8.], [100., 100., 100.]], dtype="float32")
    ref = reference.NumpyEmeans(3, 2, nepochs=1, batch_size=2, alpha0=0.2, unused_threshold=0.1)
    ref.begin(2, init)
    ref._usage[:] = [10, 0]
    state = EMAState.initialize(init)
    state.counts[:] = [10, 0]
    state.sums[:] = init * state.counts[:, None]
    monkeypatch.setattr(reference, "_SPLIT_EPSILON", 2**-7)
    ref._apply_splits(np.array([0]), np.array([1]))
    state.revive(np.random.default_rng(0))
    assert_allclose(state.centroids(), ref.current_centroids(), atol=1e-7)
    assert_allclose(state.counts, ref._usage)


def test_full_loop_with_revival_and_closing_empty_slot(reference, monkeypatch):
    x = np.tile(np.array([[2., 4., 8.], [2.1, 4.1, 8.1]], dtype="float32"), (5, 1))
    init = np.array([[2., 4., 8.], [100., 100., 100.]], dtype="float32")
    cfg = EMeansConfig(2, epochs=1, batch_size=2, alpha0=0.2,
                      initial_mass=2**-16, dtype="float32")
    ours = EMeans(cfg).fit(x, init_centroids=init)
    ref = reference.NumpyEmeans(3, 2, nepochs=1, batch_size=2, alpha0=0.2, unused_threshold=0.1)
    monkeypatch.setattr(reference, "_SPLIT_EPSILON", 2**-7)
    ref.begin(len(x), init)
    order = np.random.default_rng(np.random.SeedSequence(0).spawn(3)[1]).permutation(len(x))
    for offset in range(0, len(x), 2):
        ref.update(ref.accumulate(x[order[offset:offset+2]]))
    ref.end_epoch()
    assert ours.n_revivals_ == 1
    assert_allclose(ours.cluster_centers_, ref.finalize(), rtol=1e-5, atol=2e-6)


def test_constant_shadow_recurrence_and_split_against_authors_primitives(reference, monkeypatch):
    # The release has no slow-EMA finalizer. Validate the specified constant
    # recurrence and own-state split using its existing arithmetic primitives;
    # this is not presented as an end-to-end authors' slow-EMA reproduction.
    init = np.array([[2., 4., 8.], [10., 20., 30.], [100., 100., 100.]], dtype="float32")
    ref = reference.NumpyEmeans(3, 3, nepochs=1, batch_size=8, alpha0=0.02, unused_threshold=0)
    ref.begin(80, init)
    ref._usage[:] = 1e-10
    state = EMAState.initialize(init)
    rng = np.random.default_rng(10)
    for _ in range(10):
        x = rng.normal(size=(8, 3)).astype("float32")
        labels = np.arange(8) % 2
        sums, counts = batch_statistics(x, labels, 3)
        state.update(sums, counts, 0.02)
        ref._apply_statistics(None, sums, counts, 0.0, 0.02, 0.98)
        assert_allclose(state.centroids(), ref.current_centroids(), atol=2e-6, rtol=2e-5)
        assert_allclose(state.counts, ref._usage, atol=1e-7, rtol=2e-6)
    monkeypatch.setattr(reference, "_SPLIT_EPSILON", 2**-7)
    state.split(np.array([0]), np.array([2]), 2**-7)
    ref._apply_splits(np.array([0]), np.array([2]))
    assert_allclose(state.centroids(), ref.current_centroids(), atol=2e-6, rtol=2e-5)
    assert_allclose(state.counts, ref._usage, atol=1e-7, rtol=2e-6)


def _isolated_reference_test(function):
    """Keep bundled Faiss/PyTorch OpenMP runtimes out of the same process.

    Each original oracle case runs unchanged in its own pytest process. Do not
    use KMP_DUPLICATE_LIB_OK, which bypasses the runtime's correctness guard.
    """
    @functools.wraps(function)
    def execute(*args, **kwargs):
        if os.getenv("EMEANS_ISOLATED_REFERENCE") == "1":
            return function(*args, **kwargs)
        node = os.environ["PYTEST_CURRENT_TEST"].removesuffix(" (call)")
        env = dict(os.environ, EMEANS_ISOLATED_REFERENCE="1")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", node], env=env,
                                text=True, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    return execute


for _name, _function in list(globals().items()):
    if _name.startswith("test_") and callable(_function):
        globals()[_name] = _isolated_reference_test(_function)
