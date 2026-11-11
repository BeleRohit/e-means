"""Historical Phase 1 centers/labels remain an immutable regression oracle."""

import json
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from e_means import EMeans, EMeansConfig, LloydKMeans, LloydConfig


def test_original_twelve_run_artifacts_are_unchanged():
    root = Path(__file__).resolve().parent / "fixtures/phase1"
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["runs"]) == 12
    for name in manifest["runs"]:
        record = json.loads((root / name).read_text())
        spec, seed = record["dataset"], record["seed"]
        rng = np.random.default_rng(seed)
        k, d = spec["n_clusters"], spec["n_features"]
        centers = rng.normal(size=(k, d)) * spec["center_scale"]
        x = centers[rng.integers(k, size=spec["n_train"])] + rng.normal(size=(spec["n_train"], d))*spec["noise_scale"]
        # Consume the original test-generation RNG stream before initialization.
        _ = centers[rng.integers(k, size=spec["n_test"])] + rng.normal(size=(spec["n_test"], d))*spec["noise_scale"]
        init = x[rng.choice(len(x), k, replace=False)].copy()
        cls, cfg = (EMeans, EMeansConfig) if record["algorithm"] == "e_means" else (LloydKMeans, LloydConfig)
        model = cls(cfg(**record["config"])).fit(x, init_centroids=init)
        with np.load(root / name.replace(".json", ".npz")) as expected:
            assert_allclose(model.cluster_centers_, expected["centroids"], atol=1e-12, rtol=1e-12)
            assert_array_equal(model.labels_, expected["labels"])
