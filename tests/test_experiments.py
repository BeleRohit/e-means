"""Exercise experiment isolation, artifact pairing and deterministic reruns."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_array_equal


def test_experiment_artifacts_and_reproducibility(tmp_path):
    config = {
        "schema_version": 1, "seeds": [12],
        "datasets": [{"name": "tiny", "n_train": 21, "n_test": 10,
                      "n_features": 3, "n_clusters": 2, "center_scale": 3., "noise_scale": 1.}],
        "lloyd": {"max_iter": 3},
        "e_means": {"epochs": 2, "batch_size": 4, "alpha0": 0.2, "drop_last": False},
        "common": {"dtype": "float64"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    runner = Path(__file__).resolve().parents[1] / "scripts/benchmark.py"
    def run(output):
        return subprocess.run([sys.executable, str(runner), "--config", str(config_path),
                               "--output", str(output)], capture_output=True, text=True)
    outputs = [tmp_path / "a", tmp_path / "b"]
    for output in outputs:
        result = run(output)
        assert result.returncode == 0, result.stderr
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["status"] == "complete"
        assert len(manifest["runs"]) == 2
        paired = [json.loads((output / name).read_text()) for name in manifest["runs"]]
        assert paired[0]["data_and_init_sha256"] == paired[1]["data_and_init_sha256"]
        assert paired[0]["source_sha256"] == paired[1]["source_sha256"] == manifest["source_sha256"]
        for record in paired:
            assert record["peak_process_rss_bytes"] > 0
            assert record["fit_wall_seconds"] >= record["training_seconds"] >= 0
            assert record["final_train_mse"] >= 0
    for algorithm in ("lloyd", "e_means"):
        with np.load(outputs[0] / f"tiny_12_{algorithm}.npz") as a, np.load(outputs[1] / f"tiny_12_{algorithm}.npz") as b:
            for name in a.files:
                assert_array_equal(a[name], b[name])
    assert run(outputs[0]).returncode != 0  # immutable run directory
