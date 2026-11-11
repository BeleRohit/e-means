"""Offline integration checks for real-data runner and seed aggregation."""

import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from numpy.testing import assert_allclose
import pytest


@pytest.fixture
def research(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return importlib.import_module("research_cpu"), importlib.import_module("summarize_cpu"), importlib.import_module("fetch_cpu_data")


def test_aggregation_and_hash_validation(research, tmp_path):
    runner, summary, fetcher = research
    assert summary.stats([1, 3]) == {"mean": 2., "std": np.sqrt(2)}
    assert summary.stats([2]) == {"mean": 2., "std": None}
    x = np.arange(12, dtype=float).reshape(6, 2)
    path = tmp_path / "data.npz"
    np.savez(path, train=x, test=x[:2])
    case = dict(path=str(path), array_sha256=fetcher.array_hash(x, x[:2]), n_clusters=2)
    a, b, init, digest = runner.data(case, 4)
    assert init.shape == (2, 2)
    assert digest == fetcher.array_hash(a, b, init)
    with pytest.raises(ValueError):
        runner.data(dict(case, array_sha256="wrong"), 4)
    with pytest.raises(ValueError):
        fetcher.prepare("optdigits", b"changed archive", tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "running"}))
    with pytest.raises(ValueError):
        summary.summarize(tmp_path)


def test_complete_cpu_runner_summary_and_coverage(research, tmp_path):
    pytest.importorskip("torch")
    runner, summary, fetcher = research
    rng = np.random.default_rng(10)
    x, test = rng.normal(size=(21, 3)), rng.normal(size=(9, 3))
    path = tmp_path / "data.npz"
    np.savez(path, train=x, test=test)
    config = dict(schema_version=2, seeds=[1, 2], repeats=1, threads=1,
                  datasets=[dict(name="tiny", path=str(path), array_sha256=fetcher.array_hash(x, test), n_clusters=2)],
                  common=dict(dtype="float64"), lloyd=dict(max_iter=3),
                  e_means=dict(epochs=2, batch_size=4, alpha0=0.2, drop_last=False), slow_ema=dict(slow_ema_factor=0.25))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    output = tmp_path / "run"
    result = subprocess.run([sys.executable, runner.__file__, "--config", str(cfg), "--output", str(output)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = summary.summarize(output)
    assert report["verified_runs"] == 12
    assert len(report["summary"]) == 6
    assert all(row["final_test_mse"]["std"] is not None for row in report["summary"])
    for family in ("lloyd", "emeans", "slow"):
        pair = [r for r in report["summary"] if r["method"].endswith("_"+family)]
        assert_allclose(pair[0]["final_test_mse"]["mean"], pair[1]["final_test_mse"]["mean"], atol=1e-12)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["runs"].pop()
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coverage"):
        summary.summarize(output)
