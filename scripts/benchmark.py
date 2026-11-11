"""Reproducible Phase 1 experiments; synthetic evidence, not paper reproduction.

Each algorithm runs in a fresh process. Peak RSS is the whole worker process
(interpreter, imports, dataset, validation and model); it is not GPU memory or
an isolated algorithm allocation. Fixed-center evaluation is excluded from
training_seconds, included in fit_wall_seconds. All runs share exact dataset
and initialization per seed. Requires an installed e_means package.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np

from e_means import EMeans, EMeansConfig, LloydConfig, LloydKMeans, mse


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def dataset(spec: dict, seed: int):
    """[UNSPECIFIED] Synthetic equal-weight Gaussian mixture; no preprocessing."""
    rng = np.random.default_rng(seed)
    k, d = spec["n_clusters"], spec["n_features"]
    centers = rng.normal(size=(k, d)) * spec["center_scale"]
    x = centers[rng.integers(k, size=spec["n_train"])] + rng.normal(size=(spec["n_train"], d)) * spec["noise_scale"]
    test = centers[rng.integers(k, size=spec["n_test"])] + rng.normal(size=(spec["n_test"], d)) * spec["noise_scale"]
    init = x[rng.choice(len(x), k, replace=False)].copy()
    digest = hashlib.sha256()
    for array in (x, test, init):
        digest.update(array.astype("<f8").tobytes())
    return x, test, init, digest.hexdigest()


def source_digest() -> str:
    """Hash implementation, runner and reference tests, including uncommitted work."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = [root / "pyproject.toml", *sorted((root / "src").rglob("*.py")),
             *sorted((root / "scripts").glob("*.py")), *sorted((root / "tests").glob("*.py"))]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def worker(config: dict, case: dict, algorithm: str, seed: int, output: Path) -> dict:
    x, test, init, data_hash = dataset(case, seed)
    cls, config_cls = (LloydKMeans, LloydConfig) if algorithm == "lloyd" else (EMeans, EMeansConfig)
    settings = config_cls(n_clusters=case["n_clusters"], seed=seed,
                          **config["common"], **config[algorithm])
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = perf_counter()
    model = cls(settings).fit(x, init_centroids=init, eval_X=test)
    fit_wall = perf_counter() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1 if sys.platform == "darwin" else 1024
    # Closing MSE is recomputed for both, including all training rows.
    result = {
        "algorithm": algorithm, "dataset": case, "seed": seed,
        "config": asdict(settings), "data_and_init_sha256": data_hash,
        "fit_wall_seconds": fit_wall,
        "training_seconds": model.history_[-1].training_seconds,
        "evaluation_seconds": model.history_[-1].evaluation_seconds,
        "finalization_seconds": getattr(model, "finalization_seconds_", 0.0),
        "peak_process_rss_bytes": int(peak_rss * unit),
        "baseline_process_high_water_rss_bytes": int(baseline_rss * unit),
        "final_train_mse": model.mse_,
        "final_test_mse": mse(test, model.cluster_centers_),
        "history": [asdict(row) for row in model.history_],
        "recipe": getattr(model, "recipe_", None),
        "source_sha256": source_digest(),
    }
    prefix = f"{case['name']}_{seed}_{algorithm}"
    arrays = {"centroids": model.cluster_centers_, "labels": model.labels_, "init_centroids": init}
    if algorithm == "e_means":
        arrays.update(last_epoch_labels=model.last_epoch_labels_, ema_centroids=model.ema_centers_)
    np.savez_compressed(output / f"{prefix}.npz", **arrays)
    write_json(output / f"{prefix}.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1_smoke.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["lloyd", "e_means"], help=argparse.SUPPRESS)
    parser.add_argument("--case", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if args.worker:
        worker(config, config["datasets"][args.case], args.worker, args.seed, args.output)
        return
    if config["schema_version"] != 1:
        raise ValueError("unsupported experiment schema")
    # Refuse overwrite: run identity and artifacts remain auditable.
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[variable] = "1"
    manifest = {
        "schema_version": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": source_digest(), "python": sys.version,
        "numpy": np.__version__, "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "thread_settings": {key: value for key, value in env.items() if key.endswith("NUM_THREADS") or key == "VECLIB_MAXIMUM_THREADS"},
        "memory_method": "fresh worker ru_maxrss; whole-process high water; bytes",
        "timing_method": "perf_counter; training excludes fixed-center epoch evaluation; fit wall includes validation and evaluation",
        "status": "running", "runs": [],
    }
    write_json(args.output / "manifest.json", manifest)
    for case_index, case in enumerate(config["datasets"]):
        for seed in config["seeds"]:
            for algorithm in ("lloyd", "e_means"):
                command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                           "--output", str(args.output.resolve()), "--worker", algorithm,
                           "--case", str(case_index), "--seed", str(seed)]
                subprocess.run(command, check=True, env=env)
                filename = f"{case['name']}_{seed}_{algorithm}.json"
                manifest["runs"].append(filename)
                write_json(args.output / "manifest.json", manifest)
                print(filename, flush=True)
    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
