"""Phase 2 isolated CPU comparisons with real data, repeats and saved artifacts.

Run: python scripts/research_cpu.py --config configs/phase2_cpu.json --output results/<new-run>
All paths in configs are relative to the repository root. No download occurs.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
import zipfile

import numpy as np

from e_means import EMeans, EMeansConfig, LloydKMeans, LloydConfig, mse
from benchmark import source_digest, write_json
from fetch_cpu_data import array_hash


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("numpy_lloyd", "numpy_emeans", "numpy_slow", "torch_lloyd", "torch_emeans", "torch_slow")


def data(case: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    path = ROOT / case["path"]
    with np.load(path, allow_pickle=False) as file:
        x, test = file["train"], file["test"]
    if array_hash(x, test) != case["array_sha256"]:
        raise ValueError(f"dataset hash mismatch: {path}")
    if x.ndim != 2 or test.ndim != 2 or x.shape[1] != test.shape[1] or not np.isfinite(x).all() or not np.isfinite(test).all():
        raise ValueError("invalid train/test matrices")
    init = x[np.random.default_rng(seed).choice(len(x), case["n_clusters"], replace=False)].copy()
    return x, test, init, array_hash(x, test, init)


def run_worker(config: dict, case: dict, method: str, seed: int, repeat: int, output: Path) -> dict:
    x, test, init, data_digest = data(case, seed)
    is_torch = method.startswith("torch_")
    is_lloyd = method.endswith("_lloyd")
    settings = dict(config["common"], n_clusters=case["n_clusters"], seed=seed)
    settings.update(config["lloyd"] if is_lloyd else config["e_means"])
    if method.endswith("_slow"):
        settings.update(config["slow_ema"], finalize="slow_ema")
    cfg = (LloydConfig if is_lloyd else EMeansConfig)(**settings)
    torch_version = None
    if is_torch:
        import torch
        from e_means.torch_backend import TorchEMeans, TorchLloydKMeans
        torch.set_num_threads(config["threads"])
        torch.set_num_interop_threads(1)
        model = (TorchLloydKMeans if is_lloyd else TorchEMeans)(cfg, device="cpu")
        torch_version = torch.__version__
    else:
        model = (LloydKMeans if is_lloyd else EMeans)(cfg)
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = perf_counter()
    model.fit(x, init_centroids=init, eval_X=test)
    wall = perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1 if sys.platform == "darwin" else 1024
    def array(value):
        return value.detach().cpu().numpy() if is_torch else value
    centers = array(model.cluster_centers_)
    record = {
        "schema_version": 2, "dataset": case["name"], "method": method, "seed": seed, "repeat": repeat,
        "config": asdict(cfg), "device": "cpu", "torch_version": torch_version,
        "data_and_init_sha256": data_digest, "source_sha256": source_digest(),
        "final_train_mse": mse(x, centers), "final_test_mse": mse(test, centers),
        "solver_reported_train_mse": model.mse_,
        "fit_wall_seconds": wall, "training_seconds": model.history_[-1].training_seconds,
        "evaluation_seconds": model.history_[-1].evaluation_seconds,
        "finalization_seconds": getattr(model, "finalization_seconds_", 0),
        "peak_process_rss_bytes": int(peak * unit), "baseline_high_water_rss_bytes": int(baseline * unit),
        "history": [asdict(item) for item in model.history_], "recipe": getattr(model, "recipe_", None),
        "n_revivals": getattr(model, "n_revivals_", 0), "n_iter": model.n_iter_,
    }
    stem = f"{case['name']}_{method}_seed{seed}_repeat{repeat}"
    arrays = dict(centroids=centers, labels=array(model.labels_), init_centroids=init)
    if not is_lloyd:
        arrays.update(last_epoch_labels=array(model.last_epoch_labels_), ema_centroids=array(model.ema_centers_))
        if model.slow_state_ is not None:
            arrays.update(slow_sums=array(model.slow_state_.sums), slow_counts=array(model.slow_state_.counts))
    np.savez_compressed(output / f"{stem}.npz", **arrays)
    record["artifact_sha256"] = hashlib.sha256((output / f"{stem}.npz").read_bytes()).hexdigest()
    write_json(output / f"{stem}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase2_cpu.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument("--case", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    args = parser.parse_args()
    content = args.config.read_bytes()
    config = json.loads(content)
    if args.worker:
        if source_digest() != args.source:
            raise RuntimeError("source changed during experiment")
        run_worker(config, config["datasets"][args.case], args.worker, args.seed, args.repeat, args.output)
        return
    if config["schema_version"] != 2 or not config["seeds"] or config["repeats"] < 1 or config["threads"] < 1:
        raise ValueError("invalid research experiment configuration")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("seeds must be distinct")
    for case in config["datasets"]:
        if not case["name"].replace("_", "").isalnum():
            raise ValueError("dataset name must be alphanumeric or underscore")
        data(case, config["seeds"][0])  # fail before creating run if data is missing/changed
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "config.json"
    snapshot.write_bytes(content)
    # Keep runnable source alongside its digest even without a git checkout.
    with zipfile.ZipFile(args.output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [ROOT / "pyproject.toml", ROOT / "README.md", *sorted((ROOT / "src").rglob("*.py")),
                 *sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]
        for path in paths:
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(config["threads"])
    manifest = dict(schema_version=2, status="running", config=config,
                    config_sha256=hashlib.sha256(content).hexdigest(), source_sha256=source_digest(),
                    source_archive_sha256=hashlib.sha256((args.output / "source.zip").read_bytes()).hexdigest(),
                    started_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    numpy=np.__version__, torch=importlib.metadata.version("torch"),
                    platform=platform.platform(), machine=platform.machine(), logical_cpus=os.cpu_count(),
                    threads=config["threads"], methods=list(METHODS), runs=[],
                    timing="cold fit per fresh worker; includes conversion/evaluation, excludes imports/data loading and serialization",
                    memory="ru_maxrss sampled immediately after fit; whole-process high water including imports and data; macOS bytes/Linux KiB converted to bytes",
                    aggregation="average repeated fits within each seed; then mean and sample std (ddof=1) across distinct seeds")
    write_json(args.output / "manifest.json", manifest)
    try:
        for index, case in enumerate(config["datasets"]):
            for seed in config["seeds"]:
                for repeat in range(config["repeats"]):
                    # Rotate order deterministically to reduce fixed ordering bias.
                    shift = (seed + repeat) % len(METHODS)
                    for method in METHODS[shift:] + METHODS[:shift]:
                        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--config", str(snapshot.resolve()),
                                        "--output", str(args.output.resolve()), "--worker", method, "--case", str(index),
                                        "--seed", str(seed), "--repeat", str(repeat), "--source", manifest["source_sha256"]],
                                       check=True, env=env)
                        filename = f"{case['name']}_{method}_seed{seed}_repeat{repeat}.json"
                        manifest["runs"].append(filename)
                        write_json(args.output / "manifest.json", manifest)
                        print(filename, flush=True)
        if source_digest() != manifest["source_sha256"]:
            raise RuntimeError("source changed during experiment")
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
