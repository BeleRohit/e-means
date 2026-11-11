"""Verify Phase 2 run coverage and summarize across seeds, not pseudo-replicates.

Repetitions are averaged within each seed, then sample std uses ddof=1 across
seeds. A one-seed summary has null sample std. Curves describe fixed-center
EMA/Lloyd epoch results before finalization; final readouts are separate points.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from benchmark import write_json


METRICS = ("final_train_mse", "final_test_mse", "fit_wall_seconds", "training_seconds",
           "finalization_seconds", "peak_process_rss_bytes", "n_revivals")


def stats(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def summarize(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("only complete runs can be summarized")
    config = manifest["config"]
    if hashlib.sha256((root / "config.json").read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("config snapshot hash mismatch")
    if hashlib.sha256((root / "source.zip").read_bytes()).hexdigest() != manifest["source_archive_sha256"]:
        raise ValueError("source archive hash mismatch")
    expected = {(case["name"], method, seed, repeat) for case in config["datasets"]
                for method in manifest["methods"] for seed in config["seeds"] for repeat in range(config["repeats"])}
    records, actual, hashes, groups = [], set(), {}, defaultdict(list)
    if len(manifest["runs"]) != len(set(manifest["runs"])):
        raise ValueError("duplicate result filenames")
    for name in manifest["runs"]:
        record = json.loads((root / name).read_text())
        key = (record["dataset"], record["method"], record["seed"], record["repeat"])
        if key in actual or key not in expected or record["source_sha256"] != manifest["source_sha256"]:
            raise ValueError("inconsistent result identity/source")
        actual.add(key)
        pair = (record["dataset"], record["seed"])
        digest = hashes.setdefault(pair, record["data_and_init_sha256"])
        if digest != record["data_and_init_sha256"]:
            raise ValueError("paired data/initialization hashes differ")
        if any(not math.isfinite(record[m]) or record[m] < 0 for m in METRICS):
            raise ValueError("invalid measured metric")
        artifact_path = root / name.replace(".json", ".npz")
        if not artifact_path.is_file():
            raise ValueError("missing result arrays")
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != record["artifact_sha256"]:
            raise ValueError("result array hash mismatch")
        groups[key[:2]].append(record)
        records.append(record)
    if actual != expected:
        raise ValueError("incomplete experiment coverage")
    summary, curves = [], []
    for (dataset, method), rows in sorted(groups.items()):
        by_seed = {seed: [row for row in rows if row["seed"] == seed] for seed in config["seeds"]}
        # Repeated runs should change timing/memory, not clustering outcomes.
        for seed, runs in by_seed.items():
            first = None
            for run in runs:
                stem = f"{dataset}_{method}_seed{seed}_repeat{run['repeat']}.npz"
                with np.load(root / stem, allow_pickle=False) as artifacts:
                    current = {name: artifacts[name].copy() for name in artifacts.files}
                if first is None:
                    first = current
                else:
                    for name in first:
                        assert_array_equal(current[name], first[name], err_msg="nondeterministic repeated artifacts")
        entry = {"dataset": dataset, "method": method, "seeds": config["seeds"], "repeats_per_seed": config["repeats"]}
        for metric in METRICS:
            entry[metric] = stats([float(np.mean([row[metric] for row in by_seed[seed]])) for seed in config["seeds"]])
        entry["mean_within_seed_fit_std"] = (float(np.mean([np.std([r["fit_wall_seconds"] for r in runs], ddof=1)
                                                          for runs in by_seed.values()])) if config["repeats"] > 1 else None)
        summary.append(entry)
        budget = config["lloyd"]["max_iter"] if method.endswith("_lloyd") else config["e_means"]["epochs"]
        for epoch in range(budget + 1):
            curve = {"dataset": dataset, "method": method, "epoch": epoch}
            for metric in ("train_mse", "test_mse", "training_seconds"):
                values = []
                for runs in by_seed.values():
                    # Carry a converged Lloyd endpoint forward; no extra work.
                    values.append(float(np.mean([r["history"][min(epoch, len(r["history"])-1)][metric] for r in runs])))
                curve[metric] = stats(values)
            curves.append(curve)
    return dict(schema_version=2, source_sha256=manifest["source_sha256"],
                aggregation="repeat average per seed, mean/sample std ddof=1 across seeds; no confidence interval",
                verified_runs=len(records), summary=summary, convergence=curves)


def plot(summary: dict, root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted({r["dataset"] for r in summary["summary"]})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(12, 4*len(datasets)), squeeze=False)
    colors = {"lloyd": "#2159a6", "emeans": "#c45c12", "slow": "#18805e"}
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(("train_mse", "test_mse")):
            ax = axes[i, j]
            for entry in summary["summary"]:
                if entry["dataset"] != dataset:
                    continue
                method = entry["method"]
                rows = [r for r in summary["convergence"] if r["dataset"] == dataset and r["method"] == method]
                epochs = np.array([r["epoch"] for r in rows])
                means = np.array([r[metric]["mean"] for r in rows])
                std = np.array([r[metric]["std"] or 0 for r in rows])
                color = colors[method.split("_")[1]]
                ax.plot(epochs, means, color=color, ls="--" if method.startswith("torch") else "-", label=method)
                if method.startswith("numpy"):
                    ax.fill_between(epochs, np.maximum(0, means-std), means+std, color=color, alpha=0.12)
                ax.scatter(epochs[-1], entry["final_" + metric]["mean"], color=color,
                           marker="x" if method.startswith("torch") else "o", s=35)
            ax.set(title=f"{dataset}: {metric}", xlabel="Epoch (curves before finalization)", ylabel="MSE per vector")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("CPU experiments: lines = main state; markers = final readout; bands = ±1 sample SD across seeds")
    fig.tight_layout()
    fig.savefig(root / "convergence.png", dpi=160)
    fig.savefig(root / "convergence.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.run)
    write_json(args.run / "summary.json", summary)
    if not args.no_plots:
        plot(summary, args.run)
    print(f"Verified {summary['verified_runs']} runs; wrote summary.json")


if __name__ == "__main__":
    main()
