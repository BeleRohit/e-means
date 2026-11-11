"""Verify complete GPU artifacts and aggregate repeats within seeds.

Never summarize prepared/failed jobs as measurements. Missed time-to-target
observations remain censored; reached-only timings are explicitly conditional.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from e_means.gpu_protocol import file_hash, seed_summary, write_json
from e_means.gpu_workloads import METHODS


METRICS=("train_mse","test_mse","resident_fit_wall_seconds","cuda_event_elapsed_seconds",
         "peak_allocated_bytes","peak_reserved_bytes","incremental_peak_allocated_bytes",
         "training_vectors_per_resident_fit_second","training_vectors_per_loop_second")


def summarize(root: Path) -> dict:
    manifest=json.loads((root/"manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest["gpu_verified"]:
        raise ValueError("complete GPU-verified measurements required; preparation is not a result")
    config=manifest["config"]
    if file_hash(root/"config.json") != manifest["config_sha256"] or file_hash(root/"source.zip") != manifest["source_archive_sha256"]:
        raise ValueError("configuration/source snapshot integrity failure")
    gate=json.loads((root/"validation_report.json").read_text())
    if gate["status"] != "cuda_verified" or gate["source_sha256"] != manifest["source_sha256"] or gate["hardware"] != manifest["hardware"]:
        raise ValueError("validation report does not match measurements")
    if file_hash(root/"validation_suite.xml") != gate["junit_sha256"]:
        raise ValueError("validation XML integrity failure")
    expected={(c["name"],s,d,m,r) for c in config["workloads"] for s in config["seeds"]
              for d in config["dtypes"] for m in METHODS for r in range(config["repeats"])}
    actual=set();groups=defaultdict(list);digests={}
    for name in manifest["runs"]:
        row=json.loads((root/name).read_text())
        key=(row["workload"],row["seed"],row["dtype"],row["method"],row["repeat"])
        if key in actual or key not in expected:
            raise ValueError("duplicate/unknown measurement identity")
        actual.add(key)
        if row["status"] != "gpu_measured" or row["source_sha256"] != manifest["source_sha256"] or row["hardware"] != manifest["hardware"]:
            raise ValueError("mixed hardware/source or unverified measurement")
        if file_hash(root/name.replace(".json",".npz")) != row["artifact_sha256"]:
            raise ValueError("result arrays changed")
        pair=key[:3]
        if digests.setdefault(pair,row["data_and_init_sha256"]) != row["data_and_init_sha256"]:
            raise ValueError("paired inputs differ")
        if any(not np.isfinite(row[m]) or row[m] < 0 for m in METRICS):
            raise ValueError("nonfinite or negative measurement")
        calibration=json.loads((root/f"{key[0]}_seed{key[1]}_{key[2]}_calibration.json").read_text())
        cal_path=root/f"{key[0]}_seed{key[1]}_{key[2]}_calibration.npz"
        if file_hash(cal_path) != calibration["artifact_sha256"] or calibration["data_and_init_sha256"] != row["data_and_init_sha256"]:
            raise ValueError("calibration integrity failure")
        for metric in ("train","test"):
            if row[f"{metric}_time_to_target"]["target"] != calibration[f"{metric}_target"]:
                raise ValueError("contenders used different targets")
        groups[(row["workload"],row["dtype"],row["method"])].append(row)
    if actual != expected or len(manifest["runs"]) != len(expected):
        raise ValueError("incomplete GPU measurement coverage")
    output=[]
    for (workload,dtype,method),rows in sorted(groups.items()):
        entry=dict(workload=workload,dtype=dtype,method=method)
        for metric in METRICS:
            entry[metric]=seed_summary(rows,metric)
        for metric in ("train","test"):
            for prefix in ("", "returned_"):
                field=f"{prefix}{metric}_time_to_target"
                hits=[r for r in rows if r[field]["reached"]]
                # Do not replace censored observations with zero/last epoch and
                # do not hide failed seeds behind a conditional latency mean.
                entry[field]=dict(reached_runs=len(hits),total_runs=len(rows),
                    reached_seeds=len({r["seed"] for r in hits}),
                    hit_stages={stage:sum(r[field].get("stage")==stage for r in hits) for stage in ("main_state","final_readout")},
                    conditional_mean_seconds=float(np.mean([r[field]["seconds"] for r in hits])) if hits else None,
                    interpretation="conditional mean among reached repetitions only; not an unconditional speedup estimate")
        output.append(entry)
    return dict(status="gpu_measured",verified_runs=len(actual),summary=output,
                aggregation="repeat means within each seed, then mean/sample SD across seeds",
                target_observations="censored misses retained; threshold hit occurs only at completed measured evaluations")


def plots(root: Path, result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest=json.loads((root/"manifest.json").read_text())
    for case in manifest["config"]["workloads"]:
        for dtype in manifest["config"]["dtypes"]:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
            for method in METHODS:
                # Thin lines preserve all seed/repeat curves, avoiding an
                # invented interpolated mean along different wall timelines.
                rows=[json.loads((root/n).read_text()) for n in manifest["runs"]]
                rows=[r for r in rows if r["workload"]==case["name"] and r["dtype"]==dtype and r["method"]==method]
                color={"lloyd":"#2159a6","emeans":"#c45c12","slow_ema":"#18805e"}[method]
                for i,row in enumerate(rows):
                    main=[p for p in row["convergence"] if p["stage"]=="main_state"]
                    for ax,xkey in zip(axes,("epoch","wall_seconds")):
                        ax.plot([p[xkey] for p in main],[p["test_mse"] for p in main],color=color,alpha=.35,label=method if i==0 else None)
                        final=row["convergence"][-1]
                        ax.scatter(final[xkey],final["test_mse"],color=color,s=12)
            axes[0].set_xlabel("Epoch")
            axes[1].set_xlabel("Resident fit wall seconds (includes metric overhead)")
            for ax in axes:
                ax.set_ylabel("Test MSE per vector");ax.grid(alpha=.2);ax.legend()
            fig.suptitle(f"{case['name']} / {dtype}: individual seed/repeat curves; dots = final readouts")
            fig.tight_layout();fig.savefig(root/f"{case['name']}_{dtype}_convergence.png",dpi=160);plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--plots",action="store_true")
    args=parser.parse_args()
    result=summarize(args.run)
    write_json(args.run/"summary.json",result)
    if args.plots:
        plots(args.run,result)
    print(f"Verified {result['verified_runs']} actual GPU measurements")


if __name__ == "__main__":
    main()
