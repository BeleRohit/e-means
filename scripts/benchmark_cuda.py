"""Strictly gated, synchronized CUDA baseline measurements, without optimization.

--prepare-only validates/hashes inputs locally and emits zero GPU measurements.
Actual runs require validate_cuda.py's successful same-source/device full-suite
report. No CPU timing can be written as a GPU measurement by this entry point.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import zipfile

import numpy as np

from e_means.gpu_protocol import (configure_torch, cuda_identity, file_hash, source_hash, source_paths,
                                  time_to_target, verify_gate, write_json)
from e_means.gpu_workloads import METHODS, load_workload, settings, validate_config

ROOT = Path(__file__).resolve().parents[1]


def checked_gate(path: Path, hardware: dict) -> dict:
    report = json.loads(path.read_text())
    verify_gate(report, ROOT, hardware)
    if file_hash(path.parent / "suite.xml") != report["junit_sha256"]:
        raise ValueError("validation XML was modified or is missing")
    return report


def measure(config, case, method, seed, dtype, x, test, init):
    """Actual-device timing, including final metric evaluation, excluding H2D.

    CUDA-event interval includes host submission gaps/synchronizations; it is
    not pure kernel time. Whole fit wall time is the primary latency metric.
    GPU peaks include resident input tensors; baseline/increment are separate.
    """
    import torch
    from e_means.torch_backend import TorchEMeans, TorchLloydKMeans
    cls = TorchLloydKMeans if method == "lloyd" else TorchEMeans
    cfg = settings(config, case, seed, dtype, method)
    class Traced(cls):
        def _record(self, *args, **kwargs):
            record = super()._record(*args, **kwargs)
            self.points.append(dict(epoch=record.epoch, stage="main_state", train_mse=record.train_mse,
                                    test_mse=record.test_mse, wall_seconds=perf_counter()-self.origin))
            return record
    model = Traced(cfg, device="cuda:0")
    model.points = []
    torch.cuda.synchronize()
    gc.collect()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    model.origin = perf_counter()
    start_event.record()
    model.fit(x, init_centroids=init, eval_X=test)
    final_test = model._assignment(test, model.cluster_centers_)[1].double().mean().item()
    model.points.append(dict(epoch=model.n_iter_, stage="final_readout", train_mse=model.mse_,
                             test_mse=final_test, wall_seconds=perf_counter()-model.origin))
    end_event.record()
    torch.cuda.synchronize()
    elapsed = perf_counter()-model.origin
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    n_seen = (len(x)*model.n_iter_ if method == "lloyd" else
              (len(x) if not cfg.drop_last else len(x)//model.recipe_["batch_size"]*model.recipe_["batch_size"])*cfg.epochs)
    record = dict(method=method, seed=seed, dtype=dtype, config=asdict(cfg),
                  resident_fit_wall_seconds=elapsed, cuda_event_elapsed_seconds=start_event.elapsed_time(end_event)/1000,
                  baseline_allocated_bytes=baseline_allocated, baseline_reserved_bytes=baseline_reserved,
                  peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                  incremental_peak_allocated_bytes=max(0,allocated-baseline_allocated),
                  train_mse=model.mse_, test_mse=final_test, convergence=model.points,
                  training_loop_seconds=model.history_[-1].training_seconds,
                  evaluation_seconds=model.history_[-1].evaluation_seconds,
                  finalization_seconds=getattr(model,"finalization_seconds_",0),
                  training_vectors_seen=n_seen, training_vectors_per_resident_fit_second=n_seen/elapsed,
                  training_vectors_per_loop_second=n_seen/model.history_[-1].training_seconds,
                  history=[asdict(r) for r in model.history_], recipe=getattr(model,"recipe_",None))
    arrays = dict(centroids=model.cluster_centers_.cpu().numpy(), labels=model.labels_.cpu().numpy())
    del model
    return record, arrays


def cohort(config: dict, index: int, seed: int, dtype: str, output: Path, gate_path: Path) -> list[str]:
    import torch
    from e_means.torch_backend import TorchEMeans, TorchLloydKMeans
    configure_torch()
    hardware = cuda_identity()
    gate = checked_gate(gate_path, hardware)
    case = config["workloads"][index]
    x_cpu, test_cpu, init_cpu, digest = load_workload(case, seed, dtype, ROOT)
    torch.cuda.synchronize()
    start = perf_counter()
    x, test, init = (torch.as_tensor(a, device="cuda:0") for a in (x_cpu,test_cpu,init_cpu))
    torch.cuda.synchronize()
    transfer_seconds = perf_counter()-start
    # Target rule is predeclared. One independent Lloyd calibration, saved
    # before any timed contender, determines a common per-seed/dtype target.
    reference = TorchLloydKMeans(settings(config,case,seed,dtype,"lloyd"),device="cuda:0").fit(x,init_centroids=init,eval_X=test)
    train_target = reference.mse_*(1+config["target_relative_slack"])
    test_target = reference._assignment(test,reference.cluster_centers_)[1].double().mean().item()*(1+config["target_relative_slack"])
    prefix = f"{case['name']}_seed{seed}_{dtype}"
    target_arrays = output / f"{prefix}_calibration.npz"
    np.savez_compressed(target_arrays, centroids=reference.cluster_centers_.cpu().numpy(), init_centroids=init_cpu)
    write_json(output/f"{prefix}_calibration.json", dict(status="gpu_measured",train_target=train_target,test_target=test_target,
               target_rule="terminal Lloyd calibration MSE * (1 + predeclared relative slack)",
               target_relative_slack=config["target_relative_slack"],data_and_init_sha256=digest,
               config=asdict(reference.config),artifact_sha256=file_hash(target_arrays),source_sha256=gate["source_sha256"]))
    del reference
    gc.collect()
    # Every method gets identical-data warmups, then fresh models per repeat.
    # Context/kernel warmup is excluded; allocator cache is not emptied between
    # contenders, so baseline reserved memory is explicitly recorded.
    for method in METHODS:
        for _ in range(config["warmups"]):
            cls = TorchLloydKMeans if method == "lloyd" else TorchEMeans
            warm = cls(settings(config,case,seed,dtype,method),device="cuda:0").fit(x,init_centroids=init,eval_X=test)
            del warm
    torch.cuda.synchronize()
    names = []
    for repeat in range(config["repeats"]):
        shift = (seed+repeat) % len(METHODS)
        for method in METHODS[shift:]+METHODS[:shift]:
            if source_hash(ROOT) != gate["source_sha256"]:
                raise RuntimeError("source/config changed during benchmark")
            record, arrays = measure(config,case,method,seed,dtype,x,test,init)
            record.update(status="gpu_measured",workload=case["name"],repeat=repeat,
                          source_sha256=gate["source_sha256"],data_and_init_sha256=digest,
                          hardware=hardware,initial_host_to_device_seconds=transfer_seconds,
                          train_time_to_target=time_to_target(record["convergence"],train_target,"train_mse"),
                          test_time_to_target=time_to_target(record["convergence"],test_target,"test_mse"))
            # Main EMA monitoring and the selected finalizer's returned result
            # are different estimators. Preserve both instead of presenting an
            # early main-state hit as an early slow/closing readout.
            final_points=[p for p in record["convergence"] if p["stage"]=="final_readout"]
            record.update(target_observation_scope="monitored main state or final readout; stage explicitly tagged",
                          returned_train_time_to_target=time_to_target(final_points,train_target,"train_mse"),
                          returned_test_time_to_target=time_to_target(final_points,test_target,"test_mse"))
            name = f"{prefix}_{method}_repeat{repeat}"
            array_path = output/f"{name}.npz"
            np.savez_compressed(array_path, **arrays)
            record["artifact_sha256"] = file_hash(array_path)
            write_json(output/f"{name}.json",record)
            names.append(f"{name}.json")
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True,help="new run directory")
    parser.add_argument("--gate",type=Path)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--worker",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--seed",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--dtype",help=argparse.SUPPRESS)
    args = parser.parse_args()
    content = args.config.read_bytes()
    config = json.loads(content)
    validate_config(config)
    if args.worker is not None:
        names = cohort(config,args.worker,args.seed,args.dtype,args.output,args.gate)
        write_json(args.output/f"cohort_{args.worker}_{args.seed}_{args.dtype}.json",dict(runs=names))
        return 0
    hardware, gate = None, None
    if not args.prepare_only:
        if args.gate is None:
            raise ValueError("--gate required; run the complete NVIDIA validation suite first")
        configure_torch()
        hardware = cuda_identity()
        gate = checked_gate(args.gate,hardware)
    plan = []
    for case in config["workloads"]:
        for seed in config["seeds"]:
            for dtype in config["dtypes"]:
                x,test,init,digest = load_workload(case,seed,dtype,ROOT)
                plan.append(dict(workload=case["name"],seed=seed,dtype=dtype,train_shape=list(x.shape),
                                 test_shape=list(test.shape),data_and_init_sha256=digest,
                                 resident_input_bytes=x.nbytes+test.nbytes+init.nbytes))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"config.json").write_bytes(content)
    manifest = dict(schema_version=3,status="prepared" if args.prepare_only else "running",
                    gpu_verified=not args.prepare_only,config=config,config_sha256=file_hash(args.output/"config.json"),
                    source_sha256=source_hash(ROOT),hardware=hardware,workloads=plan,runs=[],
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    timing="synchronized resident fit + final test evaluation; H2D separate; CUDA events include submission gaps",
                    targets="per-seed/dtype separate Lloyd calibration; first observed metric hit, misses censored; no interpolation",
                    memory="CUDA allocator peaks include resident inputs and epoch evaluation; baseline/reserved/increment separate; not nvidia-smi total")
    if gate is not None:
        write_json(args.output/"validation_report.json",gate)
        (args.output/"validation_suite.xml").write_bytes((args.gate.parent/"suite.xml").read_bytes())
    with zipfile.ZipFile(args.output/"source.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths(ROOT):
            archive.write(path,path.relative_to(ROOT))
        for path in ROOT.glob("*.md"):
            archive.write(path,path.relative_to(ROOT))
        for path in (ROOT/"docs").rglob("*"):
            if path.is_file():archive.write(path,path.relative_to(ROOT))
    manifest["source_archive_sha256"] = file_hash(args.output/"source.zip")
    write_json(args.output/"manifest.json",manifest)
    if args.prepare_only:
        print(f"Prepared {len(plan)} workload/seed/dtype combinations; zero GPU measurements")
        return 0
    try:
        for i,case in enumerate(config["workloads"]):
            for seed in config["seeds"]:
                for dtype in config["dtypes"]:
                    subprocess.run([sys.executable,str(Path(__file__).resolve()),"--config",str((args.output/"config.json").resolve()),
                                    "--output",str(args.output.resolve()),"--gate",str(args.gate.resolve()),"--worker",str(i),
                                    "--seed",str(seed),"--dtype",dtype],check=True,cwd=ROOT)
                    cohort_record=json.loads((args.output/f"cohort_{i}_{seed}_{dtype}.json").read_text())
                    manifest["runs"].extend(cohort_record["runs"])
                    write_json(args.output/"manifest.json",manifest)
        if source_hash(ROOT) != manifest["source_sha256"]:
            raise RuntimeError("source changed during benchmark")
        manifest.update(status="complete",completed_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        manifest.update(status="failed",error=str(error))
        raise
    finally:
        write_json(args.output/"manifest.json",manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
