"""GPU experiment protocol; CPU-testable bookkeeping, never simulated results."""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


def write_json(path: Path, value: dict) -> None:
    """Atomic progress records: incomplete experiments must not look complete."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    """Canonical shape/dtype-aware data/initialization identity."""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array, dtype=np.dtype(array.dtype).newbyteorder("<"))
        digest.update(json.dumps([list(array.shape), array.dtype.str]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_paths(root: Path) -> list[Path]:
    paths = [root / "pyproject.toml"]
    for folder, patterns in (("src", ("*.py",)), ("scripts", ("*.py",)), ("tests", ("*.py",)),
                             ("configs", ("*.json",)), ("runpod", ("*",))):
        for pattern in patterns:
            paths.extend(p for p in (root / folder).rglob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    fixtures=root/"tests/fixtures/phase1"
    if fixtures.exists():
        paths.extend(p for p in fixtures.iterdir() if p.is_file() and p.suffix in (".json",".npz"))
    return sorted(set(paths))


def source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in source_paths(root):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def configure_torch() -> None:
    """Explicit experiment policy; never silently relax deterministic errors."""
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cuda_identity() -> dict:
    """Require an actual CUDA build/device and execute a device probe."""
    import torch
    if not torch.cuda.is_available() or torch.version.cuda is None or torch.version.hip is not None:
        raise RuntimeError("NVIDIA CUDA hardware/build unavailable: GPU execution gate is closed")
    device = torch.device("cuda:0")
    if (torch.ones(3, device=device) + 1).sum().item() != 6:
        raise RuntimeError("CUDA execution probe failed")
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return dict(name=props.name, uuid=str(getattr(props, "uuid", "")), total_memory=props.total_memory,
                capability=list(torch.cuda.get_device_capability(device)), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                nvidia_smi=result.stdout.strip(), cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                python=sys.version, platform=platform.platform(), numpy=np.__version__,
                deterministic=torch.are_deterministic_algorithms_enabled(), tf32=False)


def junit_counts(path: Path) -> dict:
    """No skipped/failed cases can establish the full NVIDIA validation gate."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    counts = {"tests": len(cases), "skipped": 0, "failures": 0, "errors": 0, "cuda_cases": 0}
    for case in cases:
        for key, tag in (("skipped", "skipped"), ("failures", "failure"), ("errors", "error")):
            counts[key] += int(case.find(tag) is not None)
        if "test_cuda_validation" in case.get("classname", "") and "cuda" in case.get("name", ""):
            counts["cuda_cases"] += 1
    return counts


def verify_gate(report: dict, root: Path, hardware: dict) -> None:
    if report.get("status") != "cuda_verified" or report.get("source_sha256") != source_hash(root):
        raise ValueError("successful, source-matched NVIDIA validation report required")
    if report.get("hardware") != hardware:
        raise ValueError("GPU validation and benchmark hardware/environment differ")
    counts = report.get("suite", {})
    if counts.get("tests", 0) < 100 or counts.get("cuda_cases", 0) < 20 or any(counts.get(k, -1) != 0 for k in ("failures", "errors", "skipped")):
        raise ValueError("complete passing suite with executed CUDA cases required")


def time_to_target(points: list[dict], target: float, metric: str = "test_mse") -> dict:
    """First observed completed evaluation; no interpolation/extrapolation.

    A miss is right-censored at the final observation, never reported as zero.
    Wall times include metric overhead and initial transfer; main-state points
    and a final readout are identified separately by the producer.
    """
    if not math.isfinite(target) or target < 0 or not points:
        raise ValueError("finite nonnegative target and observations required")
    previous = -1.0
    for point in points:
        seconds = point["wall_seconds"]
        if not math.isfinite(seconds) or seconds < previous or not math.isfinite(point[metric]) or point[metric] < 0:
            raise ValueError("observations must be finite and time ordered")
        previous = seconds
    for point in points:
        if point[metric] <= target:
            return dict(reached=True, seconds=point["wall_seconds"], epoch=point["epoch"], stage=point["stage"], target=target)
    return dict(reached=False, seconds=None, censored_at_seconds=previous, target=target)


def seed_summary(records: list[dict], metric: str) -> dict:
    """Average repeats within seed, then mean/sample SD across seeds."""
    seeds = sorted({r["seed"] for r in records})
    means = [float(np.mean([r[metric] for r in records if r["seed"] == seed])) for seed in seeds]
    return dict(mean=float(np.mean(means)), std=float(np.std(means, ddof=1)) if len(means) > 1 else None,
                independent_seeds=len(seeds))
