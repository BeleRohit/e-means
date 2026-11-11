"""Execute the complete suite on actual NVIDIA hardware; emit a strict gate.

CPU-only invocation exits 2 with hardware_unavailable, never a green report.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from e_means.gpu_protocol import (configure_torch, cuda_identity, file_hash,
                                  junit_counts, source_hash, verify_gate, write_json)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new validation directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="started", source_sha256=source_hash(ROOT),
                  started_utc=datetime.now(timezone.utc).isoformat())
    output = args.output / "report.json"
    write_json(output, report)
    try:
        configure_torch()
        report["hardware"] = cuda_identity()
    except RuntimeError as error:
        report.update(status="hardware_unavailable", reason=str(error))
        write_json(output, report)
        print(str(error), file=sys.stderr)
        return 2
    try:
        if not os.environ.get("EMEANS_REFERENCE_DIR"):
            raise ValueError("EMEANS_REFERENCE_DIR required: fetch the pinned authors' oracle before validation")
        junit = args.output / "suite.xml"
        with (args.output / "pytest.log").open("w") as log:
            result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "--junitxml", str(junit.resolve())],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        counts = junit_counts(junit)
        report.update(suite=counts, junit_sha256=file_hash(junit), status="cuda_verified")
        if result.returncode != 0:
            raise RuntimeError("validation suite failed; inspect pytest.log")
        verify_gate(report, ROOT, report["hardware"])
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as error:
        report.update(status="failed", reason=str(error))
        write_json(output, report)
        print(str(error), file=sys.stderr)
        return 1
    write_json(output, report)
    print(f"NVIDIA validation passed: {counts}; gate: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
