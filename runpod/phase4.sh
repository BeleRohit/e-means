#!/usr/bin/env bash
# Execute only on an explicitly approved GPU Pod under an external deadline.
set -euo pipefail
emeans_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$emeans_root"
emeans_output="${1:?Usage: bash runpod/phase4.sh /workspace/results/new-run [quality-config] [profile-config]}"
emeans_quality="${2:-configs/phase4_pilot.json}"
emeans_profile="${3:-configs/phase4_pilot.json}"
case "$emeans_output" in /workspace/*) ;; *) printf '%s\n' 'Output must be on /workspace' >&2; exit 2 ;; esac
if [[ -e "$emeans_output" ]]; then printf '%s\n' 'Use a new immutable output path' >&2; exit 2; fi
mkdir -p "$emeans_output"
emeans_output="$(cd "$emeans_output" && pwd)"
export PATH="/opt/conda/bin:$PATH"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$emeans_output/matplotlib-cache" XDG_CACHE_HOME="$emeans_output/cache"
export EMEANS_REFERENCE_DIR="$emeans_output/reference"
emeans_export() {
  emeans_code=$?
  trap - EXIT
  python runpod/export_results.py "$emeans_output" --exit-code "$emeans_code" || true
  exit "$emeans_code"
}
trap emeans_export EXIT
exec > >(tee -a "$emeans_output/run.log") 2>&1
python -m pip freeze > "$emeans_output/pip-freeze.txt"
python -m pip check > "$emeans_output/pip-check.txt"
nvidia-smi -q > "$emeans_output/nvidia-smi.txt"
python scripts/fetch_reference.py "$EMEANS_REFERENCE_DIR" --native-torch
# Phase 3's old certificate cannot authorize the changed source. The complete
# new suite includes an actual CUDA profiler/event probe and must not skip.
python scripts/validate_cuda.py --output "$emeans_output/validation"
python scripts/phase4_cuda.py --mode profile --config "$emeans_profile" --gate "$emeans_output/validation/report.json" --output "$emeans_output/profiles"
python scripts/summarize_phase4.py "$emeans_output/profiles"
python scripts/phase4_cuda.py --mode benchmark --config "$emeans_quality" --gate "$emeans_output/validation/report.json" --output "$emeans_output/quality"
python scripts/summarize_phase4.py "$emeans_output/quality" --plots
python scripts/compare_native_phase4.py --config "$emeans_profile" --reference "$EMEANS_REFERENCE_DIR" --gate "$emeans_output/validation/report.json" --output "$emeans_output/official-native"
