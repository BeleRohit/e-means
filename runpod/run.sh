#!/usr/bin/env bash
# Run only after explicit GPU provisioning. Never start benchmarks on failure.
set -euo pipefail
emeans_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$emeans_root"
emeans_output="${1:?Usage: bash runpod/run.sh /workspace/results/new-run [configs/gpu_smoke.json]}"
emeans_config="${2:-configs/gpu_smoke.json}"
case "$emeans_output" in
  /workspace/*) ;;
  *) printf '%s\n' 'Output must be on the mounted /workspace persistent volume' >&2; exit 2 ;;
esac
if [[ -e "$emeans_output" ]]; then
  printf '%s\n' 'Use a new run directory; existing results are immutable' >&2
  exit 2
fi
mkdir -p "$emeans_output"
emeans_output="$(cd "$emeans_output" && pwd)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR="$emeans_output/matplotlib-cache"
export XDG_CACHE_HOME="$emeans_output/cache"
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
python scripts/fetch_reference.py "$EMEANS_REFERENCE_DIR"
# Complete suite must pass on NVIDIA before either data-dependent calibration
# or benchmark warmups. Every skipped case causes the gate to remain closed.
python scripts/validate_cuda.py --output "$emeans_output/validation"
if [[ ! -f data/glove100_subset.npz ]]; then
  python scripts/fetch_gpu_embeddings.py
fi
python scripts/benchmark_cuda.py --config "$emeans_config" --gate "$emeans_output/validation/report.json" --output "$emeans_output/benchmark"
python scripts/summarize_cuda.py "$emeans_output/benchmark" --plots
