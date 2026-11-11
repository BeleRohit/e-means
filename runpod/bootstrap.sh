#!/usr/bin/env bash
set -euo pipefail
emeans_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$emeans_root"
python - <<'PY'
import platform, sys, torch
assert platform.system() == 'Linux' and platform.machine() == 'x86_64', 'Use pinned Linux AMD64 image'
assert sys.version_info[:2] == (3,11), 'Pinned image expects Python 3.11'
assert torch.__version__.split('+')[0] == '2.5.1', 'Use the pinned official PyTorch image'
assert torch.version.cuda == '12.4', 'CUDA 12.4 runtime required by pinned image'
PY
python -m pip install --no-deps -r runpod/requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
printf '%s\n' 'Pinned Python overlay installed. NVIDIA correctness is not established until validate_cuda.py succeeds.'
