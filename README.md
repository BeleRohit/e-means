# e-means research implementation

Independent NumPy and PyTorch implementation of [e-means: outperforming k-means (in one-tenth the epochs)](https://arxiv.org/abs/2611.09501v1), Bessa et al. (2026). The paper is the specification; the [authors' implementation](https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means) resolves bookkeeping ambiguities and supplies an optional test oracle. No authors' source is vendored or used by the library. Paper and reference provenance is recorded in [docs/sources.json](docs/sources.json).

**Current milestone:** validated NumPy/PyTorch CPU algorithms plus CUDA validation/benchmark preparation. Only CPU execution has been verified locally. CUDA tests, pinned workloads, event/synchronized timing, allocator measurements and Runpod scripts are ready for the NVIDIA execution gate. No GPU measurement, custom kernel or distributed clustering result is claimed. This is a tested research implementation, not a paper-scale performance reproduction.

Local verification: **146 tests passed, 32 CUDA cases skipped** on a CPU-only macOS environment. The real-data CPU study completed 120 fits; paired FP64 NumPy/PyTorch centers and labels matched exactly in those tested runs. These finite CPU observations do not establish CUDA parity or the paper's large-scale performance claims. Generated datasets/results and local research notes are excluded from this Git repository; small numeric regression fixtures are retained under `tests/fixtures/phase1/`.

## Install and test

Python 3.10+; NumPy is the only runtime dependency.

```bash
git clone https://github.com/BeleRohit/e-means.git
cd e-means
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

Optional authors' comparisons (network access is needed only for installation and fetching):

```bash
.venv/bin/python -m pip install -e '.[reference]'
.venv/bin/python scripts/fetch_reference.py /tmp/emeans-reference-2b648f
EMEANS_REFERENCE_DIR=/tmp/emeans-reference-2b648f .venv/bin/python -m pytest -q
```

Reference files are pinned and SHA256 checked. These tests use the authors' caller-driven API with matched initialization, batch order, and settings. They do not assert equality between different RNG streams or paper/refitted scaling laws. Without the reference directory, seven optional cases skip explicitly. Oracle tests execute in clean subprocesses to avoid loading Faiss and PyTorch's bundled OpenMP runtimes together on macOS.

## Use

```python
import numpy as np
from e_means import EMeans, EMeansConfig, LloydKMeans, LloydConfig

x = np.random.default_rng(7).normal(size=(1000, 32))
init = x[:8].copy()
lloyd = LloydKMeans(LloydConfig(n_clusters=8, max_iter=25))
ema = EMeans(EMeansConfig(n_clusters=8, epochs=3, seed=7))
lloyd.fit(x, init_centroids=init)
ema.fit(x, init_centroids=init)
print(lloyd.mse_, ema.mse_)
labels = ema.predict(x)
```

`batch_size=None` and `alpha0=None` resolve the **published Table 2** laws. Set either explicitly to override it. The laws were fitted to large, high-dimensional embedding data; tiny synthetic examples are numerical checks, not evidence of optimal automatic tuning.

`fit(X, init_centroids=..., eval_X=...)` returns the estimator. Fits are deterministic for fixed data, seed and NumPy execution environment; fitting again starts a fresh run. Inputs are not mutated. Arrays must be finite, real, nonempty matrices, and `1 <= k <= n`. Unrepresentable distance arithmetic raises `FloatingPointError`.

- `cluster_centers_`, `labels_`: final centers and fresh nearest-center labels for **all** input rows.
- `mse_`: mean squared Euclidean norm per vector (Eq. 1); `inertia_`: summed squared distances.
- `history_`: epoch zero plus fixed-center train/test MSE after each epoch, cumulative training/evaluation seconds, update/split counts, and separately named moving-pass MSE for e-means. The last history row precedes finalization; `mse_` is the final result.
- `recipe_`: resolved e-means settings; `ema_centers_` and `ema_state_`: pre-finalization positions and EMA sums/counts.
- `last_epoch_labels_`, `last_epoch_sums_`, `last_epoch_counts_`: assignments/statistics produced while centers moved during the final training pass.

The default `drop_last=True` follows §3.1's `M=T floor(n/B)`; shuffled remainder rows are omitted from each training epoch. Their last-pass labels are `-1`. With `drop_last=False`, all rows train, and `M=T ceil(n/B)` follows the modern reference. Last-epoch means use visited rows only. Both final evaluation and prediction always include all rows.

The implementation deliberately uses published paper values where the modern reference differs: initial mass `1e-10`, split perturbation `2**-7` (BF16 epsilon), and the rounded Table 2 scaling coefficients. The reference uses initial mass `2**-16`, split perturbation `1e-4`, refitted coefficients, and different batch/low-dimensional defaults. Controlled comparisons align these settings and batch order. The cosine schedule follows Eq. 5 literally with zero-based `t/M`, so its last executed step stays above the theoretical floor.

`dtype="float64"` prioritizes correctness. `float32` is supported and tested, with its usual accumulation limits. Direct squared differences avoid the cancellation in `||x||² + ||c||² - 2x·c`. Assignment tiles bound distance scratch to `O(assignment_chunk_size * centroid_chunk_size)`; the implementation still performs `O(nkd)` work. It is a baseline for future backends, not a high-throughput kernel.

## Experiments

```bash
.venv/bin/python scripts/benchmark.py --config configs/phase1_smoke.json --output results/local/my_run
```

Each algorithm/seed/dataset runs in a fresh process with one-thread environment settings. JSON records include configuration, seed, dataset/initialization hashes, source digest, Python/NumPy/platform metadata, convergence histories, final train/test MSE, timings and RSS. NPZ artifacts preserve final centroids, labels, initialization and e-means moving labels. Existing output directories are rejected. The JSON manifest stays `running` if execution fails and becomes `complete` only after all workers succeed.

`training_seconds` excludes fixed-center epoch evaluation; `fit_wall_seconds` includes validation, initialization, evaluation and final assignments. RSS is the whole worker process high-water mark, including interpreter, data and imports. Do not interpret it as isolated optimizer allocations. `resource` memory measurement supports macOS/Linux; the NumPy library itself is portable.

## Layout and next phases

`src/e_means/` separates configurations, exact assignment, schedules, EMA state, estimators and GPU experiment bookkeeping. `tests/` contains independent numerical oracles, pinned-reference checks, CPU parity tests and guarded CUDA cases. `scripts/` provides data preparation, experiment, validation and summary entry points. `configs/` retains reproducible experiment settings; `runpod/` contains the pinned environment and execution scripts. Experiment outputs are generated locally under ignored `results/` paths.

Next: execute the guarded CUDA correctness matrix on actual NVIDIA hardware, inspect the GPU smoke results, then run the larger predeclared workloads. Custom kernels, distributed clustering and performance optimization remain deferred.

## PyTorch backend and slow EMA

```bash
.venv/bin/python -m pip install -e '.[test,torch,experiments]'
```

```python
import torch
from e_means import TorchEMeans, TorchLloydKMeans, EMeansConfig, LloydConfig

x = torch.randn(1000, 32, dtype=torch.float64)
lloyd = TorchLloydKMeans(LloydConfig(8), device="cpu").fit(x)
model = TorchEMeans(EMeansConfig(8, finalize="slow_ema", slow_ema_factor=0.25)).fit(x)
print(model.mse_, model.cluster_centers_.device)
# On a CUDA-equipped host, explicitly select device="cuda" or pass a CUDA tensor.
```

Constructors share NumPy configurations and methods, adding only optional `device`. With no device, tensors retain their device and non-tensors use CPU. Config `dtype` determines arithmetic precision. Fitted arrays and predictions are tensors; loss and history scalars remain Python values. Inputs are detached, never mutated, and no autograd graph is retained. Both assignment dimensions are tiled; scratch stays O(row_chunk * centroid_chunk). CUDA operations include explicit metric/timing synchronizations; compatibility is not a measured speed claim.

The Phase 1 NumPy RNG streams control initial row selection, batch order, and donor sampling for both backends. Tensor updates, distance calculations, reductions, and splits run in PyTorch. Revival transfers k masses to the host to preserve controlled random choices. This is intentional for parity; a future device-native RNG would need separate reproducibility semantics. CUDA `index_add_` reductions may have device-specific roundoff or nondeterminism; CPU equality tests do not establish CUDA determinism.

Select `finalize="slow_ema"` on either backend. It maintains independent count/sum EMAs throughout training, consuming the **main** model's assignments and donor/target choices. It never affects assignment. The constant step is `slow_ema_factor * alpha_min_factor * alpha0`; omitted factor means 1, an explicit research default. Alternatively set `slow_half_life=h` in **logical batch updates**, giving `alpha_slow = 1-exp(-log(2)/h)`. These options are mutually exclusive. Invalid speeds are rejected rather than clipped. Each shadow donor splits its own center and mass. `slow_state_` and `slow_centers_` expose the shadow state, or are None for other finalizers.

The paper specifies the constant count/sum recurrence, but not every half-life/split bookkeeping detail. The pinned authors' package has no slow-EMA finalizer. Independent scalar tests and comparisons against authors' EMA/split primitives validate the explicit interpretation above; end-to-end equality to an unavailable authors' slow-EMA routine is not claimed. Last-epoch means minimize SSE for the recorded moving assignments, which does not guarantee lower freshly reassigned MSE than the final main EMA. Multiplicative revival cannot separate an all-zero donor.

## Real-data CPU experiments

```bash
.venv/bin/python scripts/fetch_cpu_data.py --output data
.venv/bin/python scripts/research_cpu.py --config configs/phase2_cpu.json --output results/local/phase2_run
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/summarize_cpu.py results/local/phase2_run
```

The two public UCI digits datasets use original train/test splits. Archive and processed-array hashes are pinned; class labels are discarded. Range scaling uses fixed published ranges, with no test-set fitted preprocessing. Downloading is explicit; tests and workers remain offline. [scripts/fetch_cpu_data.py](scripts/fetch_cpu_data.py) records the source URLs, DOI, archive hashes and scaling rules, and generates local provenance JSON alongside the prepared data.

The configuration runs NumPy/PyTorch × Lloyd/main e-means/slow e-means with five independent seeds and two fresh-process repeated fits per seed. Each pair shares exact data/initialization. Records preserve all settings, seed/repeat IDs, epoch histories, losses, timing, process RSS and checksummed NPZ arrays. A config snapshot and checksummed source archive permit reproduction without relying on a git checkout. Mid-run source changes cause failure. Historical Phase 1 artifacts are retained as regression oracles.

`summary.json` averages repetitions **within each seed**, then reports mean and sample standard deviation (`ddof=1`) across distinct seeds. It verifies complete method/seed/repeat coverage, shared data hashes, artifact integrity and deterministic repeated outcomes. It also reports within-seed timing variation. Final train/test MSE is recomputed in FP64 against returned centers for a common evaluator. Curves show the main state before finalization; final-readout points are separate. PNG/SVG curves include ±1 sample-SD bands, which are not confidence intervals.

Fit timings include conversion, initialization and epoch evaluation but exclude process startup/imports, dataset loading and output serialization. Loop and finalization times are separate. RSS is the whole process high-water mark through fit, including imports/data; PyTorch runtime overhead makes it unsuitable as an algorithm-only memory comparison. These CPU datasets are smaller than the paper's fitted embedding regime; no published quality/speed claim is inferred.

## CUDA and Runpod execution gate

The pinned image is recorded in [runpod/image.json](runpod/image.json); the exact dependency overlay is [runpod/requirements.lock](runpod/requirements.lock). That Linux CUDA environment has not been executed locally. Provision a single compatible NVIDIA Ampere/Hopper GPU and a persistent network volume mounted at `/workspace` before following the commands below. No cloud resource is created by these repository scripts.

Local preparation requires no GPU:

```bash
.venv/bin/python scripts/fetch_gpu_embeddings.py
.venv/bin/python scripts/benchmark_cuda.py --config configs/gpu_smoke.json --output results/local/gpu_plan --prepare-only
```

Synthetic workloads span 128/256/1024 dimensions; the real GloVe-100 subset has range/array SHA256 pins and original held-out splits. Prepared manifests contain **zero** GPU measurements. On actual NVIDIA hardware, `scripts/validate_cuda.py` must pass the full suite without skips before `scripts/benchmark_cuda.py` accepts its matching report. The harness saves calibration targets, warmups/repeats, per-epoch metrics, separate monitored/returned time-to-target observations, synchronized/event latency, allocated/reserved peaks and throughput. CPU-only execution fails closed. No Triton optimization or distributed mode is used.

Real-embedding source ranges, array hashes and attribution are retained in [data/glove100_source.json](data/glove100_source.json). Only the selected ranges and arrays are SHA256-pinned; the unselected remainder of the parent corpus was not hashed. Prefix selection can introduce bias, and squared Euclidean clustering MSE is distinct from the source site's angular retrieval metric.

Build a transfer archive locally:

```bash
.venv/bin/python runpod/package.py /tmp/e-means-gpu-source.tar.gz
```

Use the exact image named in `runpod/image.json` for the Pod, transfer/extract the archive under `/workspace`, then run inside that container:

```bash
cd /workspace/e-means
bash runpod/bootstrap.sh
mkdir -p /workspace/e-means-results
nohup bash runpod/run.sh /workspace/e-means-results/gpu-smoke-001 configs/gpu_smoke.json > /workspace/e-means-results/gpu-smoke-001-launcher.log 2>&1 &
```

The orchestrator fetches the pinned reference, executes the complete zero-skip NVIDIA suite, and benchmarks only after a source/environment/device-matched validation report succeeds. It saves failure logs as well as successful results, and exports a checksummed archive next to the run directory. Use new output paths. Inspect the smoke run before invoking `configs/gpu_research.json`, which plans 162 measured fits plus calibration and warmups. None of those GPU fits has been executed by this project yet.

Synchronized resident-fit wall time includes host control, epoch metrics and final test evaluation; H2D transfer is separately timed. CUDA-event intervals may contain host submission gaps and are not pure kernel time. Allocator peaks include resident inputs and evaluation, with allocated/reserved baselines recorded separately. Targets come from an independent Lloyd calibration with a predeclared `.001` relative slack. Time-to-target uses completed observations without interpolation; misses remain censored. Early main-EMA hits are reported separately from the selected finalizer's returned result. Repeated measurements are averaged within seed before reporting mean/sample SD across seeds.

## Citation

```bibtex
@misc{bessa2026emeans,
  title={e-means: outperforming k-means (in one-tenth the epochs)},
  author={Bessa, Swann and Mazaré, Pierre-Emmanuel and Szilvasy, Gergely and Touvron, Hugo and Picard, David and Douze, Matthijs and Jégou, Hervé},
  year={2026},
  eprint={2611.09501},
  archivePrefix={arXiv}
}
```
