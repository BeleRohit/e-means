# Experiment log

> Final archive notice, 12 November 2026: Phases 1–4 are closed. Historical preparation/access statements and Phase 5 proposals below are dated records, not current pending work. No further experiments or optimization are authorized or planned. See [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) for the final synthesis.

All user-facing dates use Asia/Kolkata. Machine-readable manifests use UTC timestamps. Recorded results are evidence from actual executions, not predictions.

## 11 November 2026 — Phase 1 numerical validation

Environment: Python 3.13.7, NumPy 2.5.4, pytest 9.2.0, optional faiss-cpu 1.16.0; macOS ARM64. See the run manifest for exact platform/Python strings. Paper parsing packages were installed as development tools; they are not package runtime dependencies.

Commands:

```bash
.venv/bin/python -m pip install -e '.[test]'
EMEANS_REFERENCE_DIR=/tmp/emeans-reference-2b648f .venv/bin/python -m pytest -q
```

Outcome: **54 passed**. Six executed optional cases compare against authors' source at `2b648f0605f2ef3d14c4705350f8a0b214463878`, verified by SHA256 before import. The remaining 48 tests independently cover distance/stats arithmetic, scalar full-loop oracle, Lloyd descent, reproducibility, validation, split weighting/geometry/cadence, finalization, two precisions, degenerate data, and actual experiment subprocess/artifact behavior. Normal offline tests explicitly skip the six reference cases if no reference source is supplied.

Additional checks: all walkthrough notebook code cells executed successfully; the installed package ran from outside the repository without importing Faiss or PyTorch. The offline test variant completed with **48 passed, 6 skipped**. All 12 result JSON/NPZ records, shared per-pair data hashes, and the manifest's source digest were verified against the final implementation. Source digest: `5272b26ac263aa5b2964389861ff8aad26f1beab519d6f8a85b6ac8610e65d83`.

Findings: paper and modern reference use different initialization, split delta, fitted coefficients, low-dimensional defaults, and tail policies. Tests align relevant values rather than assert equivalence between different recipes. Eq. 5's t/M schedule does not reach the floor on its final executed update. Closing means minimize fixed-code SSE; they should not be tested against an unsupported monotonicity claim for freshly evaluated final EMA centers.

## 11 November 2026 — paired synthetic CPU smoke experiment

Command:

```bash
.venv/bin/python scripts/benchmark.py --config configs/phase1_smoke.json --output results/phase1_smoke_2026-11-11
```

Status: **complete, 12 runs**. [Manifest and environment](artifacts/public/runs/phase1/manifest.json). Each pair shares exact train/test matrices and initialization; artifacts contain hashes and centers/labels. Each solver uses a fresh process. Config: seeds 0,1,2; n_train=1200, n_test=400, d=16, k=8; Lloyd up to 25 iterations with exact stopping; e-means 3 epochs, B=64, alpha0=.2, keep tails, paper revival/default finalizer; FP64. The experiment deliberately supplies B/alpha, rather than representing the fitted large-data laws as optimal for these tiny mixtures.

Medians across three seeds; MSE is mean squared norm per vector. Timing is single-run local measurement, not a statistically established speedup.

| Dataset | Algorithm | Final train MSE | Final test MSE | Solver loop (ms) | Entire fit (ms) | Peak process RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|
| Separated mixture | Lloyd | 42.235696 | 44.246065 | 7.576 | 16.908 | 36.344 |
| Separated mixture | e-means | 54.698006 | 52.518848 | 6.774 | 11.044 | 36.578 |
| Overlapping mixture | Lloyd | 14.909558 | 15.397616 | 20.559 | 43.052 | 36.328 |
| Overlapping mixture | e-means | 15.225443 | 15.609053 | 6.959 | 11.211 | 36.594 |

Interpretation: at these explicit budgets, e-means' median runtime is lower and its median MSE is higher on both mixtures. This does not reproduce the paper's speed/quality claim. Random row initialization can leave some true mixture modes unrepresented; neither method is guaranteed to find the global solution. Additional epochs, hyperparameter experiments and realistic embedding data belong to later evaluation.

Measurement limits: loop time excludes fixed-center epoch evaluation and initialization; total fit includes them. E-means closing time is recorded separately. RSS is whole-process `ru_maxrss` in bytes, including Python/NumPy imports, data and evaluation; it cannot establish a meaningful algorithm-only memory advantage at this scale. Timings are short, unrepeated and susceptible to scheduler/background contention. No confidence interval or hardware-normalized conclusion is supported. The complete per-epoch curves are retained in each JSON `history` rather than collapsed into the median table.

## Phase 1 follow-up plan (historical)

The original follow-up plan was automatic-law sweeps, held-out real embeddings, repeated timings, matched-MSE comparisons, slow-EMA train/test tradeoffs, and PyTorch parity before CUDA work. Some items are completed by Phase 2 below; realistic embedding-scale and CUDA measurements remain future work.

## 11 November 2026 — Phase 2 CPU correctness and regression gate

Environment: Python 3.13.7, NumPy 2.5.4, PyTorch 2.15.0, pytest 9.2.0, faiss-cpu 1.16.0, matplotlib 3.11.2; macOS ARM64. Exact versions/platform are retained in [phase2_validation_environment.json](docs/phase2_validation_environment.json).

Command:

```bash
EMEANS_REFERENCE_DIR=/tmp/emeans-reference-2b648f .venv/bin/python -m pytest -q
```

Outcome: **101 passed, 4 skipped**. All original 54 cases pass, plus historical centers/labels from all twelve Phase 1 runs match. Four FP32/FP64 CUDA Lloyd/e-means cases skip because `torch.cuda.is_available()` is false. CPU checks include per-update sum/count parity, exact controlled donor IDs, full histories/moving codes, all finalizers, half-life/speed conversion, independent scalar slow-EMA recurrence, validation, no-grad/input ownership and actual research subprocess execution.

Seven optional cases execute against pinned authors' code, including a new slow recurrence/split **primitive** comparison. No released end-to-end authors' slow finalizer was found in the audited files, so its missing policy oracle is an explicit limit. The full suite initially exposed a bundled Faiss/PyTorch OpenMP conflict on macOS; isolating reference cases in subprocesses resolved it without an unsafe duplicate-runtime override. Paper/reference differences documented in Phase 1 remain in force.

FP64 CPU parity assertions use atol/rtol=2e-12; FP32 uses 2e-5 over multi-epoch trajectories. These are tested tolerances for controlled inputs, not a guarantee of identical assignments near every distance tie or on every device/version.

Additional verification: the offline variant completed with **94 passed, 11 skipped** (seven optional authors' cases plus four CUDA cases). Editable installation with all Phase 2 extras succeeded. The public lazy PyTorch API ran from outside the repository; base `import e_means` imported neither Torch nor Faiss. All Phase 1/2 notebook code cells executed successfully, and exported convergence plots were visually inspected. Final source digest still matches all 120 measured records and the manifest.

## 11 November 2026 — Phase 2 public-data CPU experiment

Commands:

```bash
.venv/bin/python scripts/fetch_cpu_data.py --output data
.venv/bin/python scripts/research_cpu.py --config configs/phase2_cpu.json --output results/phase2_cpu_2026-11-11
MPLCONFIGDIR=.mplconfig .venv/bin/python scripts/summarize_cpu.py results/phase2_cpu_2026-11-11
```

The executed data preparation used the same verified archives through `--archive-dir /tmp`; the command above downloads those pinned archives when no local cache is supplied.

Status: **complete, 120 measured fits, all verified**. [Manifest](artifacts/public/runs/phase2/manifest.json), [full summary and convergence statistics](artifacts/public/phase2/summary.json), [PNG curves](docs/figures/cpu_convergence.png), [SVG curves](docs/figures/cpu_convergence.svg). The directory also retains 120 JSON/NPZ pairs, config.json, and a checksummed source.zip snapshot. Source digest: `4fed06289da4ffbafd68e0e639dee39f2aab55754f10893d66727fce256a1b3d`.

Datasets: UCI optical digits (3823 train / 1797 test, 64 features) and pen digits (7494 / 3498, 16 features), original provided splits. Labels are discarded. Feature ranges are scaled by fixed published constants 16 and 100, respectively. Archive/member/array hashes and attributions are in [data/README.md](data/README.md) and `data/*.json`; no test-based fitting or selection occurs.

Predeclared settings: k=32, FP64, shared initialization per dataset/seed, seeds 0–4, two repeated fresh-process fits per seed, one-thread settings, Lloyd up to 25 iterations with exact stopping, e-means 5 epochs at B=128 and alpha0=.2, retained tails and paper revival parameters. Slow EMA uses factor .25 of the theoretical cosine floor (.005 constant step); it is fixed a priori, not selected using test results. Both backends use the same controlled seed streams. These are explicit development configurations, not automatic-law experiments.

Each row below reports **mean ± sample SD across five seed means**. Repeated fits are averaged within each seed first; this avoids treating ten repetitions as ten independent seeds. MSE is recomputed in FP64 against returned centers. Runtime is entire fit in milliseconds; RSS is whole-process high-water MiB through fit, including framework imports and dataset memory.

| Dataset | Backend / finalizer | Train MSE | Test MSE | Fit ms | Peak process MiB |
|---|---|---:|---:|---:|---:|
| Optical | NumPy Lloyd | 1.836249 ± .004449 | 2.072305 ± .010209 | 1336.38 ± 52.26 | 45.68 ± .05 |
| Optical | NumPy e-means closing | 1.824642 ± .012599 | 2.034950 ± .020987 | 324.84 ± .52 | 46.04 ± .08 |
| Optical | NumPy e-means slow | 1.830756 ± .012879 | 2.037737 ± .019625 | 339.68 ± 14.79 | 46.18 ± .16 |
| Optical | PyTorch Lloyd | 1.836249 ± .004449 | 2.072305 ± .010209 | 1553.57 ± 90.72 | 206.66 ± 3.63 |
| Optical | PyTorch e-means closing | 1.824642 ± .012599 | 2.034950 ± .020987 | 433.01 ± 5.65 | 209.90 ± .59 |
| Optical | PyTorch e-means slow | 1.830756 ± .012879 | 2.037737 ± .019625 | 447.13 ± 5.67 | 209.92 ± .11 |
| Pen | NumPy Lloyd | .254665 ± .004793 | .272405 ± .004186 | 689.09 ± .40 | 42.55 ± .11 |
| Pen | NumPy e-means closing | .249024 ± .002097 | .265048 ± .007126 | 175.46 ± .37 | 42.67 ± .12 |
| Pen | NumPy e-means slow | .251278 ± .002191 | .265268 ± .006571 | 182.33 ± 2.01 | 42.62 ± .09 |
| Pen | PyTorch Lloyd | .254665 ± .004793 | .272405 ± .004186 | 814.09 ± 20.72 | 206.41 ± .35 |
| Pen | PyTorch e-means closing | .249024 ± .002097 | .265048 ± .007126 | 260.03 ± .79 | 206.49 ± .72 |
| Pen | PyTorch e-means slow | .251278 ± .002191 | .265268 ± .006571 | 311.59 ± 49.83 | 207.12 ± .27 |

Findings: all **60 NumPy/PyTorch paired FP64 center arrays and final label arrays were identical** in these runs (maximum absolute center difference 0). Repeated artifacts within each seed were also identical. Under these explicit budgets, e-means closing has lower mean train/test MSE and fit time than Lloyd at its iteration cap on both datasets. Slow EMA at factor .25 has slightly higher mean test MSE than closing on both; this setting does not reproduce a generalization improvement here. The unoptimized PyTorch CPU implementation is slower and has higher whole-process RSS than NumPy on these small problems.

Interpretation limits: observations are limited to two small real feature datasets, five seeds, two timings per seed, one macOS CPU environment and explicit B/alpha. They are not the paper's high-dimensional embedding workloads, matched-MSE time-to-target trials, automatic scaling-law validation, statistical confidence intervals or GPU results. Main-state convergence lines precede finalization; endpoint markers are final readouts. The slow and closing variants share the same main trajectory. RSS largely includes PyTorch runtime overhead and cannot establish an isolated algorithm memory cost. No CUDA, RunPod, Triton or distributed experiment ran.

Recommended next measurement gate: actual CUDA correctness with FP32/FP64 and slow revival, synchronized/event timings plus allocator peaks, then pinned realistic embeddings at the fitted scale with predeclared matched-MSE targets. GPU optimization follows measured baseline correctness.

## 11 November 2026 — Phase 3 local preparation, not GPU results

Status: **local preparation complete; zero GPU measurements; NVIDIA gate closed**. Host remains macOS ARM64, Torch 2.15.0, `torch.version.cuda=None`, `torch.cuda.is_available()=False`. No Pod was provisioned and no GPU calibration, warmup or contender ran.

```bash
EMEANS_REFERENCE_DIR=/tmp/emeans-reference-2b648f .venv/bin/python -m pytest -q
bash -n runpod/bootstrap.sh runpod/run.sh
```

Outcome: **146 passed, 32 skipped**. The original 101 passing Phase 2 cases remain passing. New coverage is 28 CPU rehearsals of the NVIDIA matrix and 17 protocol/config/data/timing-order checks. The 28 new NVIDIA cases and four original CUDA cases skip explicitly. Timer mocks check ordering/units/peak-reset semantics, not actual CUDA behavior, and do not save GPU verification certificates. Numeric clustering algorithms are unchanged.

Primitive tolerances are FP32 atol/rtol=3e-6 and FP64=2e-12; full-trajectory CUDA budgets are 1e-4 and 1e-10, with exact controlled assignments/donor identities still required. These remain prepared assertions, not GPU-validated tolerances. NVIDIA failures must be investigated before benchmarks proceed; tolerances are not automatically widened.

Real data: ANN-Benchmarks GloVe-100 HDF5 layout was inspected with h5py 3.16.0. A slow full download was interrupted after obtaining the needed training prefix. Only selected raw ranges and their prepared arrays are SHA256 verified: 16,384 original train and 2,048 original test vectors, 100 dimensions, no renormalization/fitted preprocessing. The test range returned exact HTTP 206/Content-Range and the recorded ETag. The entire 485,413,888-byte parent file is **not** SHA256-verified. Pins, layout and credit are in [data/glove100_source.json](data/glove100_source.json); prepared array hash is `a85b3b6fa7a7616130acc82d3a4ab7709bd086765d4df91f7970093875f1e404`. Squared Euclidean clustering results are not ANN angular retrieval metrics.

Executed commands:

```bash
.venv/bin/python scripts/fetch_gpu_embeddings.py --cache-dir /tmp
.venv/bin/python scripts/benchmark_cuda.py --config configs/gpu_smoke.json --output results/phase3_preparation_2026-11-11/release_smoke --prepare-only
.venv/bin/python scripts/benchmark_cuda.py --config configs/gpu_research.json --output results/phase3_preparation_2026-11-11/release_research --prepare-only
.venv/bin/python scripts/validate_cuda.py --output results/phase3_preparation_2026-11-11/release_gate
```

The release preparation manifests cover four smoke and eighteen research workload/seed/dtype combinations, with empty run lists and `gpu_verified: false`. Source/config/script/regression-fixture fingerprint: `04cc5d83d51c18581423449ef7008d9ecaa5fcc5e12c9aa032f303b7105bfea8`. Earlier non-release folders retain intermediate preparation-only snapshots. The hardware gate exits **2**, writes `hardware_unavailable`, and never executes pytest or issues a successful CUDA report. This is an expected closed gate, not a failed numerical GPU experiment.

The harness requires a complete, zero-skip NVIDIA suite, source/environment/device matching and checksummed XML. Prepared measurements separate target calibration, warmups, rotated repeats, synchronized/event timing, allocator peaks/baselines, MSE, throughput and per-epoch observations. Monitored main-state target hits and returned-finalizer target hits are distinct; misses remain censored. Summaries average repeats within seed before calculating sample SD, and preserve conditional target-hit counts. See [docs/RUNPOD.md](docs/RUNPOD.md) for all definitions and exact commands.

The official image tag `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` resolved through Docker registry metadata to `sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`. The Linux AMD64/Python 3.11 image and exact dependency overlay are **not container/GPU validated locally**. Bootstrap and the complete numerical suite must run on the eventual host. Shell syntax and local protocol tests passed. No Triton optimization, torch.compile, ANN assignment or distributed clustering was introduced.

## 11 November 2026 — requested Runpod agent setup

Fetched and followed [the official guide](https://agents.runpod.io/), using its Codex route:

```text
codex plugin marketplace add https://github.com/runpod/runpod-plugins-official.git
codex plugin add runpod@runpod
```

Verified with Codex CLI 0.160.0: `runpod@runpod` **installed, enabled**, version **2.0.0**; 17 nonempty skill files; hosted MCP endpoint `https://mcp.runpod.io/` configured through the plugin. No duplicate server, API key, separate CLI or Flash SDK was added. User-level plugin settings were changed as requested; no credentials were printed or copied into the repository.

Authentication/account access is **unverified**. The remaining guide step is to fully quit/reopen Codex Desktop, approve OAuth when prompted, then verify by listing Pods. This session did not list Pods or provision resources. Plugin installation is not evidence of authenticated account connectivity or CUDA access.

## 11 November 2026 — requested GPU execution: stopped at access gate

Reviewed checkout `c1b104b` (clean tracked working tree), CUDA tests, pinned image/dependencies, both GPU configurations, `validate_cuda.py`, and the Runpod bootstrap/orchestration scripts. Requested infrastructure is one RTX 4090 (24 GB), with a total budget below $10 including storage, subject to explicit paid-provisioning approval.

Read-only connection checks: Runpod plugin remains enabled and configured for `https://mcp.runpod.io/`, but this session exposes no Runpod MCP tools. `runpodctl` is absent and `RUNPOD_API_KEY` is not set. No credential values were read or printed. Account authentication, balance, active Pods, GPU stock and current pricing could not be verified. No resources were created/modified, no dependencies were installed remotely, and no CUDA tests or GPU benchmarks ran. This is an access-blocked attempt, not an experimental failure or measured GPU result.

The existing `runpod/run.sh` enforces the numerical gate and exports results, but does not enforce a dollar cap or terminate the cloud Pod. Before paid execution, establish a conservative current-rate budget including storage, an external deadline/cleanup mechanism, artifact retrieval and verified Pod termination. The user's $10 ceiling is not treated as explicit permission to provision.

Next action: fully quit/reopen Codex Desktop, approve Runpod OAuth when prompted, then verify the connection by listing Pods/account balance. After read-only access succeeds, obtain a live RTX 4090/storage quote and request explicit approval for that concrete spending/teardown plan. Existing preparation and all prior artifacts are preserved.

## 11 November 2026 — CLI authentication verified; provisioning not authorized

`runpodctl 3.1.0-5ab53b9` account read succeeded with network access enabled: balance $10, current spend $0/hour. `runpodctl pod list --all` returned an empty list. MCP tools remain absent; authenticated CLI access is available. These observations supersede the earlier access-blocked audit. No credential values are recorded here.

The live GPU catalog including unavailable entries reports RTX 4090 (24 GB) unavailable, with listed Secure $0.89/hour and Community $0.34/hour GPU rates. A40 (48 GB) reports available, Secure $0.59/hour, low stock in CA-MTL-1 and EU-SE-1. These are catalog GPU rates, not complete storage-inclusive quotes or reservations.

Re-reviewed pinned image/bootstrap, smoke/research configs and execution orchestration. Stop at the user's explicit provisioning-approval gate: no resources created, CUDA tests executed, or GPU measurements obtained. GPU selection, storage pricing, budget deadline and verified teardown still require resolution before execution.

## 11 November 2026 — completed actual Runpod A40 CUDA milestone

Explicit user approval: one A40 with a two-hour initial session cap and total budget below $10. Pod `0nuwjg528hyuf1`, temporary network volume `48m37issf3` (20 GB), CA-MTL-1 Secure Cloud. Pinned official image from `runpod/image.json`; A40 compute capability 8.6, driver 595.91.07, Torch 2.5.1+cu124, CUDA runtime 12.4, cuDNN 90100, Python 3.11.10, NumPy 1.26.4. Device UUID and complete metadata are in both validation reports. TF32 disabled, deterministic algorithms enabled, thread counts fixed to one.

Deployment findings: extraction on the network filesystem rejected restoration of macOS ownership; `tar --no-same-owner` resolved it. SSH shells needed `/opt/conda/bin` on PATH. Base-image Ninja 1.11.1.1 failed pip 25.0.1 check with an unsupported-platform wheel warning. Exact overlay replacement `ninja==1.11.1.3` passed pip check and is now tracked in `runpod/requirements.lock`. The initial `a40-smoke-001` dependency-check failure was preserved; it ran no CUDA validation or measurements. No algorithm, numerical tolerance or benchmark configuration was changed.

Both successful runs executed the entire suite: **178 passed, zero failures/errors/skips** (smoke 35.28 s). This includes 28 new CUDA matrix cases and the four retained CUDA cases, plus every Phase 1/2 test and pinned authors' comparisons. Tested FP32/FP64 primitives and full controlled trajectories match NumPy and Torch CPU within predeclared tolerances; exact controlled labels and donor IDs are checked. Primitive atol/rtol: FP32 3e-6, FP64 2e-12. Full-trajectory atol/rtol: FP32 1e-4, FP64 1e-10. Slow EMA remains the documented interpretation; an unavailable authors' complete slow finalizer is not claimed verified.

Smoke: `configs/gpu_smoke.json`, one seed, two precisions, three methods, two measured repetitions, **24 GPU measurements**. Research: unchanged `configs/gpu_research.json`, three workloads, three seeds, two precisions, three methods, three measured repetitions, **162 GPU measurements**. Both include separate Lloyd target calibrations and untimed warmups. Total **186 measured fits**, with complete paired-input hashes, target records, convergence points, synchronized/event timing, throughput, allocated/reserved peaks and source/config snapshots. Research completed 2026-11-11T17:58:00Z (23:28 IST). Source fingerprint for both studies: `42baee29f1e331c4f2ea9acc69ef44cae97479ccfdd80a5d28c146f870589ec5`.

Research FP32 results, mean ± sample SD across three independent seeds after averaging repetitions within seed:

| Workload | Method | Train MSE | Test MSE | Resident fit seconds | Mean peak allocated MiB |
|---|---|---:|---:|---:|---:|
| gaussian1024 | emeans | 1105.1 ± 33.01 | 1124.26 ± 27.32 | 4.52839 ± 0.3309 | 49.942 |
| gaussian1024 | lloyd | 1189.16 ± 63.69 | 1213.46 ± 61.09 | 1.74672 ± 0.1006 | 48.422 |
| gaussian1024 | slow_ema | 1108.86 ± 37.09 | 1127.78 ± 31.89 | 4.383 ± 0.1866 | 50.192 |
| gaussian256 | emeans | 275.77 ± 5.876 | 281.543 ± 5.942 | 2.77117 ± 0.3511 | 24.817 |
| gaussian256 | lloyd | 289.677 ± 7.439 | 294.608 ± 6.73 | 1.45046 ± 0.05498 | 24.282 |
| gaussian256 | slow_ema | 277.217 ± 4.883 | 283.017 ± 4.626 | 2.95543 ± 0.2559 | 24.942 |
| glove100 | emeans | 25.4488 ± 0.01727 | 26.0416 ± 0.05295 | 2.5028 ± 0.1075 | 18.492 |
| glove100 | lloyd | 25.7463 ± 0.07604 | 26.3105 ± 0.0887 | 3.71372 ± 0.2179 | 18.230 |
| glove100 | slow_ema | 25.5181 ± 0.01608 | 26.0393 ± 0.03676 | 2.91549 ± 0.2429 | 18.541 |

For these fixed budgets, closing-mean e-means reduced FP32 mean test MSE versus Lloyd by 4.43% (Gaussian256), 7.35% (Gaussian1024), and 1.02% (GloVe100). Mean resident wall time was 1.91×/2.59× Lloyd on the Gaussian workloads; GloVe was 1.48× faster than Lloyd. Slow EMA is separately recorded; it is not uniformly better than closing means. These are measured development results, not paper-reported or paper-scale claims. Lloyd uses max_iter=25 (may converge earlier), e-means five epochs with explicit alpha0=.2 and B=256. No claim of equal training compute or tuned published scaling-law reproduction. All research returned test-target records reached the separately calibrated targets (9/9 repetitions per workload/dtype/method). Target latencies and stages remain separately reported rather than inferred from aggregate fit times.

Both completed checksummed archives were downloaded, safely extracted and independently audited locally for complete coverage, paired inputs, per-fit arrays, calibration/source/config hashes, hardware and passing JUnit identity. Full FP32/FP64 tables, curves, all metrics and scope limits: `results/runpod/a40-20261111/REPORT.md`; flat metrics: `metrics.csv`; original immutable archives and failure/setup logs are retained alongside them. No generated data/results were added to Git.

Cleanup: Pod deletion verified by fresh 404/not_found; volume deletion likewise verified; subsequent Pod and volume inventories empty. Controller wrote `cleanup-complete`; local watchdog exits after normal cleanup. Account read at 18:00:12Z: balance $9.7524077472, currentSpendPerHr=0, observed decrease from initial $10 of $0.2475922528. This is an observed balance decrease, approximately $0.25, not a finalized invoice: billing bucket reads still returned empty arrays. No resources remain billing according to the fresh account/inventory reads. The local two-hour caffeinate/watchdog depended on local network availability; it was not a provider-enforced dollar cap.

Limitations and next work: single A40/runtime; three small development workloads and three seeds; fixed-prefix GloVe subset; MSE is squared Euclidean per vector; full parent embedding file not SHA256 hashed. Frequent evaluation and Python/per-dimension loops affect resident timings. No Triton, distributed clustering, paper-scale reproduction or universal hardware/parity claim. Next milestone: profile host/launch/evaluation overhead, compare equal-target quality and matched-budget sweeps with explicit scaling-law configurations, extend seeds/data scale under a separately approved budget, and strengthen provider-independent cleanup controls before larger campaigns.

## Phase 4 — local preparation, GPU execution not yet authorized (12 November 2026)

Re-audited all 186 original A40 measurements. Lloyd executed only 4 Gaussian1024 iterations and 6–8 Gaussian256 iterations, versus five complete e-means epochs. Existing broad wall timers show substantial epoch-evaluation cost; they do not identify GPU kernel or host bottlenecks. Original fixed-budget timing ratios compare unequal final MSE and are not matched-quality speedups.

Implemented a benchmark-only observation layer and fresh returned-horizon protocol, without changing any NumPy/PyTorch optimizer update. Primary latencies are unprofiled; exact diagnostic replays provide profiling/distance accounting. Profiling emits CPU self-time/operator and synchronization candidates, overlapping inclusive ranges, CUDA event spans and Chrome traces; actual captured CUDA events are required for GPU attribution. Main EMA snapshots cannot stand in for returned closing/slow finalizers. Discrete first-horizon target attainment, grid-search cost and censored misses are explicit. Ratios require both paired methods to meet identical targets under the same observation policy.

Prepared `configs/phase4_pilot.json` (312 primary fits, one seed/two repetitions), `configs/phase4_research.json` (936 fits, three seeds/two repetitions), nine full baseline-budget profiles, and six paired official native comparisons. Paper automatic B/alpha and coupled half/double sensitivities are predeclared. Formula outputs/plans are not measured scaling-law quality results. CPU rehearsals verified complete coverage, target/array/config/source/calibration hashes, plotting and profiling output under CPU-only statuses.

Actual local official Faiss CPU primary study: 24 fits, two small smoke workloads, three seeds, two repeats, identical FP32 initialization/order, mass 2**-16, retained tails and revival off. All 12 paired center arrays exactly equal; lower author CPU solver latency on these fixtures. Single host/CPU evidence is not GPU performance evidence. Primary artifacts: `results/phase4/local/official-cpu-003`; older exploratory CPU runs retained separately. Optional native-Torch files from the pinned authors commit are SHA256 checked; native compatibility tested on CPU with explicit FP32 typed-input override to suppress default CUDA BF16 casting. No Triton/CuTe invocation or official slow-EMA result is fabricated.

New pipeline: `scripts/audit_phase3.py`, `scripts/phase4_cuda.py`, `scripts/summarize_phase4.py`, `scripts/compare_official_phase4.py`, `scripts/compare_native_phase4.py`, `runpod/phase4.sh`, local `runpod/watchdog.py`. `PHASE4_REPORT.md` separates measured baseline/CPU findings from pending GPU tables and gives exact commands/interpretation limits. The current CUDA gate must be rerun; the historical Phase 3 certificate does not authorize changed source. No paid Phase 4 infrastructure has been created.

Read-only Runpod check: balance $9.7524077472, current rate $0/hour, Pods/volumes empty; A40 Secure $0.59/hour, low stock CA-MTL-1/EU-SE-1. Proposed initial two-hour pilot about $1.20 including 30 GB container/20 GB standard network storage, subject to explicit new paid-resource approval and current readback pricing. External local watchdog/process timeouts need an awake/online controller and are not backend dollar enforcement; verify backup and both deletions on completion. Full three-seed work remains a separate review decision after the pilot. No new GPU bottleneck ranking, matched-MSE speedup, scaling-law quality outcome or official native CUDA compatibility is claimed yet.

Final Phase 4 local validation: **166 passed, 33 NVIDIA skips** (199 collected). Source SHA256 `df38023f14ce1594f5706ebc8e6b99e71b7906163bdc09d948d821d108466093`. Both final prepare-only plans match this source, with empty measurement lists and gpu_verified=false. CPU protocol/profile summaries and the primary 24-fit official CPU study passed independent local integrity audits. No new GPU execution or paid resource was started.


## 12 November 2026 — Phase 4 actual A40 execution, verified export and teardown

Explicit user approval authorized one A40, the 312-fit pilot/profiling/scaling/reference studies, a $2 total cap including storage, and deletion after export. The 936-fit campaign was excluded and was not run. Existing Phase 4 scripts/configurations were used without changing algorithms, kernels or numerical tolerances.

Session: `results/phase4/runpod/a40-20261112-001/`. Secure Cloud CA-MTL-1, A40 UUID `GPU-7c01fb29-ab82-970f-1796-7c6cc8bb284d`, driver 595.91.07, PyTorch 2.5.1+cu124, Python 3.11.10, NumPy 1.26.4; deterministic algorithms, TF32 off. Exact image/dependency overlay and source fingerprint `df38023f14ce1594f5706ebc8e6b99e71b7906163bdc09d948d821d108466093` recorded. Session-local CLI 3.1.1-72afce1 checksum verified; no secrets transferred.

Executed `bash runpod/bootstrap.sh`, then `timeout --signal=TERM --kill-after=60s 6000 bash runpod/phase4.sh /workspace/results/phase4-pilot-001`. Full CUDA gate ran 05:51:29–05:52:18 UTC (11:21–11:22 IST): **199 tests passed, no skips/failures/errors**, including FP32/FP64 behavior, EMA, revival, cosine, both finalizers and actual profiler/event capture. Main execution/export status succeeded at 06:34:35 UTC, followed by archive compression. The isolated CPU comparison began after all main GPU/export work completed.

Completed **9 profiles**, **312 primary quality fits**, **6 native GPU paired records** (12 full fits plus assignment micro-comparisons), and **24 official Faiss CPU fits**. Warmups, independent calibrations and exact-output replays are additional execution, not part of 312. All requested pilot stages completed. All three FP32 seed-zero input/init hashes match Phase 3. All 132 every-epoch/final-only policy pairs produced exactly equal centers and labels.

Final-only first returned horizon meeting BOTH train and test calibration × 1.001, mean seconds across two timing repetitions:

| Workload | Lloyd | Closing fixed | Closing automatic | Slow fixed | Slow automatic |
|---|---:|---:|---:|---:|---:|
| Gaussian256 | .4018 (T4) | .4396 (T1) | .7966 (T1) | .5086 (T1) | .8335 (T1) |
| Gaussian1024 | .6558 (T3) | .7923 (T1) | 2.6679 (T1) | 1.3641 (T2) | 2.6070 (T1) |
| GloVe100 | 1.5810 (T25) | .8176 (T2) | .4709 (T2) | .8829 (T2) | .5389 (T2) |

These are fresh budget-dependent finalized fits, not online stopping or identical terminal losses. Two repeats are not two independent seeds. Automatic closing e-means on GloVe has a **3.36× ratio of mean common-threshold times** versus Lloyd; fixed e-means remains slower on both Gaussians. Across all target factors/policies/variants/train/test/joint scopes, **104/648 target records are censored** (4 at 1.001, 100 at .99); all 312 fits completed. At .99 joint, Lloyd misses on all workloads and slow fixed/automatic miss on GloVe. No ratio is inferred from a censored pair.

Assignment inclusive host ranges occupy 65–99% of instrumented fit intervals. Gaussian1024 e-means launches 370,762 kernels, 248,800 in training assignment. Kernel-launch CPU self time is 1.864 s; GPU activity union occupies about 7% of the instrumented span (not unprofiled utilization). EMA/revival/finalizer/sync tables are retained; finalizer readout alone is below 1 ms, while shadow updates add training work. This identifies assignment/dispatch as the next investigation target without attributing all residual wall time to one host component.

Scaling-law application is workload/horizon dependent. At T3, automatic B/alpha are 512/.5128, 256/.5163 and 1024/.7005 for Gaussian256/1024/GloVe. Automatic settings improve Gaussian quality; small T1 batches worsen Gaussian latency. Larger-batch sensitivity reduces latency with non-monotone quality; doubled alpha caps at 1. All variants are retained, not selected by test quality. This does not validate fitted exponents outside their original regime.

Peak allocated GPU memory across quality fits is 24.28–25.70 / 48.42–51.20 / 18.23–19.25 MiB; maximum reserved is 44/66/66 MiB. These include resident inputs and exclude driver/context. Throughput and CUDA-event/wall timing remain in records and derived tables.

Native reference commit `2b648f0605f2ef3d14c4705350f8a0b214463878`: explicit native FP32, no BF16/Triton/CuTe; B256/alpha .2/T3, retained tails, mass 2^-16, revival off on both sides. All six initial assignment-label arrays match exactly; full Gaussian centers match; GloVe maximum center difference .00275086 and slightly different MSE are documented. Raw assignment means ours/authors: .073719/.002808 s, .148749/.001413 s, .070744/.007803 s, respectively. This is backend/control-path headroom, not a paper algorithmic speedup. Faiss CPU 1.10.0 ran single-thread after GPU work: 24 fits, 12/12 exact paired centers. No Faiss-GPU or complete authors slow-finalizer comparison was executed.

Both exports were independently verified locally before teardown: 721 main and 65 CPU members, whole-archive SHA256 plus every member digest. Complete 5.094 GiB raw traces are retained losslessly in the compressed main archive. Source/data transfer archive SHA256: `53db3eaeb217252d0cff5b04ac7fe9c6031af7be3eb976736a8d7cb722619f04`. Original Phase 3 artifacts are unchanged. Metadata, arrays, plots, launch correlations, source/config/environment snapshots, original archives and downstream analysis scripts remain under the session path. See PHASE4_REPORT.md for full readable tables, commands, figures and limitations.

Budget: approximately $0.596/hour including disk/network storage; external two-hour watchdog, process timeout and repeated balance checks. After verified local exports, Pod `wy0bzxf36j7wtd` and temporary 20 GB volume `nzrqbbfov4` were deleted. Both fresh reads returned 404, inventories were empty, and account spend was $0/hour at 06:47:26 UTC. Initial balance $9.7524077472 → $9.2006167611: **observed debit $0.5517909861**, including storage account charges, under $2. Itemized billing arrays were still empty; this is not a final invoice. The watchdog cleanup marker was written. No cloud resources are retained.

Phase 5 recommendation: independently validate an opt-in shared assignment backend against direct-distance oracles, then measure with identical observation policies and broader predeclared seeds under separate approval. No optimization or 936-fit campaign started.


## Final archive — 12 November 2026

Project closed through Phase 4 at the user’s instruction. No Phase 5 optimization, further GPU experiments or cloud calls were performed. Final local suite: **166 passed, 33 NVIDIA-only skips**. All 12 validated src modules match the executed Phase 4 source ZIP byte for byte. Re-audited all 120 CPU fits, 60 exact backend pairs, 186 Phase 3 GPU fits, 312 Phase 4 quality fits, nine profiles and six native GPU pairs; retained 104 censored target observations. Reverified complete raw archive checksums and every member of both Phase 4 exports (721/65).

Public documentation, raw numeric arrays/records, historical independent source snapshots, notebook, configs and figures are retained under an explicit publication manifest. Supported secret-pattern checks found no credential values in public content/nested source ZIPs, local reachable Git blobs or Git config. Caches, the downloaded CLI binary and four byte-identical preview files were removed with a cleanup inventory; no unique measured result was removed. The local environment remains ignored so tests can be rerun. Final report: REPRODUCTION_REPORT.md. No repository push, visibility change or remote archive hosting is claimed. The code license remains an owner choice.
