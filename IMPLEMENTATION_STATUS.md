# Implementation status

> Final archive notice, 12 November 2026: Phases 1–4 are closed. Historical preparation/access statements and Phase 5 proposals below are dated records, not current pending work. No further experiments or optimization are authorized or planned. See [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) for the final synthesis.

Updated: 12 November 2026 (Asia/Kolkata). Current milestone: Phase 4 A40 pilot measured and exported; cloud resources deleted. Historical access/preparation entries below are retained as dated records.

## Phase 1 — implemented and locally verified

- [x] Package with NumPy-only runtime dependency, editable installation verified.
- [x] Lloyd full-batch assignment/means, deterministic initialization, empty-cluster policy, convergence history and stopping.
- [x] Explicit sum/count e-means EMA, tiny initial mass, literal cosine equation, and separate stochastic/fixed-center losses.
- [x] Every-five-update dead-cluster checks, weighted live donors without replacement, alternating split perturbations, count/sum consistency.
- [x] Published scaling-law coefficients and explicit batch/step overrides.
- [x] Last-epoch moving-assignment means; preserve empty centers; fresh final labels kept separately.
- [x] Float64 baseline and float32 option; bounded assignment tiles; finite-input/shape/config checks.
- [x] 48 independent numerical/API/experiment tests plus six pinned authors' comparisons: **54 passed** locally.
- [x] Paired synthetic convergence, train/test MSE, runtime and process RSS experiment: two datasets × three seeds × two algorithms, **12 runs**.
- [x] Source/config/data hashes, immutable run directories, serialized configurations, environment metadata, JSON histories and NPZ artifacts.
- [x] PAPER_MAPPING.md, REPRODUCTION_NOTES.md, EXPERIMENT_LOG.md and runnable walkthrough.

The Phase 1 correctness gate is satisfied for the tested cases. This does not establish large-scale performance, optimal scaling-law behavior on tiny data, or general parity for different random streams. Local validation used Python 3.13 on macOS ARM64; the full supported Python/NumPy version matrix remains unverified.

## Phase 2 — implementation and CPU correctness verified

- [x] Independent PyTorch Lloyd/e-means with shared configuration/API semantics and optional CPU/CUDA device selection.
- [x] FP32/FP64 tensor arithmetic, two-dimensional chunked direct-distance assignment, detached inputs and tensor outputs.
- [x] Matched NumPy initialization/order/donor RNG streams; tests compare every update, actual revival identities, whole trajectories and final readouts.
- [x] Slow count/sum EMA on both backends, independent constant-speed scalar oracle, optional explicitly defined half-life, and mirrored main-model splits.
- [x] Slow recurrence/split arithmetic compared with pinned authors' primitives; missing complete authors' slow finalizer is documented.
- [x] Historical twelve-run NumPy result regression checks; original 54 tests preserved.
- [x] Complete suite: **101 passed, 4 CUDA cases skipped** on this CPU-only macOS ARM64 environment.
- [x] Optional Faiss reference cases isolated in subprocesses to avoid conflicting bundled OpenMP runtimes.
- [x] Pinned public UCI optical/pen digits acquisition, original splits, no labels in training, fixed feature-range scaling.
- [x] Research runner with five seeds, two repetitions, six methods, source/config snapshots, checksummed artifacts and explicit timing/RSS definitions.
- [x] Coverage/integrity verification, per-seed aggregation before mean/sample-SD calculation, and exportable train/test convergence plots.

- [x] Real-data experiment completed: **120 fits**, two datasets × six methods × five seeds × two repetitions, with verified coverage/integrity and deterministic repeated arrays.
- [x] All 60 paired NumPy/PyTorch FP64 center/label outputs identical in those runs; measured mean/sample-SD results and convergence curves retained.

Actual research-run outcomes are recorded in EXPERIMENT_LOG.md. Phase 2 CPU results and the later A40 CUDA study are separate evidence. Numerical parity covers controlled fixtures, not every near-tie input or hardware/software version.

## Phase 3 — A40 CUDA validation and GPU studies complete

- [x] New correctness matrix: 28 CPU rehearsal cases and 28 matching guarded NVIDIA cases, covering FP32/FP64 assignment/statistics, literal cosine updates, revival/own-state shadow splits, finalizers, full Lloyd/e-means trajectories, tails, empty clusters and no-grad behavior.
- [x] NumPy plus PyTorch CPU references; predeclared primitive and trajectory tolerances, with exact controlled labels/donor identities. Existing four CUDA cases retained.
- [x] Full suite locally: **146 passed, 32 skipped**. All 32 skips require NVIDIA CUDA. Original Phase 1/2 numerical algorithms remain unchanged.
- [x] Validation runner requiring an actual CUDA device probe and the entire passing, zero-skip suite; report binds source/config/scripts, GPU/driver/runtime identity and validation XML.
- [x] CUDA benchmark harness with separate calibration, identical data/initialization, warmups, repeated fresh models, synchronized wall time, CUDA events, CUDA allocated/reserved peaks, throughput and per-epoch train/test observations.
- [x] Predeclared Lloyd-relative target, no time interpolation, explicitly censored misses, and separate monitored-state versus returned-finalizer availability.
- [x] Reproducible 128/256/1024-dimensional Gaussian workload configurations and pinned real GloVe-100 train/test prefixes.
- [x] Selected real embedding byte ranges and prepared arrays verified locally by SHA256. The entire 485 MB parent file is explicitly not claimed SHA256-verified.
- [x] Preparation-only mode emits zero GPU measurements; actual benchmark/summary paths reject missing/failed/mismatched validation and incomplete artifacts.
- [x] Immutable official CUDA image digest resolved, exact dependency overlay, bootstrap/validate-before-benchmark script, persistent failure/success exports, transfer package and [Runpod instructions](docs/RUNPOD.md).
- [x] Official Runpod Codex plugin 2.0.0 installed/enabled; 17 skill files and hosted MCP endpoint wiring verified.
- [x] Authenticated Runpod CLI 3.1.0; explicit user approval for one A40 and a two-hour initial session.
- [x] Pinned Linux CUDA container bootstrapped on actual A40; complete suite **178 passed, zero skips/failures/errors**. Ninja compatibility fix pinned in overlay.
- [x] Actual GPU smoke study: **24 measured fits**, checksummed export downloaded and independently audited locally.
- [x] Research study: **162 measured fits**, three seeds × two precisions × three workloads × three methods × three repeats; complete artifact audit passed.
- [x] Both exports downloaded and independently audited; Pod/volume deletion verified, inventories empty, current spend $0/hour. Observed account decrease approximately **$0.25**; finalized billing buckets not yet populated.

Actual Runpod A40 execution uses Torch 2.5.1+cu124, Python 3.11.10, NumPy 1.26.4, driver 595.91.07, deterministic algorithms and TF32 disabled. The local macOS host remains CPU-only. GPU parity is established for the executed tests; no paper-scale performance reproduction is claimed. Custom kernels, Triton optimization and distributed clustering have not been introduced.

After the hardware gate passes, inspect smoke results and only then execute the larger predeclared workloads. Scaling-law validation at paper-sized real embeddings and broader version/hardware CI remain later research work; these prepared development workloads do not reproduce the paper's largest scales.

### GPU execution request — access blocked, 11 November 2026

Inspected published checkout `c1b104b`. Preferred hardware: RTX 4090 24 GB; total task ceiling: below $10 including storage. No live Runpod tools are exposed, no CLI is installed, and no API-key environment variable is available. The installed plugin's endpoint configuration is not authenticated-access evidence. No active Pod, balance, stock, rate or NVIDIA execution has been verified; no paid infrastructure was provisioned.

Required next step is desktop restart/OAuth and a read-only Pod/account check. Paid provisioning still requires explicit approval after a concrete live-priced plan is available. The existing script must be paired with external budget/deadline control and verified Pod termination: its current EXIT trap exports artifacts only. GPU validation and benchmark milestones remain pending, with no new GPU results.

## Known limits

- Default dropped tails follow the paper's floor update count; retained tails are explicit and tested.
- Multiplicative zero-vector splits can remain coincident; no non-paper additive displacement is introduced.
- Only integer epoch budgets and Euclidean unweighted clustering are implemented.
- In-memory input/state, no resumable training, spherical clustering or ANN assignment.
- Process RSS includes interpreter/data/evaluation and is not isolated optimizer memory.
- PyTorch revival uses host-side seeded donor control, verified in controlled A40 CUDA tests. Device-native RNG is not silently substituted.
- Complete authors' slow-EMA routine is unavailable in the audited release; shadow split/half-life conventions are explicit interpretations.
- Small real-data CPU experiments and sample standard deviations are not confidence intervals or paper-scale embedding reproduction.
- No production release or paper table reproduction is claimed; actual A40 development benchmarks are recorded separately.

## 11 November 2026 — latest execution gate

Runpod CLI account authentication now verified; balance $10, current spend $0/hour, no Pods. This supersedes the previous access-blocked status. RTX 4090 is unavailable in the live catalog; A40 Secure Cloud is available at $0.59/hour GPU-only. Explicit paid provisioning approval is still absent. No NVIDIA validation/benchmark results exist. Storage-inclusive budget guard, artifact retrieval and verified termination remain prerequisites.

## Final A40 execution status — 11 November 2026

Actual NVIDIA evidence supersedes historical access/provisioning-blocked entries above. Complete suite passed 178 tests with zero skips before each successful study. 24 smoke + 162 research = 186 actual measured GPU fits. A40 FP32 closing e-means lowered mean test MSE on all three research workloads; it was slower on Gaussian256/1024 and faster on GloVe100 at the declared budgets. Exact mean/sample-SD metrics, target observations, memory, throughput and scope are in `results/runpod/a40-20261111/REPORT.md` and original artifacts. Only tracked code change is an exact Ninja compatibility overlay pin; numeric algorithms/configurations/tolerances remain unchanged. README reflects actual GPU execution. Pod and volume are gone; current spend zero; remaining balance $9.7524077472. No RTX 4090, Triton, distributed or paper-scale results are claimed.

## Phase 4 — A40 pilot executed and independently verified

- [x] Preserved/re-audited all 186 Phase 3 baseline fits; Phase 4 seed-zero FP32 input/init hashes exactly match that baseline.
- [x] Full current suite on actual A40: **199 passed, zero skips/failures/errors** before benchmarking; unchanged predeclared tolerances and numerical algorithms.
- [x] Nine CUDA profiles, separately timed unprofiled references, exact-output diagnostics, CUDA-event ranges, operator self times and lossless raw traces.
- [x] **312 primary matched-MSE fits**: one independent seed, FP32, two timing repeats, fixed/automatic recipes, batch/alpha sensitivity and both observation policies.
- [x] All 132 observation-policy pairs return bit-identical centers/labels. 104/648 target observations censored and retained; no primary fit incomplete.
- [x] Six native GPU pairs (12 full fits plus assignment diagnostics): exact initial assignment labels in all pairs; exact full Gaussian centers, small documented GloVe differences.
- [x] Additional isolated official Faiss CPU comparison on the Runpod host: 24 fits, 12/12 exact paired centers. Earlier local CPU evidence preserved separately.
- [x] All exports downloaded/checksummed; every member streaming-hashed (721 main/65 CPU). 5.094 GiB raw traces retained in verified compressed archive.
- [x] PHASE4_REPORT.md, experiment log, reproduction notes and mapping updated with measured quality, censoring, profiling, memory, official controls and limitations.
- [x] Observed campaign account debit **$0.5517909861**, below $2. Pod/temporary volume deleted and 404-verified; inventories empty; current spend $0/hour. Billing buckets pending, so debit is not a final invoice.
- [x] Archive closed without multi-seed expansion: the **936-fit campaign was not executed**.
- [x] No Phase 5 implementation or further optimization; project finalized at the user’s request.

Measured assignment/dispatch overhead motivates Phase 5. At a common returned train/test threshold, automatic e-means took .471 s versus Lloyd 1.581 s on GloVe; current e-means remained slower on Gaussian256/1024. One seed/two repeats does not establish general speedups or scaling-law exponents. FP32/FP64 correctness was verified; this performance pilot is FP32. No Faiss-GPU, Triton, distributed or paper-scale performance reproduction is claimed.

Evidence: `results/phase4/runpod/a40-20261112-001/`; source SHA256 `df38023f14ce1594f5706ebc8e6b99e71b7906163bdc09d948d821d108466093`. See PHASE4_REPORT.md for exact commands and audited tables. This measured status supersedes the earlier Phase 4 preparation/approval-pending notes.


## Final archive — 12 November 2026

Project closed through Phase 4 at the user’s instruction. No Phase 5 optimization, further GPU experiments or cloud calls were performed. Final local suite: **166 passed, 33 NVIDIA-only skips**. All 12 validated src modules match the executed Phase 4 source ZIP byte for byte. Re-audited all 120 CPU fits, 60 exact backend pairs, 186 Phase 3 GPU fits, 312 Phase 4 quality fits, nine profiles and six native GPU pairs; retained 104 censored target observations. Reverified complete raw archive checksums and every member of both Phase 4 exports (721/65).

Public documentation, raw numeric arrays/records, historical independent source snapshots, notebook, configs and figures are retained under an explicit publication manifest. Supported secret-pattern checks found no credential values in public content/nested source ZIPs, local reachable Git blobs or Git config. Caches, the downloaded CLI binary and four byte-identical preview files were removed with a cleanup inventory; no unique measured result was removed. The local environment remains ignored so tests can be rerun. Final report: REPRODUCTION_REPORT.md. No repository push, visibility change or remote archive hosting is claimed. The code license remains an owner choice.
