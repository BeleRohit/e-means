# Reproduction notes and ambiguity audit

> Final archive notice, 12 November 2026: Phases 1–4 are closed. Historical preparation/access statements and Phase 5 proposals below are dated records, not current pending work. No further experiments or optimization are authorized or planned. See [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) for the final synthesis.

Specification: Bessa et al., [arXiv:2611.09501v1](https://arxiv.org/pdf/2611.09501v1). The full paper including Appendices A–M was acquired and parsed; numerical formulas were checked against the rendered paper because automatic equation extraction was incomplete. No neural-network architecture, gradient optimizer or preprocessing recipe applies to this clustering implementation.

Phases 1–2 implement the optimization contribution independently in NumPy and PyTorch, including Lloyd, both finalization strategies, numerical tests, and synthetic/real-data CPU experiments. This establishes tested CPU algorithmic behavior, not the paper's dataset-scale quality or GPU runtime claims.

## Paper versus current reference

Reference pinned at [`2b648f0605f2ef3d14c4705350f8a0b214463878`](https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means). [base.py](https://github.com/facebookresearch/faiss/blob/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means/base.py#L19) records several differences explicitly.

| Detail | Paper / this implementation | Pinned reference |
|---|---|---|
| Initial EMA mass | 1e-10 | 2^-16 |
| Split delta | BF16 machine epsilon, 2^-7 (Appendix H.1) | 1e-4 |
| Step coefficients | (.824, -.46, .90, -.45, -.11) | (.8244, -.4617, .9003, -.4498, -.1057), lower clamp 1e-4 |
| Batch law | Published Table 2, cap n; /8 above 2M | Refitted coefficients, different caps and dimension-based choice |
| Low dimension revival | tau=.1 at every dimension | Disabled by default at d<=2 |
| Epoch tails | Drop by default; explicit keep option | Keep |
| Precision | float64 default; float32 option | float32 |
| Random streams | Independent Generator streams | RandomState, epoch-specific seeds, Gumbel donor keys |

We implement the paper's values rather than silently inheriting later reference defaults. Overrides permit controlled comparisons. No wholesale source copying or import delegation is used in the library.

## Ambiguities and explicit choices

| Component / classification | Choice | Alternative / consequence |
|---|---|---|
| Cosine indexing: partially specified | Eq. 5, zero-based t/M | Algorithm box increments first; prose suggests exact final floor. M-1 interpolation would change the formula |
| Batch tails: partially specified | Default drop, M=T floor(n/B) | Keep via `drop_last=False`, M=T ceil(n/B). Default last means exclude shuffled leftover rows |
| Batch rounding: from official code | Closest power in log space, ties upward, then cap n | Arithmetic rounding may select another power |
| Large-n coefficient: partly rounded | Divide .035 formula by exactly 8 above 2M | Table's .0044 is rounded; do not treat it as extra precision |
| Donor candidate/cap: from official code | Only positive live donors, without replacement, at most one split per donor per check | All clusters as candidates can select dying centroids; repeating donors violates no-replacement |
| Donor sampling: unspecified RNG | Weighted Generator.choice, counts normalized in float64 | Reference Gumbel top-k gives same sampling family but different exact seed trajectories |
| Zero-mass readout: from official code | Retain last valid position, still store literal EMA sums/counts | Eq. 4 is undefined for 0/0 when alpha=1 empties a slot; adding epsilon would modify all ratios |
| Empty final centroids: from official code | Preserve final EMA location | Resample, zero fill, or additional Lloyd pass would alter algorithm |
| Closing totals on revival: from official code | Keep accumulated last-pass sums/counts unchanged | Reset/splitting would no longer average the recorded moving assignments |
| Ties: unspecified | First centroid index | Any nearest minimizer satisfies Eq. 1, but labels can differ |
| Numeric strategy: unspecified | Direct tiled differences, FP64 default, FP64 loss reduction | Norm/dot identity is faster but can cancel at large offsets |
| Seeds: unspecified | SeedSequence streams for init, ordering, revival | Authors use different streams; identical integer seed does not mean identical ordering |
| Lloyd empties/stopping: unspecified | Preserve empty center; absolute maximum displacement <= tol | Donor split or relative loss tolerance changes the baseline trajectory |
| Evaluation frequency: engineering choice | Epoch zero and each epoch against fixed centers | Extra evaluation cost is timed separately; online loss is separately named |
| Tiny CPU experiment: unspecified | Seeded Gaussian mixtures, explicit B/alpha | These data are outside the fitted scaling-law regime; no claim of law optimality |

Reference details: [donor selection/splitting](https://github.com/facebookresearch/faiss/blob/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means/e_means.py#L261), [EMA/closing accumulation](https://github.com/facebookresearch/faiss/blob/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means/e_means.py#L456), [finalization](https://github.com/facebookresearch/faiss/blob/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means/e_means.py#L533).

## Numerical and interpretation limits

Multiplicative Eq. 8 cannot separate an all-zero donor. The modern reference adds a nextafter fallback for coincident children; this implementation follows the paper's equation and tests the zero-vector limitation. Revival is not a guarantee of k nonempty clusters, especially for duplicate data or more dead slots than live donors.

The direct sum/count representation is retained for auditability. Extremely small decayed masses can underflow; zero-mass positions retain their previous readout. Float32 large-count/sum accuracy has its usual limits (integer counts beyond 2^24 are not all exactly representable). Extreme input magnitudes can overflow even in float64; distance overflow raises rather than silently selecting invalid labels. Mixed precision, compensated accumulation and large out-of-core data are not Phase 1 features.

Last-epoch finalization minimizes SSE for **moving-pass codes**, not necessarily fresh nearest assignments to pre-finalization EMA positions. The last history row measures EMA centers; the estimator's final `mse_` measures returned centers. Lloyd MSE is monotone under our empty-preserving policy. Stochastic e-means MSE need not be monotone.

## Verification and remaining reproduction work

Tests cover Eqs. 1–8, hand arithmetic, independent scalar training, two precisions, dropped/retained tails, deterministic refits, degenerate inputs, and controlled comparisons against the pinned authors' implementation. Test tolerances allow FP32 roundoff between explicit sums/counts and the authors' algebraically equivalent centroid/mass update. Donor selection distributions share weighted no-replacement semantics; exact multi-donor RNG trajectories are not claimed identical.

Full reproduction still requires exact embedding datasets/splits, published scale/configuration sweeps, hardware-matched timing, and actual CUDA verification. No paper table number has been reproduced. See EXPERIMENT_LOG.md for measured local evidence and IMPLEMENTATION_STATUS.md for the phase gate.

## Phase 2 slow-EMA audit

Primary specification: §3.2 and Appendix I of [the paper](https://arxiv.org/pdf/2611.09501v1). Appendix I describes a constant-speed accumulator of assigned points, with speed relative to the theoretical cosine-floor step. The main text mentions half-life without fixing its units. We inspected pinned `base.py`, `e_means.py`, `e_means_torch.py` and `benchs/e_means/paper_bench.py`. None exposes a slow-EMA finalizer; no authors' end-to-end implementation was available in these sources.

| Detail | Status | Implemented convention |
|---|---|---|
| Shadow statistics | Specified §3.2 / Appendix I | EMA of **batch sums/counts**, not an EMA of centroid positions |
| Assignment source | Specified §3.2 | Main EMA alone determines batch assignments |
| Speed | Specified Appendix I | Constant `factor * epsilon * alpha0`, independent of cosine progress |
| Default factor | Explicit research choice | 1 if omitted; not asserted optimal. Experiments fix .25 a priori, with no test-based selection |
| Half-life units | Unspecified | Optional `slow_half_life` is logical batch updates; `gamma^h=1/2`. Alternative epoch/sample units would imply a different step |
| Initial state | Partially specified | Same prescribed centers and tiny mass as main EMA. Alternative zero-mass initialization changes early weighting |
| Revival identities | Partially specified | Follow main-state donor/target IDs; do not make independent donor draws |
| Shadow split geometry | Partially specified | Perturb its own donor position and halve its own mass using Eq. 8. Alternative copying main geometry changes the accumulated estimator |
| Zero-mass/empty readout | From official main-state behavior | Same retained-position fallback as Phase 1 |
| Tail batch speed | Explicit convention | One update per logical batch, including a retained tail; half-life is not normalized by tail length |
| Speed outside (0,1] | Engineering validation | Reject, without undocumented clipping |

The shadow runs throughout every epoch; it is not just a closing-pass accumulator. Main histories are unchanged by selecting slow finalization. A fitted slow run exposes both main and shadow centers. There is no automatic tuning on validation or test data.

Validation has three levels: independent scalar count/sum recurrence and half-life identities; NumPy/PyTorch per-step and full-trajectory parity including shadow splits; and a pinned authors' arithmetic comparison using constant EMA updates and explicit split IDs. The last level establishes primitive agreement, not agreement with an unavailable complete slow-EMA reference. Paper-level empirical generalization claims remain unverified.

## Phase 2 tensor and experiment conventions

PyTorch is optional and lazy. Tensor arithmetic uses the config's FP32/FP64 dtype; output arrays reside on the selected CPU/CUDA device. NumPy controls only seeded row/permutation/donor choices, using the Phase 1 streams. Revival copies k masses to the host for identical threshold/weight arithmetic. Ordinary NumPy configurations and trajectories remain unchanged, verified against historical centers/labels.

FP64 CPU trajectory comparisons use absolute and relative tolerance 2e-12; FP32 uses 2e-5 to allow repeated accumulation roundoff. Exact IDs/codes are checked on the controlled cases, whose assignment margins tolerate those errors. These finite examples do not guarantee identical decisions on every near-tie or every platform. CPU tests do not prove CUDA parity. CUDA `index_add_` may have device-dependent accumulation order; guarded CUDA tests must run on actual hardware before claims. No unsafe global deterministic/runtime settings are applied by the library.

On this macOS environment, importing bundled Faiss and PyTorch OpenMP runtimes into one process aborted execution. Optional reference cases now run unchanged in isolated pytest subprocesses, avoiding `KMP_DUPLICATE_LIB_OK`. This is test isolation; the package never imports Faiss. CPU experiments similarly use fresh workers.

UCI optical/pen digits use original writer-separated .tra/.tes files, fixed published feature-range scaling and no class-label input. They are real-data development evidence, not the paper's large embedding workloads. Five seeds and two repetitions are configured; repetitions are averaged per seed before sample-standard-deviation reporting. Final metrics use a common FP64 evaluator; epoch curves represent pre-finalization main centers. Timing/memory meanings and explicit budgets are saved with each run. No confidence interval, production throughput guarantee, or CUDA performance claim is inferred.

## Phase 3 preparation: NVIDIA gate, tolerances and measurement semantics

The original NumPy/PyTorch numeric algorithms and default configs are unchanged. Additional modules are workload/measurement/validation tooling. All new assertions run once on CPU for local test-design validation and have separate guarded NVIDIA cases. Primitive checks allow atol/rtol 3e-6 (FP32) or 2e-12 (FP64); multi-update CUDA trajectory budgets allow 1e-4 or 1e-10, respectively. These account for accumulated reduction-order roundoff in the controlled small fixtures. Exact codes, donor IDs and revival counts still must match. There is no automatic tolerance widening, no claim of universal stability near assignment boundaries, and no unexecuted CUDA pass implied by a CPU rehearsal.

The gate sets deterministic algorithms explicitly, disables TF32 and probes an actual NVIDIA CUDA tensor operation. Unsupported deterministic operations fail; they do not fall back silently. The complete suite must have zero failures/errors/skips and execute the new CUDA matrix. A checksummed XML/report is bound to source/config/script and hardware/runtime identity. The Runpod environment is a digest-pinned Linux image plus an exact package overlay, distinct from the locally tested macOS environment. That environment still needs actual bootstrap and validation.

Timing uses synchronized wall-clock intervals and CUDA events. The event interval includes host submission gaps/synchronizations and is not represented as pure GPU kernel time. Inputs are device-resident before calibration/warmup/contenders; H2D time is recorded separately. Timed resident fits include initialization, host RNG/revival, epoch metrics, finalization and final test evaluation. Solver-loop time and finalization are separately retained. Warmup/calibration and artifact serialization are excluded from contender time.

Allocator peaks are reset per measured fit after synchronization and resident-input setup. Absolute allocated/reserved peaks, baselines and incremental allocated peaks are separate. The allocator cache is retained after identical-data warmups; these metrics are not total process/driver GPU memory. Throughput numerator counts only training vectors actually visited, with early stopping/tail semantics respected.

A separate fixed-budget Lloyd calibration per workload/seed/dtype establishes train/test targets with predeclared .001 relative slack. All contenders use the same target and initial centers. Observed main-state hits and final-readout hits carry explicit stages; separate returned-readout target statistics prevent an early main-state hit from being misreported as an early slow/closing solution. No retrospective finalizer, interpolation, extrapolation or silent omission of misses is used. Misses are right-censored. Reached-only latency averages are labeled conditional and include reached-run/seed counts.

The real GloVe-100 subset is pinned by SHA256 for its selected byte ranges and complete arrays. HDF5 layout/shape was inspected locally; only 16,384 original train and 2,048 original test vectors are retained with no fitted preprocessing/renormalization. The full parent corpus was not SHA256-verified. Source ranges, ETag/length, SHA256 values and credits are retained in `data/glove100_source.json`. Squared Euclidean clustering MSE is not ANN-Benchmarks' angular retrieval metric. Fixed prefixes can have selection bias and are development workloads, not a paper-scale reproduction.

Locally, the NVIDIA execution gate remains closed. Prepared input hashes/source bundles and CPU protocol tests are preparation evidence only. No GPU performance, correctness or expected speedup is asserted until actual NVIDIA execution and review succeed.

### Execution access and budget audit (11 November 2026)

The requested Runpod execution was checked against checkout `c1b104b` and stopped at authentication/access: plugin enabled/configured, but no exposed MCP tools, no `runpodctl`, and no `RUNPOD_API_KEY` environment variable. No live pricing, balance, Pod state or GPU availability was observed. No NVIDIA numerical validation or GPU measurement was attempted, and no paid resource was created. The requested RTX 4090 and below-$10 total budget are requirements, not observations or provisioning approval.

The current orchestration script does not enforce total cloud spending or terminate the Pod; its EXIT handler exports files. A future authorized execution needs a live-priced time/storage allowance, external deadline/cleanup, persistent artifact recovery and verified termination before it can satisfy the budget/teardown requirements. No dollar estimate or hardware timing is inferred from the CPU preparation runs.

### Access recheck (11 November 2026)

Runpod CLI 3.1.0-5ab53b9 authenticated successfully after allowing network access; an initial sandbox DNS error was a network restriction, not evidence of invalid credentials. Account read: $10 balance, $0/hour current spend. Pod list --all: empty. Live catalog: RTX 4090 unavailable (listed Secure $0.89/hour, Community $0.34/hour); A40 Secure available at $0.59/hour with low stock in CA-MTL-1/EU-SE-1. GPU catalog rates exclude any separately billed storage. No paid infrastructure was provisioned because the user requires explicit approval. Existing run.sh still lacks spending enforcement and cloud termination. No actual CUDA correctness or GPU performance evidence was produced.

### Actual A40 reproduction and teardown — 11 November 2026

CUDA suite: 178 passed, zero skips/failures/errors, executed twice before successful smoke and research studies. NVIDIA A40 SM86, driver 595.91.07; Python 3.11.10, Torch 2.5.1+cu124, NumPy 1.26.4, runtime CUDA12.4. Both studies bind source SHA256 `42baee29f1e331c4f2ea9acc69ef44cae97479ccfdd80a5d28c146f870589ec5`. The pre-fix transfer archive is not the post-fix source snapshot: use each benchmark's source.zip and config.json for the exact executed source, including the Ninja overlay fix.

On the pinned container, use `export PATH=/opt/conda/bin:$PATH` in SSH shells. Extract Mac-built source archives with `tar --no-same-owner -xzf /workspace/e-means-source.tar.gz -C /workspace`. Overlay includes `ninja==1.11.1.3` because the immutable base's 1.11.1.1 wheel failed pip 25 check. Do not ignore the failing pip check or validation; the initial failed attempt is preserved separately.

Executed: `bash runpod/bootstrap.sh`, then `bash runpod/run.sh /workspace/results/a40-smoke-002 configs/gpu_smoke.json`, reviewed successful smoke, then `bash runpod/run.sh /workspace/results/a40-research-001 configs/gpu_research.json`. Smoke/research commands were wrapped with timeout (90/80 minutes), plus a detached local two-hour Pod-deletion watchdog under caffeinate. The run script itself still does not enforce dollars or terminate cloud resources. The local controller's availability/network dependence is an unresolved limit for unattended operation.

Download each tar.gz and its .sha256; verify SHA256 before extraction. The local independent `scripts.summarize_cuda.summarize` audit checked all 24/162 records, shared data/init identities, calibrations, array hashes and source/config/JUnit/hardware identity. `results/runpod/a40-20261111` retains both original complete archives, extracted data, REPORT.md, metrics.csv, source/config snapshots, curves, setup/failure logs, account-after.json, local-audit.json, cleanup.json and controller scripts. No unverified preparation records are summarized as results. Research tables average repeat means within seed, then sample SD across three seeds; these are not confidence intervals.

After successful local backup, `runpodctl pod delete 0nuwjg528hyuf1` and `runpodctl network-volume delete 48m37issf3` succeeded. Fresh get calls returned not_found, both lists were empty, and the account later reported $0/hour. Observed initial-to-final balance decrease: $0.2475922528; remaining balance $9.7524077472. Billing bucket arrays had not populated, so the amount is an account observation rather than a final itemized bill. The temporary network volume was deleted after verification, ending retained-storage charges.

### Phase 4 local protocol and approval gate — 12 November 2026

See PHASE4_REPORT.md for exact commands and evidence paths. Preserve `results/runpod/a40-20261111` unchanged; `scripts/audit_phase3.py` reads/audits it retrospectively without claiming a new device run. Local Phase 4 evidence is under `results/phase4/local`: original-evidence timing audit/plot, zero-measurement plans, CPU-only benchmark/profile rehearsals, and pinned authors CPU primary study `official-cpu-003`. Earlier exploratory CPU runs remain archived; do not pool them into the primary timing study.

Required optional reference fetch for the new complete suite: `python scripts/fetch_reference.py "$EMEANS_REFERENCE_DIR" --native-torch`. Additional pinned files: e_means_torch.py SHA256 cddc5ed1f1121c96406055fe6f50e93c049d0e09be27dc07dbe03e5c8e7ef4d3, torch_backend.py eb7dc8a41fc098604e86d0dcaa28106df5eed79670d278fe5bfbd573a605355c, same commit 2b648f0605f2ef3d14c4705350f8a0b214463878. Native files have lazy optional kernel imports; the adapter explicitly selects native and binds FP32 without cast. Do not silently benchmark authors default BF16 versus independent FP32, change donor RNG, or invent an authors slow-EMA mode. Main native comparison sets revival off on both sides and aligns mass/tails/chunks/init/order; those are control settings, not changes to paper defaults.

Primary GPU latencies exclude profiler/hooks/counters. Each diagnostic replay must reproduce exact arrays; its range timings overlap and event spans include host gaps. Main-state diagnostic points are not returned-finalizer matched-quality outcomes. Every horizon retrains because schedule/recipe/finalizer depend on T. Returned grid hits are first ascending horizon, not fastest repeat or continuous online stopping; grid-search expense and censored failures remain explicit. Train/test/joint targets are shared independent-calibration factors; test outcomes never select a best recipe. Equal target attainment need not mean exactly identical terminal MSE, and finite-grid overshoot is a limit.

For batch-factor sensitivity, B changes with the automatic alpha law recomputed; for alpha factors B stays at the automatic batch. All variants, caps, policies and seeds are retained, not cherry-picked. Pilot one seed/two repetitions supports protocol/rough runtime checks only. Confirmatory three-seed config is prepared but not executed/approved. Current tiny datasets lie below the fitted reference regime; no general exponent/scaling-law or paper-speedup validation is claimed from them.

`runpod/phase4.sh` uses the pinned environment, full new NVIDIA gate, nine baseline-budget profiles, quality sweep and native comparison, and the existing persistent exporter. Pair it with remote process timeout and local `runpod/watchdog.py SESSION_DIR` under caffeinate. The session budget.json must have explicit user approval and a finite deadline; save only the new session's own Pod create response as pod.json. No create action is performed by the guard. At deadline it reads/deletes/read-verifies that ID only. Normal cleanup verifies local archive checksums/audits before deleting Pod and network volume, then writes cleanup-complete. Controller/API availability is required; this is not a hard backend dollar cap. No API/private SSH keys should be put into the Pod or source archive.

Read-only account/catalog check: $9.7524077472 remaining, zero current spend, no Pods/volumes; A40 Secure $0.59/hour and low stock in CA-MTL-1/EU-SE-1. Proposed initial session ≤2 hours, approximately $1.20 including 30 GB container and 20 GB standard network storage, requires explicit new paid-resource approval. CLI update check advertised 3.1.1; Homebrew updater reported latest cask already installed and actual CLI remained 3.1.0. Recheck/pin current tooling before creation. No Phase 4 paid resource was provisioned. Historical Phase 3 approval is not treated as authorization for this new campaign.

Final Phase 4 local validation: **166 passed, 33 NVIDIA skips** (199 collected). Source SHA256 `df38023f14ce1594f5706ebc8e6b99e71b7906163bdc09d948d821d108466093`. Both final prepare-only plans match this source, with empty measurement lists and gpu_verified=false. CPU protocol/profile summaries and the primary 24-fit official CPU study passed independent local integrity audits. No new GPU execution or paid resource was started.


### Phase 4 actual execution and reproduction — 12 November 2026

This supersedes the Phase 4 approval-pending status above. Explicit approval authorized the A40 pilot under a new $2 cap. Full current suite: **199 passed, zero skips**, on actual A40 before benchmarking. Source hash `df38023f14ce1594f5706ebc8e6b99e71b7906163bdc09d948d821d108466093`; hardware/image/environment identity and exact executed commands are in PHASE4_REPORT.md. Algorithms and tolerances remain unchanged.

Session `results/phase4/runpod/a40-20261112-001/` contains verified main/CPU archives, source/data archive, setup/execution logs, budget/controller records, source ZIPs/configurations, environment/GPU/reference metadata, validation JUnit and per-run JSON/NPZ arrays. The full CUDA gate ran at 05:51 UTC on 12 November; the main pipeline exited 0. Nine profiles, 312 primary quality fits, six native paired records and 24 isolated Faiss CPU fits completed. The 936-fit configuration was not run. Seed-zero FP32 inputs/init exactly match Phase 3 hashes.

Verify downloaded archives and every member with session `verify_export.py`. Raw Chrome traces occupy 5.094 GiB before compression and remain losslessly archived; extraction omits only `*_trace.json` to save controller disk. Every member's SHA256/size/extraction flag is recorded in `extracted/*.members.json`. `study_analysis.py` calls existing summarizers, substituting preverified member digests only for raw trace files retained in the archive. Arrays/configs/source/gates/coverage are checked from disk. `trace_analysis.py` streams actual traceEvents from tar.gz and reports kernel activity union/span, launch correlation attribution, synchronization API intervals and CPU annotations. Instrumented activity fraction is not unprofiled utilization; nested host/event ranges overlap. PHASE4_REPORT.md provides local commands to reproduce tables/plots without new cloud spending.

All 132 observation-policy pairs have bit-identical centers/labels. All 312 fits completed; 104/648 target observations are censored and retained. Matched times are first ascending fresh-horizon returned solutions, not continuous first passage or minimum latency over horizons. Two timing repetitions of one GPU seed provide no between-seed uncertainty. Per-record recipes preserve caps and T dependence. FP32/FP64 correctness passed; the performance pilot uses FP32. Dataset size/prefix selection and finite-grid overshoot limit generalization.

Official native comparison is actual GPU execution; official Faiss comparison is actual CPU execution on the Runpod host after GPU measurements and main archive compression. Faiss 1.10.0, one thread, 24 fits/12 exact paired centers. Keep this separate from the earlier macOS Faiss 1.16.0 study. Native GPU controls suppress the authors' BF16 cast and disable revival, align mass/tails/order/chunks, and use their native backend only. All six initial assignment labels match; Gaussian full centers match; GloVe full centers differ by up to .00275086. No Faiss-GPU, optimized CuTe/Triton or complete authors slow-EMA comparison is claimed.

Exports passed independent local audit before Pod/volume deletion. Cleanup IDs: Pod `wy0bzxf36j7wtd`, network volume `nzrqbbfov4`; both fresh reads returned 404, inventories empty, current spend zero. Observed debit $0.5517909861 including storage; balance $9.2006167611. Itemized billing buckets remained empty, so this is not a final invoice. The cleanup marker released the external watchdog. No ongoing cloud resource or 936-fit execution. See session `cleanup.json`/`usage.jsonl` and PHASE4_REPORT.md.
