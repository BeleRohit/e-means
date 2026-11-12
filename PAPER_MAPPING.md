# Paper-to-code mapping

> Final archive notice, 12 November 2026: Phases 1–4 are closed. Historical preparation/access statements and Phase 5 proposals below are dated records, not current pending work. No further experiments or optimization are authorized or planned. See [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) for the final synthesis.

Primary specification: [arXiv:2611.09501v1](https://arxiv.org/pdf/2611.09501v1), §3, Algorithm 1, Table 2, Appendix H.1. All implementations are independent. Source identity is in [sources.json](docs/sources.json); ambiguity decisions are in [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md).

| Paper location | Definition | Implementation | Verification |
|---|---|---|---|
| §1 Eq. 1 | Mean squared Euclidean norm, nearest assignment | `assignment.assign_nearest`, `assignment.mse` | Broadcast distance oracle, ties, large offsets, per-vector MSE |
| §1 Lloyd description | Full assignment / means iteration | `estimators.LloydKMeans.fit` | Hand result, empty cluster, monotone fixed-center MSE |
| §3.1 / Algorithm 1 lines 1–2 | Random initial rows; N=1e-10, S=N*c | `_NumpyEstimator._inputs`, `EMAState.initialize` | Hand state, fixed initialization, reproducibility |
| §3.1 Eq. 3 | Per-cluster batch counts and sums | `assignment.batch_statistics` | Hand sums/counts; scalar full-loop oracle |
| §3.1 Eq. 2 | N=(1-alpha)N+alpha*n; S=(1-alpha)S+alpha*s | `EMAState.update` | Hand arithmetic, per-batch authors' parity at both initial masses |
| §3.1 Eq. 4 | c=S/N | `EMAState.centroids` | Ratio identity; alpha=1 empty-cluster handling |
| §3.1 Eq. 5 | alpha0*[epsilon+(1-epsilon)*(1+cos(pi*t/M))/2] | `schedules.cosine_alpha` | Literal values, monotonicity, one-step budget, authors' parity |
| Algorithm 1 lines 3–4 | Fresh shuffled batches per epoch | `EMeans.fit` | Independent scalar oracle, dropped/retained tails, deterministic refits |
| Algorithm 1 lines 9–10 | Revival every five global updates | `EMeans.fit`, `EMAState.revive` | Revival execution and full-loop authors' parity at fifth update |
| Appendix H.1 Eq. 8 | Weighted donors, opposite alternating multiplicative perturbations, halve mass | `EMAState.revive` | Exact split geometry, donor exclusion/uniqueness/cap, zero-vector limitation, aligned authors' split |
| §3.2 / Algorithm 1 closing | Last-epoch moving-assignment means | `state.last_epoch_mean`, `EMeans.fit` | Fixed-code SSE minimizer, empty fallback, scalar oracle, authors' full loops |
| §3.3 Eq. 6 / Table 2 | alpha0=min(.824*k^-.46*B^.90*n^-.45*T^-.11, 1) | `schedules.paper_alpha0` | Formula/override tests |
| §3.3 Eq. 7 / Table 2 | B=.035*k^.13*n^.88*T^.81, /8 above 2M, round/cap | `schedules.paper_batch_size` | Boundary and formula tests |
| §3.2 / Appendix I | Additional constant-speed sum/count EMA | `schedules.slow_alpha`, optional shadow in both e-means estimators | Scalar oracle, half-life identity, assignment isolation, shadow splitting, authors' update/split primitives |
| §3.4 / Appendices J, M | Approximate assignment, precision kernels, sharding | Deferred | No GPU optimization in Phase 1 |
| §4 | Large-scale embedding experiments | Future reproduction | Phase 1 synthetic experiment records are separate evidence |

## Phase 2 backend mapping

`torch_backend.py` independently implements Eq. 1 with tiled direct differences, Eq. 3 with `index_add_`/`bincount`, Eqs. 2–4 with explicit `TorchEMAState`, Eq. 8 with tensor splitting, and §3.2 with tensor finalization. The shared literal schedule and recipe functions avoid conflicting conventions. Lloyd and e-means preserve NumPy's API/defaults and seeded control streams. Tests compare FP32/FP64 per-update statistics, exact controlled donor IDs, moving codes, fixed-center histories, main/shadow readouts, and final labels. The original twelve-run artifacts provide a separate NumPy regression oracle.

Appendix I's speed multiplier is applied to the theoretical cosine floor `epsilon*alpha0`, not the last executed cosine step. The alternative half-life unit is explicitly batch updates. Both shadows initialize at the paper's tiny mass and split their own geometry/mass using the main donor IDs. These bookkeeping interpretations are documented choices, not hidden claims of fully specified authors' behavior.

The pinned authors' NumPy/PyTorch classes expose no slow-EMA finalizer. The optional shadow reference test validates constant recurrence/split arithmetic through those authors' primitives. It does not manufacture an end-to-end authors' slow-EMA result. Real-data CPU measurements use the UCI fixed splits and are recorded separately from the paper's embedding benchmarks.

## Phase 3 CUDA preparation and measurement mapping

No paper algorithm equations changed. `tests/test_cuda_validation.py` rehearses each assertion on CPU and separately guards actual NVIDIA cases for Eq. 1 assignment/ties/tiling, Eq. 3 statistics, Eqs. 2–5 EMA/cosine updates, Appendix H.1 revival geometry/masses, and §3.2 closing/slow finalizers. The full trajectory matrix compares both CPU references, including tails and observed revivals. CPU rehearsal passes do not count as CUDA verification.

`scripts/validate_cuda.py` plus `gpu_protocol.py` require actual device execution and complete-suite success before recording `cuda_verified`. `scripts/benchmark_cuda.py` instruments the existing tensor estimators through a benchmark-only subclass, preserving initialization/RNG/algorithm behavior. It captures main-state observations and the final readout separately, and measures rather than estimates device memory/timing when NVIDIA execution becomes available.

The `.001` target slack, independent Lloyd calibration, warmup/repeat counts, deterministic-algorithm policy, range-pinned real subsets, synthetic distributions and time-to-target censoring are **experiment design choices**, not new paper claims or algorithmic modifications. `configs/gpu_smoke.json` and `configs/gpu_research.json` predeclare them. Published Table 2 laws remain available; these initial GPU workloads explicitly fix B/alpha for controlled development measurements.

No GPU results are reproduced locally. The milestone's preparation-only manifests and closed hardware gate are kept distinct from the historical measured CPU experiments.

## Interpretation details

The cosine expression's denominator is **M**, not M-1. We use Eq. 5's zero-based indexing even though Algorithm 1 increments t before calculating decay. The last executed update is above the floor. This also matches the reference's schedule.

For default dropped tails, M follows §3.1 literally. For retained tails, it counts actual updates. This resolves a conflict between the paper's floor count and its all-points-per-epoch description without silently changing either policy.

Closing assignments were produced against different centroids throughout the final epoch. Means minimize SSE **for those fixed codes**. Fresh reassignment of the returned means can only improve that fixed-code loss, but the paper's statement does not by itself prove improvement relative to a separate fresh evaluation of the final EMA positions. Tests avoid that unsupported comparison.

## Phase 4 measurement and interpretation mapping

| Specification/evidence | Phase 4 implementation | Verification/scope |
|---|---|---|
| Eq. 1 nearest-center MSE | Unchanged estimators and common returned-output evaluation | No main-EMA substitution for closing/slow readouts |
| Eq. 5 denominator and T-dependent schedule | Fresh fit at each declared horizon | Prefix of a longer run is explicitly not a shorter-budget algorithm |
| Eqs. 6–7 / Table 2 | `quality_protocol.recipe_config` delegates to existing paper law functions | Each horizon resolves B/alpha; batch-factor sensitivity recomputes coupled step; cap recorded |
| §3.2 / Appendix I | Unchanged closing and documented constant shadow finalizers | Independent returns; no authors full slow routine exists in audited source |
| §4.1 scoring / fixed Lloyd reference | Independent per-input Lloyd calibration, threshold factors 1.001/.99 | Discrete grid and early-fixed-point convention differ from a forced 25-pass baseline; not paper-table reproduction |
| §4/Appendix J GPU performance | `profiling.Observation`, unprofiled `quality_experiments.timed_fit`, profiler/event traces | Actual CUDA activity and new complete gate required; no custom/low-precision kernel implemented |
| Backend versus optimizer contribution | Official Faiss CPU controlled driver and optional native-Torch adapter | FP32, retained tails, 2**-16 mass, revival disabled; exact init/order; norm/dot versus direct distance and representation override documented |
| Experimental distance work | Actual assignment counters by training/evaluation/readout scope | Point-centroid pairs, not claimed FLOPs; exact independent formulas tested |

The paper GPU setup uses H100 80GB and optimized CuTe/BF16/FP8/INT8 paths (paper §4.1/Appendix J). Our Phase 3 A40 direct FP32/FP64 implementation and small n/k regime do not implement that setup. Lower final MSE supports only a qualitative direction; unmatched timing does not validate or refute the stated speedup magnitude. Phase 4 GPU measurements are now independently audited; CPU rehearsals and deterministic formula outputs remain separate evidence. No undocumented source change/optimization is introduced in the core solvers.


### Measured Phase 4 evidence — 12 November 2026

The new full A40 gate passed **199 tests without skips** before nine profiles, 312 quality fits and six native paired records. Source/input/gate/array/trace audits passed. The three Phase 4 seed-zero FP32 data/initialization hashes match Phase 3. Eqs. 6–7 were applied at every horizon with predeclared sensitivity and caps; measured quality/latency tables are in PHASE4_REPORT.md. This tests recipe application on small data, not fitted exponents.

Common-threshold GloVe timings support a workload-specific runtime advantage; Gaussian results show fewer returned epochs without a runtime advantage in our backend. Assignment host ranges dominate actual profiles, with many small coordinate-kernel launches. The official native FP32 diagnostic shows backend/control-path headroom under explicit nondefault mass/revival/representation controls; it is not a paper-default optimized-kernel replication. The Faiss reference executed on CPU only; all 12 paired centers match exactly. No authors full slow-finalizer result is fabricated. Matched-grid censoring (104/648 observations), one independent GPU seed, two timing repeats, FP32-only performance and profiler observer cost are explicit limits. No 936-fit campaign or optimization was executed.
