# Phase 4 — measured A40 profiling and matched-MSE pilot

> Final archive notice, 12 November 2026: Phases 1–4 are closed. Historical preparation/access statements and Phase 5 proposals below are dated records, not current pending work. No further experiments or optimization are authorized or planned. See [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md) for the final synthesis.

Status: **completed on actual NVIDIA hardware, 12 November 2026**. The full CUDA gate passed **199 tests, zero skips/failures/errors**, before all benchmark execution. Nine profiles, **312 primary quality fits**, six paired native-GPU comparison records and 24 isolated official-Faiss CPU fits completed and passed independent local artifact audits. No kernels or validated numerical algorithms were modified. The **936-fit campaign was not run**.

The main result is workload dependent: at a predeclared common train-and-test threshold, fixed e-means remained slower than Lloyd on both Gaussian workloads; automatic e-means reached the GloVe threshold in 0.471 s versus Lloyd's 1.581 s. Profiling establishes assignment/dispatch as the first performance investigation target. These are one-seed development measurements, not a reproduction of the paper's headline speedup.

## Hardware, execution and cost

- Runpod Secure Cloud, CA-MTL-1, NVIDIA A40, SM 8.6, 84 SMs; GPU UUID `GPU-7c01fb29-ab82-970f-1796-7c6cc8bb284d`. Driver 595.91.07, visible memory 46,068 MiB.
- Python 3.11.10, PyTorch 2.5.1+cu124, CUDA 12.4, cuDNN 9.1, NumPy 1.26.4. Deterministic algorithms enabled; TF32 disabled; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; numerical CPU thread counts one.
- Immutable image `pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`, existing exact dependency overlay and successful `pip check`. Source fingerprint `df38023f14ce1594f5706ebc8e6b99e71b7906163bdc09d948d821d108466093` remained fixed.
- CUDA validation ran 05:51:29–05:52:18 UTC (11:21–11:22 IST), followed by profiles, quality sweep and native comparison. Main execution/export status was successful at 06:34:35 UTC; export compression and then the isolated CPU comparison followed. Exports were verified locally before teardown.
- Read-back rate was $0.59/hour GPU, approximately $0.596/hour including 30 GB container disk and 20 GB temporary network storage. A two-hour external Pod-deletion watchdog, process timeout and repeated balance observations bounded the session; early stop threshold was a $1.60 debit.
- **Observed total account debit $0.5517909861**, including storage in the account balance, against the $2 campaign cap. Initial balance $9.7524077472; post-cleanup balance $9.2006167611. Itemized billing buckets were still empty, so this is an observed balance debit rather than a final invoice.
- Pod `wy0bzxf36j7wtd` and volume `nzrqbbfov4` were deleted. Both fresh reads returned 404, both inventories were empty, and account spend was **$0/hour** at 06:47:26 UTC. The watchdog was released only after cleanup. No retained cloud volume remains.

## Data and scientific protocol

| Workload | Training / test rows | Dimensions | Clusters |
|---|---:|---:|---:|
| gaussian256 | 8,192 / 2,048 | 256 | 64 |
| gaussian1024 | 4,096 / 1,024 | 1,024 | 32 |
| glove100 | 16,384 / 2,048 | 100 | 64 |

All three FP32 seed-zero data/initialization hashes exactly match the Phase 3 baseline. GloVe uses the existing SHA256-pinned ANN-Benchmarks byte-range/array subset, original held-out split and no fitted preprocessing. The full parent corpus is not claimed SHA256-verified. Phase 3's original 186 fits and artifacts are preserved separately.

The pilot uses seed 0, two timing repetitions, one warmup, fresh models and identical data/initial centers for all methods. Primary wall latency is an **unprofiled synchronized resident fit plus returned train/test evaluation**; loading/H2D and serialization are excluded. CUDA-event durations, baseline/peak allocated/reserved memory, exact executed epochs and separate distance-accounting replay are saved. Recorded randomized job order uses a separate design RNG. Replays must reproduce exact centroids and labels.

Lloyd horizons are [1,2,3,4,8,25]; closing/slow e-means horizons [1,2,3,5]. Each horizon is a fresh run because T changes the cosine schedule, automatic recipe and finalization. Fixed B=256/alpha0=.2 reproduces Phase 3; automatic variants resolve the unchanged published Table 2 laws at each T. Both every-epoch and final-only observation policies are measured. All **132 policy pairs returned bit-identical centers and labels**; wall-time ratios ranged 1.11–2.49 across pairs. This confirms substantial measurement cost without changing the solver outputs in these tests.

Independent Lloyd calibration (max 25, original exact-fixed-point stopping) sets common train/test thresholds at factors **1.001** and **0.99**. Train, test and joint attainment remain separate. Time to target means the latency of the **first ascending declared horizon** whose returned finalized model meets the target. It is not online stopping, interpolation, the fastest successful horizon or cumulative grid-search cost. The full search expense is recorded separately. Test outcomes do not select a best recipe.

## Matched-quality measurements

Below: final-only observation, joint train AND test target = 1.001 × independent calibration. All listed rows reached in both repetitions. SD is variation between two timing repetitions of **one seed**, not between-seed uncertainty or a confidence interval. Training throughput counts actual training-vector visits divided by primary wall time; fewer passes can give better time to quality despite lower vectors/s.

| Workload | Method | First horizon | Seconds, mean ± repeat SD | Returned train / test MSE | Training vectors/s |
|---|---|---:|---:|---:|---:|
| gaussian256 | Lloyd | 4 | 0.4018 ± 0.0278 | 293.0391 / 296.1237 | 81,755 |
| gaussian256 | e fixed | 1 | 0.4396 ± 0.0044 | 289.2421 / 295.7051 | 18,637 |
| gaussian256 | e automatic | 1 | 0.7966 ± 0.0290 | 282.1682 / 288.2601 | 10,291 |
| gaussian256 | slow fixed | 1 | 0.5086 ± 0.0559 | 289.2558 / 295.6569 | 16,205 |
| gaussian256 | slow automatic | 1 | 0.8335 ± 0.0014 | 281.4573 / 287.8264 | 9,829 |
| gaussian1024 | Lloyd | 3 | 0.6558 ± 0.0790 | 1171.4398 / 1195.9818 | 18,874 |
| gaussian1024 | e fixed | 1 | 0.7923 ± 0.0312 | 1172.4976 / 1186.9393 | 5,174 |
| gaussian1024 | e automatic | 1 | 2.6679 ± 0.2245 | 1164.8273 / 1185.5949 | 1,541 |
| gaussian1024 | slow fixed | 2 | 1.3641 ± 0.0024 | 1170.2751 / 1184.9837 | 6,005 |
| gaussian1024 | slow automatic | 1 | 2.6070 ± 0.0878 | 1157.5561 / 1180.9347 | 1,572 |
| glove100 | Lloyd | 25 | 1.5810 ± 0.0381 | 25.7549 / 26.3121 | 259,151 |
| glove100 | e fixed | 2 | 0.8176 ± 0.0159 | 25.5798 / 26.1202 | 40,088 |
| glove100 | e automatic | 2 | 0.4709 ± 0.0720 | 25.5366 / 26.0859 | 70,405 |
| glove100 | slow fixed | 2 | 0.8829 ± 0.0192 | 25.6793 / 26.1546 | 37,122 |
| glove100 | slow automatic | 2 | 0.5389 ± 0.0325 | 25.6733 / 26.1320 | 60,914 |

The common train/test thresholds respectively are 293.2223/296.2069 (Gaussian256), 1172.6113/1197.1778 (Gaussian1024), and 25.7807/26.3384 (GloVe). Terminal MSE can overshoot the common threshold; these are equal-threshold comparisons, not identical terminal losses.

Using ratios of the two-repeat mean latencies, fixed e-means took **1.09× / 1.21× Lloyd's time** on Gaussian256/1024. Automatic e-means took **1.98× / 4.07×** respectively. On GloVe, fixed and automatic e-means reached the threshold **1.93× / 3.36× sooner** than Lloyd. Slow EMA is reported independently; it is not substituted for the closing estimator after observing test quality.

Of **648** predeclared target-observation records, **104 are censored**: four at factor 1.001 and 100 at factor .99, across all variants/policies/train/test/joint scopes. Every one of the 312 fits completed. A censored target is not a failed fit. At the harder .99 joint threshold, Lloyd is censored on all three workloads; closing e-means reaches within its declared grid, while fixed/automatic slow EMA both remain censored on GloVe through T=5. No runtime ratio is claimed where Lloyd or its paired contender misses. The complete reached/censored table is in `analysis/matched_targets.csv`, with target values and per-record search costs in the original quality summary.

Figures: [matched target latency](docs/figures/matched_mse.png), plus returned train/test MSE versus horizon, wall time and training distance pairs for [Gaussian256](docs/figures/gaussian256_quality.png), [Gaussian1024](docs/figures/gaussian1024_quality.png) and [GloVe](docs/figures/glove100_quality.png). Points are separately finalized budget fits, not prefixes of an online trajectory. Main-state epoch observations are additionally retained with their stage labels.

## GPU profiling and bottleneck evidence

Profiles use the Phase 3 baseline budgets (Lloyd max 25; e/slow five passes, B=256, alpha=.2, every-epoch metrics). Each has separate unprofiled timing and exact-output diagnostic replay. Assignment host ranges consumed **65–99%** of the instrumented observed fit across nine profiles. They are the dominant measured interval, especially on Gaussian1024. This is inclusive host range attribution, not a pure kernel percentage.

| Workload | Method | Kernels | GPU activity union / span (s) | Assignment kernel time (s) | CUDA sync API time (s) |
|---|---|---:|---:|---:|---:|
| gaussian1024 | emeans | 370,762 | 0.5808 / 8.2095 | 0.5550 | 0.0073 |
| gaussian1024 | lloyd | 142,938 | 0.2392 / 3.1760 | 0.2367 | 0.0017 |
| gaussian1024 | slow_ema | 378,301 | 0.5931 / 8.5812 | 0.5543 | 0.0094 |
| gaussian256 | emeans | 211,327 | 0.3493 / 4.5481 | 0.2997 | 0.0123 |
| gaussian256 | lloyd | 101,990 | 0.1919 / 3.0061 | 0.1880 | 0.0017 |
| gaussian256 | slow_ema | 226,586 | 0.3750 / 4.9555 | 0.2999 | 0.0171 |
| glove100 | emeans | 202,754 | 0.3364 / 5.3720 | 0.2376 | 0.0239 |
| glove100 | lloyd | 288,594 | 0.5325 / 7.0700 | 0.5077 | 0.0074 |
| glove100 | slow_ema | 233,573 | 0.3871 / 5.3395 | 0.2376 | 0.0330 |

All kernel events had runtime correlation IDs. The trace analysis maps launches to scoped host annotations and retains an explicit other/unattributed category. Activity union includes kernel/copy/memset activity; span is first-to-last GPU activity. The 6–8% activity fractions are **instrumented diagnostics**, not unprofiled GPU utilization. Profiler overhead expands observed fits (e.g. Gaussian1024 e-means 8.227 s versus unprofiled 5.217 s). Neither nested host ranges nor overlapping event spans should be summed into a fictitious additive breakdown.

Gaussian1024 closing e-means emitted **248,800 training-assignment kernels**, plus 93,012 evaluation-assignment kernels, within 370,762 total kernels. Its top three kernel families are coordinate arithmetic (117,760–117,952 launches per family). `cudaLaunchKernel` CPU self time was **1.864 s**; assignment/training annotation self time was 1.630 s in that instrumented run. Captured assignment kernel durations total about .555 s, versus 7.888 s across its assignment host ranges. Together with the independent native-backend experiment below, this establishes a launch/dispatch-heavy assignment path as the first engineering target. It does not separately prove the exact unprofiled cost of Python, the dispatcher, allocator or driver.

For closing e-means, instrumented host EMA/revival ranges were **.237/.042 s** (Gaussian256), **.128/.019 s** (Gaussian1024), and **.586/.112 s** (GloVe). Slow EMA approximately doubled EMA work (.486/.250/.990 s); shadow splitting also added work. CUDA-event spans and nested statistics/split/readout ranges are preserved in the profile records. The independent closing/slow finalizer replay took .81–.89 ms of host range time; integrated unprofiled finalizer timers were .48–.62 ms. The extra shadow updates throughout training, rather than the final readout alone, matter for slow-EMA cost.

Host CUDA synchronization API union time was **.0017–.0330 s** across profiles, despite 118–6,730 `cudaStreamSynchronize` calls. The explicit solver synchronization wrapper is narrower still. Synchronization call count alone does not explain the seconds-long gap; scalar validations/readbacks also entail dispatch and bookkeeping, which overlap assignment ranges. Full raw traces, CUDA-event spans and CPU self-time operator tables remain available for independent inspection.

A concrete work-count explanation agrees with the measurements: at the first Gaussian1024 common-threshold hit, fixed e-means needs one pass but 16 B=256 assignment tiles; Lloyd needs three passes × four 1024-row tiles = 12. Automatic T=1 uses B=64, raising that count to 64. Gaussian256 automatic T=1 uses B=128 rather than fixed 256. Thus fewer point-centroid comparisons can still cost more dispatch. These tile counts describe the unchanged code and measured recipe, not an optimization experiment.

See [host-range figure](docs/figures/profile_ranges.png), `analysis/profile_ranges.csv`, `analysis/trace_findings.json` and `profiles/profile_operators.json`.

## Scaling laws and sensitivity

T=3 closing-e-means outcomes below are predeclared final-only variants, mean seconds across two timing repetitions. B changes recompute the coupled alpha law; alpha-only changes hold automatic B fixed. No test-set winner is selected.

| Workload | Variant | B | alpha0 | Seconds | Test MSE |
|---|---|---:|---:|---:|---:|
| Gaussian256 | fixed | 256 | .2000 | 1.1252 | 287.0066 |
| Gaussian256 | automatic | 512 | .5128 | .6644 | 274.0710 |
| Gaussian256 | batch half | 256 | .2748 | 1.1778 | 275.8449 |
| Gaussian256 | batch double | 1024 | .9569 | .3495 | 284.5781 |
| Gaussian256 | alpha half | 512 | .2564 | .5992 | 290.6842 |
| Gaussian256 | alpha double | 512 | 1.0000 (cap) | .6087 | 271.0242 |
| Gaussian1024 | fixed | 256 | .2000 | 1.9256 | 1184.2964 |
| Gaussian1024 | automatic | 256 | .5163 | 2.0267 | 1121.6208 |
| Gaussian1024 | batch half | 128 | .2767 | 3.7001 | 1105.1525 |
| Gaussian1024 | batch double | 512 | .9635 | 1.0640 | 1097.7893 |
| Gaussian1024 | alpha half | 256 | .2582 | 1.9217 | 1149.7840 |
| Gaussian1024 | alpha double | 256 | 1.0000 (cap) | 1.9968 | 1094.4022 |
| GloVe | fixed | 256 | .2000 | 1.1219 | 26.0780 |
| GloVe | automatic | 1024 | .7005 | .3496 | 26.1171 |
| GloVe | batch half | 512 | .3754 | .5971 | 26.0785 |
| GloVe | batch double | 2048 | 1.0000 (cap) | .2903 | 26.1021 |
| GloVe | alpha half | 1024 | .3502 | .3426 | 26.0621 |
| GloVe | alpha double | 1024 | 1.0000 (cap) | .3443 | 26.1120 |

Automatic settings improve Gaussian T=3 quality relative to fixed .2, and improve GloVe latency substantially at similar quality. At T=1 they make Gaussian latency worse because small B increases update/dispatch count. Their advantage is not uniform over horizons or workload. Larger batches improve runtime here while quality changes non-monotonically. Full closing/slow train/test/timing rows are retained in `analysis/sensitivity.csv`; caps and per-horizon recipes are in every original JSON record. The 4K–16K training sizes are below the fitted large-data regime: this evaluates applying the published law, **not its fitted exponents or universal optimality**.

## GPU memory and throughput

Across all primary quality fits, absolute PyTorch peak allocated memory ranged **24.28–25.70 MiB** (Gaussian256), **48.42–51.20 MiB** (Gaussian1024), and **18.23–19.25 MiB** (GloVe). Maximum reserved memory was 44/66/66 MiB respectively; resident-input allocation baselines were at least 10.06/20.13/7.06 MiB. These allocator figures include resident inputs; they exclude driver/context memory and are not `nvidia-smi` process memory. The allocator cache is retained after warmup. This study is not VRAM-capacity limited. Raw trace construction consumes substantial **host RAM/disk**, distinct from solver GPU memory. Per-fit baselines/peaks, CUDA-event/wall time and actual work are preserved, with matched-target throughput in the table above and `analysis/memory.csv` for coverage-wide memory.

## Official implementations: controlled comparisons

Pinned authors commit: `2b648f0605f2ef3d14c4705350f8a0b214463878`. All fetched reference files are SHA256 checked and preserved in the export. The native CUDA comparison explicitly selects the authors' **native PyTorch backend**, suppressing default FP32→BF16 via the recorded pinned `_bind_input_dtype(..., cast=False)` adapter. No Triton/CuTe path is invoked.

The six GPU pairs use B=256, alpha0=.2, three passes, retained tails, mass 2**-16 and **revival off on both sides**, identical initialization/order, FP32 and common returned-MSE evaluation. These controls differ from the main pilot defaults. The full lifecycle also differs in validation/publishing/host bookkeeping. The assignment micro-comparison uses identical initial centers and row/center chunks, includes norm preparation, and uses synchronized wall timing. Authors use norm/dot distances; ours uses direct differences.

| Workload | Full fit ours / authors (s) | Assignment ours / authors (s) | Full-center max difference | Test MSE ours / authors |
|---|---:|---:|---:|---:|
| Gaussian256 | 1.18728 / .21464 | .073719 / .002808 | 0 | 302.39681 / 302.39681 |
| Gaussian1024 | 2.18429 / .22935 | .148749 / .001413 | 0 | 1220.13585 / 1220.13585 |
| GloVe | 1.04931 / .29418 | .070744 / .007803 | .00275086 | 27.584932 / 27.584995 |

These are means across two timing repeats of one seed. Initial assignment labels agree **exactly in all six pairs**; max squared-distance differences are .0006714/.0053711/.00003815 for Gaussian256/1024/GloVe. Full Gaussian centers match exactly; GloVe centers differ slightly, so no exact-full-quality speedup is asserted there. Assignment raw time ratios (~26×/~105×/~9×) establish backend/control-path headroom, not a causal pure-kernel ratio or the paper's algorithmic speedup. Cancellation/near-tie behavior, numerical guards and precision must be independently validated before changing our assignment backend.

The official **Faiss CPU** driver was additionally run after GPU measurement/export completion to avoid concurrent CPU interference: **24 fits**, two smoke workloads × two implementations × three seeds × two repeats. All **12 paired centers match exactly**. Single-thread Faiss 1.10.0 versus NumPy primitives had mean solver times .006508/.017185 s (Gaussian128 smoke) and .006453/.015337 s (GloVe smoke), averaging repeats within each seed. Original summaries retain between-seed SD and separate common FP64 evaluation latency. This is CPU evidence, distinct from the actual native-PyTorch GPU experiment. A Faiss-GPU/custom optimized-kernel benchmark and a full authors slow-EMA finalizer were **not executed**; the audited package exposes no complete slow routine. Earlier macOS Faiss CPU results remain separate, not pooled.

## Reproducibility and verified exports

All permanent evidence is under `results/phase4/runpod/a40-20261112-001/` (ignored by the code-publishing policy):

- `phase4-pilot-001.tar.gz` + SHA256: full CUDA gate, 9 profiles/raw Chrome traces, 312 JSON/NPZ primary fit records, warmup/calibration/diagnostic records, native pairs, plots, source ZIPs, config/environment/reference hashes and execution logs.
- `phase4-official-faiss-cpu-001.tar.gz` + SHA256: 24 CPU comparison fits and independent environment/source snapshots.
- `e-means-phase4-source-20261112.tar.gz` + SHA256: transferred source/data; SHA256 `53db3eaeb217252d0cff5b04ac7fe9c6031af7be3eb976736a8d7cb722619f04`.
- `extracted/`: verified metadata/arrays/figures. Raw traces total **5.094 GiB**, retained losslessly in the compressed main archive rather than duplicated on disk. Whole-archive checksums and **every member's** streaming SHA256 were verified: 721 main-export members and 65 CPU-export members. Member digests and extraction flags are saved alongside the extracted roots.
- `analysis/`: independent full coverage/hash audits, exact policy parity, original-input identity, matched/censored target CSV, sensitivity/memory/range tables, trace launch correlation analysis and derived figures. `verify_export.py`, `study_analysis.py`, `trace_analysis.py` and `presentation_analysis.py` reproduce downstream analysis. Archive-backed trace digests substitute only for absent extracted trace files during the unchanged summarizer audit.
- Session logs, redacted usage observations, read-back provisioning metadata, pinned CLI checksum, bootstrap log, budget/watchdog and `cleanup.json`. No API/private SSH keys were transferred into the source or Pod.

Executed inside the approved Pod:

```bash
export PATH=/opt/conda/bin:$PATH
cd /workspace/e-means
bash runpod/bootstrap.sh
timeout --signal=TERM --kill-after=60s 6000 bash runpod/phase4.sh /workspace/results/phase4-pilot-001
# After GPU study AND its export compression completed:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/compare_official_phase4.py --config configs/gpu_smoke.json   --reference /workspace/results/phase4-pilot-001/reference   --output /workspace/results/phase4-official-faiss-cpu-001
python runpod/export_results.py /workspace/results/phase4-official-faiss-cpu-001 --exit-code 0
```

Local reanalysis commands (no paid resources needed):

```bash
SESSION=results/phase4/runpod/a40-20261112-001
.venv/bin/python "$SESSION/verify_export.py" "$SESSION/phase4-pilot-001.tar.gz" --output "$SESSION/extracted"
.venv/bin/python "$SESSION/study_analysis.py" "$SESSION/extracted/phase4-pilot-001"   --members "$SESSION/extracted/phase4-pilot-001.tar.gz.members.json" --output "$SESSION/analysis"
.venv/bin/python "$SESSION/trace_analysis.py" "$SESSION/phase4-pilot-001.tar.gz" --output "$SESSION/analysis/trace_findings.json"
.venv/bin/python "$SESSION/presentation_analysis.py"
```

## Findings versus paper and Phase 5 recommendation

The pilot supports faster **statistical** convergence: fewer returned epochs reach the common threshold, and closing e-means reaches quality levels Lloyd did not reach on its tested grid. It supports a GloVe equal-threshold runtime improvement on this A40. It **does not support a universal runtime advantage** for the current independent backend: Gaussian matched-threshold times remain slower. It neither validates nor refutes the paper's advertised magnitude on H100, optimized low-precision kernels and much larger n/k regimes. Automatic laws and slow EMA require workload/horizon-specific interpretation.

Phase 5 should first propose one opt-in, launch-efficient assignment backend used by **both Lloyd and e-means**, preserving the validated direct-distance path as oracle. Validate FP32/FP64 ties, large offsets/cancellation, finite/overflow guards, near ties, trajectories, revival and finalizers before timing. Separately ablate repeated input validation and host readback without silently changing public correctness guarantees. Keep evaluator policy fixed and profile the new backend before any further kernel work. Then seek separate approval for a multi-seed confirmation campaign with predeclared targets, confidence bounds and consistent precision/control settings. Larger embedding/n/k regimes and comparable optimized authors baselines are needed before paper-speedup claims. No Triton, distributed clustering or premature kernel optimization has been introduced.

No requested pilot stage is incomplete. Remaining limits: one independent GPU seed, two timing repeats, small biased real-data prefixes, FP32-only performance study (FP32/FP64 correctness passed), finite-horizon threshold overshoot/censoring, profiler observer cost, no Faiss-GPU/optimized authors slow-EMA comparison, and no multi-seed 936-fit execution. The external controller is not a provider-enforced dollar cap; verified deletion closed this session's spend.

Sources: [paper v1](https://arxiv.org/html/2611.09501v1), [pinned authors package](https://github.com/facebookresearch/faiss/tree/2b648f0605f2ef3d14c4705350f8a0b214463878/contrib/e_means). Measurements above are our exported A40/CPU evidence, not paper-reported values.
