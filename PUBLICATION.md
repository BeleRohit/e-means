# Publication package and project descriptions

The repository is finalized through Phase 4. No Phase 5 optimization, further GPU execution or larger 936-fit campaign is part of this archive. Preparing this package does not change GitHub visibility, push a commit or create a release.

## Suggested GitHub description

Independent NumPy/PyTorch e-means reproduction with numerical parity tests, audited A40 benchmarks, matched-MSE comparisons and GPU profiling.

Suggested topics: `clustering`, `k-means`, `pytorch`, `numpy`, `research-reproduction`, `cuda`, `reproducibility`.

## Resume description

Built an independent NumPy/PyTorch reproduction of e-means with 199 tests passing on NVIDIA A40; designed reproducible CPU/GPU experiments, matched-MSE comparisons and profiling that identified assignment/dispatch bottlenecks.

Optional second bullet:

Executed and audited a 312-fit GPU quality pilot; measured a 3.36× GloVe time-to-threshold improvement while documenting slower Gaussian results, censored targets and one-seed uncertainty.

## Technical blog description

**Title:** Reproducing e-means: when fewer epochs do—and do not—mean faster clustering

I implemented Lloyd’s K-Means and e-means independently in NumPy and PyTorch, then checked EMA updates, revival, schedules and finalization against scalar oracles and pinned authors’ code. After CPU and actual A40 CUDA validation, I compared returned train/test quality under shared initialization, separate evaluation policies and synchronized timing. A 312-fit pilot showed automatic e-means reaching a common GloVe threshold 3.36× sooner than Lloyd, while remaining slower on two Gaussian workloads. Profiling exposed a launch-heavy assignment path. This is a transparent reproduction and engineering study; it does not reproduce the paper’s reported 10–45× speedups or introduce a new algorithm.

Link to REPRODUCTION_REPORT.md, the original paper/code, the exact pilot configuration and measured figures. Retain the one-seed/two-repeat qualification whenever using the 3.36× figure. Do not relabel raw assignment-backend ratios as optimizer speedups.

## Release checklist

- [x] Source, tests, notebook, configs, reports and measured numeric artifacts included by the publication inventory.
- [x] Downloaded datasets, credentials, private controller/account files, virtual environments, caches and large raw exports excluded.
- [x] Historical exact source/config/environment hashes retained; finalization changes no validated algorithms.
- [x] Final local suite: 166 passed, 33 NVIDIA-only skips. Historical actual A40 suite: 199 passed, zero skips.
- [x] Raw export SHA256/member checks and current publication content/history scan recorded locally.
- [x] README/report distinguish paper claims, measured results and unexecuted work.
- [x] Cloud resources previously deleted; no cloud access during finalization.
- [ ] Owner selects a code license before calling the repository open source (see NOTICE.md).
- [ ] Optional later action by owner: choose a durable public host for reviewed raw archives. No public raw-archive URL is currently available.
- [ ] Explicit publication action: review the public inventory, commit and push, then choose repository visibility/release settings. No automatic push or visibility change was requested here.

Re-run the offline verifier after any content change; refresh the manifest only after reviewing the intended publication files. Never force-add ignored raw results, `.env` files, keys or account exports.
