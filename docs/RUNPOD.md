# Historical Runpod reproduction procedure

**Archived:** Phase 3/4 A40 execution is complete; the latest actual suite passed 199 tests without skips. Pods and temporary volumes were deleted after verified export. No GPU experiments or paid resources are part of finalization. Commands below are retained for reproducibility, not permission to launch another campaign.

## Recorded environment

Use the immutable image in [runpod/image.json](../runpod/image.json), with exact overlay [runpod/requirements.lock](../runpod/requirements.lock). The observed environment was Linux x86_64, Python 3.11.10, Torch 2.5.1+cu124, NumPy 1.26.4, driver 595.91.07 and NVIDIA A40. Hardware/runtime metadata and source certificates are retained in the public run manifests. A later source/configuration/hardware change requires a fresh complete CUDA gate.

## Prepare a transfer package

```bash
.venv/bin/python scripts/fetch_gpu_embeddings.py
.venv/bin/python runpod/package.py /tmp/e-means-gpu-source.tar.gz
```

This package includes the pinned local subset if available. It excludes credentials and unrelated workspace data, but differs from the public publication bundle because it includes experiment data. Keep the SHA256 sidecar. No Pod is created or uploaded by these commands.

## Historical execution commands

Only reuse on separately authorized compatible NVIDIA infrastructure. Obtain credentials through provider tooling outside the repository; never embed keys in source, environment exports, notebooks or datasets. Verify live availability/rates, sufficient balance, storage cost and an external cleanup deadline before any paid resource is created.

```bash
export PATH=/opt/conda/bin:$PATH
cd /workspace/e-means
bash runpod/bootstrap.sh
# Each wrapper requires the current complete zero-skip CUDA suite.
bash runpod/run.sh /workspace/results/new-smoke configs/gpu_smoke.json
bash runpod/run.sh /workspace/results/new-research configs/gpu_research.json
# Separate Phase 4 pilot; the 936-fit configuration was never executed.
timeout --signal=TERM --kill-after=60s 6000 bash runpod/phase4.sh /workspace/results/new-phase4-pilot
```

Inspect smoke before larger workloads. A failing or skipped gate stops benchmarking. Fresh output directories are mandatory. `bootstrap.sh` validates the pinned software and runs pip check; Ninja 1.11.1.3 resolves the base image’s dependency-metadata failure. The source ZIP/config/source hash, JUnit, hardware/environment and measured artifacts bind each run to its gate.

## Budget, export and teardown

The wrappers export success/failure artifacts through the existing exporter; they do not create/delete resources or enforce a dollar cap. Historical sessions used explicit process timeouts, a separate two-hour controller watchdog under caffeinate and repeated balance observations. A controller watchdog depends on network/host availability and is not a provider-enforced budget guarantee.

Download archives and SHA256 sidecars; verify whole-file digests, study coverage, configs/source/gate/array hashes, shared input/init identities and every raw-trace member digest before deleting storage. Then delete only the authorized session’s Pod and temporary volume, verify both reads return not found, and check inventories/current spend. Retained network storage continues charging after a Pod is stopped.

The completed Phase 4 session’s observed balance debit was approximately $0.552 including storage; billing buckets had not populated, so that is not a final invoice. Both resources were deleted and spend read back as zero. No raw private account/controller metadata is included in the public package.

See [REPRODUCTION_REPORT.md](../REPRODUCTION_REPORT.md), [PHASE4_REPORT.md](../PHASE4_REPORT.md) and [artifact policy](../artifacts/README.md) for measured results, frozen evidence and exact scope. No automated cloud access occurs during offline publication checks.
