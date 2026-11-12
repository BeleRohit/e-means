# Evidence archive and publication policy

`public/` contains measured evidence suitable for Git publication. Preparation-only plans are not mixed into its benchmark summaries. All figures in `docs/figures/` derive from these completed studies.

## Public evidence

| Directory | Contents |
|---|---|
| `public/phase1/`, `public/phase2/` | Compact original records, CPU configuration and summaries |
| `public/phase3/` | A40 research summary, metrics and configuration |
| `public/phase4/` | Matched/censored targets, sensitivity, memory, profiling/operator/launch tables, validation and official comparisons |
| `public/runs/phase1`, `phase2` | Original numeric JSON/NPZ, independent source snapshots and manifests |
| `public/runs/phase3-smoke`, `phase3-research` | 24/162 original GPU fit records/arrays, calibrations, source/configuration and validation XML |
| `public/runs/phase4-quality`, `phase4-profiles`, `phase4-native`, `phase4-official-cpu` | Original numeric records/arrays and provenance for 312 primary fits, nine profiles, six GPU pairs and 24 CPU fits |

Historical `source.zip` files contain the independent code as executed, not the authors’ fetched reference source. Reference fetching remains explicit and hash-pinned. Datasets and controller/account/SSH metadata are excluded. Source/config fingerprints refer to historical snapshots; final packaging metadata/offline scripts change the current fingerprint without changing numerical algorithms.

`PUBLICATION_MANIFEST.json` covers all public source, tests, notebook, documentation, configurations, evidence and figures, with SHA256 and byte sizes. Run `python3 scripts/verify_publication.py` to verify inventory, supported credential patterns, relative documentation links and cleared notebook outputs. This detects corruption relative to the manifest, not authenticity against a maliciously replaced manifest, and is not a universal secret detector.

Original study manifests also retain their original per-array/source/config hashes. Profiling manifests refer to raw trace files that are deliberately kept in the separate local archive. Thus a profile directory alone is not the full trace export; a fresh clone must not claim to have revalidated absent traces. The completed full-export audits and per-member trace digests are public.

## Preserved local raw evidence

The ignored `results/` tree retains the complete historical evidence and operational provenance. It is not uploaded by the publication bundle. In particular:

- `results/runpod/a40-20261111/`: original Phase 3 smoke/research exports, SHA256 sidecars, independent audits, logs and cleanup records.
- `results/phase4/runpod/a40-20261112-001/`: complete main and CPU exports, transfer source/data package, independent member digests, analysis scripts and operational logs.
- The Phase 4 main tar.gz contains **5.094 GiB of raw Chrome traces**, losslessly compressed. They are intentionally not duplicated as uncompressed files.
- `results/phase4/local/`: preparation plans, CPU rehearsals, the earlier macOS reference study and baseline audits, with distinct status labels.
- `results/finalization-20261112/`: final test XML, integrity/security/cleanup checks and the private evidence inventory.

No public hosting URL for the full raw archives is currently configured. Original filenames and hashes are recorded in `RAW_ARCHIVES.json`; private metadata and downloaded corpus bytes should be reviewed before any future raw-archive distribution. Code/data/reference licenses must be respected. Do not upload the whole `results/` directory.

## Offline bundles

`python3 scripts/archive_project.py --output archives/e-means-reproduction.tar.gz` creates a deterministic public bundle from the reviewed manifest and independently reads/hashes every archived member. Existing archives cannot be overwritten. `archives/` is ignored by Git. The SHA256 sidecar verifies the complete bundle; it does not replace per-member checks.

A separate local evidence bundle and checksum preserve the cleaned private `results/` tree, including complete raw GPU exports. It is not a GitHub release asset or a public package. Keep an independent backup outside this machine before treating any local archive as durable long-term storage. No remote backup/upload is claimed.
