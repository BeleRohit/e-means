# CPU validation datasets

These are public development benchmarks separate from the paper's embedding datasets.

| Dataset | Source | Original train/test rows | Features | Scaling |
|---|---|---:|---:|---|
| Optical digits | [UCI optical recognition](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), DOI 10.24432/C50P49 | 3823 / 1797 | 64 | divide by 16 |
| Pen digits | [UCI pen recognition](https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits), DOI 10.24432/C5MG6K | 7494 / 3498 | 16 | divide by 100 |

The UCI sources describe fixed feature ranges and different writers in the provided training/test groups. No fitting on test statistics is performed. The final class-label column is discarded. `scripts/fetch_cpu_data.py` verifies pinned archive SHA256 hashes, reads only the two named members, validates dimensions/ranges and writes label-free NPZ arrays. Tests and experiments never download data implicitly.

Credit: E. Alpaydin and C. Kaynak (1998), Optical Recognition of Handwritten Digits; E. Alpaydin and F. Alimoglu (1996), Pen-Based Recognition of Handwritten Digits, UCI Machine Learning Repository. Both source pages list CC BY 4.0 licensing. Our transformations discard labels and scale features as described above.

`optdigits.json` and `pendigits.json` retain original archive/member hashes, DOI, source URL, array-content hashes, split and scaling details. Large NPZ files are ignored by version control; regenerate explicitly with:

```bash
.venv/bin/python scripts/fetch_cpu_data.py --output data
```

You may instead supply downloaded archives in a cache directory as `uci-optdigits.zip` and `uci-pendigits.zip` and use `--archive-dir <cache>`. The verified outputs have identical array-content hashes regardless of NPZ container timestamps.

## Pinned real embeddings for GPU preparation

[glove100_source.json](glove100_source.json) pins the first 16,384 train and 2,048 test vectors in [ANN-Benchmarks' GloVe-100 HDF5 corpus](https://github.com/erikbern/ann-benchmarks#data-sets). Credit: Jeffrey Pennington, Richard Socher and Christopher D. Manning, *GloVe: Global Vectors for Word Representation* (EMNLP 2014), and the ANN-Benchmarks repackaging. Refer to [the GloVe project](https://nlp.stanford.edu/projects/glove/) for original distribution information; this subset is not one of the UCI datasets above.

```bash
.venv/bin/python scripts/fetch_gpu_embeddings.py
```

Only 7,372,800 selected raw data bytes need retrieval through exact HTTP ranges. Both ranges and the full prepared arrays are SHA256 checked. Original HDF5 layout, source length and ETag are recorded; the unselected parent file is **not** SHA256-verified. Offline cache mode accepts `emeans-glove-train.bin` / `emeans-glove-test.bin` with the same pinned bytes. The output is `glove100_subset.npz` (ignored by version control) plus provenance JSON. No normalization or fitting on test data occurs. Fixed prefixes may have selection bias.

This pinned subset was used in the completed A40 Phase 3/4 studies; see REPRODUCTION_REPORT.md for independently measured results. The CUDA harness measures squared Euclidean clustering MSE rather than the source site's angular nearest-neighbor metric.
