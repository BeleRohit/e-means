# API and numerical conventions

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

