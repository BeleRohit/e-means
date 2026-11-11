"""Reproducible in-memory benchmark inputs; no implicit network access."""

from pathlib import Path
import numpy as np

from e_means.config import EMeansConfig, LloydConfig, positive_int
from e_means.gpu_protocol import array_hash


METHODS = ("lloyd", "emeans", "slow_ema")


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 3:
        raise ValueError("GPU workload schema_version must be 3")
    for section in ("common", "lloyd", "e_means", "slow_ema"):
        if set(config[section]) & {"n_clusters", "seed", "dtype"}:
            raise ValueError("workload/matrix alone controls clusters, seed and dtype")
    for name in ("repeats", "warmups"):
        positive_int(name, config[name])
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("distinct nonempty seeds required")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in config["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    if not config["dtypes"] or len(set(config["dtypes"])) != len(config["dtypes"]) or any(x not in ("float32", "float64") for x in config["dtypes"]):
        raise ValueError("distinct FP32/FP64 dtypes required")
    slack = config["target_relative_slack"]
    if isinstance(slack,bool) or not np.isfinite(slack) or slack < 0:
        raise ValueError("target_relative_slack must be finite and nonnegative")
    if not config["workloads"] or len({c["name"] for c in config["workloads"]}) != len(config["workloads"]):
        raise ValueError("distinct nonempty workloads required")
    for case in config["workloads"]:
        if not case["name"].replace("_", "").isalnum():
            raise ValueError("workload name must be alphanumeric/underscore")
        positive_int("n_clusters", case["n_clusters"])
        if case["kind"] == "synthetic":
            for key in ("n_train", "n_test", "n_features", "data_seed"):
                if key == "data_seed" and case[key] == 0 and not isinstance(case[key],bool):
                    continue
                positive_int(key, case[key])
            for key in ("center_scale", "noise_scale"):
                if not np.isfinite(case[key]) or case[key] <= 0:
                    raise ValueError("positive finite mixture scales required")
            if case["n_clusters"] > case["n_train"]:
                raise ValueError("too many clusters")
        elif case["kind"] != "embedding":
            raise ValueError("workload kind must be synthetic or embedding")
        for dtype in config["dtypes"]:
            for method in METHODS:
                settings(config, case, config["seeds"][0], dtype, method)


def settings(config: dict, case: dict, seed: int, dtype: str, method: str):
    values = dict(config["common"], n_clusters=case["n_clusters"], seed=seed, dtype=dtype)
    values.update(config["lloyd"] if method == "lloyd" else config["e_means"])
    if method == "slow_ema":
        values.update(config["slow_ema"], finalize="slow_ema")
    if method not in METHODS:
        raise ValueError("unknown method")
    return (LloydConfig if method == "lloyd" else EMeansConfig)(**values)


def load_workload(case: dict, seed: int, dtype: str, root: Path):
    if case["kind"] == "synthetic":
        # Dataset seed is fixed across solver seeds; only initialization and
        # optimization randomness vary. Test draws are independent of train.
        rng = np.random.default_rng(case["data_seed"])
        k, d = case["n_clusters"], case["n_features"]
        centers = rng.normal(size=(k,d))*case["center_scale"]
        def sample(n):
            return centers[rng.integers(k, size=n)] + rng.normal(size=(n,d))*case["noise_scale"]
        x, test = sample(case["n_train"]), sample(case["n_test"])
    else:
        with np.load(root / case["path"], allow_pickle=False) as file:
            x, test = file["train"], file["test"]
        if array_hash(x, test) != case["array_sha256"]:
            raise ValueError("pinned embedding subset hash mismatch")
        for key in ("train_limit", "test_limit"):
            if key in case:
                positive_int(key, case[key])
        if case.get("train_limit", len(x)) > len(x) or case.get("test_limit", len(test)) > len(test):
            raise ValueError("embedding prefix limit exceeds pinned data")
        x = x[:case.get("train_limit", len(x))]
        test = test[:case.get("test_limit", len(test))]
    x, test = np.asarray(x, dtype=dtype), np.asarray(test, dtype=dtype)
    if x.ndim != 2 or test.ndim != 2 or min(x.shape) < 1 or min(test.shape) < 1 or x.shape[1] != test.shape[1] or not np.isfinite(x).all() or not np.isfinite(test).all():
        raise ValueError("invalid workload matrices")
    if not 1 <= case["n_clusters"] <= len(x):
        raise ValueError("too many clusters")
    init = x[np.random.default_rng(seed).choice(len(x), case["n_clusters"], replace=False)].copy()
    return x, test, init, array_hash(x, test, init)
