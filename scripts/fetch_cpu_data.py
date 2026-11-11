"""Fetch pinned public UCI digits archives; prepare label-free, fixed splits.

Only download this explicitly. Tests and benchmark workers stay offline.
Published feature-range scaling (divide by 16/100) uses no test statistics.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile

import numpy as np


SOURCES = {
    "optdigits": {
        "url": "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip",
        "sha256": "0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0",
        "doi": "10.24432/C50P49", "scale": 16, "features": 64, "rows": [3823, 1797],
    },
    "pendigits": {
        "url": "https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip",
        "sha256": "1e02bea023613c2b11c9492f6f34caf975420455934f3527d270cee9a1f03b64",
        "doi": "10.24432/C5MG6K", "scale": 100, "features": 16, "rows": [7494, 3498],
    },
}


def array_hash(*arrays: np.ndarray) -> str:
    """Shape-aware hash of canonical little-endian FP64 array contents."""
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def prepare(name: str, archive: bytes, output: Path) -> dict:
    """Verify original archive, use only named train/test members, discard labels."""
    spec = SOURCES[name]
    if hashlib.sha256(archive).hexdigest() != spec["sha256"]:
        raise ValueError(f"archive hash mismatch: {name}")
    arrays, members = [], {}
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        for extension, rows in zip(("tra", "tes"), spec["rows"]):
            member = f"{name}.{extension}"
            raw = zipped.read(member)
            matrix = np.loadtxt(io.BytesIO(raw), delimiter=",")
            if matrix.shape != (rows, spec["features"] + 1):
                raise ValueError(f"unexpected shape in {member}")
            x = matrix[:, :-1] / spec["scale"]
            if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                raise ValueError(f"invalid feature range in {member}")
            arrays.append(np.ascontiguousarray(x))
            members[member] = hashlib.sha256(raw).hexdigest()
    metadata = dict(spec, name=name, array_sha256=array_hash(*arrays), member_sha256=members,
                    preprocessing=f"discard last class-label column; divide all features by {spec['scale']}",
                    split="original UCI .tra/.tes; never reshuffled across splits")
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{name}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as old:
            if array_hash(old["train"], old["test"]) != metadata["array_sha256"]:
                raise FileExistsError(f"refusing to replace different dataset: {target}")
    else:
        np.savez_compressed(target, train=arrays[0], test=arrays[1])
    (output / f"{name}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--archive-dir", type=Path, help="offline cache named uci-optdigits.zip and uci-pendigits.zip")
    args = parser.parse_args()
    for name, spec in SOURCES.items():
        if args.archive_dir:
            archive = (args.archive_dir / f"uci-{name}.zip").read_bytes()
        else:
            with urlopen(spec["url"], timeout=120) as response:
                archive = response.read()
        metadata = prepare(name, archive, args.output)
        print(name, metadata["array_sha256"])


if __name__ == "__main__":
    main()
