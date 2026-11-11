"""Prepare a SHA256-pinned real embedding subset with small HTTP range reads.

Only the selected bytes and resulting arrays are cryptographically pinned.
No claims about hashing the entire 485 MB corpus are made. Existing different
outputs are never overwritten; tests/benchmarks never download implicitly.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from e_means.gpu_protocol import array_hash, write_json

ROOT = Path(__file__).resolve().parents[1]


def prepare(source: dict, blocks: dict[str, bytes], output: Path) -> dict:
    arrays = {}
    for name, spec in source["ranges"].items():
        payload = blocks[name]
        if len(payload) != spec["bytes"] or hashlib.sha256(payload).hexdigest() != spec["sha256"]:
            raise ValueError(f"pinned byte range mismatch: {name}")
        array = np.frombuffer(payload,dtype="<f4").reshape(spec["rows"],spec["dimensions"]).copy()
        if not np.isfinite(array).all():
            raise ValueError("nonfinite embedding vector")
        arrays[name] = array
    digest = array_hash(arrays["train"],arrays["test"])
    if digest != source["array_sha256"]:
        raise ValueError("prepared embedding hash mismatch")
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        with np.load(output,allow_pickle=False) as old:
            if array_hash(old["train"],old["test"]) != digest:
                raise FileExistsError("refusing to overwrite different embedding data")
    else:
        np.savez_compressed(output,**arrays)
    metadata = dict(source,prepared_array_sha256=digest,status="data_prepared_not_gpu_verified")
    write_json(output.with_suffix(".json"),metadata)
    return metadata


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/glove100_subset.npz")
    parser.add_argument("--cache-dir",type=Path,help="offline cache containing emeans-glove-train.bin and emeans-glove-test.bin")
    args=parser.parse_args()
    source=json.loads((ROOT/"data/glove100_source.json").read_text())
    blocks={}
    for name,spec in source["ranges"].items():
        if args.cache_dir:
            blocks[name]=(args.cache_dir/f"emeans-glove-{name}.bin").read_bytes()
        else:
            end=spec["offset"]+spec["bytes"]-1
            expected=f"bytes {spec['offset']}-{end}/{source['source_length_bytes']}"
            request=Request(source["source_url"],headers={"Range":f"bytes={spec['offset']}-{end}","If-Match":source["source_etag"]})
            with urlopen(request,timeout=180) as response:
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("server did not honor the exact pinned byte range")
                blocks[name]=response.read(spec["bytes"]+1)
    metadata=prepare(source,blocks,args.output)
    print(f"Prepared pinned real data: {metadata['prepared_array_sha256']}; no GPU measurement")


if __name__ == "__main__":
    main()
