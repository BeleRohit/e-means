"""Fetch two pinned authors' files for optional tests. No vendoring/import in core.

Usage: python scripts/fetch_reference.py /tmp/emeans-reference-2b648f
This script requires network access; pytest itself never downloads anything.
"""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

COMMIT = "2b648f0605f2ef3d14c4705350f8a0b214463878"
HASHES = {
    "base.py": "bcffd7829ebfc25e1354308770b49e6f99d8566ba5e474a41a22e9ec414b17cd",
    "e_means.py": "b1710504fdbef21efdd33d8489a56f8e7fc6056182afbed3f355985302c2e77c",
}
NATIVE_HASHES = {
    "e_means_torch.py": "cddc5ed1f1121c96406055fe6f50e93c049d0e09be27dc07dbe03e5c8e7ef4d3",
    "torch_backend.py": "eb7dc8a41fc098604e86d0dcaa28106df5eed79670d278fe5bfbd573a605355c",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--native-torch", action="store_true", help="also fetch pinned lazy native-Torch reference files")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes=dict(HASHES,**(NATIVE_HASHES if args.native_torch else {}))
    for filename, expected in hashes.items():
        url = f"https://raw.githubusercontent.com/facebookresearch/faiss/{COMMIT}/contrib/e_means/{filename}"
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"SHA256 mismatch for {filename}")
        target = args.output_dir / filename
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError(f"refusing to overwrite different file: {target}")
        target.write_bytes(data)
        print(f"Verified {filename}: {expected}")


if __name__ == "__main__":
    main()
