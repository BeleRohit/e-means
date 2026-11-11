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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected in HASHES.items():
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
