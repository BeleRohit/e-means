"""Bundle source, regression fixtures and small pinned embedding subset for transfer.

No credentials, virtual environments, arbitrary workspace files or prior large
experimental results are included. Does not create a Pod or upload anything.
"""

import argparse
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from e_means.gpu_protocol import file_hash, source_paths


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new package path")
    paths=source_paths(ROOT)+list(ROOT.glob("*.md"))
    paths.extend(p for p in (ROOT/"docs").rglob("*") if p.is_file())
    paths.extend(p for p in (ROOT/"data").glob("*.json"))
    paths.append(ROOT/"data/README.md")
    data=ROOT/"data/glove100_subset.npz"
    if data.exists():
        paths.append(data)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,"w:gz") as tar:
        for path in sorted(set(paths)):
            if path.is_file():
                tar.add(path,arcname=str(Path("e-means")/path.relative_to(ROOT)),recursive=False)
    args.output.with_suffix(args.output.suffix+".sha256").write_text(f"{file_hash(args.output)}  {args.output.name}\n")
    print(f"Source package: {args.output}; sha256={file_hash(args.output)}")


if __name__ == "__main__":
    main()
