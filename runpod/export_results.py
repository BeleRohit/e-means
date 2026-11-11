"""Persist an exit status plus checksummed archive for complete or failed runs."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import tarfile

from e_means.gpu_protocol import file_hash, write_json


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--exit-code",type=int,required=True)
    args=parser.parse_args()
    root=args.run.resolve()
    write_json(root/"execution_status.json",dict(exit_code=args.exit_code,
               state="completed" if args.exit_code==0 else "failed",exported_utc=datetime.now(timezone.utc).isoformat(),
               note="execution status is not a substitute for the CUDA validation/measurement manifests"))
    archive=root.parent/f"{root.name}.tar.gz"
    if archive.exists():
        raise FileExistsError("refusing to overwrite an existing result export")
    with tarfile.open(archive,"w:gz") as tar:
        tar.add(root,arcname=root.name)
    archive.with_suffix(archive.suffix+".sha256").write_text(f"{file_hash(archive)}  {archive.name}\n")
    print(f"Persistent export: {archive}")


if __name__ == "__main__":
    main()
