"""Build and read-verify a publication bundle from the reviewed offline manifest."""
import argparse
import gzip
import hashlib
import io
from pathlib import Path
import tarfile
from verify_publication import ROOT, MANIFEST, verify, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.with_name(args.output.name + '.sha256').exists():
        raise FileExistsError('choose a new archive path; existing archives are immutable')
    manifest = verify()
    expected = {name: value['sha256'] for name, value in manifest['files'].items()}
    expected[MANIFEST.as_posix()] = digest(ROOT / MANIFEST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open('xb') as stream:
            with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w') as archive:
                    for name in sorted(expected):
                        data = (ROOT / name).read_bytes()
                        if hashlib.sha256(data).hexdigest() != expected[name]:
                            raise ValueError(f'file changed during packaging: {name}')
                        info = tarfile.TarInfo('e-means/' + name)
                        info.size = len(data)
                        info.mode = 0o755 if name.endswith('.sh') else 0o644
                        info.mtime = 0
                        archive.addfile(info, io.BytesIO(data))
        observed = {}
        with tarfile.open(args.output, 'r:gz') as archive:
            for member in archive:
                if not member.isfile() or not member.name.startswith('e-means/'):
                    raise ValueError('unexpected archive member')
                name = member.name[len('e-means/'):]
                if name in observed:
                    raise ValueError('duplicate archive member')
                observed[name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        if observed != expected:
            raise ValueError('archive member integrity mismatch')
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    checksum = digest(args.output)
    args.output.with_name(args.output.name + '.sha256').write_text(f'{checksum}  {args.output.name}\n')
    print(f'Verified {len(expected)} members: {args.output}; SHA256 {checksum}')


if __name__ == '__main__':
    main()
