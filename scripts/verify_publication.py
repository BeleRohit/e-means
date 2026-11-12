"""Offline publication inventory, SHA256, credential and documentation audit.

No model fitting, cloud access, uploads, Git mutations or paid actions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('artifacts/PUBLICATION_MANIFEST.json')
FOLDERS = ('src', 'tests', 'configs', 'scripts', 'runpod', 'notebooks', 'docs', 'artifacts/public')
EXTENSIONS = {'.py', '.sh', '.json', '.lock', '.md', '.ipynb', '.npz', '.png', '.svg', '.csv', '.xml', '.zip', '.txt'}
PATTERNS = {
    'private key': rb'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----',
    'GitHub credential': rb'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})\b',
    'Runpod credential': rb'\brpa_[A-Za-z0-9_\-]{20,}\b',
    'AWS access key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'literal credential': rb'''(?i)["']?(?:api_key|apiKey|password|secret_access_key|access_token)["']?\s*[:=]\s*["'][A-Za-z0-9_\-/+=]{20,}["']''',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def paths(root=ROOT):
    selected = {p for p in root.glob('*.md') if p.is_file()}
    selected.update(root / name for name in ('.gitignore', 'pyproject.toml', 'artifacts/README.md', 'artifacts/RAW_ARCHIVES.json', 'data/README.md', 'data/glove100_source.json', 'data/optdigits.json', 'data/pendigits.json'))
    for folder in FOLDERS:
        selected.update(p for p in (root / folder).rglob('*') if p.is_file()
                        and p.suffix in EXTENSIONS and '__pycache__' not in p.parts
                        and not any(part.endswith('.egg-info') for part in p.parts))
    for p in selected:
        if p.is_symlink():
            raise ValueError(f'symlink excluded from publication: {p.relative_to(root)}')
    return sorted(selected)


def scan_blob(data, label):
    for kind, pattern in PATTERNS.items():
        if re.search(pattern, data):
            raise ValueError(f'possible {kind} in {label}; value suppressed')


def audit_content(files, root=ROOT):
    for p in files:
        label = p.relative_to(root).as_posix()
        if p.suffix == '.zip':
            with zipfile.ZipFile(p) as archive:
                for member in archive.infolist():
                    if not member.is_dir():
                        scan_blob(archive.read(member), f'{label}:{member.filename}')
        else:
            scan_blob(p.read_bytes(), label)
        if p.suffix == '.md':
            for target in re.findall(r'\]\(([^\s)]+)(?:\s+[^)]*)?\)', p.read_text()):
                target = target.strip('<>').split('#')[0]
                if not target or '://' in target or target.startswith(('mailto:', '#')):
                    continue
                resolved = (p.parent / target).resolve()
                if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                    raise ValueError(f'broken/local-private documentation link: {label} -> {target}')
                if resolved.is_file() and resolved not in files:
                    raise ValueError(f'link target excluded from public inventory: {label} -> {target}')
        if p.suffix == '.ipynb':
            nb = json.loads(p.read_text())
            for cell in nb['cells']:
                if cell.get('outputs') or cell.get('execution_count') is not None:
                    raise ValueError(f'clear notebook outputs before publication: {label}')


def verify(root=ROOT):
    files = paths(root)
    manifest = json.loads((root / MANIFEST).read_text())
    actual = {p.relative_to(root).as_posix(): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in files}
    if manifest['files'] != actual:
        changed = sorted(k for k in set(actual) | set(manifest['files']) if actual.get(k) != manifest['files'].get(k))
        raise ValueError(f'publication inventory/hash mismatch: {changed}')
    audit_content(files, root)
    return manifest


def scan_git_history():
    blobs = subprocess.check_output(['git', 'rev-list', '--objects', '--all'], cwd=ROOT, text=True)
    count = 0
    for row in blobs.splitlines():
        oid, _, label = row.partition(' ')
        kind = subprocess.check_output(['git', 'cat-file', '-t', oid], cwd=ROOT, text=True).strip()
        if kind == 'blob':
            scan_blob(subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT), f'Git blob {oid}:{label}')
            count += 1
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-manifest', action='store_true', help='explicitly refresh after reviewing intended publication files')
    p.add_argument('--check-git-history', action='store_true', help='also scan reachable local Git blobs; no remote access')
    args = p.parse_args()
    if args.write_manifest:
        files = paths()
        audit_content(files)
        payload = {'schema_version': 1, 'scope': 'public source, documentation, measured numeric evidence and figures; private raw/controller exports excluded',
                   'files': {f.relative_to(ROOT).as_posix(): {'sha256': digest(f), 'bytes': f.stat().st_size} for f in files}}
        (ROOT / MANIFEST).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    result = verify()
    history = scan_git_history() if args.check_git_history else None
    print(json.dumps({'status': 'verified', 'files': len(result['files']), 'git_blobs_scanned': history,
                      'credential_scan': 'no matches in supported patterns; not a guarantee of all possible secrets',
                      'document_links': 'verified', 'notebook_outputs': 'clear'}))


if __name__ == '__main__':
    main()
