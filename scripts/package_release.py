"""Package the inspected working tree, never .env, databases or local history.

python scripts/package_release.py
python scripts/package_release.py --verify output/release/ValuationAgent-advance.zip
"""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('src', 'tests', 'scripts', 'docs', 'examples', 'web/src', 'web/public', 'web/scripts', 'web/tests', 'web/dist')
TOP_FILES = ('README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml', 'environment.yml', 'requirements.lock', '.gitignore', '.env.example')
# Defense in depth for recognized credential formats, not an exhaustive secret
# detector. Keep messages value-free even when a match is only a test fixture.
SECRET = re.compile(rb'(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|tvly-(?:dev|prod)-[A-Za-z0-9_-]{24,})')
EXCLUDED_DIRS = {
    '__pycache__', 'node_modules', '.git', '.hg', '.svn', '.history',
    '.ssh', '.aws', '.azure', '.gnupg', '.gcloud', '.pytest_cache',
}
KEY_SUFFIXES = {'.pem', '.key', '.pfx', '.p12', '.p7b', '.p7c', '.p8',
                '.crt', '.cer', '.der', '.jks', '.keystore', '.csr', '.kdb', '.pub'}
DATABASE_NAME = re.compile(r'\.(?:db|db3|sqlite|sqlite3|duckdb|mdb|accdb)(?:-(?:wal|shm|journal))?$')
CREDENTIAL_NAMES = re.compile(r'^(?:credentials?|secrets?|service[-_]account)(?:\.(?:json|toml|ya?ml|ini|cfg|conf|txt))?$')


def release_exclusion(name):
    """Return a reason to exclude a relative archive path, never file content."""
    relative = str(name).replace('\\', '/')
    parts = PurePosixPath(relative).parts
    lowered = [part.casefold() for part in parts]
    if any(part in EXCLUDED_DIRS or part.endswith('.egg-info') for part in lowered):
        return 'local runtime or credential directory'
    # The only intentional environment template is the explicit top-level
    # allowlist entry. A similarly named file deeper in the tree is not trusted.
    if relative != '.env.example' and any(
        part.startswith('.env') or part.endswith('.env') or '.env.' in part
        for part in lowered
    ):
        return 'environment configuration'
    if any(part in {'credentials', 'secrets'} for part in lowered[:-1]):
        return 'credential directory'
    filename = lowered[-1] if lowered else ''
    while PurePosixPath(filename).suffix in {'.bak', '.backup', '.old', '.orig', '.tmp'}:
        filename = filename.rsplit('.', 1)[0]
    suffix = PurePosixPath(filename).suffix
    if suffix in {'.pyc', '.pyo'}:
        return 'compiled Python cache'
    if DATABASE_NAME.search(filename) or filename in {'-wal', '-shm'}:
        return 'database or journal'
    if suffix in KEY_SUFFIXES or filename in {'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519'}:
        return 'key or certificate'
    if CREDENTIAL_NAMES.fullmatch(filename) or filename in {'.netrc', '_netrc', '.npmrc', '.pypirc'}:
        return 'credential configuration'
    return None


def validate_entry(name, data):
    reason = release_exclusion(name)
    if reason:
        raise ValueError('Excluded release entry: ' + name + ' (' + reason + ')')
    if SECRET.search(data):
        raise ValueError('Potential credential found in ' + name + '; packaging stopped without printing the value')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('RELEASE_MANIFEST.json'))
        if set(archive.namelist()) != {*manifest['files'], 'RELEASE_MANIFEST.json'}:
            raise ValueError('Unexpected or missing archive entries')
        for name, entry in manifest['files'].items():
            if name.startswith('/') or '..' in Path(name).parts:
                raise ValueError('Unsafe archive path')
            data = archive.read(name)
            validate_entry(name, data)
            if sha(data) != entry['sha256'] or len(data) != entry['size']:
                raise ValueError('Release integrity mismatch: ' + name)
        if archive.testzip():
            raise ValueError('Corrupt ZIP')
    return {'verified': True, 'files': len(manifest['files']), 'sha256': sha(path.read_bytes())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'output/release/ValuationAgent-advance.zip')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(args.verify), indent=2))
        return
    if args.output.exists():
        raise FileExistsError('Choose a new output filename; existing releases are not overwritten')
    if not (ROOT / 'web/dist/index.html').is_file():
        raise ValueError('Build the frontend before packaging')
    paths = [ROOT / name for name in TOP_FILES]
    paths += [p for name in DIRECTORIES if (ROOT / name).is_dir() for p in (ROOT / name).rglob('*') if p.is_file()]
    paths += [p for p in (ROOT / 'web').iterdir() if p.is_file() and p.suffix in {'.json', '.js', '.mjs', '.html', '.md'}]
    entries = {}
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        if release_exclusion(name):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Unsafe source path: ' + str(path))
        entries[name] = path.read_bytes()
    for name, data in entries.items():
        validate_entry(name, data)
    manifest = {'schema': 'valuationagent-release-v1', 'version': '0.6.0.dev0',
                'source': 'inspected working tree, including uncommitted changes',
                'excludes': ['environment configuration except root .env.example', 'recognized credentials',
                             'keys and certificates', 'user data directories', 'databases and journals',
                             'node_modules', 'git history'],
                'files': {name: {'sha256': sha(data), 'size': len(data)} for name, data in sorted(entries.items())}}
    entries['RELEASE_MANIFEST.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    print(json.dumps({'archive': str(args.output.resolve()), **verify(args.output)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
