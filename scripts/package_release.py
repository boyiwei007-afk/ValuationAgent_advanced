"""Package the inspected working tree, never .env, databases or local history.

python scripts/package_release.py --include-current-acceptance
python scripts/package_release.py --verify output/release/ValuationAgent-20260927-v11.zip
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
    parser.add_argument('--output', type=Path, default=ROOT / 'output/release/ValuationAgent-20260927-v11.zip')
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--include-acceptance', action='store_true', help='Include only this dated synthetic acceptance fixture and public-source probe metadata')
    parser.add_argument('--include-current-acceptance', action='store_true', help='Include only 2026-09-26 synthetic reports and bounded live-check metadata')
    parser.add_argument('--include-real-acceptance', action='store_true', help='Include 2026-09-27 real-company summaries and public source hashes, never annual-report PDFs or databases')
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
    if args.include_acceptance:
        from valuationagent.storage.sqlite import SQLiteRunStore
        fixture = ROOT / 'var/delivery-live-final-20260925'
        for name in ('acceptance.json', 'synthetic-report.json', 'synthetic-report.xlsx', 'synthetic-report.pdf'):
            latest = 'synthetic-report-delivery.xlsx' if name == 'synthetic-report.xlsx' and (fixture / 'synthetic-report-delivery.xlsx').is_file() else name
            entries['acceptance/' + name] = (fixture / latest).read_bytes()
        package = json.loads(entries['acceptance/synthetic-report.json'])
        store = SQLiteRunStore(fixture)
        for ref in package['source_manifest']:
            if ref.get('role') == 'search_lead':
                continue
            meta = store.get_file(ref['file_id'])
            original = Path(meta['storage_path'])
            if not original.is_absolute():
                original = ROOT / original
            if not original.resolve().is_relative_to(fixture):
                raise ValueError('Acceptance source is outside the synthetic fixture')
            content = original.read_bytes()
            if sha(content) != ref['sha256']:
                raise ValueError('Acceptance source hash mismatch')
            entries['acceptance/sources/' + ref['file_id'] + original.suffix] = content
        for source, target in (
            ('var/delivery-benchmark-20260925/metrics.json', 'acceptance/local-performance.json'),
            ('var/delivery-source-20260925/source-check.json', 'acceptance/official-source-probe.json'),
        ):
            entries[target] = (ROOT / source).read_bytes()
    if args.include_current_acceptance:
        for folder, target, names in (
            ('output/acceptance-20260926', 'acceptance/v8/no-data', ('acceptance.json', 'no-data-outcome.pdf', 'no-data-outcome.html', 'no-data-outcome.json', 'unfinished-run-diagnostic.pdf')),
            ('var/live-upload-20260926', 'acceptance/v8/upload', ('acceptance.json', 'synthetic-valuation.pdf', 'synthetic-valuation.xlsx', 'synthetic-valuation.json')),
            ('var/live-no-upload-20260926-release', 'acceptance/v8/public-source', ('acceptance.json',)),
            ('var/performance-20260926', 'acceptance/v8/performance', ('metrics.json',)),
        ):
            for name in names:
                entries[target + '/' + name] = (ROOT / folder / name).read_bytes()
        # Ship the synthetic uploaded source as well as the computed bundle,
        # so reviewers can verify the original text and its bound file hash.
        from valuationagent.storage.sqlite import SQLiteRunStore
        fixture = ROOT / 'var/live-upload-20260926'
        store = SQLiteRunStore(fixture)
        package = json.loads(entries['acceptance/v8/upload/synthetic-valuation.json'])
        for ref in package['source_manifest']:
            if ref.get('role') == 'search_lead':
                continue
            meta = store.get_file(ref['file_id'])
            original = Path(meta['storage_path'])
            if not original.is_absolute():
                original = ROOT / original
            if not original.resolve().is_relative_to(fixture):
                raise ValueError('Acceptance source is outside the synthetic fixture')
            content = original.read_bytes()
            if sha(content) != ref['sha256']:
                raise ValueError('Acceptance source hash mismatch')
            entries['acceptance/v8/upload/sources/' + ref['file_id'] + original.suffix] = content
    if args.include_real_acceptance:
        real_root = ROOT / 'var/real-company-20260926'
        for code in ('000333', '300750', '600887'):
            entries[f'acceptance/v9/sources/{code}/manifest.json'] = (real_root / 'sources' / code / 'manifest.json').read_bytes()
            entries[f'acceptance/v9/upload/{code}/acceptance.json'] = (real_root / 'final-20260927' / code / 'acceptance.json').read_bytes()
        entries['acceptance/v9/zero-upload/600887/acceptance.json'] = (real_root / 'zero-upload-final-20260927/600887/acceptance.json').read_bytes()
        entries['acceptance/v9/resumed/000333/acceptance.json'] = (real_root / 'resumed-20260927/000333/acceptance.json').read_bytes()
        for code in ('000333', '300750', '600887'):
            entries[f'acceptance/v9/verified-upload/{code}/acceptance.json'] = (real_root / 'verified-20260927' / code / 'acceptance.json').read_bytes()
        entries['acceptance/v9/verified-zero-upload/600887/acceptance.json'] = (real_root / 'zero-upload-verified-20260927/600887/acceptance.json').read_bytes()
        entries['acceptance/v9/final-closure/000333/acceptance.json'] = (real_root / 'closure-20260927/000333/acceptance.json').read_bytes()
        entries['acceptance/v9/final-zero-upload-closure/600887/acceptance.json'] = (real_root / 'zero-upload-closure-20260927/600887/acceptance.json').read_bytes()
    for name, data in entries.items():
        validate_entry(name, data)
    manifest = {'schema': 'valuationagent-release-v1', 'version': '0.5.0-real-data-20260927-v11',
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
