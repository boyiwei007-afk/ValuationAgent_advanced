"""Offline packaging guards; no user secrets, network calls or real databases."""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


def load_packager():
    source = Path(__file__).resolve().parents[1] / 'scripts' / 'package_release.py'
    spec = importlib.util.spec_from_file_location('release_safety_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensitive_paths_are_excluded_inside_allowlisted_directories():
    packager = load_packager()
    filenames = [
        '.env', '.env.local', '.env.production.backup', '.ENV', '.envrc',
        '.env.example', 'production.env', 'app.env.local',
        'runs.db', 'runs.db-wal', 'runs.db-shm', 'runs.db-journal',
        'runs.sqlite', 'runs.sqlite3-wal', 'runs.sqlite3-shm.bak',
        'runs.duckdb', 'runs.DB.backup', 'private.pem', 'private.key',
        'private.key.old', 'client.p12', 'client.pfx', 'client.crt',
        'client.cer', 'client.der', 'client.jks', 'client.keystore',
        'id_rsa', 'id_ed25519', 'id_ed25519.pub', '.netrc', '.npmrc',
        'credentials', 'credentials.json', 'secrets.toml', 'service-account.json',
    ]
    for name in filenames:
        assert packager.release_exclusion('docs/' + name), name
    for name in ('docs/.aws/config', 'src/.ssh/config', 'tests/secrets/sample.json',
                 'examples/credentials/token.txt', 'src/__pycache__/cache.pyc'):
        assert packager.release_exclusion(name), name


def test_explicit_template_and_legitimate_source_names_remain_packagable():
    packager = load_packager()
    for name in ('.env.example', 'environment.yml', 'requirements.lock',
                 'tests/test_release_safety.py', 'tests/test_credentials.py',
                 'src/credentials.py', 'src/secrets.py', 'src/valuationagent/storage/sqlite.py',
                 'docs/security.md', 'web/package-lock.json', 'examples/structured_request.json'):
        assert packager.release_exclusion(name) is None, name


def test_packaging_omits_sensitive_files_but_keeps_safe_template_and_tests(tmp_path, monkeypatch, capsys):
    packager = load_packager()
    safe = {
        'README.md': '# Synthetic package fixture',
        '.env.example': 'API_KEY=replace-me',
        'web/dist/index.html': '<html>fixture</html>',
        'tests/test_credentials.py': 'def test_example(): pass',
    }
    hidden = ('docs/.env.local', 'docs/runs.sqlite3-wal', 'docs/private.pem',
              'docs/credentials.json', 'tests/.env.example')
    for name, body in {**safe, **dict.fromkeys(hidden, 'fixture-only private material')}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding='utf-8')
    output = tmp_path / 'release.zip'
    monkeypatch.setattr(packager, 'ROOT', tmp_path)
    monkeypatch.setattr(packager, 'TOP_FILES', ('README.md', '.env.example'))
    monkeypatch.setattr(packager, 'DIRECTORIES', ('docs', 'tests', 'web/dist'))
    monkeypatch.setattr(sys, 'argv', ['package_release.py', '--output', str(output)])
    packager.main()
    assert json.loads(capsys.readouterr().out)['verified']
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {*safe, 'RELEASE_MANIFEST.json'}
    assert packager.verify(output)['verified']


def test_verify_rejects_forbidden_file_even_when_manifest_hash_is_valid(tmp_path):
    packager = load_packager()
    body = b'not a real database'
    name = 'docs/private.db-shm'
    manifest = {'files': {name: {'sha256': packager.sha(body), 'size': len(body)}}}
    path = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(name, body)
        archive.writestr('RELEASE_MANIFEST.json', json.dumps(manifest))
    with pytest.raises(ValueError, match='Excluded release entry'):
        packager.verify(path)


def test_credential_failure_never_discloses_matched_value():
    packager = load_packager()
    for prefix in ('sk-', 'sk-proj-', 'sk-svcacct-', 'tvly-dev-', 'tvly-prod-'):
        token = prefix + ('x' * 40)
        with pytest.raises(ValueError) as captured:
            packager.validate_entry('docs/example.txt', token.encode())
        assert token not in str(captured.value)
        assert 'without printing the value' in str(captured.value)
