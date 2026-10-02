import pytest
from fastapi.testclient import TestClient

from valuationagent.api.access import COOKIE, validate_bind
from valuationagent.api.main import create_app


TOKEN = "synthetic-access-test-credential-not-a-real-secret"


def configured(tmp_path, monkeypatch, enabled=True, **client_options):
    if enabled:
        monkeypatch.setenv("VALUATION_ACCESS_TOKEN", TOKEN)
    else:
        monkeypatch.delenv("VALUATION_ACCESS_TOKEN", raising=False)
    monkeypatch.setenv("VALUATION_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")
    return TestClient(create_app(tmp_path), **client_options)


def test_single_operator_protects_listing_delete_upload_and_model_configuration(tmp_path, monkeypatch):
    client = configured(tmp_path, monkeypatch)
    for method, path in [("GET", "/api/workspaces"), ("POST", "/api/files"), ("POST", "/api/model-sessions"),
                         ("DELETE", "/api/workspaces/any"), ("GET", "/docs")]:
        response = client.request(method, path)
        assert response.status_code == 401
        assert response.headers["Cache-Control"] == "no-store"
        assert TOKEN not in response.text
    assert client.get("/health").status_code == 200
    assert client.get("/api/workspaces?token=" + TOKEN).status_code == 401
    assert client.get("/api/workspaces", headers={"Authorization": "Bearer " + TOKEN}).status_code == 200


def test_cookie_login_logout_and_same_origin_protection(tmp_path, monkeypatch):
    client = configured(tmp_path, monkeypatch)
    page = client.get("/", headers={"Accept": "text/html"})
    assert page.status_code == 401 and "工作区访问凭证" in page.text
    assert "frame-ancestors 'none'" in page.headers["Content-Security-Policy"]
    assert TOKEN not in page.text
    response = client.post("/api/access/login", json={"token": TOKEN})
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"] and "SameSite=strict" in response.headers["set-cookie"]
    cookie = client.cookies.get(COOKIE)
    assert cookie != TOKEN
    assert client.get("/api/workspaces").status_code == 200
    assert client.post("/api/workspaces", json={}, headers={"Origin": "https://attacker.invalid"}).status_code == 403
    assert client.post("/api/access/logout", headers={"Origin": "http://testserver"}).status_code == 204
    assert client.get("/api/workspaces", headers={"Cookie": COOKIE + "=" + cookie}).status_code == 401


def test_login_attempts_are_bounded_and_validation_does_not_echo_secrets(tmp_path, monkeypatch):
    client = configured(tmp_path, monkeypatch)
    for attempt in range(5):
        assert client.post("/api/access/login", json={"token": "wrong"}).status_code == 401
    assert client.post("/api/access/login", json={"token": TOKEN}).status_code == 429
    response = client.post("/api/access/login", json={"token": TOKEN * 20})
    assert response.status_code == 422 and TOKEN not in response.text


def test_unauthenticated_mode_is_loopback_only_even_when_asgi_is_bound_externally(tmp_path, monkeypatch):
    local = configured(tmp_path / "local", monkeypatch, enabled=False)
    assert local.get("/api/workspaces").status_code == 200
    assert local.post("/api/workspaces", json={}, headers={"Origin": "http://attacker.invalid"}).status_code == 403
    assert local.get("/api/workspaces", headers={"Host": "rebinding.invalid"}).status_code == 400
    remote = configured(tmp_path / "remote", monkeypatch, enabled=False, client=("192.0.2.10", 32100))
    assert remote.get("/api/workspaces").status_code == 403
    assert remote.get("/health").status_code == 200
    validate_bind("127.0.0.1")
    with pytest.raises(ValueError, match="VALUATION_ACCESS_TOKEN"):
        validate_bind("0.0.0.0")


def test_remote_login_requires_https_and_cookie_is_secure(tmp_path, monkeypatch):
    insecure = configured(tmp_path / "http", monkeypatch, client=("192.0.2.10", 32100))
    assert insecure.post("/api/access/login", json={"token": TOKEN}).status_code == 403
    secure = configured(tmp_path / "https", monkeypatch, client=("192.0.2.10", 32100), base_url="https://testserver")
    login = secure.post("/api/access/login", json={"token": TOKEN})
    assert login.status_code == 200 and "Secure" in login.headers["set-cookie"]
    assert secure.get("/api/workspaces").status_code == 200
    validate_bind("0.0.0.0")


def test_restart_invalidates_browser_sessions(tmp_path, monkeypatch):
    first = configured(tmp_path / "first", monkeypatch)
    first.post("/api/access/login", json={"token": TOKEN})
    restarted = configured(tmp_path / "second", monkeypatch)
    assert restarted.get("/api/workspaces", headers={"Cookie": COOKIE + "=" + first.cookies.get(COOKIE)}).status_code == 401


def test_default_hosts_exclude_test_and_external_domains(tmp_path, monkeypatch):
    monkeypatch.delenv("VALUATION_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("VALUATION_ACCESS_TOKEN", raising=False)
    client = TestClient(create_app(tmp_path), base_url="http://localhost")
    assert client.get("/health").status_code == 200
    for host in ["testserver", "rebinding.invalid"]:
        assert client.get("/health", headers={"Host": host}).status_code == 400


@pytest.mark.parametrize("token", ["too-short", " leading-space-credential-of-at-least-32-characters"])
def test_bad_configuration_fails_closed(tmp_path, monkeypatch, token):
    monkeypatch.setenv("VALUATION_ACCESS_TOKEN", token)
    with pytest.raises(ValueError, match="VALUATION_ACCESS_TOKEN"):
        create_app(tmp_path)
