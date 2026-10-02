import pytest


@pytest.fixture(autouse=True)
def explicit_test_hosts(monkeypatch):
    monkeypatch.setenv("VALUATION_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1,[::1]")
