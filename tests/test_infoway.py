import json
from datetime import date

import httpx
import pytest


def test_tushare_environment_is_visible_as_acquisition_capability(monkeypatch):
    from valuationagent.market.factory import create_history_provider

    monkeypatch.delenv("INFOWAY_API_KEY", raising=False)
    monkeypatch.setenv("TUSHARE_TOKEN", "in-memory-synthetic-token")
    provider = create_history_provider()
    assert provider.provider_id == "tushare"
    assert callable(provider.fetch_history)
from fastapi.testclient import TestClient

from valuationagent.api.main import create_app
from valuationagent.application.file_workspace import FileRead, read_file
from valuationagent.market.infoway import InfowayApiClient, InfowayApiError, InfowayDataProvider
from valuationagent.market.research_data import FinancialHistoryRequest, fetch_history
from test_multisource_extraction import runtime_at


ROW = {"symbol": "600123.SH", "periodType": "fy", "periodDate": "2025-12-31",
       "itemId": "net_income", "itemName": "Net income", "itemValue": "1234567890.1234", "ttm": None}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


def client_with(handler, **kwargs):
    clock = FakeClock()
    client = InfowayApiClient("synthetic-infoway-key", transport=httpx.MockTransport(handler),
        clock=clock.clock, sleep=clock.sleep, **kwargs)
    return client, clock


def test_request_cache_and_exact_response_values():
    requests = []
    raw = b'{"ret":200,"data":[{"symbol":"600123.SH","periodType":"fy","periodDate":"2025-12-31","itemValue":1234567890.123456789}]}'
    def handler(request):
        requests.append(request)
        assert request.headers["apiKey"] == "synthetic-infoway-key"
        assert request.url.params == httpx.QueryParams({"symbol": "600123.SH", "type": "STOCK_CN", "period_type": "fy"})
        return httpx.Response(200, content=raw)
    client, clock = client_with(handler)
    first = client.query("600123", "income")
    assert client.query("600123.SH", "income") == first
    assert len(requests) == 1 and first[0] == raw
    client.query("600123", "cashflow")
    assert len(requests) == 2 and clock.now >= 3
    assert client.connection_verified


def test_snapshot_filters_identity_period_and_preserves_missing_metadata():
    rows = [ROW, {**ROW, "periodDate": "2024-12-31"}, {**ROW, "periodDate": "2025-06-30", "periodType": "fh"},
            {**ROW, "symbol": "000999.SZ"}, {**ROW, "periodDate": "2027-12-31"}, {**ROW, "periodDate": "bad"}]
    client, clock = client_with(lambda request: httpx.Response(200, json={"ret": 200, "data": rows}))
    snapshot = InfowayDataProvider(client).fetch_history("600123", "income", [2024, 2025], date(2026, 9, 30))
    assert snapshot.accepted_records == 2 and snapshot.excluded_records == 4
    block = snapshot.blocks[0]
    assert json.loads(block["text"]) == ROW
    assert block["location"]["json_pointer"] == "/data/0"
    assert "published_at" in block["location"]["missing_metadata"]
    assert "published_at" not in block["location"]
    assert "归母净利润" not in block["text"] and "单位：元" not in block["text"]


def test_statistics_does_not_misdate_current_value_or_fill_missing_amount():
    row = {**ROW, "itemId": "total_shares_outstanding", "itemValue": None, "currentValue": 9000, "ttmStr": "-"}
    client, clock = client_with(lambda request: httpx.Response(200, json={"ret": 200, "data": [row]}))
    snapshot = InfowayDataProvider(client).fetch_history("600123", "statistics", [2025], date(2026, 9, 30))
    assert json.loads(snapshot.blocks[0]["text"])["itemValue"] is None
    assert "currentValue无独立日期" in snapshot.blocks[0]["location"]["value_semantics"]


@pytest.mark.parametrize("business_limit", [False, True])
def test_rate_limit_retries_once_then_stops(business_limit):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200 if business_limit else 429, json={"ret": 429}, headers={"Retry-After": "7"})
    client, clock = client_with(handler)
    with pytest.raises(InfowayApiError, match="PROVIDER_RATE_LIMIT"):
        client.query("600123", "income")
    assert len(calls) == 2 and clock.now >= (5 if business_limit else 7)
    assert not client._cache


def test_long_cooldown_does_not_hold_agent_or_repeat_network_calls():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "120"})
    client, clock = client_with(handler)
    for statement in ["income", "cashflow"]:
        with pytest.raises(InfowayApiError, match="PROVIDER_RATE_LIMIT"):
            client.query("600123", statement)
    assert len(calls) == 1 and clock.now == 0


def test_cancellation_interrupts_rate_wait():
    client, clock = client_with(lambda request: httpx.Response(200, json={"ret": 200, "data": []}))
    client.query("600123", "income")
    def cancel():
        if clock.now >= 0.5:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        client.query("600123", "cashflow", check_cancel=cancel)
    assert clock.now == 0.5


@pytest.mark.parametrize("response,code", [
    (httpx.Response(401, text="credential details"), "PROVIDER_HTTP_401"),
    (httpx.Response(200, text="not JSON"), "PROVIDER_FORMAT"),
    (httpx.Response(200, json={"ret": 403, "msg": "credential details", "data": []}), "PROVIDER_RESPONSE"),
    (httpx.Response(200, json={"ret": 200, "data": {}}), "PROVIDER_RESPONSE"),
    (httpx.Response(200, text="synthetic-infoway-key"), "PROVIDER_SECRET_ECHO"),
    (httpx.Response(302, headers={"Location": "https://untrusted.example/"}), "PROVIDER_HTTP_302"),
])
def test_safe_errors_without_leaking_response_or_credentials(response, code):
    client, clock = client_with(lambda request: response)
    with pytest.raises(InfowayApiError, match=code) as error:
        client.query("600123", "income")
    assert "synthetic-infoway-key" not in str(error.value) and "credential details" not in str(error.value)


def test_shared_history_tool_saves_readable_owned_sources_and_uses_cache(tmp_path):
    runtime = runtime_at(tmp_path)
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ret": 200, "data": [ROW, {**ROW, "periodDate": "2024-12-31"}]})
    client, clock = client_with(handler)
    runtime.service.attach_market(runtime.session.session_id, InfowayDataProvider(client))
    args = FinancialHistoryRequest(years=[2025, 2024], statements=["income"])
    result = fetch_history(runtime, args)
    assert result["provider"] == "infoway" and result["documents"][0]["accepted_records"] == 2
    assert fetch_history(runtime, args)["documents"][0]["cached"]
    assert len(requests) == 1 and not runtime.session.facts
    document = runtime.session.documents[0]
    reading = read_file(runtime.service.store, runtime.session, FileRead(file_id=document.file_id))
    assert reading["blocks"] and "1234567890.1234" in str(reading)
    assert document.sha256 and document.warnings and document.authority_tier == "B"
    assert runtime.service.data_service_status(runtime.session.session_id)["market"]["connection_status"] == "verified"
    narrower = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income"]))
    assert narrower["documents"][0]["accepted_records"] == 1 and len(requests) == 1


def test_peer_acquisition_retains_subject_and_does_not_change_target(tmp_path):
    runtime = runtime_at(tmp_path)
    client, clock = client_with(lambda request: httpx.Response(200, json={"ret": 200, "data": [{**ROW, "symbol": "000999.SZ"}]}))
    runtime.service.history_provider = InfowayDataProvider(client)
    result = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income"], ticker="000999"))
    assert result["documents"][0]["accepted_records"] == 1
    assert runtime.session.documents[0].role == "comparable_financials" and runtime.session.draft.ticker == "600123"


def test_api_configuration_is_memory_only_and_not_a_connection_verification(tmp_path, monkeypatch):
    monkeypatch.delenv("INFOWAY_API_KEY", raising=False)
    app = create_app(tmp_path)
    workspace = app.state.workspaces.create()
    another = app.state.workspaces.create()
    with TestClient(app) as client:
        response = client.post(f"/api/workspaces/{workspace.workspace_id}/data-services", json={"infoway_api_key": "memory-only-test-key"})
        assert response.status_code == 200, response.text
        assert response.json()["market"]["connection_status"] == "configured"
        assert "statistics" in response.json()["market"]["statements"]
        assert not client.get(f"/api/workspaces/{another.workspace_id}/data-services").json()["market"]["available"]
        rejected = client.post(f"/api/workspaces/{workspace.workspace_id}/data-services",
            json={"infoway_api_key": "memory-only-test-key", "tushare_token": "another-secret"})
        assert rejected.status_code == 422
        assert "memory-only-test-key" not in rejected.text and "another-secret" not in rejected.text
    assert "memory-only-test-key" not in json.dumps(app.state.workspaces.snapshot(workspace.workspace_id), ensure_ascii=False)


def test_environment_history_provider_is_not_the_calculator_data_resolver(tmp_path, monkeypatch):
    monkeypatch.setenv("INFOWAY_API_KEY", "environment-only-test-key")
    app = create_app(tmp_path)
    workspace = app.state.workspaces.create()
    assert isinstance(app.state.research.history_provider, InfowayDataProvider)
    assert app.state.runner.data is not app.state.research.history_provider
    assert app.state.research.data_service_status(workspace.research_session_id)["market"]["available"]
