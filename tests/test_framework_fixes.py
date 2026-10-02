from __future__ import annotations
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from decimal import Decimal as D
from unittest.mock import patch
import httpx
import pytest
from fastapi.testclient import TestClient
from valuationagent.api.main import create_app
from valuationagent.application.runner import ValuationRunner
from valuationagent.core.data import demo_financials, demo_peers
from valuationagent.finance.reference import ReferenceFinancialModel
from valuationagent.llm.client import (
    LlmError,
    ModelSessionRegistry,
    OpenAICompatibleClient,
)
from valuationagent.schemas.models import (
    ModelConnectionInput,
    RevisionInput,
    ValuationRequest,
)
from valuationagent.storage.sqlite import SQLiteRunStore


def request(**changes):
    data = dict(
        company={"name": "验收公司"},
        valuation_date="2026-09-12",
        mode="snapshot",
        financials=demo_financials(),
        peers=demo_peers(),
    )
    data.update(changes)
    return ValuationRequest(**data)


@pytest.fixture
def runner(tmp_path):
    return ValuationRunner(
        SQLiteRunStore(tmp_path / "runtime"), ReferenceFinancialModel()
    )


def test_sqlite_store_releases_database_handles_after_each_operation(tmp_path):
    data_dir = tmp_path / "disposable-store"
    store = SQLiteRunStore(data_dir)
    store.list_runs()

    # This is the failure mode on Windows when a sqlite3.Connection was used
    # as a transaction context manager but never explicitly closed.
    shutil.rmtree(data_dir)
    assert not data_dir.exists()


def test_default_mode_uses_submitted_data_and_preserves_decimal(runner):
    amount = D("20000000000.123456789")
    data = request(
        financials=demo_financials().model_copy(update={"revenue": amount})
    ).model_dump()
    data.pop("mode")
    r = runner.run(ValuationRequest(**data))
    assert r.request.mode == "snapshot"
    assert r.request.financials.revenue == amount
    assert r.result.forecast[0].revenue == (amount * D("1.08")).quantize(D("0.0001"))
    assert D(json.loads(r.request.model_dump_json())["financials"]["revenue"]) == amount


def test_demo_rejects_real_financials(runner):
    r = runner.run(
        request(
            mode="demo",
            financials=demo_financials().model_copy(
                update={"source_label": "real_report"}
            ),
        )
    )
    assert r.status == "waiting_review" and r.result is None


def test_uploaded_assumptions_cannot_fall_back(runner):
    r = runner.run(request(assumption_source="upload"))
    assert r.status == "waiting_review" and r.result is None


def test_json_file_roles_and_assumptions_are_used(runner):
    historical = runner.store.save_upload(
        "facts.json",
        "historical_financials",
        None,
        json.dumps(
            {
                "financials": demo_financials().model_dump(mode="json"),
                "peers": [p.model_dump(mode="json") for p in demo_peers()],
            }
        ).encode(),
    )
    assumptions = runner.store.save_upload(
        "assumptions.json",
        "assumptions",
        None,
        b'{"wacc":"0.08","terminal_growth":"0.025"}',
    )
    r = runner.run(
        request(
            data_source="upload",
            financials=None,
            peers=[],
            file_ids=[historical["file_id"]],
            assumption_source="upload",
            assumption_file_ids=[assumptions["file_id"]],
        )
    )
    assert r.status == "completed"
    assert r.result.assumptions.wacc == D(".08")
    assert r.result.assumption_evidence["wacc"][0].file_id == assumptions["file_id"]
    assert (
        r.result.effective_financials.evidence["revenue"][0].file_id
        == historical["file_id"]
    )


def test_review_replaces_invalid_assumption_and_preserves_old_pause(runner):
    old = runner.run(request(assumptions={"wacc": ".03", "terminal_growth": ".04"}))
    assert old.status == "waiting_review"
    new = runner.revise(
        old.run_id,
        RevisionInput(reason="纠正折现率", changes={"assumptions": {"wacc": ".095"}}),
    )
    assert new.result is not None
    assert runner.store.get_run(old.run_id).status == "waiting_review"


def test_legal_base_keeps_result_when_optimistic_scenario_invalid(runner):
    r = runner.run(request(assumptions={"wacc": ".04", "terminal_growth": ".03"}))
    assert r.status == "completed_with_warnings"
    assert (
        r.result.dcf.range_low
        <= r.result.dcf.per_share_value
        <= r.result.dcf.range_high
    )
    assert any("乐观" in w for w in r.result.warnings)


def test_no_available_method_waits_for_data(runner):
    r = runner.run(request(methods=["pe"], peers=[]))
    assert r.status == "waiting_review" and r.review["code"] == "NO_VALID_VALUATION"


def test_relative_only_ignores_unused_dcf_constraint(runner):
    r = runner.run(
        request(methods=["pe"], assumptions={"wacc": ".03", "terminal_growth": ".04"})
    )
    assert r.status == "completed" and r.result.forecast == []
    assert r.result.relative[0].status == "success"


def test_protocol_only_plugin_supports_demo(tmp_path):
    ref = ReferenceFinancialModel()

    class Other:
        plugin_id = "other"
        version = "test"
        validate = ref.validate
        resolve_assumptions = ref.resolve_assumptions
        forecast = ref.forecast
        dcf = ref.dcf
        relative = ref.relative
        sensitivity = ref.sensitivity
        reconcile = ref.reconcile

    r = ValuationRunner(SQLiteRunStore(tmp_path), Other()).run(
        request(mode="demo", financials=None)
    )
    assert r.status == "completed"


def test_resume_after_failure_uses_persistent_checkpoints(runner):
    with patch.object(
        runner.finance, "dcf", side_effect=RuntimeError("temporary test failure")
    ):
        old = runner.run(request())
    assert old.status == "failed" and old.result is None
    fresh = ValuationRunner(
        SQLiteRunStore(runner.store.data_dir), ReferenceFinancialModel()
    )
    done = fresh.execute(old.run_id)
    assert done.status == "completed"
    assert any(
        e.type == "tool.cached" and e.tool == "forecast_financials"
        for e in fresh.store.list_events(done.run_id)
    )
    before = len(fresh.store.list_events(done.run_id))
    assert fresh.execute(done.run_id).result == done.result
    assert len(fresh.store.list_events(done.run_id)) == before


def test_concurrent_run_lease(runner):
    started = threading.Event()
    release = threading.Event()
    original = runner.finance.forecast

    def held(*args):
        started.set()
        assert release.wait(10)
        return original(*args)

    record = runner.create_run(request())
    with (
        patch.object(runner.finance, "forecast", side_effect=held),
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        future = pool.submit(runner.execute, record.run_id)
        assert started.wait(10)
        try:
            with pytest.raises(ValueError, match="正在执行"):
                runner.execute(record.run_id)
        finally:
            release.set()
        assert future.result().result is not None


class ToolLLM:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        name, args = next(self.actions)
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{len(self.calls)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        }


def test_unknown_tool_never_executes(runner):
    from valuationagent.application.research import ResearchService
    from valuationagent.schemas.research import ResearchTurn
    llm = ToolLLM([("execute_python", {"code": "bad"}), ("finish_response", {"answer": "不执行任意代码。"})])
    service = ResearchService(runner.store)
    session = service.create(llm=llm)
    result = service.turn(session.session_id, ResearchTurn(content="检查资料"))
    assert result["execution"]["status"] == "completed"
    assert any(event["tool"] == "execute_python" and event["type"] == "tool.failed" for event in result["events"])
    assert not runner.store.list_runs()


def test_provider_errors_are_redacted(runner):
    from valuationagent.application.research import ResearchService
    from valuationagent.schemas.research import ResearchTurn
    sentinel = "SYNTHETIC_FAKE_KEY_DO_NOT_USE"
    config = ModelConnectionInput(model="fake", base_url="https://fake.invalid/v1", api_key=sentinel)
    original = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(401, json={"error": "invalid " + sentinel}))
    service = ResearchService(runner.store)
    session = service.create(llm=OpenAICompatibleClient(config))
    with patch("valuationagent.llm.client.httpx.Client", side_effect=lambda **kwargs: original(transport=transport, **kwargs)):
        result = service.turn(session.session_id, ResearchTurn(content="检查资料"))
    assert sentinel not in json.dumps(result)
    assert result["execution"]["status"] == "failed"
    assert result["session"]["last_issue"]["code"] == "LLM_HTTP_401"


def test_connection_wire_contains_tools_and_session_revoke():
    requests = []

    def handler(req):
        requests.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "test",
                                    "type": "function",
                                    "function": {
                                        "name": "connection_check",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        )

    sessions = ModelSessionRegistry()
    config = ModelConnectionInput(model="fake", api_key="SYNTHETIC_FAKE")
    session = sessions.create(config)
    client = sessions.client(session.session_id)
    original = httpx.Client
    with patch(
        "valuationagent.llm.client.httpx.Client",
        side_effect=lambda **kw: original(transport=httpx.MockTransport(handler), **kw),
    ):
        assert client.test_connection().startswith("OK")
        assert requests[0]["tools"] and requests[0]["tool_choice"] == "required"
        sessions.delete(session.session_id)
        with pytest.raises(LlmError, match="REVOKED"):
            client.complete([{"role": "user", "content": "hello"}])


def test_forecast_years_and_partial_period_policy():
    model = ReferenceFinancialModel()
    req = request()
    assumptions = model.resolve_assumptions(req, req.financials)
    rows = model.forecast(req, req.financials, assumptions)
    assert [r.year for r in rows] == [2026, 2027, 2028, 2029, 2030]
    assert 0 < rows[0].cash_flow_fraction < 1
    assert rows[1].cash_flow_fraction == 1
    assert rows[0].discount_period < rows[1].discount_period
    result = model.dcf(req, req.financials, assumptions, rows)
    # Independent present-value reconstruction from the explicit period contract.
    discount = 1 + assumptions.wacc
    pv = sum(
        r.fcff * r.cash_flow_fraction / (discount**r.discount_period) for r in rows
    )
    tv = (
        rows[-1].fcff
        * (1 + assumptions.terminal_growth)
        / (assumptions.wacc - assumptions.terminal_growth)
    )
    pv += tv / (discount ** (rows[-1].discount_period + D(".5")))
    assert abs(result.enterprise_value - pv) < D(".0001")


@pytest.mark.parametrize(
    "update",
    [
        {"published_at": date(2026, 10, 1)},
        {
            "statement_items": {
                "total_assets": D(100),
                "total_liabilities": D(60),
                "total_equity": D(30),
            }
        },
        {"currency": "USD"},
        {"period_end": date(2023, 12, 31)},
    ],
)
def test_financial_contract_blocks_conflicts(update):
    model = ReferenceFinancialModel()
    req = request()
    financials = req.financials.model_copy(update=update)
    assert any(f.severity == "blocking" for f in model.validate(req, financials))


def test_readonly_run_metadata_and_sse_errors(tmp_path):
    from test_unified_workspace_agent import ScriptedModel, completed_workspace
    app, service, workspace, _ = completed_workspace(tmp_path, ScriptedModel())
    record = app.state.store.get_run(workspace.active_run_id)
    with TestClient(app) as client:
        result = client.get(f"/api/runs/{record.run_id}/results").json()
        assert len(result["effective_input_hash"]) == 64
        assert client.get("/api/runs/absent/events").status_code == 404
        assert client.get(f"/api/runs/{record.run_id}/events", headers={"Last-Event-ID": "bad"}).status_code == 422
        stream = client.get(f"/api/runs/{record.run_id}/events", headers={"Last-Event-ID": "10"})
        ids = [int(line[3:]) for line in stream.text.splitlines() if line.startswith("id:")]
        assert ids and min(ids) > 10 and ids == sorted(set(ids))
        assert client.get(f"/api/runs/{record.run_id}/artifacts").json()


def test_api_validation_does_not_echo_secret(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/api/model-sessions",
            json={"api_key": "SYNTHETIC_SECRET", "model": "", "timeout_seconds": -1},
        )
        assert response.status_code == 422
        assert "SYNTHETIC_SECRET" not in response.text


def test_user_pause_then_resume(runner):
    old_forecast = runner.finance.forecast
    current = runner.create_run(request())

    def pause_after_forecast(*args):
        result = old_forecast(*args)
        runner.request_pause(current.run_id)
        return result

    with patch.object(runner.finance, "forecast", side_effect=pause_after_forecast):
        paused = runner.execute(current.run_id)
    assert paused.status == "waiting_review"
    assert paused.review["code"] == "USER_PAUSED"
    assert paused.result is None
    assert runner.execute(current.run_id).result is not None


def test_expired_lease_can_be_recovered(runner):
    import time
    from valuationagent.schemas.models import RunStatus

    record = runner.create_run(request())
    runner.store.update_run(record.run_id, status=RunStatus.RUNNING)
    with runner.store._connect() as db:
        db.execute(
            "INSERT INTO leases VALUES(?,?,?)",
            (record.run_id, "crashed-worker", time.time() - 1),
        )
    assert runner.execute(record.run_id).status == "completed"


def test_revenue_revision_recomputes_forecast(runner):
    old = runner.run(request())
    new = runner.revise(
        old.run_id,
        RevisionInput(
            reason="调整收入增长",
            changes={"assumptions": {"revenue_growth": [".10"] * 5}},
        ),
    )
    calls = {(e.tool, e.type) for e in runner.store.list_events(new.run_id)}
    assert ("forecast_financials", "tool.started") in calls
    assert ("calculate_relative_valuation", "tool.cached") in calls
    assert new.result.forecast[0].revenue > old.result.forecast[0].revenue
