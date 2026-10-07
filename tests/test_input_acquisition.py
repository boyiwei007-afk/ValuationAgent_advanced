import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from test_input_workspace import fixture
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.input_acquisition import AcquireFinancialInputs, acquire_financial_inputs
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.turn_control import guard_tool, resolve_control
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.market.tushare import TushareApiClient, TushareDataProvider
from valuationagent.schemas.control import TurnDecision


def acquisition_workspace(tmp_path, changes=None, duplicate=False):
    app, runtime = fixture(tmp_path)
    session = runtime.session
    session.draft.company = "Synthetic issuer"
    session.draft.ticker = "600123"
    session.draft.industry = "家用电器"
    session.draft.methods = ["pe", "ps"]
    session.draft.valuation_date = date(2026, 10, 3)
    session.information_cutoff_date = date(2026, 10, 3)
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        ticker, statement = payload["params"]["ts_code"], payload["api_name"]
        calls.append((ticker, statement))
        multiple = 50 if ticker == "600123.SH" else (int(ticker[3:6]) - 100) * 10
        if statement == "stock_basic":
            records = [{"ts_code": ticker, "name": "Synthetic issuer" if ticker == "600123.SH" else f"Synthetic peer {int(ticker[3:6]) - 101}",
                "industry": "家用电器", "list_date": "20100101"}]
        elif statement == "income":
            records = [{"ts_code": ticker, "ann_date": f"{year + 1}0403", "f_ann_date": f"{year + 1}0403",
                "end_date": f"{year}1231", "report_type": "1", "revenue": "6000000" if ticker == "600123.SH" else "10000000",
                "n_income_attr_p": "500000" if ticker == "600123.SH" else "1000000"} for year in (2024, 2025)]
        else:
            records = [{"ts_code": ticker, "trade_date": "20260930", "total_share": "100",
                "close": str(multiple), "total_mv": str(multiple * 100)}]
        for record in records:
            record.update((changes or {}).get((ticker, statement), {}))
        if duplicate and ticker == "600123.SH" and statement == "income":
            records.append({**records[-1], "n_income_attr_p": "900000"})
        fields = list(records[0])
        return httpx.Response(200, json={"code": 0, "data": {"fields": fields,
            "items": [[record[field] for field in fields] for record in records]}})

    runtime.service.attach_market(session.session_id,
        TushareDataProvider(TushareApiClient("synthetic-test-secret", transport=httpx.MockTransport(handler))))
    args = AcquireFinancialInputs(years=[2024, 2025], comparables=[{
        "ticker": f"600{index + 101}", "name": f"Synthetic peer {index}",
        "rationale": "用于合成测试的同类制造业务，业务结构差异仍须人工审阅"} for index in range(5)])
    return app, runtime, args, calls


def test_unsupported_peer_error_identifies_the_offending_sample_without_network_calls(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path)
    args.comparables[-1].ticker = "01810.HK"
    args.comparables[-1].name = "港股样本"
    with pytest.raises(ValueError, match="INPUT_PEER_MARKET_UNSUPPORTED.*港股样本.*01810.HK"):
        acquire_financial_inputs(runtime, args)
    assert calls == []
    assert runtime.session.input_dataset is None


def test_one_data_task_admits_target_and_peers_then_calculates_reports_and_replays(tmp_path):
    app, runtime, args, calls = acquisition_workspace(tmp_path)
    result = acquire_financial_inputs(runtime, args)
    assert result["status"] == "inputs_acquired", result
    assert len(calls) == 18 and result["new_input_count"] == 30
    assert result["pricing_date"] == "2026-09-30"
    assert runtime.session.draft.peer_pricing_date == date(2026, 9, 30)
    assert "共同可得日" in runtime.session.draft.peer_pricing_rationale
    assert all(row.source.kind == "provider" and row.assertion == "reported" for row in runtime.session.input_dataset.records)
    assert not runtime.session.facts
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] == "completed_with_warnings", outcome
    record = app.state.store.get_run(outcome["run_id"])
    assert {item.method: item.per_share_value for item in record.result.relative} == {"pe": Decimal(15), "ps": Decimal(18)}
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    report = write_report(app.state.research, runtime.session, ReportWrite(format="pdf"))
    assert report["numeric_result_available"]
    assert app.state.store.get_artifact(runtime.session.session_id, report["artifact_id"])[1].startswith(b"%PDF")


def test_repeat_data_task_is_idempotent_and_does_not_request_api_again(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path)
    first = acquire_financial_inputs(runtime, args)
    second = acquire_financial_inputs(runtime, args)
    assert len(calls) == 18 and second["new_input_count"] == 0
    assert second["active_input_count"] == first["active_input_count"]
    assert all(source["cached"] for source in second["sources"])
    state = runtime.working_state()
    assert len(state["provider_sources"]["sources"]) == 18
    assert state["input_acquisition"]["active_input_count"] == 30


def test_history_years_do_not_silently_limit_the_latest_available_baseline(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path)
    args.years = [2024]
    result = acquire_financial_inputs(runtime, args)
    assert result["years"] == [2024, 2025]
    assert result["requested_history_years"] == [2024]
    assert result["available_annual_periods"]["600123.SH"] == ["2025-12-31", "2024-12-31"]
    assert result["baseline_selection"] == {"policy": "latest_available", "period_end": "2025-12-31"}
    assert len(calls) == 18
    assert {row.period_end.year for row in runtime.session.input_dataset.records if row.period_end} == {2024, 2025}
    assert prepare_dataset(runtime.session, ["pe"]).financials.period_end == date(2025, 12, 31)


def test_unavailable_requested_pricing_date_does_not_lock_future_acquisition(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path)
    result = acquire_financial_inputs(runtime, args.model_copy(update={"pricing_date": date(2026, 10, 3)}))
    assert result["observed_common_dates"] == ["2026-09-30"]
    assert any(issue["code"] == "INPUT_PRICING_DATE_UNAVAILABLE" for issue in result["issues"])
    assert runtime.session.draft.peer_pricing_date is None
    assert not any(row.as_of for row in runtime.session.input_dataset.active_records())
    corrected = acquire_financial_inputs(runtime, args.model_copy(update={"pricing_date": date(2026, 9, 30)}))
    assert len(calls) == 18
    assert corrected["status"] == "inputs_acquired"
    assert runtime.session.draft.peer_pricing_date == date(2026, 9, 30)


def test_missing_market_day_for_one_peer_cannot_be_a_common_day(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path, {("600101.SH", "daily_basic"): {"total_mv": None}})
    result = acquire_financial_inputs(runtime, args)
    assert result["observed_common_dates"] == []
    assert result["pricing_date"] is None
    assert runtime.session.draft.peer_pricing_date is None
    assert any(issue["code"] == "INPUT_NO_COMMON_MARKET_DATE" for issue in result["issues"])


def test_conflicting_raw_records_remain_visible_and_block_calculation(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path, duplicate=True)
    result = acquire_financial_inputs(runtime, args)
    assert result["status"] == "partial"
    assert any(issue["code"] == "INPUT_CONFLICT" for issue in result["issues"])
    profits = [row.value for row in runtime.session.input_dataset.records
        if row.entity_ticker == "600123.SH" and row.metric == "net_income_parent" and row.period_end == date(2025, 12, 31)]
    assert profits == [Decimal(500000), Decimal(900000)]
    with pytest.raises(ValueError, match="INPUT_CONFLICT"):
        prepare_dataset(runtime.session, ["pe"])


def test_missing_peer_profit_does_not_block_ps_or_remaining_pe_peers(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path, {("600101.SH", "income"): {"n_income_attr_p": None}})
    result = acquire_financial_inputs(runtime, args)
    assert result["status"] == "partial"
    request = prepare_dataset(runtime.session, ["pe", "ps"])
    assert sum(peer.pe is not None for peer in request.peers) == 4
    assert sum(peer.ps is not None for peer in request.peers) == 5
    assert not any(row.metric == "net_income_parent" and row.entity_ticker == "600101.SH"
        for row in runtime.session.input_dataset.records)


def test_different_market_dates_are_not_silently_mixed(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path, {("600101.SH", "daily_basic"): {"trade_date": "20260929"}})
    result = acquire_financial_inputs(runtime, args)
    assert result["pricing_date"] is None
    assert any(issue["code"] == "INPUT_NO_COMMON_MARKET_DATE" for issue in result["issues"])
    assert not any(row.as_of for row in runtime.session.input_dataset.records)
    with pytest.raises(ValueError, match="INPUTS_MISSING"):
        prepare_dataset(runtime.session, ["pe"])


def test_explicit_market_date_does_not_roll_forward_or_backfill_missing_peer(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path, {("600101.SH", "daily_basic"): {"trade_date": "20260929"}})
    args.pricing_date = date(2026, 9, 30)
    result = acquire_financial_inputs(runtime, args)
    assert result["pricing_date"] == "2026-09-30"
    assert "部分样本" in runtime.session.draft.peer_pricing_rationale
    assert "共同可得日" not in runtime.session.draft.peer_pricing_rationale
    assert any(issue.get("ticker") == "600101.SH" and issue["code"] == "INPUT_INSTANT_MISSING" for issue in result["issues"])
    assert len(prepare_dataset(runtime.session, ["pe"]).peers) == 4


@pytest.mark.parametrize("change", ["future", "target_peer", "duplicate_peer", "different_date"])
def test_invalid_data_task_has_no_network_or_input_side_effects(tmp_path, change):
    _, runtime, args, calls = acquisition_workspace(tmp_path)
    if change == "future":
        args.years = [2026]
    elif change == "target_peer":
        args.comparables[0].ticker = "600123"
    elif change == "duplicate_peer":
        args.comparables[1].ticker = args.comparables[0].ticker
    else:
        runtime.session.draft.peer_pricing_date = date(2026, 9, 30)
        args.pricing_date = date(2026, 9, 29)
    with pytest.raises(ValueError):
        acquire_financial_inputs(runtime, args)
    assert not calls and runtime.session.input_dataset is None


def test_unsupported_model_fields_are_reported_not_renamed(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path)
    args.metrics = ["ebitda"]
    result = acquire_financial_inputs(runtime, args)
    assert result["status"] == "partial"
    assert all("ebitda" in item["missing"] for item in result["coverage"])
    assert not any(row.metric == "ebitda" for row in runtime.session.input_dataset.records)


def test_tax_request_acquires_raw_dependencies_and_discloses_derived_preview(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path,
        {("600123.SH", "income"): {"total_profit": "400000", "income_tax": "100000"}})
    args.comparables = []
    args.metrics = ["tax_rate"]
    outcome = acquire_financial_inputs(runtime, args)
    assert outcome["status"] == "inputs_acquired", outcome
    assert calls == [("600123.SH", "stock_basic"), ("600123.SH", "income")]
    assert {row.metric for row in runtime.session.input_dataset.active_records()} == {"profit_before_tax", "income_tax_expense"}
    assert all(row["derived_preview"]["tax_rate"]["value"] == "0.25" for row in outcome["coverage"])
    assert all("tax_rate" not in row["admitted"] and not row["missing"] for row in outcome["coverage"])


@pytest.mark.parametrize("profit", [None, "0", "-1"])
def test_tax_request_keeps_missing_and_nonpositive_profit_as_gap(tmp_path, profit):
    _, runtime, args, _ = acquisition_workspace(tmp_path,
        {("600123.SH", "income"): {"total_profit": profit, "income_tax": "100000"}})
    args.comparables = []
    args.metrics = ["tax_rate"]
    outcome = acquire_financial_inputs(runtime, args)
    assert outcome["status"] == "partial"
    assert all("tax_rate" in row["missing"] and not row["derived_preview"] for row in outcome["coverage"])


def test_tampered_cache_is_rejected_before_persisting_a_new_batch(tmp_path):
    app, runtime, args, _ = acquisition_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    before = runtime.session.input_dataset.model_dump()
    source = runtime.session.documents[0]
    Path(app.state.store.get_file(source.file_id)["storage_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        acquire_financial_inputs(runtime, args)
    assert runtime.session.input_dataset.model_dump() == before


def test_acquisition_cannot_bypass_no_network_no_api_or_read_only_permissions(tmp_path):
    app, runtime, args, calls = acquisition_workspace(tmp_path)
    with pytest.raises(ValueError, match="network"):
        runtime.call("acquire_financial_inputs", args.model_dump_json(), lambda: acquire_financial_inputs(runtime, args))
    message = app.state.store.add_message(runtime.session.session_id, "user", "允许联网，但只阅读已有文件", "agent")
    runtime.session.turn_control, runtime.session.execution_permissions = resolve_control(runtime.session, message,
        TurnDecision(summary="只阅读而不采用财务输入", actions=["read"], permission_changes=[
            {"permission": "network", "allowed": True, "user_quote": "允许联网"}]))
    with pytest.raises(ValueError, match="inputs"):
        guard_tool(runtime.session, "acquire_financial_inputs")
    assert not calls and runtime.session.input_dataset is None
