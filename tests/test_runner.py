from __future__ import annotations

from datetime import date

from valuationagent.application.runner import ValuationRunner
from valuationagent.finance.reference import ReferenceFinancialModel
from valuationagent.schemas.models import (
    AssumptionInputs,
    CompanyInput,
    DataSourceType,
    RunMode,
    RunStatus,
    ValuationRequest,
)
from valuationagent.storage.sqlite import SQLiteRunStore


def make_runner(tmp_path):
    store = SQLiteRunStore(tmp_path / "runtime")
    return store, ValuationRunner(store, ReferenceFinancialModel())


def test_demo_run_produces_audited_result(tmp_path):
    store, runner = make_runner(tmp_path)
    request = ValuationRequest(
        company=CompanyInput(name="演示公司"),
        valuation_date=date(2026, 9, 12),
        mode=RunMode.DEMO,
    )

    record = runner.run(request)

    assert record.status == RunStatus.COMPLETED
    assert record.result is not None
    assert record.result.model_version.endswith("-reference")
    assert record.result.dcf is not None
    assert (
        record.result.dcf.range_low
        < record.result.dcf.per_share_value
        < record.result.dcf.range_high
    )
    assert len(record.result.sensitivity) == 25
    event_types = [event.type for event in store.list_events(record.run_id)]
    assert "tool.started" in event_types
    assert "tool.completed" in event_types
    assert "run.completed" in event_types
    assert len(store.list_messages(record.run_id)) >= 5


def test_invalid_terminal_assumption_waits_for_review(tmp_path):
    store, runner = make_runner(tmp_path)
    request = ValuationRequest(
        company=CompanyInput(name="非法假设样例"),
        valuation_date=date(2026, 9, 12),
        mode=RunMode.DEMO,
        assumptions=AssumptionInputs(wacc="0.03", terminal_growth="0.04"),
    )

    record = runner.run(request)

    assert record.status == RunStatus.WAITING_REVIEW
    assert record.result is None
    assert record.review["code"] == "INVALID_ASSUMPTION"
    assert any(
        event.type == "review.required" for event in store.list_events(record.run_id)
    )


def test_unconfigured_ticker_adapter_is_explicit(tmp_path):
    store, runner = make_runner(tmp_path)
    request = ValuationRequest(
        company=CompanyInput(ticker="600519.SH", name="贵州茅台"),
        valuation_date=date(2026, 9, 12),
        data_source=DataSourceType.TICKER,
        mode=RunMode.SNAPSHOT,
    )

    record = runner.run(request)

    assert record.status == RunStatus.WAITING_REVIEW
    assert record.review["code"] == "DATA_INPUT_UNAVAILABLE"
    assert "TUSHARE_TOKEN" in record.review["message"]
