import json
from datetime import date
from decimal import Decimal

import httpx
import pytest

from test_provider_inputs import provider_workspace
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.result_document import build_result_document, document_sections
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.market.research_data import FinancialHistoryRequest, fetch_history
from valuationagent.market.tushare import TushareApiClient, TushareDataProvider
from valuationagent.schemas.inputs import RecordInputs


def peer_workspace(tmp_path, *, count=5, changes=None):
    app, runtime, target_args = provider_workspace(tmp_path)
    runtime.session.draft.methods = ["pe", "ps"]
    target_args.user_values = []
    target_args.provider_values.append(target_args.provider_values[0].model_copy(update={"candidate_id": target_args.provider_values[0].candidate_id.replace("n_income_attr_p", "revenue")}))
    record_inputs(runtime, target_args)
    runtime.session.draft.peer_pricing_date = date(2026, 9, 30)
    runtime.session.draft.peer_pricing_rationale = "使用已取得的共同收盘日，保留与估值日三天差异和陈旧性风险"
    tickers = [f"600{index + 101}.SH" for index in range(count)]

    def handler(request):
        payload = json.loads(request.content)
        ticker = payload["params"]["ts_code"]
        multiple = (tickers.index(ticker) + 1) * 10
        record = ({"ts_code": ticker, "ann_date": "20260403", "f_ann_date": "20260403", "end_date": "20251231",
            "report_type": "1", "n_income_attr_p": "1000000", "revenue": "10000000"}
            if payload["api_name"] == "income" else {"ts_code": ticker, "trade_date": "20260930",
                "total_share": "100", "close": str(multiple), "total_mv": str(multiple * 100)})
        record.update((changes or {}).get((ticker, payload["api_name"]), {}))
        if payload["api_name"] == "stock_basic":
            record = {"ts_code": ticker, "name": f"Synthetic peer {tickers.index(ticker)}", "industry": "制造业", "list_date": "20100101"}
        return httpx.Response(200, json={"code": 0, "data": {"fields": list(record), "items": [list(record.values())]}})

    runtime.service.attach_market(runtime.session.session_id,
        TushareDataProvider(TushareApiClient("synthetic-test-secret", transport=httpx.MockTransport(handler))))
    selections = []
    for ticker in tickers:
        response = fetch_history(runtime, FinancialHistoryRequest(years=[2025], statements=["income", "statistics"], ticker=ticker))
        income, market = [doc["file_id"] for doc in response["documents"]]
        args = RecordInputs(comparables=[{"ticker": ticker, "name": ticker,
            "rationale": "同类制造业务作为候选；规模与盈利质量差异需要独立复核"}], provider_values=[
            {"candidate_id": f"{income}@0:n_income_attr_p"},
            {"candidate_id": f"{income}@0:revenue"},
            {"candidate_id": f"{market}@0:total_mv"},
        ])
        record_inputs(runtime, args)
        selections.append(args)
    return app, runtime, selections


def test_peer_inputs_use_frozen_source_numbers_not_old_fact_assembler(tmp_path, monkeypatch):
    app, runtime, _ = peer_workspace(tmp_path)
    monkeypatch.setattr(runtime.service.valuation_assembler, "_peers", lambda *args: pytest.fail("old peer assembler must not run"))
    request = prepare_dataset(runtime.session, ["pe", "ps"])
    assert [peer.pe for peer in request.peers] == list(map(Decimal, [10, 20, 30, 40, 50]))
    assert [peer.ps for peer in request.peers] == list(map(Decimal, [1, 2, 3, 4, 5]))
    assert request.financials.net_income_parent == Decimal("500000.123456789")
    assert request.financials.revenue == Decimal("6000000")
    assert not request.assumptions.relative_multiples
    assert len(request.input_records) == 18
    assert all(peer.financial_period_end == date(2025, 12, 31) and peer.as_of_date == date(2026, 9, 30) for peer in request.peers)
    assert all(peer.pricing_basis == "a_share_equivalent" for peer in request.peers)
    runtime.session.draft.methods = ["pe", "ps"]
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] == "completed_with_warnings", outcome
    record = app.state.store.get_run(outcome["run_id"])
    result = record.result
    prices = {row.method: row.per_share_value for row in result.relative}
    assert prices["pe"] == (Decimal("500000.123456789") * 30 / Decimal("1002500")).quantize(Decimal("0.0001"))
    assert prices["ps"] == (Decimal("6000000") * 3 / Decimal("1002500")).quantize(Decimal("0.0001"))
    assert result.data_quality.result_grade == "C"
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    report = write_report(app.state.research, runtime.session, ReportWrite(format="pdf"))
    assert report["numeric_result_available"]
    document = build_result_document(app.state.research, runtime.session)
    assert len(document["verified_facts"]) == 3
    text = str(document_sections(document))
    assert "可比选择与剔除" in text and "A股价格等值" in text
    assert "600101.SH · 可比公司" in text
    assert "数据质量等级为C" in text or result.data_quality.result_grade == "C"


@pytest.mark.parametrize("changes,excluded", [
    ({("600101.SH", "income"): {"n_income_attr_p": "-1"}}, "PE缺少"),
    ({("600101.SH", "income"): {"end_date": "20241231"}}, "PE缺少"),
    ({("600101.SH", "daily_basic"): {"trade_date": "20260929"}}, "缺少2026-09-30"),
])
def test_one_ineligible_peer_does_not_block_remaining_valid_peers(tmp_path, changes, excluded):
    _, runtime, _ = peer_workspace(tmp_path, changes=changes)
    request = prepare_dataset(runtime.session, ["pe", "ps"])
    outcome = next(row for row in request.peer_screening if row["ticker"] == "600101.SH")
    assert any(excluded in reason for reason in outcome["reasons"])
    assert sum(peer.pe is not None for peer in request.peers) == 4
    if changes.get(("600101.SH", "income"), {}).get("n_income_attr_p"):
        assert sum(peer.ps is not None for peer in request.peers) == 5


def test_selection_can_exclude_without_deleting_sources_or_mutating_previous_request(tmp_path):
    _, runtime, _ = peer_workspace(tmp_path)
    before = prepare_dataset(runtime.session, ["pe"])
    original_records = len(runtime.session.input_dataset.records)
    record_inputs(runtime, RecordInputs(comparables=[{"ticker": "600101", "name": "Excluded peer", "enabled": False,
        "rationale": "本轮业务复核发现产品与市场结构差异，因此明确剔除"}]))
    after = prepare_dataset(runtime.session, ["pe"])
    assert len(before.peers) == 5 and len(after.peers) == 4
    assert len(runtime.session.input_dataset.records) == original_records
    assert after.peer_screening[0]["status"] == "excluded"


def test_peer_cannot_replace_target_or_another_peer(tmp_path):
    _, runtime, selections = peer_workspace(tmp_path)
    old = runtime.session.input_dataset.records[0]
    selections[0].provider_values[0].replaces = [old.input_id]
    with pytest.raises(ValueError, match="INPUT_REPLACEMENT"):
        record_inputs(runtime, selections[0])
    first_peer = next(row for row in runtime.session.input_dataset.records if row.entity_ticker == "600101.SH")
    selections[1].provider_values[0].replaces = [first_peer.input_id]
    with pytest.raises(ValueError, match="INPUT_REPLACEMENT"):
        record_inputs(runtime, selections[1])


def test_source_role_does_not_let_peer_values_become_target_data(tmp_path):
    _, runtime, selections = peer_workspace(tmp_path)
    with pytest.raises(ValueError, match="extra_forbidden"):
        RecordInputs(provider_values=[{**selections[0].provider_values[0].model_dump(), "role": "historical"}])
    target = next(row for row in runtime.session.input_dataset.records if row.role == "historical")
    target.role = "comparable"
    with pytest.raises(ValueError, match="INPUTS_MISSING"):
        prepare_dataset(runtime.session, ["pe"])


@pytest.mark.parametrize("ticker", ["600123", "600123.SH"])
def test_target_cannot_be_selected_as_its_own_peer(tmp_path, ticker):
    _, runtime, _ = provider_workspace(tmp_path)
    with pytest.raises(ValueError, match="INPUT_PEER_SELECTION"):
        record_inputs(runtime, RecordInputs(comparables=[{"ticker": ticker, "name": "Target", "rationale": "不得把目标自身的市场倍数作为独立可比依据"}]))


def test_market_cap_contract_checks_share_count_times_price(tmp_path):
    with pytest.raises(ValueError, match="INPUT_PROVIDER_MARKET_CAP"):
        peer_workspace(tmp_path, changes={("600101.SH", "daily_basic"): {"total_mv": "99999"}})


def test_insufficient_peers_do_not_create_an_assumed_multiple(tmp_path):
    _, runtime, _ = peer_workspace(tmp_path, count=2)
    with pytest.raises(ValueError, match="pe_multiple_or_peers"):
        prepare_dataset(runtime.session, ["pe"])
    assert not any(row.metric.endswith("_multiple") for row in runtime.session.input_dataset.records)


def test_reviewed_document_peers_share_the_input_workspace(tmp_path):
    from test_peer_inputs import peer_component

    _, runtime, _ = peer_workspace(tmp_path, count=4)
    facts = [peer_component(runtime, "market_cap", "20", unit="亿元"),
        peer_component(runtime, "net_income_parent", "10000"),
        peer_component(runtime, "revenue", "50000")]
    record_inputs(runtime, RecordInputs(source_values=[{"fact_id": fact.fact_id} for fact in facts],
        comparables=[{"ticker": "600456", "name": "Document peer",
            "rationale": "文档候选保留独立来源，业务可比性是待进一步复核的选样判断"}]))
    request = prepare_dataset(runtime.session, ["pe", "ps"])
    peer = next(peer for peer in request.peers if peer.ticker == "600456.SH")
    assert (peer.pe, peer.ps, peer.market_cap) == (20, 4, 2000000000)
    assert peer.pricing_basis == "issuer_market_value"
    assert len(request.peers) == 5
    assert len([row for row in request.input_records if row["entity_ticker"] == "600456.SH"]) == 3
    facts[0].status = "rejected"
    with pytest.raises(ValueError, match="INPUT_SOURCE"):
        prepare_dataset(runtime.session, ["pe"])


def test_explicit_user_multiple_does_not_require_unrelated_peer_repair(tmp_path):
    _, runtime, _ = peer_workspace(tmp_path)
    peer = next(row for row in runtime.session.input_dataset.records if row.role == "comparable")
    peer.source.sha256 = "invalid-source"
    record_inputs(runtime, RecordInputs(user_values=[{"metric": "pe_multiple", "amount_text": "20倍",
        "unit": "ratio", "scope": "assumption"}]))
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.assumptions.relative_multiples["pe"] == 20
    assert not request.peers and not request.peer_screening
    assert all(row["role"] == "historical" for row in request.input_records)
