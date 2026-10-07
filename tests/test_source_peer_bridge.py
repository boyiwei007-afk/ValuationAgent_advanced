from datetime import date
from decimal import Decimal

import pytest

from test_input_peers import peer_workspace
from test_multisource_extraction import attach
from test_observation_extraction import review, submit
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.input_calculations import apply_calculations
from valuationagent.application.input_peers import prepare_input_peers
from valuationagent.application.input_peer_bridge import verify_source_peers
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.schemas.inputs import RecordInputs


BASELINE = date(2025, 12, 31)
BALANCE = date(2026, 6, 30)
BRIDGE = {"balance_date": BALANCE, "debt_includes_leases": False, "cash_includes_associates": False,
    "rationale": "合成测试采用同日完整合并资本余额，现金仅含可用于桥接部分；债务不含另列租赁，年度EBITDA已按资本化租赁口径解释。",
    "limitations": ["测试金额不是真实发行人数据，账面代理与期间资本变动风险需在正式研究中复核。"]}


def source_row(runtime, ticker, metric, value, *, period=BASELINE, instant=False, raw=False, role="comparable"):
    text = f"Synthetic {ticker}\nConsolidated\nCNY 元\n{period}\n{metric} {value}"
    block = attach(runtime, f"{ticker}-{metric}-{period}.txt", text, published="2026-09-30")[0]
    capital = instant
    args = {"file_id": block["file_id"], "anchors": {
        key: {"block_id": block["block_id"], "start_line": number}
        for key, number in [("entity", 1), ("scope", 2), ("unit", 3), ("period", 4), ("value", 5)]},
        "basis": {"entity_name": f"Synthetic {ticker}", "entity_ticker": ticker, "entity_refs": ["entity"],
            "scope": "consolidated", "scope_refs": ["scope"], "unit": "元", "currency": "CNY", "unit_refs": ["unit"]},
        "rows": [{"metric": metric, "standard_metric": metric, "raw_value": str(value), "value_ref": "value",
            "label_refs": ["value"], "period_refs": ["period"], "period_kind": "instant" if instant else "annual",
            "period_end": period, "role": role, "semantic_role": "non_operating" if capital else "operating",
            "ebit_treatment": "exclude", "fcff_treatment": "exclude" if capital else "include",
            "equity_bridge_treatment": "include" if capital else "exclude",
            "rationale": "合成测试独立披露主体、合并范围、数值、期间和人民币单位；并非真实LLM复核准确率或真实公司金额。"}]}
    extracted = submit(runtime, args)
    assert extracted["saved_count"] == 1, extracted
    fact = runtime.session.facts[-1]
    review(runtime, [fact.fact_id])
    assert not fact.warnings, fact.warnings
    record_inputs(runtime, RecordInputs(source_values=[{"fact_id": fact.fact_id, "as_raw": raw}]))
    return runtime.session.input_dataset.records[-1]


def bridge_workspace(tmp_path, *, count=1, omitted=(), includes_leases=False):
    app, runtime, _ = peer_workspace(tmp_path, count=count)
    for index in range(count):
        ticker = f"600{101 + index}.SH"
        choice = runtime.session.input_dataset.comparables[ticker]
        record_inputs(runtime, RecordInputs(comparables=[choice.model_dump() | {
            "capital_bridge": BRIDGE | {"debt_includes_leases": includes_leases}}]))
        source_row(runtime, ticker, "ebitda", 1000000)
        for metric, amount in {"cash_and_non_operating_assets": 300000,
            "interest_bearing_debt": 200000, "lease_liabilities": 50000, "minority_interest": 100000}.items():
            if metric not in omitted:
                source_row(runtime, ticker, metric, amount, period=BALANCE, instant=True)
    return app, runtime


@pytest.mark.parametrize("includes_leases,expected", [(False, 10050000), (True, 10000000)])
def test_source_ev_bridge_uses_independent_balance_date_and_leases_once(tmp_path, includes_leases, expected):
    _, runtime = bridge_workspace(tmp_path, includes_leases=includes_leases)
    peers, selected, screening = prepare_input_peers(runtime.session, ["pe", "ev_ebitda"], BASELINE)
    peer = peers[0]
    assert peer.enterprise_value == expected
    assert peer.ev_ebitda == Decimal(expected) / 1000000
    assert peer.pe == 10
    assert peer.capital_bridge["balance_date"] == "2026-06-30"
    assert peer.capital_bridge["unmeasured_items"]
    assert any("不等于零" in warning for warning in peer.capital_bridge["warnings"])
    assert set(screening[0]["methods"]) == {"pe", "ev_ebitda"}
    assert any(row.period_end == BALANCE for row in selected)
    assert len(peer.evidence["ev_ebitda"]) == 6


def test_missing_lease_is_not_zero_and_does_not_block_pe(tmp_path):
    _, runtime = bridge_workspace(tmp_path, omitted={"lease_liabilities"})
    peers, _, screening = prepare_input_peers(runtime.session, ["pe", "ev_ebitda"], BASELINE)
    assert peers[0].pe == 10 and peers[0].ev_ebitda is None
    assert any("lease_liabilities" in reason for reason in screening[0]["reasons"])


def test_missing_bridge_policy_is_an_actionable_selection_not_a_search_request(tmp_path):
    _, runtime = bridge_workspace(tmp_path)
    runtime.session.input_dataset.comparables["600101.SH"].capital_bridge = None
    peers, _, screening = prepare_input_peers(runtime.session, ["pe", "ev_ebitda"], BASELINE)
    assert peers[0].ev_ebitda is None
    assert any("record_inputs.comparables.capital_bridge" in reason for reason in screening[0]["reasons"])


def test_future_bridge_rejected_but_unused_policy_does_not_block_pe(tmp_path):
    _, runtime = bridge_workspace(tmp_path)
    runtime.session.input_dataset.comparables["600101.SH"].capital_bridge.balance_date = date(2026, 10, 1)
    with pytest.raises(ValueError, match="INPUT_PEER_DATE"):
        prepare_input_peers(runtime.session, ["ev_ebitda"], BASELINE)
    assert prepare_input_peers(runtime.session, ["pe"], BASELINE)[0][0].pe == 10


def test_peer_formulas_use_own_sources_and_are_not_applied_to_target(tmp_path):
    _, runtime = bridge_workspace(tmp_path, omitted={"interest_bearing_debt"})
    first = source_row(runtime, "600101.SH", "raw.short_debt", 150000, period=BALANCE, instant=True, raw=True)
    second = source_row(runtime, "600101.SH", "raw.long_debt", 50000, period=BALANCE, instant=True, raw=True)
    formula = {"entity_ticker": "600101.SH", "metric": "interest_bearing_debt", "period_end": BALANCE,
        "terms": [{"input_id": row.input_id, "operation": "add"} for row in [first, second]],
        "debt_includes_leases": False, "rationale": BRIDGE["rationale"], "limitations": BRIDGE["limitations"]}
    record_inputs(runtime, RecordInputs(calculations=[formula]))
    peers, _, screening = prepare_input_peers(runtime.session, ["ev_ebitda"], BASELINE)
    assert peers[0].ev_ebitda == Decimal("10.05")
    assert screening[0]["calculation_ids"]
    assert apply_calculations(runtime.session.input_dataset, BALANCE, {}, {}, {"interest_bearing_debt"})[0] == {}
    formula["entity_ticker"] = ""
    with pytest.raises(ValueError, match="INPUT_CALCULATION_SCOPE"):
        record_inputs(runtime, RecordInputs(calculations=[formula]))
    runtime.session.input_dataset.comparables["600101.SH"].capital_bridge.debt_includes_leases = True
    with pytest.raises(ValueError, match="INPUT_PEER_LEASE"):
        prepare_input_peers(runtime.session, ["ev_ebitda"], BASELINE)


def test_source_peer_ebitda_derives_from_bound_ebit_and_da(tmp_path):
    _, runtime, _ = peer_workspace(tmp_path, count=1)
    source_row(runtime, "600101.SH", "ebit", 800000)
    source_row(runtime, "600101.SH", "depreciation_amortization", 200000)
    source_row(runtime, "600101.SH", "raw.unused", 123, raw=True)
    _, selected, screening = prepare_input_peers(runtime.session, ["ev_ebitda"], BASELINE)
    assert any("capital_bridge" in reason for reason in screening[0]["reasons"])
    assert not any("正值ebitda" in reason for reason in screening[0]["reasons"])
    assert all(not row.metric.startswith("raw.") for row in selected)


def test_ev_report_and_replay_keep_capital_inputs_beyond_financial_baseline(tmp_path):
    from valuationagent.application.result_document import build_result_document, document_sections

    app, runtime = bridge_workspace(tmp_path, count=3)
    first = source_row(runtime, "600101.SH", "raw.short_debt", 150000, period=BALANCE, instant=True, raw=True)
    second = source_row(runtime, "600101.SH", "raw.long_debt", 50000, period=BALANCE, instant=True, raw=True)
    record_inputs(runtime, RecordInputs(calculations=[{"entity_ticker": "600101.SH", "metric": "interest_bearing_debt",
        "period_end": BALANCE, "terms": [{"input_id": row.input_id, "operation": "add"} for row in [first, second]],
        "debt_includes_leases": False, "rationale": BRIDGE["rationale"], "limitations": BRIDGE["limitations"]}]))
    for metric, amount in {"ebitda": 500000, "cash_and_non_operating_assets": 100000,
        "interest_bearing_debt": 0, "lease_liabilities": 0, "minority_interest": 0}.items():
        source_row(runtime, runtime.session.draft.ticker, metric, amount, instant=metric != "ebitda", role="historical")
    runtime.session.draft.methods = ["pe", "ps", "ev_ebitda"]
    request = prepare_dataset(runtime.session, runtime.session.draft.methods)
    assert len([row for row in request.input_records if row["period_end"] == "2026-06-30"]) == 14
    assert len(request.input_calculations) == 1
    verify_source_peers(request)
    for field in ["ev_ebitda", "enterprise_value", "capital_bridge", "evidence"]:
        tampered = request.model_copy(deep=True)
        if field in {"capital_bridge", "evidence"}:
            setattr(tampered.peers[0], field, {})
        else:
            setattr(tampered.peers[0], field, Decimal(1))
        with pytest.raises(ValueError, match="INPUT_PEER_REPLAY"):
            verify_source_peers(tampered)
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = app.state.store.get_run(outcome["run_id"])
    assert {row.method for row in record.result.relative if row.status == "success"} == {"pe", "ps", "ev_ebitda"}
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    text = str(document_sections(build_result_document(app.state.research, runtime.session)))
    assert "可比企业价值资本桥接" in text and "2026-06-30" in text and "10050000" in text


def test_bridge_cannot_borrow_another_period_or_ignore_withdrawn_evidence(tmp_path):
    _, runtime = bridge_workspace(tmp_path)
    fact = next(item for item in runtime.session.facts if item.standard_metric == "lease_liabilities")
    fact.status = "rejected"
    with pytest.raises(ValueError, match="INPUT_SOURCE"):
        prepare_input_peers(runtime.session, ["ev_ebitda"], BASELINE)
    assert prepare_input_peers(runtime.session, ["pe"], BASELINE)[0][0].pe == 10
    fact.status = "confirmed"
    runtime.session.input_dataset.comparables["600101.SH"].capital_bridge.balance_date = date(2025, 12, 31)
    peers, _, screening = prepare_input_peers(runtime.session, ["pe", "ev_ebitda"], BASELINE)
    assert peers[0].ev_ebitda is None
    assert any("2025-12-31显式输入" in reason for reason in screening[0]["reasons"])


def test_cash_investment_overlap_is_adjusted_once_and_negative_not_clamped(tmp_path):
    _, runtime = bridge_workspace(tmp_path)
    source_row(runtime, "600101.SH", "associates_and_non_operating_investments", 80000, period=BALANCE, instant=True)
    assert prepare_input_peers(runtime.session, ["ev_ebitda"], BASELINE)[0][0].enterprise_value == 9970000
    runtime.session.input_dataset.comparables["600101.SH"].capital_bridge.cash_includes_associates = True
    peer = prepare_input_peers(runtime.session, ["ev_ebitda"], BASELINE)[0][0]
    assert peer.enterprise_value == 10050000
    assert peer.capital_bridge["values"]["associates_and_non_operating_investments"] == "80000"
    assert peer.capital_bridge["adjustments"]["associates_and_non_operating_investments"] == "0"
    source_row(runtime, "600101.SH", "preferred_equity", -1, period=BALANCE, instant=True)
    with pytest.raises(ValueError, match="INPUT_PEER_BRIDGE"):
        prepare_input_peers(runtime.session, ["ev_ebitda"], BASELINE)
