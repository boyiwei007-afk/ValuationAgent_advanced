from datetime import date
from decimal import Decimal

import pytest

from test_input_calculations import formula, user_rows
from test_input_derivations import raw_values
from test_input_workspace import fixture
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.input_balances import verify_balance_changes
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.result_document import build_result_document, document_sections
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.schemas.inputs import RecordInputs


def test_declared_operating_balances_complete_dcf_without_forcing_cashflow_line_search(tmp_path):
    app, runtime = fixture(tmp_path)
    runtime.session.draft.methods = ["dcf"]
    runtime.session.draft.industry = "电子"
    runtime.session.draft.valuation_date = date(2026, 10, 4)
    for year in range(2022, 2026):
        numbers = raw_values()
        for key in ("inventory_decrease", "operating_receivables_decrease", "operating_payables_increase"):
            numbers.pop(key)
        numbers["revenue"] += Decimal(100 * (year - 2022))
        amounts = [(metric, str(value), "annual") for metric, value in numbers.items()]
        amounts += [("raw.operating_assets", "200", "instant"), ("raw.operating_liabilities", str(400 + 40 * (year - 2022)), "instant"),
            ("cash_and_non_operating_assets", "100", "instant"), ("interest_bearing_debt", "20", "instant"), ("lease_liabilities", "0", "instant")]
        ids = user_rows(app, runtime, amounts, year)
        calculation = formula(ids, "operating_nwc", year, "raw.operating_assets", "raw.operating_liabilities")
        calculation["rationale"] = "合成用户情景经营流动资产减经营流动负债；假设合同项目已全部纳入、没有现金金融投资和带息债务且四年范围相同，不代表现实发行人的完整性或独立审计。"
        record_inputs(runtime, RecordInputs(calculations=[calculation]))
    message = app.state.store.add_message(runtime.session.session_id, "user", "普通股10股，WACC 10%，永续增长2%。", "agent")
    record_inputs(runtime, RecordInputs(user_values=[
        {"metric": "common_shares", "amount_text": "10股", "unit": "股", "scope": "issuer", "message_id": message.message_id},
        {"metric": "wacc", "amount_text": "10%", "unit": "%", "scope": "assumption", "message_id": message.message_id},
        {"metric": "terminal_growth", "amount_text": "2%", "unit": "%", "scope": "assumption", "message_id": message.message_id},
    ]))
    request = prepare_dataset(runtime.session, ["dcf"])
    assert request.financials.change_operating_nwc == -40
    assert request.financials.statement_items["operating_nwc"] == -320
    assert request.historical_financials[0].change_operating_nwc is None
    assert len(request.financials.evidence["change_operating_nwc"]) == 4
    assert len(request.input_calculations) == 4
    assert verify_balance_changes(request)
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = app.state.store.get_run(outcome["run_id"])
    assert record.result.dcf.per_share_value > 0
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    document = build_result_document(app.state.research, runtime.session)
    changes = [row for row in document["input_derivations"] if row["metric"] == "change_operating_nwc"]
    assert len(changes) == 3 and all(row["value"] == "-40" for row in changes)
    assert "并购" in str(document_sections(document))
    assert write_report(app.state.research, runtime.session, ReportWrite(format="md"))["numeric_result_available"]
    assert not runtime.session.search_history
    changed = record.request.model_copy(deep=True)
    changed.financials.calculation_methods["previous_balance_date"] = "2023-12-31"
    with pytest.raises(ValueError, match="INPUT_BALANCE_REPLAY"):
        verify_balance_changes(changed)


def test_explicit_negative_balance_is_instant_and_cannot_mix_flows(tmp_path):
    app, runtime = fixture(tmp_path)
    ids = user_rows(app, runtime, [("operating_nwc", "-100", "instant"), ("net_income_parent", "200", "annual")], 2025)
    assert next(row for row in runtime.session.input_dataset.active_records() if row.input_id == ids["operating_nwc"]).period_kind == "instant"
    user_rows(app, runtime, [("raw.flow", "10", "annual"), ("raw.balance", "20", "instant")], 2025)
    all_ids = {row.metric: row.input_id for row in runtime.session.input_dataset.active_records()}
    with pytest.raises(ValueError, match="INPUT_CALCULATION_KIND"):
        record_inputs(runtime, RecordInputs(calculations=[formula(all_ids, "operating_nwc", 2025, "raw.flow", "raw.balance")]))
