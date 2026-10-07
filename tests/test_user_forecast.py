from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from test_input_workspace import fixture
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.forecast_inputs import verify_frozen_forecast_inputs
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.research import ProposeForecast
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.result_document import build_result_document, document_sections
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.schemas.inputs import RecordInputs
from valuationagent.schemas.research import ForecastInputs


def user_scenario(tmp_path):
    text = (Path(__file__).parent / "fixtures/user_manufacturing_scenario.txt").read_text(encoding="utf-8")
    app, runtime = fixture(tmp_path, text)
    runtime.session.draft.company = "测试制造公司A"
    runtime.session.draft.industry = "非金融制造业"
    runtime.session.draft.valuation_date = date(2025, 12, 31)
    runtime.session.draft.methods = ["dcf", "pe", "ps", "ev_ebitda"]
    historical = [("revenue", "8:1", "万元"), ("ebitda", "9:1", "万元"),
        ("depreciation_amortization", "10:1", "万元"), ("ebit", "11:1", "万元"),
        ("tax_rate", "14:2", "%"), ("net_income_parent", "15:1", "万元"),
        ("capital_expenditure", "16:1", "万元"), ("operating_nwc", "17:1", "万元"),
        ("change_operating_nwc", "18:1", "万元")]
    values = [{"metric": metric, "amount_ref": reference, "unit": unit, "period_end": f"{year}-12-31", "period_ref": 7}
        for year in (2023, 2024, 2025) for metric, reference, unit in historical]
    for metric, reference, unit in [("common_shares", "22:1", "万股"), ("diluted_shares", "22:1", "万股"),
        ("cash_and_non_operating_assets", "23:1", "万元"), ("interest_bearing_debt", "24:1", "万元"),
        ("lease_liabilities", "25:1", "万元"), ("minority_interest", "26:1", "万元"),
        ("preferred_equity", "26:1", "万元"), ("unfunded_pension", "26:1", "万元"),
        ("market_price", "28:1", "元")]:
        values.append({"metric": metric, "amount_ref": reference, "unit": unit, "as_of": "2025-12-31"})
    for name, line in zip("ABC", (47, 48, 49)):
        for ordinal, metric, unit in [(1, "revenue", "万元"), (2, "net_income_parent", "万元"),
            (3, "ebitda", "万元"), (4, "common_shares", "万股"), (5, "market_price", "元"),
            *[(6, metric, "万元") for metric in ("cash_and_non_operating_assets", "interest_bearing_debt", "lease_liabilities", "minority_interest")]]:
            value = {"role": "comparable", "entity": "可比" + name, "metric": metric,
                "amount_ref": f"{line}:{ordinal}", "unit": unit}
            if ordinal <= 3:
                value.update(period_end="2025-12-31", period_ref=46)
            else:
                value.update(as_of="2025-12-31")
            values.append(value)
    record_inputs(runtime, RecordInputs(user_basis={"unit_ref": 5, "as_of_ref": 3}, user_values=values))
    return app, runtime


def propose(runtime):
    inputs = ForecastInputs(revenue_growth=[0] * 10, ebit_margin=[Decimal(".2")] * 10,
        wacc=".1", terminal_growth=0, stable_roic=".1", terminal_tax_rate=".25", tax_transition_years=0)
    reference = next(row.input_id for row in runtime.session.input_dataset.active_records() if row.metric == "revenue")
    return runtime.service._propose_forecast(runtime.session, ProposeForecast(inputs=inputs,
        rationale="仅按用户明确提供的合成测试预测进行计算，各年收入与利润率不变，不将结果当作现实投资结论。",
        risks=["人工数据及假设没有经过现实公司来源核验。"], evidence_ids=[reference]))


def test_exact_user_scenario_forecast_calculation_report_and_replay(tmp_path):
    app, runtime = user_scenario(tmp_path)
    staged = propose(runtime)
    assert staged["status"] == "forecast_staged"
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = app.state.store.get_run(outcome["run_id"])
    assert record.request.analysis_basis == "user_scenario"
    assert record.request.mode != "demo"
    assert record.request.discount_policy == "year_end"
    assert record.request.assumptions.stable_roic == Decimal(".1")
    assert abs(record.result.dcf.per_share_value - 16) < Decimal(".000001")
    assert all(row.fcff == 1500000 for row in record.result.forecast)
    assert {row.method: row.per_share_value for row in record.result.relative} == {
        "pe": Decimal("22.5"), "ps": Decimal(15), "ev_ebitda": Decimal(26)}
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    assert write_report(runtime.service, runtime.session, ReportWrite(format="md"))["numeric_result_available"]
    document = build_result_document(runtime.service, runtime.session)
    assert document["forecast_assumptions"]["inputs"]["stable_roic"] == "0.1"
    assert "稳定期ROIC" in str(document_sections(document))
    assert "三情景共享给定的单一收入路径" in str(document_sections(document))
    runtime.session.forecast_proposal.inputs.stable_roic = Decimal(".2")
    assert build_result_document(runtime.service, runtime.session)["forecast_assumptions"]["inputs"]["stable_roic"] == "0.1"
    refs = record.request.assumption_evidence["wacc"]
    assert any(ref.source == "user_input" for ref in refs)
    assert refs[0].source == "forecast_assumption" and "经用户确认" not in refs[0].note
    changed = record.request.model_copy(deep=True)
    changed.input_records = [row for row in changed.input_records if row["input_id"] != refs[1].evidence_id]
    with pytest.raises(ValueError, match="FORECAST_INPUT_REPLAY"):
        verify_frozen_forecast_inputs(changed)


def test_input_readiness_previews_forecast_without_approving_or_calculating(tmp_path):
    from valuationagent.application.research_plan import research_plan

    _, runtime = user_scenario(tmp_path)
    propose(runtime)
    before = runtime.session.model_dump(mode="json")
    plan = research_plan(runtime.session, runtime.service.valuation_assembler)
    assert {item["method"] for item in plan["method_readiness"] if item["status"] == "inputs_ready"} == {"dcf", "pe", "ps", "ev_ebitda"}
    assert runtime.session.model_dump(mode="json") == before
    assert runtime.session.forecast_proposal.status != "confirmed"
    assert not runtime.session.valuation_run_id


def test_superseded_forecast_reference_requires_new_proposal(tmp_path):
    _, runtime = user_scenario(tmp_path)
    propose(runtime)
    reference = runtime.session.forecast_proposal.evidence_ids[0]
    record_inputs(runtime, RecordInputs(user_basis={"unit_ref": 5, "period_ref": 7}, user_values=[{
        "metric": "revenue", "amount_ref": "8:1", "unit": "万元", "period_end": "2023-12-31", "replaces": [reference]}]))
    runtime.session.forecast_proposal.status = "confirmed"
    with pytest.raises(ValueError, match="FORECAST_INPUT_CHANGED"):
        runtime.service.valuation_assembler.build(runtime.session)


@pytest.mark.parametrize("change", [{"revenue_growth": [0] * 9}, {"ebit_margin": [0] * 11},
    {"revenue_growth": [3] * 10}, {"revenue_growth_scenarios": {name: [0] * 10 for name in ("pessimistic", "base", "optimistic")}},
    {"stable_roic": 0}, {"terminal_tax_rate": 2}])
def test_forecast_paths_remain_explicit_and_bounded(change):
    with pytest.raises(ValidationError):
        ForecastInputs.model_validate({"revenue_growth": [0] * 10, "wacc": ".1", "terminal_growth": 0, **change})


def test_forecast_json_schema_exposes_path_lengths():
    path = ForecastInputs.model_json_schema()["properties"]["revenue_growth"]["anyOf"][0]
    assert path["minItems"] == path["maxItems"] == 10


@pytest.mark.parametrize("extra_peers", [0, 80])
def test_offline_forecast_reference_choices_use_existing_target_inputs(tmp_path, extra_peers):
    _, runtime = user_scenario(tmp_path)
    peer = next(row for row in runtime.session.input_dataset.records if row.role == "comparable")
    runtime.session.input_dataset.records.extend(peer.model_copy(update={"input_id": f"peer_padding_{index}"})
        for index in range(extra_peers))
    original = ProposeForecast.model_json_schema()
    _, tools = runtime.adapt_request([], [{"type": "function", "function": {"name": "propose_forecast", "parameters": original}}])
    choices = tools[0]["function"]["parameters"]["properties"]["evidence_ids"]["items"]["enum"]
    assert set(choices) == {row.input_id for row in runtime.session.input_dataset.active_records() if row.role == "historical"}
    assert "enum" not in original["properties"]["evidence_ids"]["items"]
