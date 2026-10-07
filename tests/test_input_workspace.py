from decimal import Decimal

import pytest

from valuationagent.api.main import create_app
from valuationagent.application.agent_runtime import CalculateValuation, WorkspaceAgentRuntime
from valuationagent.application.agent_runtime import AgentResponse
from valuationagent.application.reporting import ValuationReportExporter
from valuationagent.application.result_document import build_result_document, document_sections
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.application.turn_control import resolve_control
from valuationagent.application.workspace_sensitivity import SensitivityRequest, analyze_sensitivity
from valuationagent.application.workspace_artifacts import ReportWrite, write_report
from valuationagent.schemas.control import TurnDecision
from valuationagent.schemas.inputs import RecordInputs


TEXT = "按我给的数值试算，不要联网：归母净利润10亿元，普通股5亿股，PE取20倍。"


def fixture(tmp_path, text=TEXT):
    app = create_app(tmp_path)
    workspace = app.state.workspaces.create(run_policy="automatic", data_source_preference="web")
    session = app.state.store.get_research(workspace.research_session_id)
    message = app.state.store.add_message(session.session_id, "user", text, "agent")
    session.turn_control, session.execution_permissions = resolve_control(session, message,
        TurnDecision(summary="用户明确给数试算", actions=["value", "sensitivity", "report"], permission_changes=[
            {"permission": "network", "allowed": False, "user_quote": "不要联网"}]))
    session.draft.methods = ["pe"]
    session.pending_action = "valuation"
    app.state.store.save_research(session)
    runtime = WorkspaceAgentRuntime(app.state.research, session, app.state.workspaces)
    return app, runtime


def values():
    return RecordInputs(user_values=[
        {"metric": "net_income_parent", "amount_text": "10亿元", "unit": "亿元"},
        {"metric": "common_shares", "amount_text": "5亿股", "unit": "亿股", "scope": "issuer"},
        {"metric": "pe_multiple", "amount_text": "20倍", "unit": "ratio", "scope": "assumption"},
    ])


def test_direct_user_inputs_calculate_report_replay_and_sensitivity(tmp_path):
    app, runtime = fixture(tmp_path)
    class NoNetworkProvider:
        version = "never-network"

        def resolve(self, request, store):
            pytest.fail("Frozen user inputs must not call a configured market provider")

    app.state.research._market_clients[runtime.session.session_id] = NoNetworkProvider()
    result = record_inputs(runtime, values())
    assert len(result["saved_input_ids"]) == 3
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.financials.period_end is None
    assert request.financials.common_shares_as_of is None
    assert request.analysis_basis == "user_scenario" and request.mode != "demo"
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    record = app.state.store.get_run(outcome["run_id"])
    price = record.result.relative[0]
    assert price.per_share_value == Decimal(40)
    assert price.equity_value == Decimal(20000000000)
    assert price.sample_size == 0 and price.range_low is None and price.range_high is None
    assert record.result.analysis_basis == "user_scenario"
    assert "PE_EXPLICIT_PER_SHARE" in record.result.calculation_checks
    artifact = write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    assert artifact["numeric_result_available"]
    response = runtime.finish(AgentResponse(answer="计算和报告已完成", evidence_ids=[artifact["artifact_id"], result["saved_input_ids"][0]]))
    assert response["_terminal"] and "输出文件不是原始证据" in response["answer"]
    assert "未经外部核验" in response["answer"]
    document = build_result_document(app.state.research, runtime.session)
    text = str(document_sections(document))
    assert "None - None" not in text and "至少3家" not in text
    assert document["analysis_basis"] == "user_scenario"
    assert document["methods"][0]["status"] == "calculated"
    assert "仍需完成确认与财务审核" not in text
    assert "低情景输入" in text and "详见正式结果" not in text
    assert "公开资料仍须实际检索" not in text
    exporter = ValuationReportExporter()
    assert exporter.pdf(record).startswith(b"%PDF")
    assert exporter.xlsx(record).startswith(b"PK")
    assert replay_bundle(build_valuation_bundle(app.state.store, record))["passed"]
    trial = analyze_sensitivity(runtime, SensitivityRequest(method="pe", parameter="multiple", values=[15, 25]))
    assert [Decimal(row["per_share_value"]) for row in trial["scenarios"]] == [30, 50]
    metric = analyze_sensitivity(runtime, SensitivityRequest(method="pe", parameter="earnings_scale", values=["0.9", "1.1"]))
    assert [Decimal(row["per_share_value"]) for row in metric["scenarios"]] == [36, 44]
    assert app.state.store.get_run(record.run_id).result == record.result
    assert not runtime.session.search_history and not runtime.session.facts
    from rich.console import Console
    from valuationagent.cli.ui import workbench

    snapshot = app.state.workspaces.snapshot(app.state.store.workspace_for_research(runtime.session.session_id).workspace_id)
    console = Console(record=True, width=85)
    console.print(workbench(snapshot))
    assert "用户输入情景" in console.export_text()


def test_references_cannot_escape_workspace(tmp_path):
    app, runtime = fixture(tmp_path)
    other = app.state.workspaces.create()
    artifact = app.state.store.save_artifact(other.research_session_id, {"filename": "foreign.md"}, b"foreign")
    with pytest.raises(ValueError, match="引用不存在"):
        runtime.finish(AgentResponse(answer="不能引用其他任务的文件", evidence_ids=[artifact["artifact_id"]]))


def test_failed_calculation_cannot_finish_as_updated_valuation(tmp_path):
    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    runtime.session.pending_action = ""
    with pytest.raises(ValueError, match="尚未记录用户估值目标"):
        runtime.calculate(CalculateValuation())
    with pytest.raises(ValueError, match="CALCULATION_NOT_COMPLETED"):
        runtime.finish(AgentResponse(answer="已经按新数据算完了。"))
    response = runtime.finish(AgentResponse(answer="本轮尚未计算，请确认任务。", outcome="needs_input"))
    assert response["_terminal"]
    runtime.session.pending_action = "valuation"
    app.state.store.save_research(runtime.session)
    result = runtime.calculate(CalculateValuation())
    assert result["status"] == "completed_with_warnings"
    write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    assert runtime.finish(AgentResponse(answer="本轮已计算，请查看确定性结果。"))["_terminal"]


def test_input_entity_error_is_actionable_and_task_cannot_bypass_it(tmp_path):
    from valuationagent.application.agent_runtime import TaskUpdate

    _, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    with pytest.raises(ValueError, match="省略company沿用当前主体"):
        record_inputs(runtime, values().model_copy(update={"company": "不同名称"}))
    before = runtime.session.draft.model_dump()
    with pytest.raises(ValueError, match="INPUT_TASK_SCOPE"):
        runtime.update_task(TaskUpdate(draft={"company": "不同名称"}, valuation_requested=False))
    assert runtime.session.draft.model_dump() == before


def test_scenario_report_does_not_promote_unrelated_research_or_old_gaps(tmp_path):
    from valuationagent.schemas.research import FactCandidate
    from test_multisource_extraction import attach

    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    blocks = attach(runtime, "unrelated-research.txt", "旧研究利润999元")
    runtime.session.facts.append(FactCandidate(metric="unrelated_research_profit", raw_value="999", normalized_value="999",
        unit="元", period="2025", scope="consolidated", status="confirmed", block_id=blocks[0]["block_id"], quote="旧研究利润999元"))
    runtime.session.gaps.append("旧DCF缺少十年年报")
    runtime.session.summary = "旧研究结论不能代表本次用户假设"
    app.state.store.save_research(runtime.session)
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}, outcome
    document = build_result_document(app.state.research, runtime.session)
    assert document["numeric_result_available"] and document["analysis_basis"] == "user_scenario"
    assert document["verified_facts"] == [] and document["sources"] == [] and document["gaps"] == []
    assert document["counts"]["verified"] == 0 and document["source_risk_review"] is None
    assert len(document["input_records"]) == 3
    first = write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    cached = write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    assert first["download_url"] == cached["download_url"] and first["artifact_id"] == cached["artifact_id"]
    content = app.state.store.get_artifact(runtime.session.session_id, first["artifact_id"])[1].decode()
    assert "unrelated_research_profit" not in content and "旧DCF缺少十年年报" not in content


def test_record_inputs_is_idempotent_and_does_not_invent_provenance(tmp_path):
    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    assert not record_inputs(runtime, values())["saved_input_ids"]
    assert len(app.state.store.get_research(runtime.session.session_id).input_dataset.records) == 3
    assert all(row.source.kind == "user" for row in runtime.session.input_dataset.records)


@pytest.mark.parametrize("change,match", [
    ({"amount_text": "100亿元"}, "INPUT_SOURCE"),
    ({"unit_quote": "助手说净利润10亿元"}, "INPUT_UNIT"),
    ({"unit": "万元"}, "INPUT_UNIT"),
    ({"message_id": "made_up"}, "INPUT_SOURCE"),
    ({"metric": "common_shares"}, "INPUT_DIMENSION"),
    ({"replaces": ["missing"]}, "INPUT_REPLACEMENT"),
    ({"period_end": "2025-12-31"}, "INPUT_DATE_SOURCE"),
    ({"as_of": "2026-10-03"}, "INPUT_DATE_SOURCE"),
    ({"period_end": "2025-12-31", "period_quote": "2025年度"}, "INPUT_DATE_SOURCE"),
])
def test_invalid_batch_has_no_partial_writes(tmp_path, change, match):
    app, runtime = fixture(tmp_path)
    args = values().model_dump()
    args["user_values"][1] = {**args["user_values"][0], **change}
    with pytest.raises(ValueError, match=match):
        record_inputs(runtime, RecordInputs.model_validate(args))
    assert app.state.store.get_research(runtime.session.session_id).input_dataset is None
    assert runtime.session.input_dataset is None


def test_conflicting_input_needs_explicit_replacement(tmp_path):
    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    old = runtime.session.input_dataset.active_records()[0]
    message = app.state.store.add_message(runtime.session.session_id, "user", "净利润更正为12亿元", "agent")
    amended = RecordInputs(user_values=[{"metric": "net_income_parent", "amount_text": "12亿元",
        "message_id": message.message_id, "unit": "亿元"}])
    record_inputs(runtime, amended)
    with pytest.raises(ValueError, match="INPUT_CONFLICT"):
        prepare_dataset(runtime.session, ["pe"])
    conflicting = [row.input_id for row in runtime.session.input_dataset.active_records() if row.metric == old.metric]
    amended.user_values[0].replaces = conflicting
    record_inputs(runtime, amended)
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.financials.net_income_parent == Decimal(1200000000)
    assert len(runtime.session.input_dataset.records) == 5


def test_dates_require_user_evidence_and_do_not_use_retrieval_date(tmp_path):
    from datetime import date

    text = TEXT + " 这是2025年度利润，股数截至2026年6月30日。"
    app, runtime = fixture(tmp_path, text)
    args = values()
    args.user_values[0].period_end = date(2025, 12, 31)
    args.user_values[0].period_quote = "2025年度"
    args.user_values[1].as_of = date(2026, 6, 30)
    args.user_values[1].as_of_quote = "2026年6月30日"
    record_inputs(runtime, args)
    request = prepare_dataset(runtime.session, ["pe"])
    assert request.financials.period_end == date(2025, 12, 31)
    assert request.financials.common_shares_as_of == date(2026, 6, 30)


def test_file_links_are_resolved_from_actual_owned_artifacts(tmp_path):
    app, runtime = fixture(tmp_path)
    artifact = app.state.store.save_artifact(runtime.session.session_id, {"filename": "report.md"}, b"test report")
    response = runtime.finish(AgentResponse(answer="[下载](sandbox:/mnt/data/report.md)", evidence_ids=[artifact["artifact_id"]]))
    assert "sandbox:" not in response["answer"] and f"/artifacts/{artifact['artifact_id']}" in response["answer"]
    with pytest.raises(ValueError, match="REPORT_LINK_INVALID"):
        runtime.finish(AgentResponse(answer="[虚构报告](sandbox:/mnt/data/missing.pdf)"))


def test_money_display_does_not_lose_an_order_of_magnitude():
    from valuationagent.application.result_views import money

    assert money(20000000000, "CNY") == "200.00亿元"
    assert money(60000000000, "CNY") == "600.00亿元"
    assert money(40, "CNY") == "40.00元"
