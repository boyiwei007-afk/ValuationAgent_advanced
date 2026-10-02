import json

import pytest

from valuationagent.api.main import create_app
from valuationagent.application.observation_consistency import retire_contradicted_entities
from valuationagent.application.observation_extraction import REVIEW_UNRESOLVED
from valuationagent.application.research_valuation import ResearchValuationAssembler
from valuationagent.application.valuation_plan import valuation_progress
from valuationagent.application.workspace_artifacts import save_interruption_report
from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
from valuationagent.llm.agent import run_tool_loop
from valuationagent.llm.client import LlmError
from valuationagent.schemas.research import ResearchTurn
from test_multisource_extraction import runtime_at
from test_observation_extraction import example, submit, review


def test_entity_contradiction_retires_interpretation_not_source_or_target(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    submit(runtime, args)
    fact = runtime.session.facts[0]
    checks = {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")}
    checks["entity"] = "contradicted"
    output = review(runtime, checks=checks)
    assert fact.status == "rejected"
    assert output["reviews"][0]["next_action"]["tool"] == "inspect_context"
    assert fact.verification["disposition"]["reason"] == "entity_contradicted"
    assert fact.verification["reading_proof"]["file_id"] == args["file_id"]
    assert runtime.session.draft.ticker == "600123" and fact.role == "historical"
    assert len(runtime.session.documents) == 1
    assert fact not in ResearchValuationAssembler.pending_blockers(runtime.session)


@pytest.mark.parametrize("dimension,verdict", [("entity", "ambiguous"), ("period", "contradicted"), ("amount", "contradicted")])
def test_other_uncertainties_are_not_silently_discarded(tmp_path, dimension, verdict):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    checks = {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")}
    checks[dimension] = verdict
    review(runtime, checks=checks)
    fact = runtime.session.facts[0]
    assert fact.status == "proposed" and REVIEW_UNRESOLVED in fact.warnings
    assert not retire_contradicted_entities(runtime.session.facts)


def test_restored_entity_rejection_is_idempotent_and_requires_an_actual_review(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    fact = runtime.session.facts[0]
    fact.verification["semantic_review"]["checks"] = {"entity": "contradicted"}
    assert not retire_contradicted_entities([fact])
    fact.verification["semantic_review"].update(reviewer="workspace_llm", reviewed_packet_id="review_saved")
    assert retire_contradicted_entities([fact]) == [fact.fact_id]
    assert not retire_contradicted_entities([fact])


def test_status_tools_are_hidden_after_repeated_reads_until_state_changes(tmp_path):
    runtime = runtime_at(tmp_path)
    tool = ToolSpec("inspect_extraction_progress", "status", NoArguments, lambda _: {}).schema()
    finish = ToolSpec("finish_response", "finish", NoArguments, lambda _: {}).schema()
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}]
    runtime.call("inspect_extraction_progress", "{}", lambda: {})
    runtime.call("inspect_extraction_progress", "{  }", lambda: {})
    adapted, tools = runtime.adapt_request(messages, [tool, finish])
    assert adapted is messages
    assert [item["function"]["name"] for item in tools] == ["finish_response"]
    with pytest.raises(ValueError, match="REPEATED_READ"):
        runtime.call("inspect_extraction_progress", "{}", lambda: {})
    runtime.session.draft.valuation_date = None
    assert runtime.adapt_request(messages, [tool, finish])[1] == [tool, finish]


def test_tool_only_adapter_restrictions_are_honored_by_loop():
    registry = ToolRegistry([ToolSpec("inspect_requirements", "status", NoArguments, lambda _: {}),
                             ToolSpec("finish_response", "finish", NoArguments, lambda _: {"_terminal": True})])
    class Model:
        def chat(self, messages, **kwargs):
            assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["finish_response"]
            return {"tool_calls": [{"id": "call_finish", "function": {"name": "finish_response", "arguments": "{}"}}]}
    result = run_tool_loop(Model(), [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"}],
                           registry, lambda name, args, invoke: invoke(), max_rounds=1,
                           request_adapter=lambda messages, tools: (messages, tools[1:]))
    assert result["_terminal"]


def test_preparation_never_recommends_removed_tools(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    text = json.dumps(valuation_progress(runtime.session, ResearchValuationAssembler()), ensure_ascii=False)
    assert "repair_facts" not in text and "propose_facts" not in text
    assert "extract_observations" in text


@pytest.mark.parametrize("code", ["AGENT_NO_PROGRESS", "EXECUTION_TIME_LIMIT", "LLM_HTTP_503"])
def test_failed_valuation_saves_nonnumeric_report_and_original_failure(tmp_path, code):
    class FailedModel:
        def chat(self, *args, **kwargs):
            raise LlmError(code + ": test failure")
    service = create_app(tmp_path).state.workspaces
    workspace = service.create(llm=FailedModel())
    session = service.store.get_research(workspace.research_session_id)
    session.pending_action = "valuation"
    session.draft.company, session.draft.ticker = "Synthetic target", "600123"
    session.draft.methods = ["pe"]
    service.store.save_research(session)
    service.message(workspace.workspace_id, ResearchTurn(content="继续完成估值"))
    snapshot = service.snapshot(workspace.workspace_id)
    assert snapshot["execution"]["status"] == "failed"
    assert snapshot["research"]["session"]["last_issue"]["code"] == code
    assert len(snapshot["artifacts"]) == 1
    artifact, content = service.store.get_artifact(session.session_id, snapshot["artifacts"][0]["artifact_id"])
    assert artifact["kind"] == "interruption_report" and artifact["numeric_result_available"] is False
    assert artifact["valuation_run_id"] is None and "不是数值估值报告" in content.decode("utf-8")
    saved = service.store.get_research(session.session_id)
    document = service.store.research_report(session.session_id)
    again = save_interruption_report(service.research, saved, document, artifact["request_id"])
    assert again["artifact_id"] == artifact["artifact_id"]
    assert len(service.store.list_artifacts(session.session_id)) == 1


@pytest.mark.parametrize("pending,code", [("", "LLM_HTTP_503"), ("valuation", "EXECUTION_CANCELLED")])
def test_free_chat_failure_and_user_cancel_do_not_generate_valuation_artifacts(tmp_path, pending, code):
    class FailedModel:
        def chat(self, *args, **kwargs):
            raise LlmError(code + ": test failure")
    service = create_app(tmp_path).state.workspaces
    workspace = service.create(llm=FailedModel())
    session = service.store.get_research(workspace.research_session_id)
    session.pending_action = pending
    service.store.save_research(session)
    service.message(workspace.workspace_id, ResearchTurn(content="讨论"))
    assert not service.snapshot(workspace.workspace_id)["artifacts"]


def test_worklist_distinguishes_current_blockers_from_unneeded_candidate_repairs(tmp_path):
    from valuationagent.application.research_plan import research_plan
    from test_peer_inputs import peer_component

    runtime = runtime_at(tmp_path)
    runtime.session.draft.methods = ["pe"]
    submit(runtime, example(runtime))
    peer_component(runtime, "market_cap", "200000")
    peer_component(runtime, "net_income_parent", "10000")
    plan = research_plan(runtime.session, ResearchValuationAssembler())
    assert plan["repair_candidates"] and not plan["repair_candidates"][0]["blocks_selected_methods"]
    assert not any(item["kind"] == "repair" for item in plan["next_work"])
    assert plan["peer_coverage"][0]["admitted_peer_count"] == 1
    assert plan["peer_coverage"][0]["remaining"] == 2
    assert any(item["kind"] == "comparable_inputs" for item in plan["next_work"])


def test_duplicate_observations_do_not_reset_progress_or_count_as_new(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    assert submit(runtime, args)["saved_count"] == 1
    duplicate = submit(runtime, args)
    assert duplicate["ok"] and duplicate["saved_count"] == 0 and duplicate["duplicate_count"] == 1
    runtime.read_streak = 7
    runtime.progress_advisory("extract_observations", duplicate)
    assert runtime.read_streak == 7
    runtime.session.facts[0].status = "rejected"
    rejected = submit(runtime, args)
    assert not rejected["ok"] and rejected["saved_count"] == 0 and len(runtime.session.facts) == 1


def test_resolved_recovery_does_not_discard_audit_or_unbound_failure(tmp_path):
    from valuationagent.application.extraction_recovery import record_attempt, recovery_plan

    runtime = runtime_at(tmp_path)
    args = example(runtime)
    submit(runtime, args)
    fact = runtime.session.facts[0]
    result = review(runtime, checks={key: "ambiguous" for key in ("entity", "amount", "period", "unit", "scope", "mapping")})
    record_attempt(runtime.session, "review_observations", {"reviews": [{"fact_id": fact.fact_id}]}, result)
    assert recovery_plan(runtime.session)["files"]
    review(runtime)
    assert not recovery_plan(runtime.session)["files"]
    assert len(runtime.session.reading_attempts) == 1
    record_attempt(runtime.session, "extract_observations", args, {"ok": False, "rows": [{"error": "AMOUNT_NOT_FOUND"}]})
    plan = recovery_plan(runtime.session)["files"][0]
    assert plan["failure_count"] == 1 and plan["historical_failure_count"] == 2


def test_multifile_review_recovery_is_associated_with_each_failed_fact(tmp_path):
    from valuationagent.application.extraction_recovery import record_attempt, recovery_plan

    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    submit(runtime, example(runtime, name="prior.txt", year="2024"))
    selected = [fact.fact_id for fact in runtime.session.facts]
    result = review(runtime, checks={key: "ambiguous" for key in ("entity", "amount", "period", "unit", "scope", "mapping")})
    record_attempt(runtime.session, "review_observations", {"reviews": [{"fact_id": fact_id} for fact_id in selected]}, result)
    assert len(recovery_plan(runtime.session)["files"]) == 2
    runtime.session.facts[0].status = "rejected"
    remaining = recovery_plan(runtime.session)["files"]
    assert len(remaining) == 1
    assert remaining[0]["file_id"] == runtime.session.facts[1].block_id.rsplit(":", 1)[0]
    assert len(runtime.session.reading_attempts) == 2


def test_old_entity_review_failure_is_not_recommended_after_retirement(tmp_path):
    from valuationagent.application.extraction_recovery import record_attempt, recovery_plan

    runtime = runtime_at(tmp_path)
    args = example(runtime)
    submit(runtime, args)
    result = review(runtime, checks={key: "contradicted" for key in ("entity", "amount", "period", "unit", "scope", "mapping")})
    record_attempt(runtime.session, "review_observations", {"reviews": [{"fact_id": runtime.session.facts[0].fact_id}]}, result)
    attempt = runtime.session.reading_attempts[0]
    attempt.pop("failed_fact_ids")
    attempt.pop("unbound_failure")
    assert not recovery_plan(runtime.session)["files"]
    assert attempt["status"] == "failed"


def test_unknown_metric_is_not_given_invented_period_or_scope_requirements(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    args["rows"][0].update(standard_metric="unrecognized_metric", period_kind="instant", period_start=None)
    args["basis"]["scope"] = "issuer"
    result = submit(runtime, args)
    warnings = result["rows"][0]["warnings"]
    assert any(warning.startswith("MODEL_METRIC_UNKNOWN") for warning in warnings)
    assert not any(warning.startswith(("MODEL_PERIOD_KIND", "MODEL_SCOPE", "MODEL_DIMENSION")) for warning in warnings)
    assert runtime.session.facts[0].status == "proposed"
