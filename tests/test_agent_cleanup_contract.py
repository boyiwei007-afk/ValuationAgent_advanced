import inspect
import json

import pytest
from pydantic import ValidationError

from test_evidence_recovery import configured_service, item
from test_research_sessions import ScriptedModel
from valuationagent.application.agent_runtime import WorkspaceAgentRuntime
from valuationagent.application.research import ProposeFacts
from valuationagent.application.runner import ValuationRunner
from valuationagent.finance.reference import ReferenceFinancialModel
from valuationagent.schemas.models import RevisionInput, ValuationRequest
from valuationagent.schemas.research import FactCandidate, ResearchTurn
from valuationagent.storage.sqlite import SQLiteRunStore


def test_context_reports_warning_candidates_without_model_authored_counts(tmp_path):
    model = ScriptedModel([
        ("inspect_context", {"section": "overview"}),
        ("finish_response", {"answer": "该字段仍需核验。"}),
    ])
    service, session, _ = configured_service(tmp_path, model)
    session.facts.append(FactCandidate(
        **item(unit="unknown").model_dump(), fact_id="warned_candidate", warnings=["单位待确认"],
    ))
    service.store.save_research(session)
    state = service.turn(session.session_id, ResearchTurn(content="检查候选字段"))
    output = json.loads(next(message["content"] for message in model.calls[-1] if message["role"] == "tool"))
    assert output["fact_counts"] == {"proposed": 1, "confirmed": 0, "rejected": 0}
    assert state["session"]["facts"][0]["warnings"]
    assert not service.store.list_runs()


def test_verified_unit_cannot_be_overwritten_by_incompatible_replacement(tmp_path):
    service, session, _ = configured_service(tmp_path)
    runtime = WorkspaceAgentRuntime(service, session)
    runtime.facts(ProposeFacts(candidates=[item(quote="营业收入 100 90")]))
    original = session.facts[0].model_copy(deep=True)
    with pytest.raises(ValueError):
        runtime.facts(ProposeFacts(
            candidates=[item(quote="营业收入 100 90", unit="元")], replaces=[original.fact_id],
        ))
    assert session.facts == [original]


def test_deterministic_runner_has_no_model_connection_or_live_mode():
    assert "llm" not in inspect.signature(ValuationRunner.create_run).parameters
    assert "llm" not in inspect.signature(ValuationRunner.run).parameters
    assert not hasattr(ValuationRunner, "attach_model")
    assert not hasattr(ValuationRunner, "converse")
    with pytest.raises(ValidationError):
        ValuationRequest(company={"name": "Test"}, valuation_date="2026-09-30", mode="live")


def test_language_revision_preserves_financial_values_after_restart(tmp_path):
    runner = ValuationRunner(SQLiteRunStore(tmp_path), ReferenceFinancialModel())
    original = runner.run(ValuationRequest(
        company={"name": "Synthetic language fixture"}, valuation_date="2026-09-30", mode="demo",
    ))
    changed = runner.revise(original.run_id, RevisionInput(reason="Use English", changes={"language": "en-US"}))
    assert changed.result.dcf == original.result.dcf
    assert changed.result.sensitivity == original.result.sensitivity
    assert changed.result.language == "en-US"
    assert "DCF base" in changed.result.executive_summary
    fresh = SQLiteRunStore(tmp_path)
    assert fresh.get_run(changed.run_id).request.language == "en-US"
    assert fresh.get_run(original.run_id).request.language == "zh-CN"
