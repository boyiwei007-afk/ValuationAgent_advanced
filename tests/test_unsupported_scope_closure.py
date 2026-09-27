from valuationagent.application.valuation_plan import valuation_progress
from valuationagent.schemas.research import FactCandidate, ResearchTurn
from test_evidence_recovery import configured_service
from test_research_sessions import ScriptedModel


def risk_fact(**updates):
    values = dict(fact_id="fixture_risk", metric="受限货币资金", raw_value="10", normalized_value="100000",
                  unit="万元", period="2025", scope="consolidated", block_id="file_test:1",
                  quote="受限货币资金 10 9", status="proposed")
    values.update(updates)
    return FactCandidate(**values)


def test_known_unsupported_method_is_not_hidden_behind_other_missing_fields(tmp_path):
    service, session, _ = configured_service(tmp_path)
    session.facts = [risk_fact(), risk_fact(fact_id="fixture_bad", metric="普通股股数", warnings=["股数单位不明确"])]
    progress = valuation_progress(session, service.valuation_assembler)
    assert progress["status"] == "unsupported_model_scope"
    assert "桥接需专项复核" in progress["blocking_reason"]


def test_other_selected_equity_method_can_still_collect_its_own_inputs(tmp_path):
    service, session, _ = configured_service(tmp_path)
    session.draft.methods = ["dcf", "pe"]
    session.facts = [risk_fact()]
    assert valuation_progress(session, service.valuation_assembler)["status"] == "building_model"


def test_unsupported_scope_stops_tool_loop_with_saved_report(tmp_path):
    model = ScriptedModel([("propose_facts", {"candidates": [{
        "metric": "受限货币资金", "raw_value": "10", "unit": "万元", "period": "2025",
        "scope": "consolidated", "block_id": "file_test:1", "quote": "受限货币资金 10 9",
    }]})])
    service, session, source = configured_service(tmp_path, model)
    source["text"] += "\n受限货币资金 10 9"
    service.store.save_research_blocks(session.session_id, "file_test", [source])
    state = service.turn(session.session_id, ResearchTurn(content="开始自动化DCF估值"))
    assert state["session"]["question"] is None
    assert state["session"]["last_issue"] is None
    assert state["result_document"]["status"] == "insufficient_data"
    assert "桥接需专项复核" in state["session"]["summary"]
    assert state["session"]["facts"][0]["status"] == "proposed"


def test_unverified_or_older_risk_does_not_pretend_to_be_confirmed_current_exposure(tmp_path):
    service, session, _ = configured_service(tmp_path)
    session.facts = [risk_fact(warnings=["来源未通过"])]
    assert valuation_progress(session, service.valuation_assembler)["status"] != "unsupported_model_scope"
    session.facts = [risk_fact(period="2024"), risk_fact(metric="营业收入", raw_value="100", normalized_value="1000000")]
    assert valuation_progress(session, service.valuation_assembler)["status"] != "unsupported_model_scope"
