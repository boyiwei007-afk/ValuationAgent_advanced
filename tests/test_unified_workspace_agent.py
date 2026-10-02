import json
from datetime import date

from fastapi.testclient import TestClient

from valuationagent.api.main import create_app
from valuationagent.application.agent_runtime import CalculateValuation
from valuationagent.application.ledgers import build_model_spec, sync_evidence_and_fact_ledgers
from valuationagent.application.reproducibility import build_valuation_bundle, replay_bundle
from valuationagent.core.tools import canonical
from valuationagent.llm.agent import compact_tool_history
from valuationagent.schemas.models import ValuationRequest
from valuationagent.schemas.research import DocumentSummary, FactCandidate, ResearchTurn
from valuationagent.schemas.workspace import WorkspaceFact


class ScriptedModel:
    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        name, arguments = self.steps.pop(0)
        return {"tool_calls": [{"id": "call_" + str(len(self.calls)),
            "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
            }}]}


def completed_workspace(tmp_path, model=None):
    app = create_app(tmp_path)
    service = app.state.workspaces
    workspace = service.create(llm=model, run_policy="automatic", data_source_preference="web")
    record = service.runner.create_run(ValuationRequest(
        company={"name": "Synthetic fixture", "currency": "CNY"},
        valuation_date=date(2026, 9, 30), data_source="structured",
        assumption_source="automatic", mode="demo", forecast_years=5,
        methods=["dcf", "pe"], user_goal="Explicit synthetic calculation test",
    ))
    spec = build_model_spec(app.state.store, workspace, record, None, service.runner.finance.version)
    app.state.store.save_workspace_record(workspace.workspace_id, "model_spec", spec, immutable=True)
    session = app.state.store.get_research(workspace.research_session_id)
    upload = app.state.store.save_upload("source.txt", "evidence", "text/plain", b"original evidence")
    session.documents = [DocumentSummary(file_id=upload["file_id"], name="source.txt",
        role="evidence", block_count=1, sha256=upload["sha256"])]
    app.state.store.save_research_blocks(session.session_id, upload["file_id"], [
        {"block_id": upload["file_id"] + ":1", "text": "original evidence", "location": {"page": 1}}
    ])
    session.valuation_run_id = record.run_id
    app.state.store.save_research(session)
    app.state.store.freeze_run_sources(record.run_id, session)
    workspace.active_run_id = record.run_id
    app.state.store.save_workspace(workspace)
    service.execute(record.run_id)
    return app, service, workspace, upload


def test_every_turn_uses_same_agent_after_calculation_and_reconnect(tmp_path):
    first = ScriptedModel(("read_valuation", {"section": "summary"}),
                          ("finish_response", {"answer": "已读取确定性结果，不需要再次估值。"}))
    app, service, workspace, _ = completed_workspace(tmp_path, first)
    service.message(workspace.workspace_id, ResearchTurn(content="解释本次结果"))
    snapshot = service.snapshot(workspace.workspace_id)
    assert snapshot["messages"][-1]["content"] == "已读取确定性结果，不需要再次估值。"
    assert len(first.calls) == 2
    assert not any(message.content == "解释本次结果"
        for message in app.state.store.list_messages(workspace.active_run_id))
    second = ScriptedModel(("finish_response", {"answer": "新的模型仍在同一对话中。"}))
    service.research.attach(workspace.research_session_id, second)
    service.message(workspace.workspace_id, ResearchTurn(content="先聊聊方法，不重算"))
    assert service.snapshot(workspace.workspace_id)["messages"][-1]["content"] == "新的模型仍在同一对话中。"
    assert "解释本次结果" in canonical(second.calls)


def test_result_question_can_read_source_without_revising(tmp_path):
    app, service, workspace, upload = completed_workspace(tmp_path)
    model = ScriptedModel(
        ("read_document", {"file_id": upload["file_id"]}),
        ("finish_response", {"answer": "已核对保存的来源。", "evidence_ids": [upload["file_id"] + ":1"]}),
    )
    service.research.attach(workspace.research_session_id, model)
    service.message(workspace.workspace_id, ResearchTurn(content="核对结果引用的原文"))
    assert len(app.state.store.list_runs()) == 1
    assert "original evidence" in canonical(model.calls)
    answer = service.snapshot(workspace.workspace_id)["messages"][-1]["content"]
    assert answer.startswith("已核对保存的来源。")
    assert "来源定位（系统生成）" in answer and "source.txt" in answer


def test_snapshot_get_is_read_only_and_legacy_routes_are_removed(tmp_path):
    app, service, workspace, _ = completed_workspace(tmp_path)
    before = app.state.store.db_path.read_bytes()
    with TestClient(app) as client:
        for _ in range(3):
            assert client.get("/api/workspaces/" + workspace.workspace_id).status_code == 200
        assert client.post("/api/research-sessions", json={}).status_code in {404, 405}
        assert client.post("/api/runs", json={}).status_code in {404, 405}
        assert client.post(f"/api/workspaces/{workspace.workspace_id}/versions/old/activate").status_code in {404, 405}
    assert app.state.store.db_path.read_bytes() == before


def test_recalculation_freezes_inputs_and_sources_before_execute(tmp_path):
    app, service, workspace, upload = completed_workspace(tmp_path)
    child = service.revise(workspace.workspace_id, "test WACC", {"assumptions": {"wacc": "0.08"}})
    snapshot = service.snapshot(workspace.workspace_id)
    assert child.status == "created"
    assert any(item["run_id"] == child.run_id for item in snapshot["model_specs"])
    app.state.store.save_research_blocks(workspace.research_session_id, upload["file_id"], [
        {"block_id": upload["file_id"] + ":1", "text": "later changed source", "location": {}}
    ])
    service.execute(child.run_id)
    bundle = build_valuation_bundle(app.state.store, app.state.store.get_run(child.run_id))
    assert bundle["source_blocks"][upload["file_id"]][0]["text"] == "original evidence"
    assert replay_bundle(bundle)["passed"]


def test_review_policy_never_calculates_agent_parameter_changes(tmp_path):
    app, service, workspace, _ = completed_workspace(tmp_path)
    workspace = service.get(workspace.workspace_id)
    workspace.run_policy = "review"
    app.state.store.save_workspace(workspace)
    result = service.calculate_from_agent(workspace.research_session_id,
        CalculateValuation(changes={"assumptions": {"wacc": "0.08"}}))
    assert result["status"] == "approval_required"
    assert len(app.state.store.list_runs()) == 1


def test_peers_are_not_conflicted_with_each_other(tmp_path):
    app = create_app(tmp_path)
    workspace = app.state.workspaces.create()
    session = app.state.store.get_research(workspace.research_session_id)
    session.facts = [
        FactCandidate(fact_id="fact_" + ticker, metric="pe", raw_value=value,
            normalized_value=value, period="2026-09-30", scope="consolidated",
            role="comparable", peer_ticker=ticker, peer_name=ticker, multiple_basis="TTM",
            block_id="message:fixture", quote="user supplied peers", status="confirmed")
        for ticker, value in [("000001", "12"), ("000002", "18")]
    ]
    sync_evidence_and_fact_ledgers(app.state.store, workspace, session)
    facts = app.state.store.list_workspace_records(workspace.workspace_id, "fact", WorkspaceFact)
    assert {fact.issuer for fact in facts} == {"000001", "000002"}
    assert all(fact.status != "conflicted" for fact in facts)


def test_context_compaction_retains_protocol_pairs_and_bound():
    messages = [{"role": "system", "content": "instructions"}]
    for index in range(8):
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": str(index), "function": {"name": "read_document", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": str(index), "content": "data" * 1200},
        ])
    compact = compact_tool_history(messages, 9000)
    assert len(canonical(compact)) <= 9000
    for index, message in enumerate(compact):
        if message["role"] == "tool":
            assert compact[index - 1]["tool_calls"][0]["id"] == message["tool_call_id"]


def test_free_conversation_does_not_open_a_valuation_or_question_card(tmp_path):
    app = create_app(tmp_path)
    model = ScriptedModel(("finish_response", {"answer": "DCF 是折现未来自由现金流的方法。"}))
    workspace = app.state.workspaces.create(llm=model)
    turn = ResearchTurn(content="解释一下DCF", request_id="request_free_chat")
    app.state.workspaces.message(workspace.workspace_id, turn)
    count = len(app.state.store.list_messages(workspace.research_session_id))
    app.state.workspaces.message(workspace.workspace_id, turn)
    snapshot = app.state.workspaces.snapshot(workspace.workspace_id)
    assert len(app.state.store.list_messages(workspace.research_session_id)) == count
    assert snapshot["active_run"] is None
    assert "question" not in snapshot["research"]["session"]
    assert len(model.calls) == 1
