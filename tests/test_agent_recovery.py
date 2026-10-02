from datetime import date, datetime, timezone
import json
from io import StringIO
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi.testclient import TestClient
from rich.console import Console
from rich.cells import cell_len

from valuationagent.api.main import create_app
from valuationagent.application.agent_runtime import WorkspaceAgentRuntime, RepairFacts
from valuationagent.application.research import CandidateInput, ProposeFacts, ReadDocument, ResearchService, SearchSources
from valuationagent.application.research_plan import research_plan
from valuationagent.cli.ui import workbench, decision_view
from valuationagent.llm.client import LlmError
from valuationagent.schemas.agent import SearchResult
from valuationagent.schemas.research import ResearchTurn, FactCandidate, DocumentSummary
from valuationagent.search.providers import CninfoAnnouncementProvider
from valuationagent.storage.sqlite import SQLiteRunStore
from test_unified_workspace_agent import ScriptedModel


class EmptySearch:
    provider_id = "test-empty"
    version = "1"

    def __init__(self):
        self.calls = []

    def search(self, query):
        self.calls.append(query)
        return SearchResult(query=query, provider=self.provider_id, provider_version=self.version, status="no_results")


def runtime_fixture(tmp_path, methods=None):
    provider = EmptySearch()
    service = ResearchService(SQLiteRunStore(tmp_path), search_provider=provider)
    session = service.create()
    session.draft.company = "样本股份有限公司"
    session.draft.ticker = "600123"
    session.draft.valuation_date = date(2026, 9, 30)
    session.draft.methods = methods or ["pe", "ps"]
    session.data_source_preference = "web"
    return WorkspaceAgentRuntime(service, session), provider


def test_read_advisory_preserves_sources_and_resets_only_on_saved_progress(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    runtime.session.pending_action = "valuation"
    original = {"blocks": [{"text": "样本原文 100"}]}
    for _ in range(5):
        assert runtime.progress_advisory("read_file", original) == original
    advised = runtime.progress_advisory("read_file", original)
    assert advised["blocks"] == original["blocks"]
    assert "progress_advisory" not in original
    assert advised["progress_advisory"]["reads_since_extraction_or_review"] == 6
    assert "不是硬门槛" in advised["progress_advisory"]["instruction"]
    runtime.progress_advisory("extract_observations", {"saved_count": 0})
    assert runtime.read_streak == 6
    runtime.progress_advisory("extract_observations", {"saved_count": 1})
    assert runtime.read_streak == 0
    runtime.read_streak = 6
    runtime.progress_advisory("review_observations", {"reviews": [{"semantic_review": "needs_evidence"}]})
    assert runtime.read_streak == 0
    runtime.session.pending_action = None
    for _ in range(8):
        assert "progress_advisory" not in runtime.progress_advisory("read_file", original)


def test_read_document_exposes_exact_block_local_line_numbers(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    original = "样本股份有限公司\n\n合并利润表\n营业收入 100 90"
    runtime.service.store.save_research_blocks(runtime.session.session_id, "source", [
        {"block_id": "source:1", "text": original, "location": {"page": 75}},
    ])
    result = runtime.read(ReadDocument(file_id="source"))
    block = result["blocks"][0]
    assert block["text"] == original
    assert block["location"]["page"] == 75
    assert block["lines"] == [{"line": number, "text": text} for number, text in enumerate(original.splitlines(), 1)]


def test_repeated_task_updates_stop_without_mutation_and_new_state_can_progress(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    executed = []
    args = '{"draft":{"company":"样本股份有限公司"}}'
    for _ in range(2):
        runtime.call("update_task", args, lambda: executed.append("saved") or {"ok": True})
    with pytest.raises(ValueError, match="REPEATED_TASK_UPDATE"):
        runtime.call("update_task", args, lambda: executed.append("unexpected"))
    assert executed == ["saved", "saved"]
    runtime.session.draft.ticker = "600124"
    runtime.call("update_task", args, lambda: executed.append("new state") or {"ok": True})
    assert executed[-1] == "new state"


def test_search_snippet_compaction_preserves_all_source_references_and_audit():
    from valuationagent.llm.agent import model_tool_result

    original = {"provider": "fixture", "warnings": ["not verified"], "hits": [
        {"file_id": "source1", "url": "https://example.test/report", "published_at": "2026-01-01", "snippet": "source " * 1000},
        {"file_id": "source2", "snippet": "short"},
    ]}
    before = json.dumps(original)
    wire = model_tool_result(original)
    assert len(wire["hits"]) == 2 and wire["warnings"] == original["warnings"]
    assert wire["hits"][0]["url"] == original["hits"][0]["url"]
    assert len(wire["hits"][0]["snippet"]) == 800 and wire["hits"][0]["snippet_truncated"]
    assert wire["hits"][1] == original["hits"][1]
    assert json.dumps(original) == before


def test_scoped_search_limit_does_not_terminate_other_years_or_methods(tmp_path):
    runtime, provider = runtime_fixture(tmp_path)
    for index in range(3):
        runtime.search(SearchSources(query=f"样本 2026 股本公告 来源{index}", reason="核查股数", purpose="company_profile"))
    with pytest.raises(ValueError, match="SEARCH_TARGET_EXHAUSTED"):
        runtime.search(SearchSources(query="样本 2026 股本 第四个来源", reason="核查股数", purpose="company_profile"))
    result = runtime.search(SearchSources(query="同业 FY 倍数", reason="取得同业样本", purpose="comparables"))
    assert result["status"] == "no_results"
    assert not runtime.session.outcome_status
    assert len(provider.calls) == 4


def test_turn_search_budget_is_not_a_permanent_global_lock(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    for year in range(2016, 2024):
        runtime.search(SearchSources(query=f"样本 {year} 资本事项", reason="核查", purpose="company_profile"))
    with pytest.raises(ValueError, match="TURN_SEARCH_BUDGET"):
        runtime.search(SearchSources(query="样本 2024 资本事项", reason="核查", purpose="company_profile"))
    fresh = WorkspaceAgentRuntime(runtime.service, runtime.session)
    assert fresh.search(SearchSources(query="样本 2024 资本事项", reason="核查", purpose="company_profile"))["status"] == "no_results"
    assert not fresh.session.outcome_status


def test_report_range_expands_intermediate_years(tmp_path):
    runtime, _ = runtime_fixture(tmp_path, ["dcf"])
    calls = []

    class Official:
        def search_annual_reports(self, ticker, years, **kwargs):
            calls.append(years)
            return None

    runtime.service.official_search_provider = Official()
    runtime.search(SearchSources(query="样本 2022–2025 年报", reason="补齐历史", purpose="financials"))
    assert calls == [[2022, 2023, 2024, 2025]]
    plan = research_plan(runtime.session, runtime.service.valuation_assembler)
    assert plan["annual_report_years"] == list(range(2025, 2015, -1))
    assert plan["history_policy"]["minimum_automatic_dcf_years"] == 4
    runtime.session.draft.methods = ["pe", "ps"]
    relative = research_plan(runtime.session, runtime.service.valuation_assembler)
    assert relative["annual_report_years"] == [2025, 2024, 2023]
    assert relative["history_policy"]["minimum_automatic_dcf_years"] is None
    assert "ebit_margin" not in relative["required_metrics"]


def test_official_semiannual_query_filters_type_year_and_cutoff():
    requests = []
    def handle(request):
        requests.append(parse_qs(request.content.decode()))
        return httpx.Response(200, json={"announcements": [
            {"secCode": "600123", "announcementId": str(index), "announcementTitle": title,
             "adjunctUrl": f"finalpage/{published}/{index}.PDF",
             "announcementTime": int(datetime.fromisoformat(published).replace(tzinfo=timezone.utc).timestamp() * 1000)}
            for index, (title, published) in enumerate([
                ("2026年半年度报告", "2026-08-20"), ("2025年年度报告", "2026-04-20"),
                ("2026年半年度报告摘要", "2026-08-20"), ("2026年半年度报告（修订）", "2026-10-01"),
            ])]})
    provider = CninfoAnnouncementProvider(transport=httpx.MockTransport(handle))
    result = provider.search_reports("600123", [2026], report_type="semiannual", cutoff=date(2026, 9, 30))
    assert [hit.title for hit in result.hits] == ["2026年半年度报告"]
    assert requests[0]["seDate"] == ["2026-01-01~2026-09-30"]


def test_explicit_official_route_does_not_silently_fall_back_when_purpose_omitted(tmp_path):
    runtime, fallback = runtime_fixture(tmp_path)
    calls = []
    class Official:
        def search_annual_reports(self, ticker, years, **kwargs):
            calls.append((ticker, years))
            return SearchResult(query={"query": "official"}, provider="fixture-official", provider_version="1", status="completed")
    runtime.service.official_search_provider = Official()
    result = runtime.search(SearchSources(query="样本 2024 年年度报告", reason="正式年报", source_route="official_catalogue", report_years=[2024]))
    assert calls == [("600123", [2024])]
    assert result["provider"] == "fixture-official"
    assert not fallback.calls


def test_repair_dated_company_shares_keeps_real_freshness_rule(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    text = "样本股份有限公司 600123\n截至2026年5月31日，公司总股本为\n1,250,081,601股，回购专用账户内的股份数为0股。"
    runtime.service.store.save_research_blocks(runtime.session.session_id, "source", [
        {"block_id": "source:1", "text": text, "location": {"page": 1}},
    ])
    runtime.session.documents.append(DocumentSummary(file_id="source", name="shares.txt", role="evidence", block_count=1))
    previous = FactCandidate(metric="公司总股本", raw_value="1,250,081,601", unit="股", period="截至2026年5月31日",
                             scope="consolidated", block_id="source:1", quote=text, fact_id="repair_shares",
                             warnings=["发行人口径待修复"], normalized_value="1250081601")
    runtime.session.facts = [previous]
    result = runtime.repair_facts(RepairFacts(fact_ids=[previous.fact_id]))
    fixed = next(fact for fact in runtime.session.facts if fact.status == "confirmed")
    assert fixed.scope == "issuer"
    assert fixed.normalized_value == "1250081601"
    assert fixed.verification["period_end"] == "2026-05-31"
    assert previous.status == "rejected"
    runtime.session.facts.append(FactCandidate(metric="revenue", raw_value="100", unit="元", period="2025",
        scope="consolidated", block_id="source:1", quote=text, status="confirmed", normalized_value="100"))
    issue = runtime.service.valuation_assembler.capital_structure_timing_issue(runtime.session)
    assert issue["latest_verified_date"] == "2026-05-31"
    assert issue["age_days"] == 122
    assert issue["required_since"] == "2026-06-02"
    assert "只做一次" not in issue["message"]
    assert result["repairs"]
    from valuationagent.application.requirements import build_requirement_graph
    from valuationagent.schemas.workspace import ValuationWorkspace
    workspace = ValuationWorkspace(workspace_id="workspace_timing", research_session_id=runtime.session.session_id)
    graph = build_requirement_graph(workspace, runtime.session, {"capital_structure": issue})
    shares = next(node for node in graph if node.metric == "common_shares")
    assert shares.status == "pending" and "2026-05-31" in shares.resolution


def test_wrapped_numbered_income_label_can_be_revalidated(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    text = '样本股份有限公司 600123\n合并利润表\n单位：元\n项目                 2025年度  2024年度\n1.归属于母公司股东的净利润 100.00 90.00\n（净亏损以“-”号填列）'
    runtime.service.store.save_research_blocks(runtime.session.session_id, "source", [
        {"block_id": "source:1", "text": text, "location": {"page": 1}},
    ])
    runtime.session.documents.append(DocumentSummary(file_id="source", name="annual.txt", role="evidence", block_count=1))
    candidate = CandidateInput(metric='1.归属于母公司股东的净利润（净亏损以“-”号填列）', raw_value="100.00",
        unit="元", period="2025", scope="consolidated", block_id="source:1", quote=text)
    result = runtime.facts(ProposeFacts(candidates=[candidate]))
    assert result["candidates"][0]["status"] == "confirmed", result


@pytest.mark.parametrize("invalid_first", [False, True])
def test_unverified_value_does_not_invalidate_clean_fact(tmp_path, invalid_first):
    from test_evidence_recovery import configured_service, item
    service, session, _ = configured_service(tmp_path)
    runtime = WorkspaceAgentRuntime(service, session)
    valid = item(quote="营业收入 100 90", unit="万元")
    invalid = valid.model_copy(update={"raw_value": "90"})
    for candidate in [invalid, valid] if invalid_first else [valid, invalid]:
        runtime.facts(ProposeFacts(candidates=[candidate]))
    clean = [fact for fact in session.facts if fact.status == "confirmed"]
    assert len(clean) == 1 and not clean[0].warnings
    assert clean[0].normalized_value == "1000000"
    assert any(fact.status == "proposed" and fact.warnings for fact in session.facts)


def test_conflicting_verified_values_remain_blocked(tmp_path):
    from test_evidence_recovery import configured_service, item, block
    service, session, _ = configured_service(tmp_path)
    runtime = WorkspaceAgentRuntime(service, session)
    runtime.facts(ProposeFacts(candidates=[item(quote="营业收入 100 90", unit="万元")]))
    source = block("样本制造公司600123 合并报表\n单位：万元\n项目 2025年 2024年\n营业收入 110 90")
    service.store.save_research_blocks(session.session_id, "file_test", [source])
    runtime.facts(ProposeFacts(candidates=[item(raw_value="110", quote="营业收入 110 90", unit="万元")]))
    assert len(session.facts) == 2
    assert all(fact.status == "proposed" and any("数值冲突" in warning for warning in fact.warnings)
               for fact in session.facts)


def test_decision_options_and_custom_chat_use_one_turn_path(tmp_path):
    app = create_app(tmp_path)
    model = ScriptedModel(
        ("finish_response", {"answer": "请选择口径，也可以说明其他方案。", "outcome": "needs_input",
                             "decision": {"question": "采用哪个口径？", "options": [
                                 {"label": "合并口径", "description": "明确披露限制"},
                                 {"label": "分部口径", "description": "需要分部输入"}]}}),
        ("finish_response", {"answer": "已记录你的自定义约束。"}),
    )
    service = app.state.workspaces
    workspace = service.create(llm=model)
    service.message(workspace.workspace_id, ResearchTurn(content="先讨论范围"))
    state = service.snapshot(workspace.workspace_id)
    assert len(state["research"]["session"]["pending_decision"]["options"]) == 2
    service.message(workspace.workspace_id, ResearchTurn(content="我有第三种方案，只讨论方法不计算。"))
    state = service.snapshot(workspace.workspace_id)
    assert state["research"]["session"]["pending_decision"] is None
    assert not app.state.store.list_runs()


@pytest.mark.parametrize("code", ["AGENT_STEP_LIMIT", "TOOL_JSON_TRUNCATED", "TOOL_JSON_INVALID", "LLM_CONTEXT_LIMIT"])
def test_step_limit_saves_actionable_checkpoint(tmp_path, code):
    app = create_app(tmp_path)
    class Exhausted:
        def chat(self, *args, **kwargs):
            raise LlmError(code + ": fixture")
    workspace = app.state.workspaces.create(llm=Exhausted())
    app.state.workspaces.message(workspace.workspace_id, ResearchTurn(content="继续"))
    state = app.state.workspaces.snapshot(workspace.workspace_id)
    assert state["research"]["session"]["resume_context"]["reason"] == code
    assert "没有后台任务继续运行" in state["messages"][-1]["content"]


def test_large_parallel_results_keep_latest_body_and_retrievable_source_ids():
    from valuationagent.llm.agent import compact_tool_history

    messages = [{"role": "system", "content": "authority"}, {"role": "assistant", "tool_calls": [
        {"id": "first", "function": {"name": "read_file", "arguments": "{}"}},
        {"id": "last", "function": {"name": "read_file", "arguments": "{}"}},
    ]}, {"role": "tool", "tool_call_id": "first", "content": json.dumps({"blocks": [
        {"file_id": "file_one", "block_id": "file_one:3", "location": {"page": 7}, "text": "original " * 500},
    ]})}, {"role": "tool", "tool_call_id": "last", "content": json.dumps({"text": "keep this complete source"})}]
    before = json.dumps(messages)
    compacted = compact_tool_history(messages, 1500)
    notice = json.loads(compacted[2]["content"])
    assert notice["context_omitted"]
    assert notice["retrieval_references"][0]["block_id"] == "file_one:3"
    assert notice["retrieval_references"][0]["location"]["page"] == 7
    assert compacted[3] == messages[3]
    assert json.dumps(messages) == before


def test_live_state_refresh_preserves_authority_and_tool_transcript():
    from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
    from valuationagent.llm.agent import run_tool_loop

    state = {"completed": False, "permission": "upload_only"}
    seen = []
    class StatefulModel:
        def chat(self, messages, **kwargs):
            seen.append(json.loads(messages[1]["content"]))
            assert messages[0]["content"] == "trusted rules"
            if len(seen) == 1:
                name = "advance"
            else:
                assert any(message.get("role") == "tool" for message in messages)
                name = "done"
            return {"tool_calls": [{"id": name, "function": {"name": name, "arguments": "{}"}}]}
    def advance(_):
        state["completed"] = True
        return {"updated": True}
    registry = ToolRegistry([ToolSpec("advance", "advance", NoArguments, advance),
                             ToolSpec("done", "done", NoArguments, lambda _: {"_terminal": True})])
    run_tool_loop(StatefulModel(), [{"role": "system", "content": "trusted rules"},
                                   {"role": "user", "content": "old state"}], registry,
                  lambda name, arguments, invoke: invoke(), max_rounds=3,
                  state_provider=lambda: dict(state))
    assert seen == [{"completed": False, "permission": "upload_only"},
                    {"completed": True, "permission": "upload_only"}]


def test_shrinking_parameter_errors_are_progress_but_still_bounded():
    from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
    from valuationagent.llm.agent import run_tool_loop
    from valuationagent.schemas.models import ApiModel
    from test_unified_workspace_agent import ScriptedModel

    class Parameters(ApiModel):
        first: int
        second: int
        third: int
        fourth: int
    executed = []
    registry = ToolRegistry([ToolSpec("submit", "submit", Parameters, lambda args: executed.append(args) or {"saved": True}),
                             ToolSpec("finish", "finish", NoArguments, lambda _: {"_terminal": True})])
    steps = [("submit", {key: 1 for key in list(Parameters.model_fields)[:count]}) for count in range(5)]
    model = ScriptedModel(*steps, ("finish", {}))
    result = run_tool_loop(model, [], registry, lambda name, arguments, invoke: invoke(), max_rounds=6)
    assert result["_agent_trace"]["tool_errors"] == 4
    assert len(executed) == 1


def test_incompatible_saved_state_is_preserved_and_explained_without_legacy_runtime(tmp_path):
    app = create_app(tmp_path)
    workspace = app.state.workspaces.create()
    session = app.state.store.get_research(workspace.research_session_id)
    incompatible = session.model_dump(mode="json")
    incompatible["retired_state"] = {"private_field": "must-not-leak"}
    payload = json.dumps(incompatible)
    with app.state.store._connect() as database:
        database.execute("UPDATE research_sessions SET session_json=? WHERE session_id=?", (payload, session.session_id))
    with TestClient(app) as client:
        response = client.get(f"/api/workspaces/{workspace.workspace_id}")
        assert response.status_code == 409
        assert "WORKSPACE_STATE_INCOMPATIBLE" in response.json()["detail"]
        assert "must-not-leak" not in response.text
        assert client.post("/api/workspaces", json={}).status_code == 201
    with app.state.store._connect() as database:
        saved = database.execute("SELECT session_json FROM research_sessions WHERE session_id=?", (session.session_id,)).fetchone()[0]
    assert saved == payload


@pytest.mark.parametrize("width", [40, 70, 112])
def test_workbench_and_decision_fit_terminal(width):
    output = StringIO()
    target = Console(file=output, width=width)
    target.print(workbench({"plan": [{"title": "修复已有证据并补齐年度覆盖", "status": "in_progress"}]}))
    target.print(decision_view({"question": "选择口径或自由补充", "options": [{"label": "合并口径", "description": "保留限制与风险"}, {"label": "分部口径", "description": "核验独立输入"}]}))
    assert all(cell_len(line) <= width for line in output.getvalue().splitlines())
    assert "流程工作台" in output.getvalue()
