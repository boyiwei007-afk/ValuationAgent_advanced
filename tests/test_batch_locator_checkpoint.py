import pytest

from valuationagent.api.main import create_app
from valuationagent.application.agent_runtime import WorkspaceAgentRuntime
from valuationagent.application.financial_evidence import FinancialBatch, propose_batch
from valuationagent.application.research import CandidateInput, ProposeFacts
from valuationagent.llm.client import LlmError
from valuationagent.schemas.research import ResearchTurn
from test_multisource_extraction import attach, runtime_at
from test_unified_workspace_agent import ScriptedModel


def statement(runtime):
    return attach(runtime, "annual.txt", "样本科技股份有限公司600123\n合并利润表\n单位：元\n项目 2025年度 2024年度\n营业收入 1,300,029,680.192,071,807,133.32\n三、营业利润（亏损以“－”号填列） 100.00 90.00")[0]


def revenue_row(block, **locator):
    return {"metric": "营业收入", "raw_value": "1,300,029,680.19", "unit": "元", "period": "2025", "scope": "consolidated", "block_id": block["block_id"], **locator}


@pytest.mark.parametrize("locator", ["quote", "line", "both"])
def test_explicit_source_locator_never_silently_reads_first_line(tmp_path, locator):
    runtime = runtime_at(tmp_path)
    block = statement(runtime)
    location = {}
    if locator in {"quote", "both"}:
        location["quote"] = "营业收入 1,300,029,680.192,071,807,133.32"
    if locator in {"line", "both"}:
        location["start_line"] = 5
    result = propose_batch(runtime, FinancialBatch(rows=[revenue_row(block, **location)]))
    assert runtime.session.facts[0].status == "confirmed", result
    assert runtime.session.facts[0].normalized_value == "1300029680.19"
    assert "营业收入" in runtime.session.facts[0].quote


@pytest.mark.parametrize("location", [{}, {"end_line": 5}, {"start_line": 1, "quote": "营业收入 1,300,029,680.192,071,807,133.32"}])
def test_missing_or_conflicting_locator_is_an_actionable_batch_failure(tmp_path, location):
    runtime = runtime_at(tmp_path)
    result = propose_batch(runtime, FinancialBatch(rows=[revenue_row(statement(runtime), **location)]))
    assert result["ok"] is False and result["error"]["code"] == "BATCH_NO_VALID_FACTS"
    assert result["failed_rows"] == 1 and not runtime.session.facts
    assert "候选数值未出现在" not in str(result)


def test_partial_batch_preserves_the_valid_row(tmp_path):
    runtime = runtime_at(tmp_path)
    block = statement(runtime)
    result = propose_batch(runtime, FinancialBatch(rows=[revenue_row(block), revenue_row(block, start_line=5, end_line=5)]))
    assert result.get("ok") is not False
    assert result["failed_rows"] == 1 and len(runtime.session.facts) == 1


def test_failed_batches_count_towards_no_progress_guard(tmp_path):
    from observation_fixtures import fixture_observations, ObservationModel
    runtime = runtime_at(tmp_path)
    payload = fixture_observations(runtime)
    payload["anchors"]["row0_value"]["quote"] = "invented"
    model = ObservationModel([("extract_observations", payload)] * 2)
    with pytest.raises(LlmError, match="AGENT_NO_PROGRESS"):
        runtime.run(model)
    assert len(model.calls) == 2 and not runtime.session.facts

def test_checkpoint_with_real_progress_continues_without_another_user_turn(tmp_path):
    from observation_fixtures import fixture_observations, ObservationModel
    runtime = runtime_at(tmp_path)
    runtime.session.pending_action = "valuation"
    model = ObservationModel([
        ("extract_observations", fixture_observations(runtime)),
        ("finish_response", {"answer": "已保存收入观察，准备复核。", "outcome": "checkpoint", "next_steps": ["复核收入解释"]}),
        ("finish_response", {"answer": "已在第二窗口检查剩余任务；尚未计算。"})])
    result = runtime.run(model)
    assert "第二窗口" in result["answer"] and len(model.calls) == 3
    assert "最后窗口" in model.calls[-1][1]["content"]
    assert sum(event.type == "agent.checkpoint" for event in runtime.service.store.list_events(runtime.session.session_id)) == 1

def test_final_checkpoint_is_saved_and_later_answer_clears_it(tmp_path):
    app = create_app(tmp_path)
    model = ScriptedModel(
        ("finish_response", {"answer": "本轮没有新增可用证据。", "outcome": "checkpoint", "next_steps": ["改查另一年度的公开财务表"]}),
        ("finish_response", {"answer": "仅说明进度，不继续计算。"}),
    )
    service = app.state.workspaces
    workspace = service.create(llm=model)
    service.message(workspace.workspace_id, ResearchTurn(content="检查当前任务进度"))
    state = service.snapshot(workspace.workspace_id)
    assert state["research"]["session"]["resume_context"]["reason"] == "AGENT_CHECKPOINT"
    assert "本轮已停止" in state["messages"][-1]["content"]
    assert len(model.calls) == 1
    service.message(workspace.workspace_id, ResearchTurn(content="只说明进度"))
    assert not service.snapshot(workspace.workspace_id)["research"]["session"]["resume_context"]


def test_checkpoints_never_create_a_third_window(tmp_path):
    from observation_fixtures import fixture_observations, ObservationModel
    runtime = runtime_at(tmp_path)
    runtime.session.pending_action = "valuation"
    model = ObservationModel([
        ("extract_observations", fixture_observations(runtime)),
        ("finish_response", {"answer": "收入已定位待复核。", "outcome": "checkpoint", "next_steps": ["读取比较年度"]}),
        ("extract_observations", fixture_observations(runtime, [{"metric": "revenue", "raw_value": "900", "period": "2024"}])),
        ("finish_response", {"answer": "比较年度已定位待复核。", "outcome": "checkpoint", "next_steps": ["核验利润"]})])
    result = runtime.run(model)
    assert len(model.calls) == 4 and result["_resume_context"]["next_steps"] == ["核验利润"]
    assert len(runtime.session.facts) == 2

@pytest.mark.parametrize("target", ["ebit", "ebitda", "ebit_margin"])
def test_reported_operating_profit_cannot_be_directly_mapped_to_derived_profit(tmp_path, target):
    runtime = runtime_at(tmp_path)
    block = statement(runtime)
    candidate = CandidateInput(metric='三、营业利润（亏损以“－”号填列）', raw_value="100.00", period="2025", unit="元", scope="consolidated",
        block_id=block["block_id"], quote=block["text"].splitlines()[5], standard_metric=target, semantic_role="operating",
        mapping_confidence=.95, mapping_rationale="测试明确的营业利润原始行，不应越过不同利润口径。")
    runtime.facts(ProposeFacts(candidates=[candidate]))
    fact = runtime.session.facts[0]
    assert fact.status == "proposed" and any("利润口径越级" in warning for warning in fact.warnings)
    fact.status, fact.warnings = "confirmed", []
    runtime.session.draft.methods = ["dcf"]
    with pytest.raises(ValueError, match="语义映射须更正"):
        runtime.service.valuation_assembler.build(runtime.session)
    assert runtime.requirements()["repair_candidates"][0]["fact_id"] == fact.fact_id
    runtime.facts(ProposeFacts(candidates=[candidate.model_copy(update={"standard_metric": "operating_profit"})], replaces=[fact.fact_id]))
    assert fact.status == "rejected" and runtime.session.facts[-1].status == "confirmed"


def test_revenue_is_not_relabelled_as_main_business_revenue(tmp_path):
    runtime = runtime_at(tmp_path)
    block = statement(runtime)
    row = revenue_row(block, start_line=5, standard_metric="main_business_revenue", semantic_role="operating",
                      mapping_confidence=.95, mapping_rationale="营业收入和主营业务收入即使数值接近也应保留各自定义。")
    propose_batch(runtime, FinancialBatch(rows=[row]))
    assert any("收入口径冲突" in warning for warning in runtime.session.facts[0].warnings)


def test_repairing_old_confirmed_mapping_counts_as_real_progress(tmp_path):
    from observation_fixtures import fixture_observations, ObservationModel, extraction_steps
    from valuationagent.application.observation_extraction import extract_observations, ExtractObservations
    runtime = runtime_at(tmp_path)
    runtime.session.pending_action = "valuation"
    args = fixture_observations(runtime, [{"metric": "三、营业利润（亏损以“－”号填列）",
        "standard_metric": "ebit", "raw_value": "100.00"}])
    extract_observations(runtime, ExtractObservations.model_validate(args))
    previous = runtime.session.facts[0]
    previous.status, previous.warnings = "confirmed", []
    args["rows"][0].update(standard_metric="operating_profit", replaces=[previous.fact_id])
    model = ObservationModel([*extraction_steps(args),
        ("finish_response", {"answer": "已修正旧口径。", "outcome": "checkpoint", "next_steps": ["补利润总额与融资利息"]}),
        ("finish_response", {"answer": "继续补齐推导依据，尚未计算。"})])
    result = runtime.run(model)
    assert len(model.calls) == 5 and result["outcome"] == "answer"
    assert previous.status == "rejected"
