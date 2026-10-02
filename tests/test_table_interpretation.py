import copy

import pytest

from valuationagent.application.agent_runtime import AgentResponse
from valuationagent.application.financial_evidence import FinancialBatch, interpret_table, propose_batch
from valuationagent.core.table_interpretation import TableInterpretation, compile_table
from valuationagent.application.evidence_status import observation_verified
from test_multisource_extraction import attach, runtime_at


def source(runtime, *, public=False, note="附注七", unit="千元"):
    return attach(runtime, "statement.txt", f"样本科技股份有限公司600123\n财务附注中报表的单位为：{unit}\n合并利润表\n{note}    2025 年    2024 年\n营业收入 45 100,000 90,000\n利润总额 10,000 8,000\n利息费用 56 300 200", public=public)[0]


def interpretation(block, **changes):
    anchor = lambda line: {"block_id": block["block_id"], "start_line": line}
    return TableInterpretation.model_validate({
        "file_id": block["block_id"].rsplit(":", 1)[0], "data_block_ids": [block["block_id"]],
        "header": anchor(4), "scope_evidence": anchor(3), "scope": "consolidated",
        "unit_evidence": anchor(2), "unit": "千元", "unit_extent": "financial_statements",
        "columns": [{"period": "2025", "label": "2025 年"}, {"period": "2024", "label": "2024 年"}],
        "rationale": "财务报表统一使用千元，合并利润表的两列按原文顺序对应本年和上年，附注号不是金额。",
        **changes,
    })


def revenue(block, **changes):
    return {"block_id": block["block_id"], "start_line": 5, "metric": "营业收入", "raw_value": "100,000",
            "period": "2025", "scope": "consolidated", "unit": "千元", **changes}


@pytest.mark.parametrize("note", ["附注七", "附注十九", "项目 附注七", "资产 附注十九"])
def test_llm_interpretation_reuses_shared_headers_and_repairs_existing_facts(tmp_path, note):
    runtime = runtime_at(tmp_path)
    block = source(runtime, note=note)
    first = propose_batch(runtime, FinancialBatch(rows=[revenue(block)]))
    assert first["ok"] is False
    assert first["error"]["code"] == "TABLE_BINDING_REQUIRED"
    previous = runtime.session.facts[0]
    assert not observation_verified(previous)
    result = interpret_table(runtime, interpretation(block))
    fact = runtime.session.facts[-1]
    assert fact.status == "confirmed" and not fact.warnings, result
    assert previous.status == "rejected"
    assert fact.normalized_value == "100000000"
    assert fact.verification["year_column"] == 2025
    assert fact.verification["table_interpretation"]["table_id"] == result["table_id"]
    runtime.service.store.save_research(runtime.session)
    restored = runtime.service.store.get_research(runtime.session.session_id)
    assert restored.table_interpretations == runtime.session.table_interpretations
    propose_batch(runtime, FinancialBatch(defaults={"table_id": result["table_id"]}, rows=[revenue(block, raw_value="90,000", period="2024")]))
    assert runtime.session.facts[-1].status == "confirmed"


def test_observation_can_be_verified_while_financing_treatment_needs_review(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    row = revenue(block, start_line=7, metric="利息费用", raw_value="300", standard_metric="interest_expense",
                  semantic_role="unknown", mapping_confidence=.9, mapping_rationale="原文利息费用可核验，但业务归属及是否属于融资利息仍需查附注。")
    propose_batch(runtime, FinancialBatch(rows=[row]))
    old = runtime.session.facts[-1]
    result = interpret_table(runtime, interpretation(block))
    fact = runtime.session.facts[-1]
    assert old.status == "rejected"
    assert observation_verified(fact) and fact.status == "proposed", result
    assert fact.verification["source_assessment"]["binding"] == "verified"
    assert fact.verification["source_assessment"]["admission"] == "blocked"
    assert any("融资利息准入冲突" in warning for warning in fact.warnings)
    plan = runtime.requirements()
    assert plan["evidence_counts"]["observations_verified"] == 1
    assert not plan["table_repairs"]
    assert plan["annual_coverage"][0]["bound_metrics"] == ["利息费用"]
    assert plan["annual_coverage"][0]["model_review_count"] == 1


def test_table_structure_does_not_promote_public_source_or_allow_wrong_year(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime, public=True)
    table = interpret_table(runtime, interpretation(block))
    propose_batch(runtime, FinancialBatch(defaults={"table_id": table["table_id"]}, rows=[revenue(block), revenue(block, period="2024")]))
    correct, wrong_year = runtime.session.facts
    assert observation_verified(correct) and correct.status == "proposed"
    assert correct.verification["source_assessment"]["source_tier"] == "C"
    assert any("PUBLIC_SOURCE" in warning for warning in correct.warnings)
    assert not observation_verified(wrong_year)
    assert any("年度列冲突" in warning for warning in wrong_year.warnings)


@pytest.mark.parametrize("change,code", [
    ({"columns": [{"period": "2024", "label": "2024 年"}, {"period": "2025", "label": "2025 年"}]}, "TABLE_COLUMN_ORDER"),
    ({"columns": [{"period": "2025", "label": "2024 年"}]}, "TABLE_PERIOD_CONFLICT"),
    ({"columns": [{"period": "2025", "label": "2025 年"}]}, "TABLE_COLUMN_COVERAGE"),
    ({"unit": "元"}, "TABLE_UNIT_CONFLICT"),
    ({"scope": "parent"}, "TABLE_SCOPE_AMBIGUOUS"),
])
def test_interpretation_cannot_fabricate_unit_period_scope_or_column_order(tmp_path, change, code):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    with pytest.raises(ValueError, match=code):
        interpret_table(runtime, interpretation(block, **change))
    assert not runtime.session.table_interpretations


def test_cannot_borrow_other_file_headers_or_changed_snapshots(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    other = attach(runtime, "other.txt", "单位：千元")[0]
    with pytest.raises(ValueError, match="TABLE_SOURCE_MISMATCH"):
        interpret_table(runtime, interpretation(block, unit_evidence={"block_id": other["block_id"], "start_line": 1}))
    table = interpret_table(runtime, interpretation(block))
    blocks = runtime.service.store.research_blocks(runtime.session.session_id, block["file_id"])
    blocks[0]["text"] += "\n改变后的原文"
    runtime.service.store.save_research_blocks(runtime.session.session_id, block["file_id"], blocks)
    result = propose_batch(runtime, FinancialBatch(defaults={"table_id": table["table_id"]}, rows=[revenue(block)]))
    assert "TABLE_SOURCE_CHANGED" in str(result)
    assert not runtime.session.facts


def test_multiple_tables_in_one_block_require_explicit_data_boundary(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    block["text"] += "\n母公司利润表\n单位：元\n2025 年 2024 年\n营业收入 9 8"
    blocks = {block["block_id"]: block}
    with pytest.raises(ValueError, match="TABLE_BOUNDARY"):
        compile_table(interpretation(block), blocks)
    bounded = interpretation(block, data_ranges=[{"block_id": block["block_id"], "start_line": 5, "end_line": 7}])
    compiled = compile_table(bounded, blocks)
    assert compiled["data_ranges"][block["block_id"]] == [5, 7]


def test_quarter_and_duplicate_restatement_columns_are_not_annual_values(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    quarter = copy.deepcopy(block)
    quarter["text"] = quarter["text"].replace("2025 年", "2025 年6月30日")
    with pytest.raises(ValueError, match="TABLE_PERIOD_CONFLICT"):
        compile_table(interpretation(quarter, columns=[{"period": "2025", "label": "2025 年6月30日"}, {"period": "2024", "label": "2024 年"}]), {quarter["block_id"]: quarter})
    repeated = copy.deepcopy(block)
    repeated["text"] = repeated["text"].replace("2024 年", "2024 年 2024 年")
    with pytest.raises(ValueError, match="TABLE_COLUMN_AMBIGUOUS"):
        compile_table(interpretation(repeated), {repeated["block_id"]: repeated})


def test_citations_use_stored_page_not_block_suffix(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    blocks = runtime.service.store.research_blocks(runtime.session.session_id, block["file_id"])
    blocks[0]["location"]["page"] = 142
    runtime.service.store.save_research_blocks(runtime.session.session_id, block["file_id"], blocks)
    result = runtime.finish(AgentResponse(answer="已读取原文，尚未计算。", evidence_ids=[block["block_id"]]))
    assert "第142页" in result["answer"]


def test_quarter_header_cannot_be_truncated_to_a_year_label(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    block["text"] = block["text"].replace("2025 年", "2025 年6月30日")
    with pytest.raises(ValueError, match="TABLE_PERIOD_CONFLICT"):
        compile_table(interpretation(block), {block["block_id"]: block})


def test_local_units_override_financial_section_default(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    block["text"] = block["text"].replace("合并利润表", "合并利润表\n单位为：元")
    spec = interpretation(block, header={"block_id": block["block_id"], "start_line": 5})
    with pytest.raises(ValueError, match="TABLE_UNIT_CONFLICT"):
        compile_table(spec, {block["block_id"]: block})


def test_table_unit_may_immediately_precede_its_title_but_not_come_from_another_table(tmp_path):
    runtime = runtime_at(tmp_path)
    block = source(runtime)
    assert compile_table(interpretation(block, unit_extent="table"), {block["block_id"]: block})["unit"] == "千元"
    block["text"] = block["text"].replace("合并利润表", "无关表格数据 10 20\n合并利润表")
    spec = interpretation(block, unit_extent="table", scope_evidence={"block_id": block["block_id"], "start_line": 4},
                          header={"block_id": block["block_id"], "start_line": 5})
    with pytest.raises(ValueError, match="TABLE_UNIT_EXTENT"):
        compile_table(spec, {block["block_id"]: block})


def test_verified_observation_and_table_proof_are_frozen_in_evidence_ledger(tmp_path):
    from valuationagent.application.ledgers import sync_evidence_and_fact_ledgers
    from valuationagent.schemas.workspace import EvidenceRecord, ValuationWorkspace

    runtime = runtime_at(tmp_path)
    block = source(runtime)
    result = interpret_table(runtime, interpretation(block))
    propose_batch(runtime, FinancialBatch(defaults={"table_id": result["table_id"]}, rows=[revenue(block,
        start_line=7, metric="利息费用", raw_value="300", standard_metric="interest_expense", semantic_role="unknown",
        mapping_confidence=.9, mapping_rationale="明确的原文利息费用金额，但归属业务及融资模型处理仍需单独审查。")]))
    workspace = ValuationWorkspace(workspace_id="workspace_table_audit", research_session_id=runtime.session.session_id)
    runtime.service.store.create_workspace(workspace)
    sync_evidence_and_fact_ledgers(runtime.service.store, workspace, runtime.session)
    records = runtime.service.store.list_workspace_records(workspace.workspace_id, "evidence", EvidenceRecord)
    assert records[0].status == "verified" and not records[0].adopted_fact_ids
    assert records[0].binding_proof["table_interpretation"]["table_id"] == result["table_id"]
    assert records[0].binding_proof["table_interpretation"]["anchors"]["unit_evidence"]["text"] == "财务附注中报表的单位为：千元"
    assert runtime.session.facts[-1].status == "proposed"


def test_single_llm_interprets_notes_without_registering_an_old_table(tmp_path):
    from observation_fixtures import ObservationModel, extraction_steps
    from test_observation_extraction import example
    runtime = runtime_at(tmp_path)
    model = ObservationModel(extraction_steps(example(runtime)))
    runtime.run(model)
    assert len(model.calls) == 4 and runtime.session.facts[-1].status == "confirmed"
    assert not runtime.session.table_interpretations

def test_downloading_more_files_does_not_extend_a_common_binding_stall(tmp_path, monkeypatch):
    runtime = runtime_at(tmp_path)
    runtime.session.pending_action = "valuation"
    block = source(runtime)
    propose_batch(runtime, FinancialBatch(rows=[revenue(block), revenue(block, start_line=6, metric="利润总额", raw_value="10,000"),
                                                 revenue(block, start_line=7, metric="利息费用", raw_value="300")]))
    calls = []
    def loop(*args, **kwargs):
        calls.append(True)
        attach(runtime, "extra.txt", "新的年报资料，但没有解决已有表头问题。")
        return runtime.finish(AgentResponse(answer="保存尚未修复的表头问题。", outcome="checkpoint", next_steps=["解释已有表格"] ))
    monkeypatch.setattr("valuationagent.application.agent_runtime.run_tool_loop", loop)
    result = runtime.run(object())
    assert len(calls) == 1 and "本轮已停止" in result["answer"]


def test_common_binding_failures_are_grouped_and_annual_priorities_differ(tmp_path):
    runtime = runtime_at(tmp_path)
    runtime.session.draft.methods = ["dcf", "pe"]
    block = source(runtime)
    propose_batch(runtime, FinancialBatch(rows=[revenue(block), revenue(block, start_line=6, metric="利润总额", raw_value="10,000")]))
    plan = runtime.requirements()
    assert plan["table_repairs"][0]["affected_count"] == 2
    assert len(plan["annual_coverage"]) == 10
    assert plan["annual_coverage"][0]["priority"] == "baseline"
    assert plan["annual_coverage"][-1]["priority"] == "long_term_trend"
    assert "interest_bearing_debt" not in plan["annual_coverage"][-1]["priority_metrics"]
    assert {item["method"] for item in plan["method_readiness"]} == {"dcf", "pe"}
