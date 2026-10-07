import copy
import json

import pytest

from test_context_budget_state import workspace_state
from valuationagent.core.tools import canonical
from valuationagent.llm.agent import compact_tool_history, research_checkpoint
from valuationagent.llm.client import LlmError


def action_state():
    state = workspace_state()
    state["context"]["task_state"]["turn_control"] = {"effects": ["inputs", "calculate"], "run_policy": "review"}
    state["research_plan"] = {
        "methods": ["pe", "ps", "ev_ebitda"],
        "method_readiness": [
            {"method": "pe", "status": "inputs_ready", "reason": "已保存输入可计算"},
            {"method": "ps", "status": "inputs_ready", "reason": "已保存输入可计算"},
            {"method": "ev_ebitda", "status": "blocked", "reason": "缺少完整现金口径解释"},
        ],
        "method_completion": {"requested_methods": ["pe", "ps", "ev_ebitda"],
            "run_id": None, "completed_in_run": [], "current_inputs_match": False,
            "completed_methods": [], "remaining_methods": {"pe": "尚未计算", "ps": "尚未计算", "ev_ebitda": "缺少完整现金口径解释"},
            "all_requested_methods_completed": False},
        "next_work": [{"kind": "calculate", "methods": ["pe", "ps"], "tool": "calculate_valuation",
            "instruction": "输入就绪不代表获得批准，review模式仍须用户批准。"},
            {"method": "ev_ebitda", "status": "blocked", "reason": "缺少完整现金口径解释"}],
        "annual_coverage": [{"year": 2025, "diagnostic": "可检索诊断数据 " * 8000}],
    }
    return state


def compact(state, budget=4200):
    messages = [{"role": "system", "content": "authoritative policy"}, {"role": "user", "content": canonical(state)}]
    before = copy.deepcopy(messages)
    result = compact_tool_history(messages, budget)
    assert messages == before and len(canonical(result)) <= budget
    return json.loads(result[1]["content"])


def test_compaction_keeps_ready_and_remaining_actions_without_bulk_plan():
    state = action_state()
    result = compact(state)
    kept = result["research_plan"]
    assert kept["context_omitted"] and kept["retrieve_with"] == "inspect_requirements"
    assert "annual_coverage" not in kept
    for field in ("methods", "method_readiness", "method_completion", "next_work"):
        assert kept[field] == state["research_plan"][field]
    for field in ("draft", "memory", "pending_decision", "turn_control", "information_cutoff_date"):
        assert result["context"]["task_state"][field] == state["context"]["task_state"][field]


def test_blocked_checkpoint_does_not_invent_ready_methods_or_actions():
    state = action_state()
    state["research_plan"] = {
        "methods": ["ev_ebitda"], "method_readiness": [{"method": "ev_ebitda", "status": "blocked", "reason": "缺少租赁负债"}],
        "next_work": [], "annual_coverage": state["research_plan"]["annual_coverage"],
    }
    kept = compact(state)["research_plan"]
    assert kept["methods"] == ["ev_ebitda"]
    assert kept["method_readiness"] == state["research_plan"]["method_readiness"]
    assert kept["next_work"] == [] and "method_completion" not in kept
    assert "calculate_valuation" not in canonical(kept)


def test_long_diagnostics_are_bounded_but_method_status_and_real_next_arguments_survive():
    state = action_state()
    plan = state["research_plan"]
    plan["method_readiness"][2]["reason"] = "INPUTS_MISSING: " + "source diagnostic " * 3000
    plan["method_completion"]["remaining_methods"]["ev_ebitda"] = plan["method_readiness"][2]["reason"]
    action = {"kind": "interpret_inputs", "method": "ev_ebitda", "tool": "inspect_inputs",
        "arguments": {"ticker": "600123.SH", "period_end": "2025-12-31", "offset": 12},
        "instruction": "读取现有保存值，不联网补造。"}
    plan["next_work"] = [plan["next_work"][0], action, *[
        {"kind": "repair", "metric": f"raw.component_{index}", "reason": "diagnostic " * 3000}
        for index in range(15)]]
    kept = compact(state, 6000)["research_plan"]
    assert kept["method_readiness"][2]["status"] == "blocked"
    assert kept["method_readiness"][2]["reason"].startswith("INPUTS_MISSING:")
    assert len(kept["method_readiness"][2]["reason"]) <= 400
    assert kept["method_completion"]["remaining_methods"]["ev_ebitda"].startswith("INPUTS_MISSING:")
    assert kept["next_work"][1] == action
    assert 0 < len(kept["next_work"]) <= 3
    assert kept["next_work_omitted"] == len(plan["next_work"]) - len(kept["next_work"])


def test_action_checkpoint_cannot_displace_user_constraints_to_force_fit():
    state = action_state()
    state["context"]["task_state"]["memory"][0]["content"] = "完整保留用户原始限制 " * 4000
    with pytest.raises(LlmError, match="CONTEXT_BUDGET"):
        compact(state)


def test_checkpoint_reprojection_preserves_partial_status_and_omission_counts():
    plan = action_state()["research_plan"]
    plan["method_completion"].update(run_id="run_actual", completed_in_run=["pe"], completed_methods=["pe"], current_inputs_match=True)
    plan["method_completion"]["remaining_methods"] = {"ps": "尚未计算", "ev_ebitda": "missing evidence " * 1000}
    plan["next_work"] *= 10
    first = research_checkpoint(plan)
    assert research_checkpoint(first) == first
    assert first["method_completion"]["completed_methods"] == ["pe"]
    assert first["method_completion"]["all_requested_methods_completed"] is False
    assert first["next_work_omitted"] == 17


def test_oversized_action_arguments_are_not_truncated_into_a_different_call():
    plan = {"next_work": [{"tool": "inspect_inputs", "arguments": {"input_ids": ["input_long_" + str(index) for index in range(500)]}}]}
    kept = research_checkpoint(plan)["next_work"][0]
    assert "arguments" not in kept and kept["details_omitted"] == ["arguments"]
    assert research_checkpoint({"next_work": [kept]})["next_work"] == [kept]
