import json

import pytest

from valuationagent.core.tools import canonical
from valuationagent.llm.document_focus import FileTask, FileTaskEnd, READING_TOOLS
from valuationagent.schemas.research import ResearchTurn
from test_multisource_extraction import runtime_at
from test_observation_extraction import example
from observation_fixtures import ObservationModel, extraction_steps


def test_same_loop_focus_extracts_reviews_and_returns_to_main_tools(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    model = ObservationModel([
        ("begin_file_task", {"file_id": args["file_id"], "entity_ticker": "600123", "role": "historical", "objective": "只核验本文件完整年度合并营业收入，不联网，不启动估值。", "metrics": ["revenue"]}),
        ("read_file", {"file_id": args["file_id"]}),
        *extraction_steps(args),
        ("end_file_task", {"summary": "已经完成合成来源阅读及复核；这不是实际LLM正确率验收。"}),
    ])
    runtime.service._clients[runtime.session.session_id] = model
    runtime.service.store.save_research(runtime.session)
    runtime.service.turn(runtime.session.session_id, ResearchTurn(content="只核验这个文件的收入，不联网，不估值。"))
    state = runtime.service.store.get_research(runtime.session.session_id)
    assert state.facts[0].status == "confirmed"
    assert {tool["function"]["name"] for tool in model.kwargs[1]["tools"]} == READING_TOOLS
    assert {tool["function"]["name"] for tool in model.kwargs[4]["tools"]} == {"review_observations"}
    assert "search_sources" in {tool["function"]["name"] for tool in model.kwargs[-1]["tools"]}
    assert "不联网" in canonical(model.calls[1])
    assert state.valuation_run_id is None


def test_focus_permissions_and_local_budget_are_enforced_not_only_prompted(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    focus = runtime.document_focus
    task = FileTask(file_id=args["file_id"], entity_ticker="600123", role="historical", objective="核验本文件的完整年度合并营业收入", metrics=["revenue"])
    focus.begin(task)
    with pytest.raises(ValueError, match="FILE_TASK_ACTIVE"):
        focus.begin(task)
    for name, params in [("search_sources", {}), ("update_task", {}), ("calculate_valuation", {}),
                         ("read_file", {"file_id": "different-file"}), ("prepare_observation_review", {"fact_ids": ["foreign"]}),
                         ("reject_candidates", {"fact_ids": ["foreign"], "reason": "不是当前文件"})]:
        with pytest.raises(ValueError, match="FILE_TASK_SCOPE"):
            focus.guard(name, json.dumps(params))
    for counter in range(16):
        focus.guard("read_file", {"file_id": args["file_id"]})
    with pytest.raises(ValueError, match="FILE_TASK_BUDGET"):
        focus.guard("read_file", {"file_id": args["file_id"]})
    focus.guard("end_file_task", {})
    assert focus.end(FileTaskEnd(summary="原文仍有歧义，保留缺口不使用猜测值。"))["facts"] == []
    assert focus.task is None


def test_focus_omits_other_file_data_and_stale_assistant_narrative(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    runtime.document_focus.begin(FileTask(file_id=args["file_id"], entity_ticker="600123", role="historical", objective="读取当前文件年度收入，保留证据与歧义", metrics=["revenue"]))
    messages = [{"role": "system", "content": "main"}, {"role": "user", "content": canonical({"context": {
        "summary": "OLD_WRONG_ASSERTION", "task_state": {"facts": [{"secret_other_file": "OTHER_FILE"}]},
        "recent_turns": [{"role": "user", "content": "只读取，不计算"}, {"role": "assistant", "content": "OLD_WRONG_ASSERTION"}]}})}]
    focused, _ = runtime.document_focus.adapt(messages, [])
    assert "OLD_WRONG_ASSERTION" not in canonical(focused) and "OTHER_FILE" not in canonical(focused)
    assert "只读取，不计算" in canonical(focused)
    assert "rows[].role必须填comparable" in focused[0]["content"]


def test_declared_peer_scope_cannot_be_overridden_by_historical_defaults(tmp_path):
    from valuationagent.application.observation_extraction import ExtractObservations
    from valuationagent.core.tools import ToolSpec

    runtime = runtime_at(tmp_path)
    args = example(runtime)
    runtime.session.draft.ticker = "600999"
    task = FileTask(file_id=args["file_id"], entity_ticker="600123", role="comparable",
                    objective="读取可比公司的完整年度营业收入", metrics=["revenue"])
    with pytest.raises(ValueError, match="FILE_TASK_ENTITY"):
        runtime.document_focus.begin(task.model_copy(update={"role": "historical"}))
    runtime.document_focus.begin(task)
    schemas = [ToolSpec("extract_observations", "extract", ExtractObservations, lambda args: args).schema()]
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}]
    _, focused_tools = runtime.document_focus.adapt(messages, schemas)
    definitions = focused_tools[0]["function"]["parameters"]["$defs"]
    assert definitions["Observation"]["properties"]["role"]["const"] == "comparable"
    assert "role" in definitions["Observation"]["required"]
    assert definitions["ReadingBasis"]["properties"]["entity_ticker"]["const"] == "600123"
    assert "const" not in schemas[0]["function"]["parameters"]["$defs"]["Observation"]["properties"]["role"]
    with pytest.raises(ValueError, match="FILE_TASK_ROLE"):
        runtime.document_focus.guard("extract_observations", args)
    args["rows"][0]["role"] = "comparable"
    runtime.document_focus.guard("extract_observations", args)
    args["basis"]["entity_ticker"] = "600999"
    with pytest.raises(ValueError, match="FILE_TASK_ENTITY"):
        runtime.document_focus.guard("extract_observations", args)
    assert not runtime.session.facts and runtime.session.draft.ticker == "600999"


def test_focus_priorities_never_force_other_observations_into_wrong_metric_or_period(tmp_path):
    from valuationagent.application.observation_extraction import ExtractObservations
    from valuationagent.core.tools import ToolSpec

    runtime = runtime_at(tmp_path)
    args = example(runtime)
    runtime.document_focus.begin(FileTask(file_id=args["file_id"], entity_ticker="600123", role="historical",
                                          objective="核对该文件披露的普通股股数及独立截止日", metrics=["common_shares"]))
    tools = [ToolSpec("extract_observations", "extract", ExtractObservations, lambda args: args).schema()]
    _, selected = runtime.document_focus.adapt([{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}], tools)
    properties = selected[0]["function"]["parameters"]["$defs"]["Observation"]["properties"]
    assert "enum" not in properties["standard_metric"]
    assert "const" not in properties["period_kind"]
    assert properties["period_start"] != {"type": "null"}
    assert "const" not in properties["period_end"]
    runtime.document_focus.guard("extract_observations", args)
    assert not runtime.session.facts
    runtime.document_focus.end(FileTaskEnd(summary="当前原文为收入而非股数，返回主循环继续寻找股数。"))
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}]
    returned_messages, returned_tools = runtime.document_focus.adapt(messages, tools)
    assert returned_messages is messages and returned_tools is tools
    assert "const" not in returned_tools[0]["function"]["parameters"]["$defs"]["Observation"]["properties"]["role"]


def test_reference_focus_reads_unknown_or_multiple_entities_without_assigning_target(tmp_path):
    from valuationagent.core.tools import NoArguments, ToolSpec
    from valuationagent.llm.document_focus import EXPLORATION_TOOLS

    runtime = runtime_at(tmp_path)
    args = example(runtime)
    task = FileTask(file_id=args["file_id"], entity_ticker="", role="reference",
                    objective="先阅读行业文章识别可比公司，不将所有数据当作目标公司", metrics=["market_cap"])
    runtime.document_focus.begin(task)
    tools = [ToolSpec(name, name, NoArguments, lambda _: {}).schema() for name in READING_TOOLS]
    _, selected = runtime.document_focus.adapt([{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}], tools)
    assert {tool["function"]["name"] for tool in selected} == EXPLORATION_TOOLS
    runtime.document_focus.guard("read_file", {"file_id": args["file_id"]})
    with pytest.raises(ValueError, match="FILE_TASK_SCOPE"):
        runtime.document_focus.guard("extract_observations", args)
    assert runtime.session.draft.ticker == "600123" and not runtime.session.facts
