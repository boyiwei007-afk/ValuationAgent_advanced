import json

import pytest

from valuationagent.core.tools import canonical
from valuationagent.llm.document_focus import FileTask, FileTaskEnd, READING_TOOLS
from valuationagent.schemas.research import ResearchTurn
from test_multisource_extraction import runtime_at
from test_observation_extraction import example
from observation_fixtures import ObservationModel, extraction_steps


def test_extraction_schema_exposes_the_actual_metric_catalog_and_currency(tmp_path):
    from valuationagent.application.observation_extraction import ExtractObservations
    from valuationagent.core.tools import ToolSpec
    from valuationagent.llm.context import DOCUMENT_PROMPT

    runtime = runtime_at(tmp_path)
    example(runtime)
    schema = ToolSpec("extract_observations", "extract", ExtractObservations, lambda args: args).schema()
    _, projected = runtime.adapt_request([{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}], [schema])
    definitions = projected[0]["function"]["parameters"]["$defs"]
    choices = definitions["Observation"]["properties"]["standard_metric"]["anyOf"]
    assert {"revenue", "total_revenue", "net_income_parent", "common_shares"} <= set(choices[0]["enum"])
    assert "net_profit" not in choices[0]["enum"]
    assert choices[1]["pattern"] == r"^raw\.[a-z][a-z0-9_]*$"
    assert "currency" in definitions["ReadingBasis"]["required"]
    assert "currency" not in schema["function"]["parameters"]["$defs"]["ReadingBasis"]["required"]
    assert len(DOCUMENT_PROMPT) < 3000


def test_historical_focus_requires_an_explicit_target_but_reference_does_not(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    runtime.session.draft.company = ""
    runtime.session.draft.ticker = ""
    task = FileTask(file_id=args["file_id"], entity_ticker="", role="historical",
        objective="先读取文件并识别真实主体与相关财务数据", metrics=["revenue"])
    with pytest.raises(ValueError, match="FILE_TASK_TARGET_MISSING"):
        runtime.document_focus.begin(task)
    assert runtime.document_focus.task is None
    runtime.document_focus.begin(task.model_copy(update={"role": "reference"}))
    assert runtime.document_focus.task.role == "reference"
    assert runtime.session.draft.company == ""


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
    assert "search_sources" not in {tool["function"]["name"] for tool in model.kwargs[-1]["tools"]}
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


def test_phase_mismatch_returns_to_parent_without_executing_rejected_arguments(tmp_path):
    from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
    from valuationagent.llm.agent import run_tool_loop
    from valuationagent.llm.client import ToolPhaseError

    runtime = runtime_at(tmp_path)
    args = example(runtime)
    task = FileTask(file_id=args["file_id"], entity_ticker="", role="reference",
        objective="只读原始文件并识别主体后保存观察，不联网不估值", metrics=["revenue"])
    runtime.document_focus.begin(task)
    executed = []

    class Model:
        def __init__(self):
            self.offered = []

        def chat(self, messages, **kwargs):
            self.offered.append({tool["function"]["name"] for tool in kwargs["tools"]})
            if len(self.offered) == 1:
                raise ToolPhaseError("TOOL_PHASE_MISMATCH: rejected outside file phase")
            return {"tool_calls": [{"id": "new_decision", "function": {"name": "finish", "arguments": "{}"}}]}

    model = Model()
    registry = ToolRegistry([ToolSpec("end_file_task", "end", NoArguments, lambda _: executed.append("unexpected")),
        ToolSpec("finish", "finish", NoArguments, lambda _: executed.append("finish") or {"_terminal": True})])
    result = run_tool_loop(model, [{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}],
        registry, lambda name, arguments, invoke: invoke(), request_adapter=runtime.document_focus.adapt,
        phase_recovery=runtime.document_focus.recover_phase)
    assert executed == ["finish"] and result["_agent_trace"]["phase_returns"] == [args["file_id"]]
    assert model.offered == [{"end_file_task"}, {"end_file_task", "finish"}]
    assert runtime.document_focus.task is None and not runtime.session.facts
    events = runtime.service.store.list_events(runtime.session.session_id)
    assert any(event.type == "agent.file_phase_return" for event in events)
    runtime.document_focus.begin(task)
    assert runtime.document_focus.recover_phase() is None
    assert runtime.document_focus.task is not None


def test_non_phase_protocol_error_does_not_change_file_scope(tmp_path):
    from valuationagent.core.tools import NoArguments, ToolRegistry, ToolSpec
    from valuationagent.llm.agent import run_tool_loop
    from valuationagent.llm.client import ToolProtocolError

    runtime = runtime_at(tmp_path)
    args = example(runtime)
    runtime.document_focus.begin(FileTask(file_id=args["file_id"], entity_ticker="", role="reference",
        objective="只读取已有原始文件，不联网不估值", metrics=["revenue"]))

    class Model:
        def chat(self, *args, **kwargs):
            raise ToolProtocolError("TOOL_JSON_TRUNCATED: fixture")

    registry = ToolRegistry([ToolSpec("end_file_task", "end", NoArguments, lambda _: {"_terminal": True})])
    with pytest.raises(ToolProtocolError):
        run_tool_loop(Model(), [], registry, lambda name, arguments, invoke: invoke(),
            phase_recovery=runtime.document_focus.recover_phase)
    assert runtime.document_focus.task is not None
    assert not runtime.document_focus.phase_returns
