from datetime import date
from types import SimpleNamespace

import pytest

from test_agent_recovery import runtime_fixture
from valuationagent.application.turn_control import resolve_control
from valuationagent.core.tools import NoArguments, ToolSpec
from valuationagent.llm.tool_catalog import LoadTools
from valuationagent.schemas.control import TurnDecision


def set_control(runtime, actions, denied=()):
    message = SimpleNamespace(message_id="routing", content="本轮测试明确限制")
    runtime.session.turn_control, runtime.session.execution_permissions = resolve_control(
        runtime.session, message, TurnDecision(summary="工具路由测试，执行仍须遵守本轮权限", actions=actions,
            permission_changes=[{"permission": permission, "allowed": False, "user_quote": message.content}
                for permission in denied]))


def route(runtime):
    tools = [ToolSpec(name, name, LoadTools if name == "load_tools" else NoArguments, lambda _: {}).schema()
        for name in ("load_tools", "update_task", "acquire_financial_inputs", "fetch_financial_history", "search_sources")]
    before = runtime.session.model_dump(mode="json")
    messages, selected = runtime.adapt_request([
        {"role": "system", "content": "policy"}, {"role": "user", "content": "{}"},
    ], tools)
    assert runtime.session.model_dump(mode="json") == before
    return messages, {tool["function"]["name"] for tool in selected}


def test_valuation_exposes_only_bound_acquisition_not_raw_fetch(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    set_control(runtime, ["research", "value"])
    runtime.tool_catalog.loaded = ["fetch_financial_history"]
    _, names = route(runtime)
    assert "acquire_financial_inputs" in names
    assert "fetch_financial_history" not in names
    assert "fetch_financial_history" not in runtime.tool_catalog.available
    with pytest.raises(ValueError, match="TOOL_PREREQUISITE.*acquire_financial_inputs"):
        runtime.tool_catalog.load(LoadTools(names=["fetch_financial_history"]))
    assert runtime.session.input_dataset is None and not runtime.session.documents


@pytest.mark.parametrize("missing", ["ticker", "valuation_date"])
def test_acquisition_requires_registered_target_and_date_without_fetch_detour(tmp_path, missing):
    runtime, _ = runtime_fixture(tmp_path)
    set_control(runtime, ["research", "value"])
    setattr(runtime.session.draft, missing, None)
    messages, names = route(runtime)
    assert "acquire_financial_inputs" not in names and "fetch_financial_history" not in names
    assert "update_task" in names
    assert missing in runtime.tool_catalog.deferred["acquire_financial_inputs"]
    assert "update_task" in messages[0]["content"]
    with pytest.raises(ValueError, match="TOOL_PREREQUISITE.*update_task"):
        runtime.tool_catalog.load(LoadTools(names=["acquire_financial_inputs"]))
    setattr(runtime.session.draft, missing, "600123" if missing == "ticker" else date(2026, 9, 30))
    _, names = route(runtime)
    assert "acquire_financial_inputs" in names and "fetch_financial_history" not in names
    assert "acquire_financial_inputs" not in runtime.tool_catalog.deferred


def test_research_without_current_valuation_keeps_pure_fetch(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    runtime.session.pending_action = "valuation"
    set_control(runtime, ["research", "read"])
    route(runtime)
    assert "fetch_financial_history" in runtime.tool_catalog.available
    runtime.tool_catalog.load(LoadTools(names=["fetch_financial_history"]))
    _, names = route(runtime)
    assert "fetch_financial_history" in names
    invoked = []
    result = runtime.call("fetch_financial_history", "{}", lambda: invoked.append("raw_fetch") or {"status": "sources_saved"})
    assert invoked == ["raw_fetch"] and result["status"] == "sources_saved"
    assert runtime.session.input_dataset is None


@pytest.mark.parametrize("actions,denied", [
    (["read"], []), (["discuss"], []), (["value"], []),
    (["research", "value"], ["network"]), (["research", "value"], ["structured_data"]),
])
def test_routing_does_not_restore_disallowed_api_tools(tmp_path, actions, denied):
    runtime, _ = runtime_fixture(tmp_path)
    runtime.tool_catalog.loaded = ["fetch_financial_history", "acquire_financial_inputs"]
    runtime.session.pending_action = "valuation"
    set_control(runtime, actions, denied)
    _, names = route(runtime)
    for tool in ("fetch_financial_history", "acquire_financial_inputs"):
        assert tool not in names and tool not in runtime.tool_catalog.available
        assert tool not in runtime.tool_catalog.deferred
    assert runtime.session.input_dataset is None


def test_api_only_value_keeps_acquisition_without_web_access(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    set_control(runtime, ["research", "value"], ["web"])
    _, names = route(runtime)
    assert "acquire_financial_inputs" in names
    assert "search_sources" not in runtime.tool_catalog.available
    assert "fetch_financial_history" not in runtime.tool_catalog.available


def test_missing_file_permission_does_not_upgrade_raw_fetch_to_input_acquisition(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    set_control(runtime, ["research", "value"], ["files"])
    route(runtime)
    assert "fetch_financial_history" in runtime.tool_catalog.available
    assert "acquire_financial_inputs" not in runtime.tool_catalog.available
    assert "acquire_financial_inputs" not in runtime.tool_catalog.deferred


def test_upload_only_platform_policy_keeps_both_api_routes_unavailable(tmp_path):
    runtime, _ = runtime_fixture(tmp_path)
    runtime.session.data_source_preference = "upload"
    set_control(runtime, ["research", "value"])
    _, names = route(runtime)
    assert "acquire_financial_inputs" not in names
    assert "fetch_financial_history" not in runtime.tool_catalog.available
    assert "acquire_financial_inputs" not in runtime.tool_catalog.deferred
