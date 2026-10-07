from types import SimpleNamespace

import pytest

from test_agent_recovery import runtime_fixture
from test_input_acquisition import acquisition_workspace
from test_input_calculations import formula, workspace
from valuationagent.application.agent_runtime import WorkspaceAgentRuntime
from valuationagent.application.input_acquisition import acquire_financial_inputs
from valuationagent.application.input_views import InspectInputs, inspect_inputs
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.turn_control import guard_tool, resolve_control
from valuationagent.core.tools import NoArguments, ToolSpec
from valuationagent.llm.tool_catalog import LoadTools
from valuationagent.schemas.control import TurnDecision
from valuationagent.schemas.inputs import RecordInputs


def set_control(runtime, actions, permissions=None):
    message = SimpleNamespace(message_id="input-handoff", content="Explicit test permissions")
    runtime.session.turn_control, runtime.session.execution_permissions = resolve_control(
        runtime.session, message, TurnDecision(summary="Test only schema visibility, not financial execution", actions=actions,
            permission_changes=[{"permission": permission, "allowed": allowed, "user_quote": message.content}
                for permission, allowed in (permissions or {}).items()]))


def offered(runtime):
    tools = [ToolSpec(name, name, schema, lambda _: {}).schema() for name, schema in (
        ("load_tools", LoadTools), ("record_inputs", RecordInputs), ("inspect_inputs", InspectInputs),
        ("read_user_input", NoArguments), ("calculate_valuation", NoArguments),
        ("acquire_financial_inputs", NoArguments), ("search_sources", NoArguments),
    )]
    before = runtime.session.model_dump(mode="json")
    _, selected = runtime.adapt_request([
        {"role": "system", "content": "policy"}, {"role": "user", "content": "{}"},
    ], tools)
    assert runtime.session.model_dump(mode="json") == before
    return {tool["function"]["name"]: tool for tool in selected}


def test_successful_acquisition_exposes_input_declaration_without_second_load(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path)
    set_control(runtime, ["research", "value"], {"network": True})
    assert "record_inputs" not in offered(runtime)
    output = acquire_financial_inputs(runtime, args)
    assert output["new_input_count"] > 0 and calls
    before = runtime.session.model_dump(mode="json")
    result = runtime.progress_advisory("acquire_financial_inputs", output)
    assert result["reading_handoff"]["tools"] == ["record_inputs"]
    assert result["reading_handoff"]["executed"] is False
    assert "record_inputs" in offered(runtime)
    assert runtime.session.model_dump(mode="json") == before
    assert not runtime.session.input_dataset.calculations and runtime.session.valuation_run_id is None


def test_inspected_raw_operands_offer_full_declaration_schema_without_mapping(tmp_path):
    _, runtime, _ = workspace(tmp_path)
    before = runtime.session.model_dump(mode="json")
    output = inspect_inputs(runtime.session, InspectInputs())
    result = runtime.progress_advisory("inspect_inputs", output)
    assert result["reading_handoff"]["tools"] == ["record_inputs"]
    schema = offered(runtime)["record_inputs"]["function"]["parameters"]
    assert "calculations" in schema["properties"]
    assert "InputCalculationDraft" in schema["$defs"]
    assert not runtime.session.input_dataset.calculations
    assert runtime.session.model_dump(mode="json") == before


def test_failed_load_and_later_read_cannot_hide_editing_existing_inputs(tmp_path):
    _, runtime, _ = workspace(tmp_path)
    offered(runtime)
    with pytest.raises(ValueError, match="TOOL_UNAVAILABLE"):
        runtime.tool_catalog.load(LoadTools(names=["record_inputs", "unavailable_tool"]))
    runtime.tool_catalog.load(LoadTools(names=["inspect_inputs"]))
    runtime.progress_advisory("read_file", {"blocks": [{"text": "Untrusted document content"}]})
    assert "record_inputs" in offered(runtime)
    assert not runtime.session.input_dataset.calculations


def test_resumed_runtime_recovers_input_schema_from_persisted_state(tmp_path):
    _, previous, _ = workspace(tmp_path)
    runtime = WorkspaceAgentRuntime(previous.service, previous.session, previous.workspaces)
    assert not runtime.reading_tools and not runtime.tool_catalog.loaded
    assert "record_inputs" in offered(runtime)
    assert runtime.session.valuation_run_id is None


@pytest.mark.parametrize("name,output", [
    ("acquire_financial_inputs", {"new_input_count": 10, "active_input_count": 10}),
    ("inspect_inputs", {"items": [{"input_id": "invented"}], "active_count": 1}),
    ("acquire_financial_inputs", {"ok": False, "new_input_count": 10}),
    ("inspect_inputs", {"items": [], "active_count": 0}),
])
def test_output_claims_do_not_create_an_input_handoff_without_saved_inputs(tmp_path, name, output):
    runtime, _ = runtime_fixture(tmp_path)
    result = runtime.progress_advisory(name, output)
    assert "reading_handoff" not in result
    assert "record_inputs" not in offered(runtime)
    assert runtime.session.input_dataset is None


def test_read_only_turn_cannot_promote_existing_raw_inputs_to_editing(tmp_path):
    _, runtime, _ = workspace(tmp_path)
    set_control(runtime, ["read"])
    runtime.tool_catalog.loaded = ["record_inputs"]
    result = runtime.progress_advisory("inspect_inputs", inspect_inputs(runtime.session, InspectInputs()))
    assert "reading_handoff" not in result
    assert "record_inputs" not in offered(runtime)
    with pytest.raises(ValueError, match="inputs"):
        guard_tool(runtime.session, "record_inputs", {"calculations": []})


def test_input_handoff_does_not_restore_network_calculation_or_source_file_access(tmp_path):
    _, runtime, _ = workspace(tmp_path)
    set_control(runtime, ["ingest"], {"network": False, "calculation": False, "files": False})
    names = offered(runtime)
    assert "record_inputs" in names
    assert not {"calculate_valuation", "acquire_financial_inputs", "search_sources"} & names.keys()
    with pytest.raises(ValueError, match="files"):
        guard_tool(runtime.session, "record_inputs", {"source_values": [{"fact_id": "fact"}]})


def test_declared_calculation_success_keeps_input_schema_for_remaining_operands(tmp_path):
    _, runtime, ids = workspace(tmp_path)
    result = record_inputs(runtime, RecordInputs(calculations=[formula(ids)]))
    before = runtime.session.model_dump(mode="json")
    runtime.progress_advisory("record_inputs", result)
    runtime.progress_advisory("record_inputs", {"saved_input_ids": [ids["revenue"]]})
    assert "record_inputs" in offered(runtime)
    assert runtime.session.model_dump(mode="json") == before
