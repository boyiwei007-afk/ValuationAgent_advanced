from types import SimpleNamespace

from valuationagent.application.turn_control import resolve_control
from valuationagent.core.tools import canonical
from valuationagent.schemas.control import TurnDecision


def control_reply(kwargs, actions=("research", "value", "sensitivity", "report")):
    if {tool["function"]["name"] for tool in kwargs.get("tools", [])} != {"set_turn_plan"}:
        return None
    return {"tool_calls": [{"id": "fixture_control", "type": "function", "function": {
        "name": "set_turn_plan", "arguments": canonical(TurnDecision(
            summary="Explicit scripted permissions for tool execution tests, not an intent-understanding evaluation.",
            actions=list(actions)))}}]}


def allow_tool_testing(session, store=None):
    message = store.add_message(session.session_id, "user", "Execute fixture tools", "agent") if store else SimpleNamespace(
        message_id="fixture", content="Execute fixture tools")
    session.turn_control, session.execution_permissions = resolve_control(session,
        message,
        TurnDecision(summary="Explicit tool-test scope", actions=["research", "value", "sensitivity", "report"]))
