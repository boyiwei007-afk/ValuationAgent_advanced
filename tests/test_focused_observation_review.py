import json

from valuationagent.core.tools import ToolRegistry, ToolSpec, canonical
from valuationagent.llm.observation_review import focused_review_request
from valuationagent.application.observation_extraction import ReviewObservations, prepare_reviews, ObservationSelection
from test_observation_extraction import example, submit, review
from test_multisource_extraction import runtime_at


def test_focus_removes_extractor_narrative_but_keeps_full_proof_and_context(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    packet = prepare_reviews(runtime, ObservationSelection(fact_ids=[runtime.session.facts[0].fact_id]))
    messages = [{"role": "system", "content": "main task authority"},
                {"role": "user", "content": "earlier conclusions must not prime review"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "review_packet", "function": {
                    "name": "prepare_observation_review", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "review_packet", "content": canonical(packet)}]
    tools = ToolRegistry([ToolSpec("review_observations", "review", ReviewObservations, lambda _: None)]).schemas()
    before = canonical(messages)
    original_tools = canonical(tools)
    focused, restricted = focused_review_request(messages, tools)
    assert len(focused) == 4 and len(restricted) == 1
    schema = restricted[0]["function"]["parameters"]
    properties = schema["$defs"]["ObservationReview"]["properties"]
    assert properties["fact_id"]["enum"] == [packet["packets"][0]["fact_id"]]
    assert properties["packet_id"]["enum"] == [packet["packets"][0]["packet_id"]]
    assert list(properties).index("rationale") < list(properties).index("checks")
    assert schema["properties"]["reviews"]["maxItems"] == 1
    assert canonical(tools) == original_tools
    assert "earlier conclusions" not in canonical(focused)
    actual = json.loads(focused[-1]["content"])
    assert actual["original_context"] == packet["original_context"]
    assert actual["packets"][0]["anchors"] == packet["packets"][0]["anchors"]
    assert "rationale" not in actual["packets"][0]["interpretation"]["row"]
    assert canonical(messages) == before


def test_new_contradiction_can_downgrade_previous_supported_review(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    review(runtime)
    fact = runtime.session.facts[0]
    assert fact.status == "confirmed"
    checks = {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")}
    checks["unit"] = "ambiguous"
    review(runtime, checks=checks)
    assert fact.status == "proposed"
    assert fact.verification["semantic_review"]["status"] == "needs_evidence"
    assert fact.verification["semantic_review"]["independent_audit"] is False


def test_review_context_is_selected_before_main_history_compaction():
    from valuationagent.core.tools import NoArguments
    from valuationagent.llm.agent import run_tool_loop

    packet = {"packets": [{"fact_id": "example", "interpretation": {"row": {"rationale": "old claim"}}}],
              "original_context": {"block": {"text": "original context " * 1000}}}
    registry = ToolRegistry([
        ToolSpec("prepare_observation_review", "read packet", NoArguments, lambda _: packet),
        ToolSpec("review_observations", "review", NoArguments, lambda _: {"_terminal": True}),
    ])
    class Model:
        calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 2:
                assert json.loads(messages[-1]["content"])["original_context"] == packet["original_context"]
                assert len(kwargs["tools"]) == 1
            name = "prepare_observation_review" if self.calls == 1 else "review_observations"
            return {"tool_calls": [{"id": name, "function": {"name": name, "arguments": "{}"}}]}
    result = run_tool_loop(Model(), [{"role": "system", "content": "task"}, {"role": "user", "content": "review"}],
                          registry, lambda name, args, invoke: invoke(), max_rounds=2, max_context_chars=2200,
                          request_adapter=focused_review_request)
    assert result["_terminal"]
