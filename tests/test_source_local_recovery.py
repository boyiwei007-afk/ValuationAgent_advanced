from valuationagent.application.extraction_recovery import record_attempt, recovery_plan
from test_multisource_extraction import runtime_at
from test_observation_extraction import example


def test_read_parameter_failure_does_not_suggest_financial_reextraction(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    for error, expected in [("VIEW_MISMATCH: conflicting arguments", "read_file"),
                            ("PAGE_NOT_FOUND: invalid page", "inspect_file")]:
        record_attempt(runtime.session, "read_file", {"file_id": args["file_id"]},
                       {"ok": False, "error": {"message": error}})
        plan = recovery_plan(runtime.session, file_ids={args["file_id"]})
        assert plan["files"][0]["next_choices"][0]["tool"] == expected
        assert not any(item["tool"] == "extract_observations" for item in plan["files"][0]["next_choices"])


def test_tool_failure_returns_only_current_source_recovery_not_all_old_failures(tmp_path):
    runtime = runtime_at(tmp_path)
    old = example(runtime, name="old.txt")
    current = example(runtime, name="current.txt")
    for args in (old, current):
        record_attempt(runtime.session, "read_file", {"file_id": args["file_id"]},
                       {"ok": False, "error": {"message": "PAGE_NOT_FOUND: invalid page"}})
    def fail():
        raise ValueError("VIEW_MISMATCH: conflicting arguments")
    result = runtime.call("read_file", {"file_id": current["file_id"]}, fail)
    assert [item["file_id"] for item in result["recovery"]["files"]] == [current["file_id"]]
    assert len(recovery_plan(runtime.session)["files"]) == 2


def test_implicit_pdf_view_is_recorded_as_actual_decoder_not_untried_strategy(tmp_path):
    runtime = runtime_at(tmp_path)
    args = example(runtime)
    runtime.session.documents[0].name = "synthetic.pdf"
    record_attempt(runtime.session, "read_file", {"file_id": args["file_id"], "page": 3},
                   {"view": "pdf_geometry", "blocks": []})
    record_attempt(runtime.session, "extract_observations", {"file_id": args["file_id"]},
                   {"ok": False, "error": {"message": "AMOUNT_NOT_FOUND: wrapped number"}})
    choices = recovery_plan(runtime.session)["files"][0]["next_choices"]
    assert runtime.session.reading_attempts[0]["view"] == "pdf_geometry"
    views = {item.get("arguments", {}).get("view") for item in choices}
    assert "pdf_geometry" not in views and {"pdf_plain", "pdf_layout"} <= views
