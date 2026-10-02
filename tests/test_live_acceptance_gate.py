import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("live_acceptance", Path(__file__).parents[1] / "scripts/live_document_acceptance.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def snapshot():
    return {"active_run": {"run_id": "run_actual", "status": "completed_with_warnings",
                           "request": {"mode": "snapshot"},
                           "result": {"dcf": None, "relative": [{"status": "success", "per_share_value": "25.10"}]}},
            "artifacts": [{"kind": "result_report", "numeric_result_available": True, "valuation_run_id": "run_actual"}],
            "execution": {"status": "completed"}}


def test_numeric_report_requires_successful_snapshot_valuation_from_same_run():
    result = module.acceptance_checks(snapshot())
    assert all(result.values())
    assert module.acceptance_exit_code(result) == 0
    state = snapshot()
    state["artifacts"][0]["valuation_run_id"] = "old_run"
    checks = module.acceptance_checks(state)
    assert checks["numeric_valuation_completed"] and not checks["numeric_report_matches_active_run"]
    assert module.acceptance_exit_code(checks) == 1


@pytest.mark.parametrize("price", [None, "NaN", "Infinity", "not calculated"])
def test_truthy_result_with_missing_or_nonfinite_price_never_passes(price):
    state = snapshot()
    state["active_run"]["result"]["relative"][0]["per_share_value"] = price
    assert not module.acceptance_checks(state)["numeric_valuation_completed"]


@pytest.mark.parametrize("field,value", [("mode", "demo"), ("status", "failed"), ("method_status", "not_applicable")])
def test_demo_failed_or_not_applicable_is_not_numeric_acceptance(field, value):
    state = snapshot()
    if field == "mode":
        state["active_run"]["request"][field] = value
    elif field == "method_status":
        state["active_run"]["result"]["relative"][0]["status"] = value
    else:
        state["active_run"][field] = value
    assert not module.acceptance_checks(state)["numeric_valuation_completed"]


def test_execution_failure_or_missing_report_fails_and_checks_are_read_only():
    state = snapshot()
    before = copy.deepcopy(state)
    assert module.acceptance_exit_code(module.acceptance_checks(state, "AGENT_NO_PROGRESS")) == 1
    assert state == before
    state["artifacts"] = [{"kind": "research_note", "numeric_result_available": False}]
    assert module.acceptance_exit_code(module.acceptance_checks(state)) == 1
    assert module.acceptance_exit_code({}) == 1


def test_evaluator_source_fingerprint_ignores_runtime_data_but_tracks_code(tmp_path):
    source = tmp_path / "src/valuationagent"
    source.mkdir(parents=True)
    program = source / "engine.py"
    program.write_text("version = 1", encoding="utf-8")
    first = module.source_revision(tmp_path)
    assert first["file_count"] == 1
    (tmp_path / ".env").write_text("private runtime settings", encoding="utf-8")
    (source / "__pycache__").mkdir()
    (source / "__pycache__/engine.pyc").write_bytes(b"cache")
    assert module.source_revision(tmp_path) == first
    program.write_text("version = 2", encoding="utf-8")
    assert module.source_revision(tmp_path) != first
    assert module.acceptance_exit_code({**module.acceptance_checks(snapshot()), "source_revision_stable": False}) == 1
