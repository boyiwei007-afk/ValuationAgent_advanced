"""A report is an outcome, never evidence of a completed numerical valuation."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("real_acceptance", Path(__file__).parents[1] / "scripts/real_company_acceptance.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def state(numeric=False, question=None):
    return {"session": {"pending_action": "valuation", "question": question},
            "result_document": {"report_id": "audit-id", "status": "valued" if numeric else "insufficient_data",
                                "numeric_result_available": numeric}}


def test_missing_data_report_is_not_a_numeric_pass():
    checks = module.acceptance_checks(state(), report_before_export=True, elapsed_seconds=25, budget=60)
    assert checks["report_available_without_export_side_effect"]
    assert not checks["numeric_valuation_completed"]
    assert checks["valuation_goal_retained"]


def test_export_creation_cannot_hide_missing_automatic_report():
    checks = module.acceptance_checks(state(), report_before_export=False, elapsed_seconds=25, budget=60)
    assert not checks["report_available_without_export_side_effect"]


def test_batch_approval_and_budget_overrun_are_visible():
    checks = module.acceptance_checks(state(question={"kind": "facts"}), report_before_export=True,
                                      elapsed_seconds=121, budget=60)
    assert not checks["no_individual_fact_approval"]
    assert not checks["within_budget_plus_60_seconds"]


def test_actual_numeric_result_passes_separate_gate():
    checks = module.acceptance_checks(state(True), report_before_export=True, elapsed_seconds=25, budget=60)
    assert checks["numeric_valuation_completed"]
