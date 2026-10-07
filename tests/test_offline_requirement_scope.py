from test_input_workspace import fixture
from valuationagent.application.input_workspace import FINANCIAL_FIELDS, RATIO_FIELDS, RAW_INPUT_FIELDS


def test_offline_user_requirements_do_not_start_a_ten_year_collection_plan(tmp_path):
    _, runtime = fixture(tmp_path)
    runtime.session.draft.methods = ["dcf", "pe", "ps", "ev_ebitda"]
    plan = runtime.requirements()
    assert plan["analysis_basis"] == "user_scenario"
    assert plan["history_policy"]["target_years"] == 0
    assert plan["acquisition_targets"] == []
    assert plan["annual_coverage"] == []
    assert "capital_expenditure" in plan["required_metrics"]
    assert runtime.session.input_dataset is None


def test_user_requirement_catalog_matches_the_exposed_input_tool(tmp_path):
    _, runtime = fixture(tmp_path)
    catalog = runtime.requirements()["metric_catalog"]
    assert {item["standard_metric"] for item in catalog} == FINANCIAL_FIELDS | RATIO_FIELDS | RAW_INPUT_FIELDS
    assert all(item["input_tool"] == "record_user_inputs" for item in catalog)
