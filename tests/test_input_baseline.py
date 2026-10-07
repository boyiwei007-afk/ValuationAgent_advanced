from datetime import date

import pytest

from test_input_acquisition import acquisition_workspace
from valuationagent.application.input_acquisition import acquire_financial_inputs
from valuationagent.application.input_workspace import prepare_dataset
from valuationagent.application.record_inputs import record_inputs
from valuationagent.schemas.inputs import BaselineInstruction, RecordInputs


def test_user_selected_historical_baseline_is_preserved_and_frozen_with_its_quote(tmp_path):
    app, runtime, args, calls = acquisition_workspace(tmp_path)
    quote = "请使用2024年度作为估值基期。"
    message = app.state.store.add_message(runtime.session.session_id, "user", quote, "agent")
    args.baseline = BaselineInstruction(period_end=date(2024, 12, 31), user_quote=quote, message_id=message.message_id)
    args.years = [2024]
    result = acquire_financial_inputs(runtime, args)
    assert result["years"] == [2024]
    assert result["newer_unselected_annual_periods"]["600123.SH"] == "2025-12-31"
    request, _, _, _ = app.state.workspaces._freeze_prevaluation_request(runtime.session, {"methods": ["pe"]})
    assert request.financials.period_end == date(2024, 12, 31)
    assert request.baseline_selection["policy"] == "user_selected"
    assert request.baseline_selection["user_quote"] == quote
    assert request.baseline_selection["observed_available_target_periods"] == ["2024-12-31", "2025-12-31"]
    assert len(calls) == 18


def test_changing_baseline_does_not_delete_or_relabel_existing_inputs(tmp_path):
    app, runtime, args, calls = acquisition_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    ids = [row.input_id for row in runtime.session.input_dataset.records]
    quote = "改用2024年度作为基期。"
    app.state.store.add_message(runtime.session.session_id, "user", quote, "agent")
    record_inputs(runtime, RecordInputs(baseline=BaselineInstruction(period_end=date(2024, 12, 31), user_quote=quote)))
    assert prepare_dataset(runtime.session, ["pe"]).financials.period_end == date(2024, 12, 31)
    assert ids == [row.input_id for row in runtime.session.input_dataset.records]
    app.state.store.add_message(runtime.session.session_id, "user", "恢复最新完整年度基期。", "agent")
    record_inputs(runtime, RecordInputs(baseline=BaselineInstruction(period_end=None, user_quote="恢复最新完整年度基期。")))
    assert prepare_dataset(runtime.session, ["pe"]).financials.period_end == date(2025, 12, 31)
    assert len(calls) == 18


def test_model_cannot_invent_a_user_instruction_to_choose_an_older_baseline(tmp_path):
    _, runtime, args, calls = acquisition_workspace(tmp_path)
    args.baseline = BaselineInstruction(period_end=date(2024, 12, 31), user_quote="请用2024年度。")
    with pytest.raises(ValueError, match="INPUT_BASELINE_SOURCE"):
        acquire_financial_inputs(runtime, args)
    assert not calls and runtime.session.input_dataset is None


def test_known_newer_year_cannot_disappear_just_by_not_admitting_its_inputs(tmp_path):
    _, runtime, args, _ = acquisition_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    runtime.session.input_dataset.records = [row for row in runtime.session.input_dataset.records if row.period_end != date(2025, 12, 31)]
    with pytest.raises(ValueError, match="INPUT_BASELINE_NEWER_AVAILABLE"):
        prepare_dataset(runtime.session, ["pe"])


def test_user_baseline_without_inputs_is_not_silently_replaced(tmp_path):
    app, runtime, args, _ = acquisition_workspace(tmp_path)
    acquire_financial_inputs(runtime, args)
    quote = "按2023年度作为基期。"
    app.state.store.add_message(runtime.session.session_id, "user", quote, "agent")
    record_inputs(runtime, RecordInputs(baseline=BaselineInstruction(period_end=date(2023, 12, 31), user_quote=quote)))
    with pytest.raises(ValueError, match="INPUT_BASELINE_MISSING"):
        prepare_dataset(runtime.session, ["pe"])


def test_historical_baseline_is_included_in_the_provider_query(tmp_path, monkeypatch):
    from valuationagent.application import input_acquisition

    app, runtime, args, _ = acquisition_workspace(tmp_path)
    quote = "请使用2024年度作为估值基期。"
    app.state.store.add_message(runtime.session.session_id, "user", quote, "agent")
    args.baseline = BaselineInstruction(period_end=date(2024, 12, 31), user_quote=quote)
    args.years = [2025]
    observed = []
    original = input_acquisition.fetch_history

    def capture(runtime, request):
        observed.append(request.years)
        return original(runtime, request)

    monkeypatch.setattr(input_acquisition, "fetch_history", capture)
    acquire_financial_inputs(runtime, args)
    assert observed and all(years == [2024, 2025] for years in observed)
    assert prepare_dataset(runtime.session, ["pe"]).financials.period_end == date(2024, 12, 31)


@pytest.mark.parametrize("change", [{"message_sha256": "changed"}, {"period_end": "2023-12-31"}])
def test_baseline_cannot_be_frozen_after_source_or_date_tampering(tmp_path, change):
    app, runtime, args, _ = acquisition_workspace(tmp_path)
    quote = "请使用2024年度作为估值基期。"
    app.state.store.add_message(runtime.session.session_id, "user", quote, "agent")
    args.baseline = BaselineInstruction(period_end=date(2024, 12, 31), user_quote=quote)
    acquire_financial_inputs(runtime, args)
    runtime.session.input_dataset.baseline_selection.update(change)
    with pytest.raises(ValueError, match="INPUT_BASELINE_SOURCE_CHANGED|INPUT_DATE"):
        app.state.workspaces._freeze_prevaluation_request(runtime.session, {"methods": ["pe"]})
