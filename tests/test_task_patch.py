from datetime import date

import pytest

from valuationagent.application.agent_runtime import TaskUpdate
from test_multisource_extraction import runtime_at
from test_observation_extraction import example, review, submit


def test_partial_task_update_preserves_identity_dates_objective_and_admitted_facts(tmp_path):
    runtime = runtime_at(tmp_path)
    runtime.session.draft.objective = "持续完成数值估值及报告，保留已确认范围"
    runtime.session.draft.information_cutoff_date = date(2026, 9, 30)
    submit(runtime, example(runtime))
    review(runtime)
    before = runtime.session.draft.model_dump()
    result = runtime.update_task(TaskUpdate(draft={"company": before["company"], "industry": "新行业分类"}, valuation_requested=True))
    assert result["valuation_requested"]
    for field in ("company", "ticker", "valuation_date", "information_cutoff_date", "objective", "methods"):
        assert getattr(runtime.session.draft, field) == before[field]
    assert runtime.session.facts[0].status == "confirmed"


@pytest.mark.parametrize("changes", [{"company": "另一家公司"}, {"ticker": "000321"}])
def test_identity_patch_cannot_pair_new_company_with_old_code(tmp_path, changes):
    runtime = runtime_at(tmp_path)
    previous = runtime.session.draft.model_copy(deep=True)
    with pytest.raises(ValueError, match="TASK_ENTITY_UPDATE"):
        runtime.update_task(TaskUpdate(draft=changes, valuation_requested=True))
    assert runtime.session.draft == previous


def test_explicit_entity_switch_still_invalidates_old_company_facts(tmp_path):
    runtime = runtime_at(tmp_path)
    submit(runtime, example(runtime))
    review(runtime)
    runtime.update_task(TaskUpdate(draft={"company": "另一家公司", "ticker": "000321"}, valuation_requested=True))
    assert runtime.session.draft.ticker == "000321"
    assert runtime.session.facts[0].status == "proposed"
    assert "研究公司已变更" in str(runtime.session.facts[0].warnings)


def test_pricing_date_patch_validates_against_preserved_valuation_date(tmp_path):
    runtime = runtime_at(tmp_path)
    runtime.update_task(TaskUpdate(draft={"peer_pricing_date": "2026-09-29",
        "peer_pricing_rationale": "引用前一日统一行情，保留一天的市场变化风险。"}, valuation_requested=True))
    assert runtime.session.draft.valuation_date == date(2026, 9, 30)
    assert runtime.session.draft.peer_pricing_date == date(2026, 9, 29)
    with pytest.raises(ValueError, match="七天内"):
        runtime.update_task(TaskUpdate(draft={"peer_pricing_date": "2026-09-01"}, valuation_requested=True))
    assert runtime.session.draft.peer_pricing_date == date(2026, 9, 29)


def test_new_valuation_requires_explicit_date_but_conversation_does_not(tmp_path):
    runtime = runtime_at(tmp_path)
    runtime.session.draft.valuation_date = None
    with pytest.raises(ValueError, match="TASK_DATE_REQUIRED"):
        runtime.update_task(TaskUpdate(draft={"methods": ["pe"]}, valuation_requested=True))
    runtime.update_task(TaskUpdate(draft={"industry": "讨论阶段"}, valuation_requested=False))
    assert runtime.session.draft.valuation_date is None
    runtime.update_task(TaskUpdate(draft={"valuation_date": "2026-09-30"}, valuation_requested=True))
    assert runtime.session.draft.valuation_date == date(2026, 9, 30)
