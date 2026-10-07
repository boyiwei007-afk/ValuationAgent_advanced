from datetime import date

import pytest

from valuationagent.application.observation_extraction import Observation


def observation(**changes):
    return Observation.model_validate({"metric": "营业收入", "standard_metric": "revenue", "raw_value": "100",
        "value_ref": "row", "label_refs": ["row"], "period_refs": ["years"], "period_kind": "annual",
        "period_end": "2025-12-31", "rationale": "这是文件明确列示的完整年度收入，仍需依据原文独立复核期间和列对应关系。", **changes})


def test_explicit_calendar_annual_kind_has_a_canonical_start_not_a_new_fact():
    assert observation().period_start == date(2025, 1, 1)
    assert observation(period_start="2025-01-01").period_start == date(2025, 1, 1)


@pytest.mark.parametrize("changes", [
    {"period_end": "2025-06-30"}, {"period_start": "2025-02-01"},
    {"period_kind": "interim"}, {"period_kind": "ttm"},
    {"period_kind": "instant", "period_start": "2025-01-01"},
])
def test_other_periods_are_never_silently_extended(changes):
    with pytest.raises(ValueError):
        observation(**changes)
