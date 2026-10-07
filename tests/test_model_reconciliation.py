from decimal import Decimal

from valuationagent.finance.team_model import FinanceTeamModel
from valuationagent.schemas.models import DcfResult, MultipleResult


def dcf(low, high):
    midpoint = (Decimal(low) + Decimal(high)) / 2
    return DcfResult(enterprise_value=midpoint, equity_value=midpoint, per_share_value=midpoint,
        range_low=Decimal(low), range_high=Decimal(high), terminal_value_share=Decimal("0.5"), bridge={})


def relative(low, high, method="pe"):
    return MultipleResult(method=method, status="success", range_low=Decimal(low), range_high=Decimal(high),
        per_share_value=(Decimal(low) + Decimal(high)) / 2, sample_size=5, sample_quality="adequate")


def test_gaps_between_methods_are_not_fabricated_overlap():
    result = FinanceTeamModel().reconcile(dcf(40, 60), [relative(10, 20), relative(80, 90, "ps")])
    assert result.relative_range == (Decimal(10), Decimal(90))
    assert result.overlap_range is None and result.combined_range is None
    assert result.method_comparison["status"] == "conflict_review_required"
    assert result.method_comparison["pe.status"] == result.method_comparison["ps.status"] == "no_overlap"
    assert "外包络" in result.conclusion


def test_overlap_is_common_to_all_methods_without_confidence_claim():
    result = FinanceTeamModel().reconcile(dcf(30, 60), [relative(10, 50), relative(40, 70, "ps")])
    assert result.overlap_range == (Decimal(40), Decimal(50))
    assert "高置信参考" not in result.conclusion
    assert "不代表独立验证" in result.conclusion


def test_separate_overlaps_are_not_joined_into_a_single_interval():
    result = FinanceTeamModel().reconcile(dcf(0, 100), [relative(10, 20), relative(80, 90, "ps")])
    assert result.overlap_range is None
    assert result.method_comparison["status"] == "mixed_methods_review_required"


def test_midpoint_gap_is_symmetric_and_handles_signed_or_zero_values():
    model = FinanceTeamModel()
    forward = model.reconcile(dcf(90, 110), [relative(110, 130)])
    reverse = model.reconcile(dcf(110, 130), [relative(90, 110)])
    assert forward.method_comparison["midpoint_gap"] == reverse.method_comparison["midpoint_gap"] == "18.18%"
    zero = model.reconcile(dcf(-1, 1), [relative(-2, 2)])
    assert zero.method_comparison["midpoint_gap"] == "undefined_zero_midpoints"
    signed = model.reconcile(dcf(-110, -90), [relative(90, 110)])
    assert signed.method_comparison["midpoint_gap"] == "200.00%"
    assert signed.method_comparison["status"] == "conflict_review_required"
