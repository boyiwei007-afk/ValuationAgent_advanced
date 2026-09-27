"""A ten-year proposal stays legible in the delivered PDF/HTML narrative."""
from valuationagent.application.result_document import _forecast_lines


def test_forecast_display_preserves_ten_values_and_review_state():
    path = [str(i / 100) for i in range(1, 11)]
    proposal = {
        "status": "proposed", "rationale": "本研究的增长率由公开披露和审慎情景推导，仍待用户确认。",
        "inputs": {
            "revenue_growth_scenarios": {name: path for name in ("pessimistic", "base", "optimistic")},
            "ebit_margin_scenarios": None, "wacc": "0.085", "terminal_growth": "0.025",
        },
        "risks": ["需求弱于预计"],
    }
    lines = _forecast_lines(proposal)
    assert "待集中确认" in lines[0]
    assert "不是历史事实" in lines[0]
    assert all(f"收入增长率·{name}（第1至10年）" in "\n".join(lines)
               for name in ("审慎", "基准", "乐观"))
    assert "10.0%" in "\n".join(lines)
    assert "WACC：8.500%" in "\n".join(lines)
    assert "永续增长率：2.500%" in "\n".join(lines)
    assert "{" not in "\n".join(lines)
