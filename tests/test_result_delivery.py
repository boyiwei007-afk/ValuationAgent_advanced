import pytest

from test_input_workspace import fixture, values
from valuationagent.application.agent_runtime import AgentResponse, CalculateValuation
from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.result_delivery import bridge_amount, qualitative_only
from valuationagent.application.workspace_artifacts import ReportWrite, write_report


def completed(tmp_path):
    app, runtime = fixture(tmp_path)
    record_inputs(runtime, values())
    outcome = runtime.calculate(CalculateValuation())
    assert outcome["status"] in {"completed", "completed_with_warnings"}
    artifact = write_report(app.state.research, runtime.session, ReportWrite(format="md"))
    return app, runtime, outcome, artifact


def test_numeric_delivery_uses_frozen_values_not_unbound_model_narration(tmp_path):
    _, runtime, outcome, artifact = completed(tmp_path)
    response = runtime.finish(AgentResponse(answer="合理股价999元，利润888亿元。[报告](https://made-up.example/api/workspaces/fake/artifacts/fake)"))
    assert "40.0000 元/股" in response["answer"]
    assert "999" not in response["answer"] and "888亿元" not in response["answer"]
    assert "made-up.example" not in response["answer"]
    assert artifact["download_url"] in response["answer"]
    assert response["delivery"]["run_id"] == outcome["run_id"]
    assert response["delivery"]["narrative_status"] == "unbound_not_published"
    assert "| 方法 | 点估值 | 每股区间 | 口径 |\n| ---" in response["answer"]


def test_qualitative_analysis_is_preserved_and_current_result_is_rendered(tmp_path):
    _, runtime, _, _ = completed(tmp_path)
    answer = "相对估值对可比选择敏感，建议先核对业务结构差异，不要机械平均方法。"
    response = runtime.finish(AgentResponse(answer=answer))
    assert answer in response["answer"]
    assert response["delivery"]["narrative_status"] == "qualitative_only"
    assert "用户提供数据及假设的情景计算" in response["answer"]


def test_free_discussion_is_not_replaced_with_old_valuation_summary(tmp_path):
    _, runtime, _, _ = completed(tmp_path)
    runtime.session.turn_control.decision.actions = ["discuss"]
    answer = "PE为20倍是本次用户指定假设，不是市场事实。"
    response = runtime.finish(AgentResponse(answer=answer))
    assert response["answer"] == answer and "delivery" not in response


def test_existing_artifact_urls_cannot_keep_an_invented_host_or_cross_workspace(tmp_path):
    _, runtime, _, artifact = completed(tmp_path)
    runtime.session.turn_control.decision.actions = ["discuss"]
    response = runtime.finish(AgentResponse(answer=f"[报告](https://invented.example{artifact['download_url']})"))
    assert "invented.example" not in response["answer"] and artifact["download_url"] in response["answer"]
    with pytest.raises(ValueError, match="REPORT_LINK_INVALID"):
        runtime.finish(AgentResponse(answer=f"[报告](/api/workspaces/wrong/artifacts/{artifact['artifact_id']})"))


@pytest.mark.parametrize("text", ["价值三十亿元", "涨幅百分之二十", "价值４０元", "PE: 20", "one million dollars", "https://invented.example/report"])
def test_unbound_quantitative_or_link_commentary_is_not_published(text):
    assert not qualitative_only(text)


def test_bridge_share_count_is_not_formatted_as_currency():
    assert bridge_amount("diluted_or_common_shares", 1000000, "CNY") == "1,000,000 股"
    assert "元" in bridge_amount("interest_bearing_debt", 0, "CNY")
