import pytest

from valuationagent.application.research import _valuation_finish_violation
from valuationagent.schemas.research import ResearchTurn
from test_evidence_recovery import configured_service
from test_research_sessions import ScriptedModel


QUESTION = "上传年报中股数无法绑定，还是允许联网用巨潮官方披露核验这几项后再计算？"
OPTIONS = ["保留上传年报并允许联网核验股本与有息债务", "交付说明报告"]


@pytest.mark.parametrize("question,options", [
    (QUESTION, OPTIONS),
    ("请选择后续路线", ["允许联网核验这些字段", "下载说明报告"]),
    ("是否继续从官方披露下载原文？", ["继续", "停止"]),
])
def test_authorized_public_research_is_not_presented_as_new_permission(question, options):
    assert "已获授权" in _valuation_finish_violation("缺少证据", question, options,
                                                      public_research_authorized=True)


def test_upload_only_session_still_needs_network_authorization():
    assert not _valuation_finish_violation("缺少证据", QUESTION, OPTIONS)


def test_real_user_choices_not_blocked():
    assert not _valuation_finish_violation("需要确认主体", "研究A股还是H股？", ["A股", "H股"],
                                           public_research_authorized=True)


@pytest.mark.parametrize("question,options", [
    ("营运资本变动三项与普通股股数的原文解析未通过校验，需你裁定提交口径", ["补证", "说明报告"]),
    ("请选择", ["营运资本变动以模型的间接法可行近似处理", "说明报告"]),
    ("请选择", ["普通股总股本以A股与H股加总口径裁定并提交", "说明报告"]),
    ("请选择", ["营运资本变动按上列原文数值，我按你确认的数值提交", "说明报告"]),
])
def test_confirmation_cannot_launder_failed_historical_evidence(question, options):
    assert "未通过来源校验" in _valuation_finish_violation("字段仍需补证", question, options)


def test_numeric_choices_are_rejected_regardless_of_acceptance_verb():
    for verb in ("采用", "采信", "认定", "选用", "依据", "沿用"):
        assert "普通澄清卡" in _valuation_finish_violation("", "哪个口径？", [verb + "7,655,955,883股", "提供资料"])
    assert not _valuation_finish_violation("", "采用哪个估值日？", ["2025-06-30", "2024-12-31"])


def test_repeated_permission_request_closes_with_report_instead_of_question(tmp_path):
    call = ("finish_response", {"answer": "存在缺失字段", "question": QUESTION, "options": OPTIONS})
    service, session, _ = configured_service(tmp_path, ScriptedModel([call, call]))
    state = service.turn(session.session_id, ResearchTurn(content="开始自动化DCF估值"))
    assert state["session"]["question"] is None
    assert state["result_document"]["status"] == "insufficient_data"
    assert not state["result_document"]["numeric_result_available"]
