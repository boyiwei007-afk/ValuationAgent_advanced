"""Deterministic acquisition coverage, separate from calculation authorization."""
from datetime import date, timedelta

from valuationagent.application.research_valuation import (
    MAX_SHARE_EVIDENCE_AGE_DAYS, METRIC_ALIASES, MIN_AUTOMATIC_HISTORY_YEARS, _period, financial_mapping_issue, mapped_financial_metric,
)
from valuationagent.schemas.models import required_financial_metrics
from valuationagent.application.evidence_status import observation_verified, binding_issues, repair_groups, evidence_counts
from valuationagent.application.valuation_plan import _build_for_methods, preview_session
from valuationagent.application.extraction_recovery import recovery_plan


def research_plan(session, assembler):
    from valuationagent.application.observation_extraction import observation_next_action

    methods = assembler._selected_methods(session) if session.draft.methods else []
    cutoff = session.information_cutoff_date or session.draft.valuation_date or date.today()
    baseline_year = cutoff.year - 1
    target_years = 10 if "dcf" in methods else 3 if methods else 0
    years = list(range(baseline_year, baseline_year - target_years, -1))
    metrics = sorted(required_financial_metrics(methods)) if methods else []
    annual_metrics = [metric for metric in metrics if metric not in {"common_shares", "diluted_shares"}]
    clean = [fact for fact in session.facts if fact.status == "confirmed" and not fact.warnings
             and fact.role == "historical" and fact.normalized_value is not None and not financial_mapping_issue(fact)
             and (fact.published_at is None or fact.published_at <= cutoff)]
    coverage = []
    for year in years:
        facts = [fact for fact in clean if fact.scope == "consolidated" and _period(fact.period) == date(year, 12, 31)]
        candidates = [fact for fact in session.facts if fact.status != "rejected" and fact.scope == "consolidated"
                      and fact.role == "historical" and _period(fact.period) == date(year, 12, 31)]
        observed = sorted({mapped_financial_metric(fact) for fact in facts} - {None})
        priority = "baseline" if year == baseline_year else "core_history" if year >= baseline_year - 3 else "long_term_trend"
        priority_metrics = annual_metrics if priority == "baseline" else [metric for metric in annual_metrics if metric in {"revenue", "net_income_parent", "ebit_margin", "ebitda"}]
        coverage.append({"year": year, "confirmed_metrics": observed,
                         "candidate_metrics": sorted({fact.metric for fact in candidates}),
                         "bound_metrics": sorted({fact.metric for fact in candidates if observation_verified(fact)}),
                         "binding_issues": sorted({issue for fact in candidates for issue in binding_issues(fact)}),
                         "model_review_count": sum(observation_verified(fact) and bool(fact.warnings) for fact in candidates),
                         "source_file_ids": sorted({fact.block_id.rsplit(":", 1)[0] for fact in candidates}),
                         "priority": priority, "priority_metrics": priority_metrics,
                         "not_directly_verified": sorted(set(priority_metrics) - set(observed)),
                         "fact_ids": [fact.fact_id for fact in facts]})
    blocking_ids = {fact.fact_id for fact in assembler.pending_blockers(session)}
    repairs = [{"fact_id": fact.fact_id, "metric": fact.metric, "standard_metric": mapped_financial_metric(fact),
                "status": fact.status,
                "role": fact.role, "peer_ticker": fact.peer_ticker, "blocks_selected_methods": fact.fact_id in blocking_ids,
                "next_action": observation_next_action(fact),
                "repair_action": "使用正确原始指标重新提交并用replaces替换，不能沿用已确认的错误语义映射" if fact.status == "confirmed" else "重检或修正已有候选",
                "period": fact.period, "scope": fact.scope, "warnings": [*fact.warnings, *([financial_mapping_issue(fact)] if financial_mapping_issue(fact) else [])],
                "block_id": fact.block_id}
               for fact in session.facts if (fact.status == "proposed" and fact.warnings)
               or (fact.status == "confirmed" and financial_mapping_issue(fact))]
    repairs.sort(key=lambda item: (not item["blocks_selected_methods"], item["standard_metric"] not in metrics, item["standard_metric"] != "common_shares"))
    readiness = []
    preview = preview_session(session)
    for method in methods:
        try:
            request = _build_for_methods(preview, assembler, [method])
            driver_repairs = assembler.recoverable_driver_repairs(session, request.financials) if method == "dcf" else []
            readiness.append({"method": method, "status": "needs_repair" if driver_repairs else "inputs_ready",
                              "reason": "原文已绑定，资本开支语义待修正" if driver_repairs else "仅表示输入准备通过，不代表已批准或已计算"})
        except ValueError as exc:
            readiness.append({"method": method, "status": "blocked", "reason": str(exc)[:1200]})
    timing = assembler.capital_structure_timing_issue(session)
    share_since = session.draft.valuation_date - timedelta(days=MAX_SHARE_EVIDENCE_AGE_DAYS) if session.draft.valuation_date else None
    current_shares = [fact for fact in clean if mapped_financial_metric(fact) == "common_shares"
                      and assembler._verified_dated_issuer_shares(fact) and share_since
                      and (period := _period(fact.period)) and share_since <= period <= cutoff]
    acquisition_targets = []
    if coverage and coverage[0]["not_directly_verified"]:
        acquisition_targets.append({"kind": "annual_baseline", "target_year": baseline_year,
            "metrics": coverage[0]["not_directly_verified"], "information_cutoff": cutoff.isoformat(),
            "instruction": "先定位该目标年度截至信息截止日可得的正式财务披露；旧年度事实用于历史趋势，不默认就是最新基期。此目标不证明年报已经发布：若尚未披露，记录可得性并采用已披露的完整年度；不改写期间。可推导项由计算器检查，不必在原文寻找派生指标同名行。"})
    if "common_shares" in metrics and share_since and not current_shares:
        acquisition_targets.append({"kind": "capital_structure", "metric": "common_shares",
            "required_since": share_since.isoformat(), "information_cutoff": cutoff.isoformat(),
            "instruction": "当前缺少该日期窗口内的已核验发行人股数。优先近期中报/季报股份表或生效资本变动披露，并保留实际截止日。修好窗口之前的旧年报股数仍不能满足当前时效；不要在旧文件中反复寻找未来日期。分红基数、库存股金额或EPS加权股数不自动等于期末总股数。若窗口与信息截止日冲突则如实报告，不搜索未来资料。"})
    for item in repairs:
        period = _period(item["period"])
        item["current_acquisition_priority"] = not (item["standard_metric"] == "common_shares" and period and share_since and period < share_since)
    relative_methods = [method for method in methods if method != "dcf"]
    peer_coverage, peer_error = [], None
    if relative_methods:
        try:
            peers = assembler._peers(preview)
        except ValueError as exc:
            peers, peer_error = [], str(exc)[:1000]
        for method in relative_methods:
            usable = [peer.ticker for peer in peers if getattr(peer, method, None) is not None and getattr(peer, method) > 0]
            peer_coverage.append({"method": method, "admitted_peer_count": len(usable), "tickers": usable,
                                  "minimum_required": 3, "remaining": max(0, 3 - len(usable)), "issue": peer_error})
    next_work = [*acquisition_targets, *[{"kind": "repair", **item} for item in repairs
                                      if item["blocks_selected_methods"] and item["current_acquisition_priority"]][:4]]
    if timing:
        next_work.append({"kind": "capital_structure", "issue": timing,
                          "instruction": "先查已有近期披露；旧年报中的相同截止日不会因重新下载或换页而变新。若股数过期，所需截止日须不早于required_since：可在更新的半年报/季报股份表或生效的资本变动公告核验。search_sources可明确report_type=semiannual/q1/q3及对应年份，不再搜索同一旧年度报告来解决时效。保持原估值日、年度利润基期和来源权限。仍无新证据时处理其他独立缺口，不造日期。"})
    for item in peer_coverage:
        if item["remaining"] or item["issue"]:
            next_work.append({"kind": "comparable_inputs", **item,
                              "instruction": "按真实可比公司分别取得定价日总市值与匹配完整年度财务，或可核验的直接FY倍数；使用comparable角色。已有利润不必反复核验，不把目标公司当自身可比。"})
    return {
        "methods": methods,
        "required_metrics": metrics,
        "annual_report_years": years,
        "history_policy": {
            "target_years": target_years,
            "minimum_automatic_dcf_years": MIN_AUTOMATIC_HISTORY_YEARS if "dcf" in methods else None,
            "instruction": "DCF目标近十年可得完整年度，四年是自动预测最低门槛，不是采集目标；相对估值目标近三年用于利润质量与趋势核对，但定价使用匹配的基期。新上市或缺失年度需披露。PE/PS不要求DCF四年门槛或预测假设。",
            "baseline_policy": "年度经营基期与股数截止日独立；中报/季报不能直接充作完整年度或混入年度收入序列。",
        },
        "annual_coverage": coverage,
        "acquisition_targets": acquisition_targets,
        "evidence_counts": evidence_counts(session.facts),
        "extraction_recovery": recovery_plan(session),
        "table_repairs": repair_groups(session.facts)[:12],
        "method_readiness": readiness,
        "capital_structure": timing,
        "peer_coverage": peer_coverage,
        "peer_pricing": {"valuation_date": str(session.draft.valuation_date or ""),
                         "pricing_date": str(session.draft.peer_pricing_date or session.draft.valuation_date or ""),
                         "rationale": session.draft.peer_pricing_rationale,
                         "instruction": "所有可比使用同一个已明确选择的行情日。没有估值日行情时，可经update_task说明依据后选择此前七天内的可核验日期，不改估值日/截止日，不混用各公司不同日期。"},
        "next_work": next_work,
        "model_scope_issue": assembler.model_scope_issue(session),
        "repair_candidates": repairs[:12],
        "repair_candidates_total": len(repairs),
        "sources": {
            "downloaded_documents": sum(doc.provenance_type not in {"search_snippet", "official_index"} for doc in session.documents),
            "search_leads": sum(doc.provenance_type in {"search_snippet", "official_index"} for doc in session.documents),
        },
        "next_actions": [
            "先处理next_work中的真实阻断。repair_candidates包含不阻塞当前方法的补充资料，不必为了清零所有候选反复修复。按next_action修正解释或复核，不调用旧表头解析器重试。",
            "以annual_report_years为覆盖目标，先补方法所需核心年度再扩展历史。可用fetch_financial_history取已连接数据，未配置则直接换网页或多年官方目录。",
            "用read_file或read_financial_evidence读取，再extract_observations批量理解。年报不是唯一载体，第三方历史字段复核后还须corroborate_facts，保留来源等级。",
            "股数、历史年度、可比样本分别推进；单项目标耗尽不代表其他缺口不可检索。",
            "只有check_preparation通过才可calculate_valuation；用户选择不能实现尚不存在的专业金融模型。",
        ],
    }


def metric_catalog():
    return [{"standard_metric": metric, "source_labels": sorted(aliases)}
            for metric, aliases in sorted(METRIC_ALIASES.items())] + [
                {"standard_metric": "market_cap", "role": "comparable", "source_labels": ["发行人全部普通股总市值"],
                 "instruction": "CNY金额、issuer时点、与明确选择的统一行情日一致；不可用流通市值替代。和同主体年度revenue/net_income_parent合成FY PS/PE，公式由程序计算。"},
                *[{"standard_metric": metric, "role": "comparable", "source_labels": [label],
                   "instruction": "直接倍数必须明确FY分母年度denominator_period_end、denominator_refs及确切定价日；不是TTM/预测倍数。"}
                  for metric, label in [("pe", "市盈率"), ("ps", "市销率"), ("ev_ebitda", "企业价值倍数")]],
            ]
