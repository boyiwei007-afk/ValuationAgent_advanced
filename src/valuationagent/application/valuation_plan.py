"""Valuation-first planning. Preview copies never authorize live calculations."""
import hashlib
import json
from itertools import combinations


def scope_key(session):
    scope = session.draft.model_dump(mode="json", exclude={"objective"})
    scope["methods"] = sorted(scope["methods"])
    scope["data_source"] = session.data_source_preference
    return hashlib.sha256(json.dumps(scope, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def preview_session(session, fact_ids=None):
    preview = session.model_copy(deep=True)
    preview.question = None
    online_ticker = session.data_source_preference == "online" and bool(session.draft.ticker)
    candidates = {f.fact_id for f in preview.facts if f.status == "proposed" and not f.warnings
                  and (not online_ticker or f.role == "assumption")
                  and (fact_ids is None or f.fact_id in fact_ids)}
    superseded = {old for key in candidates for old in preview.staged_supersessions.get(key, [])}
    for fact in preview.facts:
        if fact.fact_id in superseded:
            fact.status = "rejected"
        elif fact.fact_id in candidates:
            fact.status = "confirmed"
    if preview.forecast_proposal and preview.forecast_proposal.scope_key == scope_key(preview):
        preview.forecast_proposal.status = "confirmed"
    return preview


def _build_for_methods(session, assembler, methods):
    """Build an isolated preview for a specific subset of requested methods."""
    candidate = session.model_copy(deep=True)
    candidate.valuation_methods_override = list(methods)
    candidate.valuation_method_exclusions = {}
    return assembler.build(candidate)


def valuation_progress(session, assembler):
    """Actual assembler checks, not model-written or stale document checklists."""
    preview = preview_session(session)
    preview_facts = {f.fact_id: f for f in preview.facts}
    staged = [f.fact_id for f in session.facts if f.status == "proposed" and not f.warnings
              and preview_facts[f.fact_id].status == "confirmed"]
    candidate_counts = {
        "confirmed": sum(f.status == "confirmed" for f in session.facts),
        "staged_clean": sum(f.status == "proposed" and not f.warnings for f in session.facts),
        "needs_repair": sum(f.status == "proposed" and bool(f.warnings) for f in session.facts),
        "rejected": sum(f.status == "rejected" for f in session.facts),
    }
    request = None
    exclusions = dict(getattr(session, "valuation_method_exclusions", {}) or {})
    requested_methods = list(session.draft.methods or [])
    active_methods = list(getattr(session, "valuation_methods_override", []) or [])
    original_error = None
    try:
        if not requested_methods:
            raise ValueError("请先确认估值方法")
        request = assembler.build(preview)
    except ValueError as exc:
        original_error = str(exc)

    # Before the plan is accepted, evaluate method readiness independently.
    # Prefer the largest calculable subset, preserving the user's method order.
    # This does not authorize calculation or mutate the live research session;
    # exclusions become effective only through the same final plan confirmation.
    if request is None and not active_methods and len(requested_methods) > 1:
        probe_errors = {}
        for method in requested_methods:
            try:
                _build_for_methods(preview, assembler, [method])
            except ValueError as exc:
                probe_errors[method] = str(exc)
        for size in range(len(requested_methods) - 1, 0, -1):
            for subset in combinations(requested_methods, size):
                try:
                    request = _build_for_methods(preview, assembler, subset)
                except ValueError:
                    continue
                exclusions = {
                    method: probe_errors.get(
                        method,
                        "与当前可执行方法组合后口径不兼容；本次不纳入计算。",
                    )[:1000]
                    for method in requested_methods
                    if method not in subset
                }
                break
            if request is not None:
                break

    if request is None:
        scope_unsupported = bool(requested_methods and set(requested_methods) <= {"dcf", "ev_ebitda"}
                                 and assembler.model_scope_issue(preview))
        evidence_issues = [{"fact_id": f.fact_id, "metric": f.metric, "period": f.period,
                            "warnings": list(f.warnings)} for f in assembler.pending_blockers(preview) if f.warnings]
        detailed_error = original_error or "估值输入尚不完整"
        if evidence_issues and original_error and "待确认候选" in original_error:
            detailed_error = "必要字段尚未通过来源校验：" + "；".join(
                f"{item['period']} {item['metric']}（{'、'.join(item['warnings'][:2])}）" for item in evidence_issues[:5])
        return {"status": "unsupported_model_scope" if scope_unsupported else "building_model", "ready_for_review": False, "blocking_reason": original_error or "估值输入尚不完整",
                "blocking_detail": detailed_error, "evidence_issues": evidence_issues,
                "staged_fact_ids": staged, "confirmed_fact_count": candidate_counts["confirmed"],
                "candidate_counts": candidate_counts,
                "instruction": ("当前所选方法需要尚未实现的专业调整，继续补普通财务字段也不能解除；立即交付说明报告，不要持续检索或承诺补一项就能计算。用户可另行明确更换方法。"
                                if scope_unsupported else "只补当前模型必要输入。基期完整但历史不足时，可调用propose_forecast提出有依据的十年三情景预测，最终由用户集中确认。历史缺失不可用假设、零值或搜索摘要替代。")}
    degraded = bool(exclusions)
    return {"status": "ready_for_review", "ready_for_review": True, "blocking_reason": "",
            "staged_fact_ids": staged, "methods": request.methods,
            "requested_methods": requested_methods,
            "excluded_methods": exclusions,
            "degraded": degraded,
            "company": request.company.name or request.company.ticker,
            "valuation_date": str(request.valuation_date),
            "baseline_period": str(request.financials.period_end) if request.financials else None,
            "historical_periods": [str(f.period_end) for f in request.historical_financials],
            "financials": request.financials.model_dump(mode="json", exclude={"evidence", "statement_items"}) if request.financials else None,
            "assumptions": request.assumptions.model_dump(mode="json", exclude_none=True),
            "forecast_proposal_id": session.forecast_proposal.proposal_id if session.forecast_proposal else None,
            "forecast_rationale": session.forecast_proposal.rationale if session.forecast_proposal else "采用确定性模型的历史推导与行业参数；计算时披露假设及风险。",
            "risks": [
                *(session.forecast_proposal.risks if session.forecast_proposal else []),
                *(
                    ["部分已选估值方法因缺少可靠数据被排除；本次区间只综合可执行方法，结论置信度相应降低。"]
                    if degraded else []
                ),
            ],
             "candidate_counts": candidate_counts,
            "instruction": (
                "至少一种已选方法具备可靠输入。立即调用request_formal_valuation集中复核可执行方法及排除原因；"
                "不要让缺数据的方法继续阻塞已有估值。"
                if degraded else
                "必要输入已齐备，立即调用request_formal_valuation集中复核并提交；不要继续搜集不影响本次模型的资料。"
            )}
