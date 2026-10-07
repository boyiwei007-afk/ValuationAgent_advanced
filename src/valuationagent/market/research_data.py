import hashlib
from datetime import date
from typing import Literal

from pydantic import Field

from valuationagent.market.tushare import normalize_a_share_ticker
from valuationagent.schemas.models import ApiModel
from valuationagent.schemas.research import DocumentSummary


class FinancialHistoryRequest(ApiModel):
    years: list[int] = Field(default_factory=list, max_length=10,
        description="完整财务年度，例如[2025,2024]。仅请求按行情日期取数的statistics时省略；它使用任务截止日，不是当前年度财务快照。年度型供应商统计仍须提供完整年度。")
    statements: list[Literal["income", "balancesheet", "cashflow", "statistics"]] = Field(
        default_factory=lambda: ["income", "balancesheet", "cashflow"], min_length=1, max_length=4,
        description="财务报表按years取完整年度。statistics为供应商统计：Tushare返回截止日前30天的每日股数/市值，有独立行情日期，不按years伪造年度快照。")
    ticker: str = Field(default="", max_length=24, description="默认当前任务主体；取可比公司时显式填写其真实证券代码。")


def fetch_history(runtime, args):
    from valuationagent.application.file_workspace import source_bytes
    from valuationagent.application.provider_inputs import provider_candidates

    session, service = runtime.session, runtime.service
    if session.data_source_preference == "upload":
        raise ValueError("NETWORK_OUT_OF_SCOPE: 当前任务仅允许上传数据")
    cutoff = session.information_cutoff_date or session.draft.valuation_date or date.today()
    years = sorted(set(args.years))
    ticker = normalize_a_share_ticker(args.ticker or session.draft.ticker)
    default = getattr(getattr(runtime, "workspaces", None), "runner", None)
    with service._data_client_lock:
        provider = service._market_clients.get(session.session_id, service.history_provider or getattr(default, "data", None))
    if not callable(getattr(provider, "fetch_history", None)):
        return {"status": "not_configured", "instruction": "没有已连接的结构化财务数据服务。继续用search_sources(source_route=web)查财务网页/表格或官方披露，不必等待用户配置。"}
    date_only = set(getattr(provider, "history_date_only_statements", ()))
    if set(args.statements) - date_only and (not years or any(year < 1990 or year >= cutoff.year for year in years)):
        raise ValueError("ANNUAL_PERIOD_REQUIRED: years须为截止日之前已结束的完整年度，不得将季度当全年；仅按行情日期取数的statistics不受年度门槛限制。")
    from valuationagent.application.issuer_identity import ensure_identity

    identity, failures = None, []
    try:
        identity = ensure_identity(runtime, provider, ticker)
    except ValueError as exc:
        if str(exc).startswith(("SOURCE_CHANGED", "ISSUER_IDENTITY_CHANGED")):
            raise
        failures.append({"statement": "issuer_identity", "error": str(exc)[:800],
            "instruction": "原始财务仍可读取，但身份未核对的API字段不得直接入模；换可核验来源，不猜名称/代码，不重复相同失败接口。"})
    documents = []
    for statement in dict.fromkeys(args.statements):
        service._check_execution()
        if statement not in provider.history_statements:
            failures.append({"statement": statement, "error": "PROVIDER_CAPABILITY: 当前供应商不支持该报表", "supported": list(provider.history_statements)})
            continue
        statement_years = [] if statement in date_only else years
        selection = hashlib.sha256(f"{provider.version}:{statement_years}:{cutoff}".encode()).hexdigest()[:16]
        key = f"{provider.provider_id}:{ticker}:{statement}:{selection}"
        cached = next((doc for doc in session.documents if doc.provider == key), None)
        if cached:
            _, raw = source_bytes(service.store, session, cached.file_id)
            documents.append({"file_id": cached.file_id, "cached": True, "block_count": cached.block_count,
                "warnings": cached.warnings, "input_candidates": provider_candidates(session, cached, raw)})
            continue
        if key in runtime.data_signatures:
            failures.append({"statement": statement, "error": "本轮已尝试该接口；换来源，不重复消耗供应商配额"})
            continue
        runtime.data_signatures.add(key)
        try:
            snapshot = provider.fetch_history(ticker, statement, statement_years, cutoff, service._check_execution)
        except ValueError as exc:
            failures.append({"statement": statement, "error": str(exc)[:600]})
            continue
        meta = service.store.save_upload(f"{ticker}-{statement}.json", "evidence", "application/json", snapshot.raw)
        blocks = [{**block, "block_id": f"{meta['file_id']}:{index + 1}", "file_id": meta["file_id"]} for index, block in enumerate(snapshot.blocks)]
        service.store.save_research_blocks(session.session_id, meta["file_id"], blocks)
        target = normalize_a_share_ticker(session.draft.ticker) if session.draft.ticker else ticker
        document = DocumentSummary(file_id=meta["file_id"], name=meta["original_name"], role="historical_financials" if ticker == target else "comparable_financials",
            block_count=len(blocks), sha256=meta["sha256"], size_bytes=meta["size_bytes"],
            provenance_type="structured_provider", authority_tier="B", source_confidence=.8,
            provider=key, source_url=snapshot.source_url, parse_status="parsed" if blocks else "unreadable",
            warnings=snapshot.warnings)
        session.documents.append(document)
        documents.append({"file_id": document.file_id, "block_count": len(blocks), "accepted_records": snapshot.accepted_records,
            "excluded_records": snapshot.excluded_records, "warnings": snapshot.warnings, "catalog": snapshot.catalog,
            "input_candidates": provider_candidates(session, document, snapshot.raw)})
    instruction = "供应商原始响应已保存，input_candidates已展示有字段契约的原值、单位、主体及期间。按任务选择candidate_id交record_inputs(provider_values)，不必再次read_file或手填映射。其他科目用catalog定位原始记录供LLM解释复核，不直接改名入模。缺失只定向补证，不重复取数、不补零、不把抓取日当披露日。"
    if not session.draft.ticker:
        instruction = "原始响应已保存，请求代码有效；当前任务尚未保存目标代码，因此暂不生成目标/可比输入候选。先update_task(draft={ticker:已核对的研究目标代码})，再list_input_candidates读取已有文件，不重新调用取数或反复手抄原始JSON。"
    return {"status": "sources_saved" if documents else "unavailable", "provider": provider.provider_id,
            "documents": documents, "failures": failures, "issuer_identity": identity,
            "date_only_statements": sorted(set(args.statements) & date_only), "information_cutoff": cutoff.isoformat(),
            "target_registration_required": not bool(session.draft.ticker), "instruction": instruction}
