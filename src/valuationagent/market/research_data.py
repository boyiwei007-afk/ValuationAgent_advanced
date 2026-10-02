"""Provider records as immutable research evidence, not ready-made valuations."""
import json
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import Field

from valuationagent.market.tushare import TushareDataProvider, normalize_a_share_ticker
from valuationagent.schemas.models import ApiModel
from valuationagent.schemas.research import DocumentSummary

FIELDS = {
    "income": ("合并利润表", "33", {
        "revenue": "营业收入", "total_revenue": "营业总收入", "oper_cost": "营业成本",
        "operate_profit": "营业利润", "total_profit": "利润总额", "income_tax": "所得税费用",
        "n_income": "净利润", "n_income_attr_p": "归属于母公司股东的净利润",
        "minority_gain": "少数股东损益", "int_exp": "利息费用", "fin_exp": "财务费用",
    }),
    "balancesheet": ("合并资产负债表", "36", {
        "money_cap": "货币资金", "trad_asset": "交易性金融资产", "st_borr": "短期借款",
        "lt_borr": "长期借款", "bond_payable": "应付债券", "lease_liab": "租赁负债",
        "non_cur_liab_due_1y": "一年内到期的非流动负债", "minority_int": "少数股东权益",
        "inventories": "存货", "accounts_receiv": "应收账款", "acct_payable": "应付账款",
        "total_assets": "资产总计", "total_liab": "负债合计",
    }),
    "cashflow": ("合并现金流量表", "44", {
        "n_cashflow_act": "经营活动产生的现金流量净额",
        "c_pay_acq_const_fiolta": "购建固定资产、无形资产和其他长期资产支付的现金",
        "depr_fa_coga_dpba": "固定资产折旧、油气资产折耗、生产性生物资产折旧",
        "amort_intang_assets": "无形资产摊销", "lt_amort_deferred_exp": "长期待摊费用摊销",
    }),
}


class FinancialHistoryRequest(ApiModel):
    years: list[int] = Field(min_length=1, max_length=10)
    statements: list[Literal["income", "balancesheet", "cashflow"]] = Field(
        default_factory=lambda: list(FIELDS), min_length=1, max_length=3)


def _date(value):
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def fetch_history(runtime, args):
    session, service = runtime.session, runtime.service
    if session.data_source_preference == "upload":
        raise ValueError("NETWORK_OUT_OF_SCOPE: 当前任务仅允许上传数据")
    cutoff = session.information_cutoff_date or session.draft.valuation_date or date.today()
    years = sorted(set(args.years))
    if any(year < 1990 or year >= cutoff.year for year in years):
        raise ValueError("years须为截止日之前已结束的完整年度，不得将季度当全年")
    ticker = normalize_a_share_ticker(session.draft.ticker)
    with service._data_client_lock:
        provider = service._market_clients.get(session.session_id)
    if not isinstance(provider, TushareDataProvider):
        return {"status": "not_configured", "instruction": "没有已连接的结构化财务数据服务。继续用search_sources(source_route=web)查财务网页/表格或官方披露，不必等待用户配置。"}
    documents, failures = [], []
    for statement in dict.fromkeys(args.statements):
        service._check_execution()
        key = f"tushare:{ticker}:{statement}:{','.join(map(str, years))}:{cutoff}"
        cached = next((doc for doc in session.documents if doc.provider == key), None)
        if cached:
            documents.append({"file_id": cached.file_id, "cached": True})
            continue
        if key in runtime.data_signatures:
            failures.append({"statement": statement, "error": "本轮已尝试该接口；换来源，不重复消耗供应商配额"})
            continue
        runtime.data_signatures.add(key)
        title, doc_id, labels = FIELDS[statement]
        fields = ["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", *labels]
        try:
            rows = provider.client.query(statement, params={"ts_code": ticker,
                "start_date": f"{min(years) + 1}0101", "end_date": cutoff.strftime("%Y%m%d"), "report_type": "1"}, fields=fields)
        except ValueError as exc:
            failures.append({"statement": statement, "error": str(exc)[:600]})
            continue
        accepted = []
        for record_index, record in enumerate(rows):
            period = _date(record.get("end_date"))
            published = _date(record.get("f_ann_date") or record.get("ann_date"))
            if (str(record.get("ts_code", "")).upper() != ticker or str(record.get("report_type")) != "1"
                    or not period or period.year not in years or (period.month, period.day) != (12, 31)
                    or not published or not period <= published <= cutoff):
                continue
            accepted.append((record_index, record, period, published))
        raw = json.dumps({"provider": "tushare", "statement": statement, "rows": rows}, ensure_ascii=False, default=str).encode()
        meta = service.store.save_upload(f"{ticker}-{statement}.json", "evidence", "application/json", raw)
        blocks = []
        for record_index, record, period, published in accepted:
            prefix = f"证券代码 {ticker}\n{title}\n单位：元\n项目 {period.year}年度\n"
            for field, label in labels.items():
                if record.get(field) is None or str(record[field]).strip() == "":
                    continue
                try:
                    value = Decimal(str(record[field]))
                    if not value.is_finite():
                        continue
                except InvalidOperation:
                    continue
                blocks.append({"block_id": f"{meta['file_id']}:{len(blocks) + 1}", "file_id": meta["file_id"],
                    "text": prefix + f"{label} {format(value, 'f')}",
                    "location": {"source_type": "structured_provider", "source_url": "https://api.tushare.pro",
                        "published_at": published.isoformat(), "period_end": period.isoformat(),
                        "provider_field": field, "record_index": record_index, "statement": statement,
                        "schema_reference": f"https://tushare.pro/document/2?doc_id={doc_id}"}})
        service.store.save_research_blocks(session.session_id, meta["file_id"], blocks)
        document = DocumentSummary(file_id=meta["file_id"], name=meta["original_name"], role="historical_financials",
            block_count=len(blocks), sha256=meta["sha256"], size_bytes=meta["size_bytes"],
            provenance_type="structured_provider", authority_tier="B", source_confidence=.8,
            provider=key, source_url="https://api.tushare.pro", parse_status="parsed" if blocks else "unreadable",
            warnings=["供应商结构化转录，不是发行人原件；缺失、未来披露及不匹配主体未采纳，冲突记录不静默覆盖。"])
        session.documents.append(document)
        documents.append({"file_id": document.file_id, "block_count": len(blocks), "accepted_records": len(accepted),
                          "excluded_records": len(rows) - len(accepted)})
    return {"status": "sources_saved" if documents else "unavailable", "documents": documents, "failures": failures,
            "instruction": "JSON响应快照已保存；read_file读取后由LLM解释语义，再用extract_observations提交并复核。没有直接写入财务事实，不补零、不默认现金可全部进入权益桥。"}
