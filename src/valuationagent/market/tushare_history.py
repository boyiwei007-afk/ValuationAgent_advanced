from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from valuationagent.market.history import HistorySnapshot
from valuationagent.market.tushare_contracts import RAW_FIELDS, direct_input_fields


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


def parse_date(value):
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def fetch_history(provider, ticker, statement, years, cutoff, check_cancel):
    if statement == "statistics":
        return fetch_statistics(provider, ticker, cutoff, check_cancel)
    title, doc_id, original_labels = FIELDS[statement]
    labels = {**original_labels, **RAW_FIELDS.get(statement, {})}
    fields = ["ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type", *labels]
    check_cancel()
    response = provider.client.query_snapshot(statement, params={"ts_code": ticker,
        "start_date": f"{min(years) + 1}0101", "end_date": cutoff.strftime("%Y%m%d"), "report_type": "1"}, fields=fields)
    rows = response.records
    blocks, accepted = [], 0
    catalog, available_periods = [], set()
    for index, record in enumerate(rows):
        check_cancel()
        period = parse_date(record.get("end_date"))
        published = parse_date(record.get("f_ann_date") or record.get("ann_date"))
        if (str(record.get("ts_code", "")).upper() != ticker or str(record.get("report_type")) != "1"
                or not period or (period.month, period.day) != (12, 31)
                or not published or not period <= published <= cutoff):
            continue
        available_periods.add(period.isoformat())
        if period.year not in years:
            continue
        accepted += 1
        catalog.append({"record_index": index, "period_end": period.isoformat(), "published_at": published.isoformat(), "first_block_offset": len(blocks)})
        for field, label in labels.items():
            if record.get(field) is None or str(record[field]).strip() == "":
                continue
            try:
                value = Decimal(str(record[field]))
                if not value.is_finite():
                    continue
            except InvalidOperation:
                continue
            period_label = f"{period.isoformat()}期末余额" if statement == "balancesheet" else f"{period.year}年度"
            blocks.append({"text": f"证券代码 {ticker}\n{title}\n单位：元\n项目 {period_label}\n{label} {format(value, 'f')}",
                "location": {"source_type": "structured_provider", "source_url": "https://api.tushare.pro",
                    "published_at": published.isoformat(), "period_end": period.isoformat(),
                    "period_kind": "instant" if statement == "balancesheet" else "annual",
                    "provider_field": field, "record_index": index, "statement": statement,
                    "json_pointer": f"/data/items/{index}",
                    "value_pointer": f"/data/items/{index}/{response.fields.index(field)}",
                    "projection": "provider_schema_field_labels_v2",
                    "schema_reference": f"https://tushare.pro/document/2?doc_id={doc_id}"}})
    return HistorySnapshot(raw=response.raw, source_url="https://api.tushare.pro", blocks=blocks,
        accepted_records=accepted, excluded_records=len(rows) - accepted,
        warnings=["保存供应商原始响应；中文标签视图来自接口字段说明，不是发行人原件。缺失、未来披露及不匹配主体未采纳，冲突记录不静默覆盖。"],
        catalog={"records": catalog, "fields": response.fields, "date_format": "YYYYMMDD", "record_path": "/data/items", "record_columns_path": "/data/fields", "direct_input_fields": direct_input_fields(statement, response.fields), "schema_reference": f"https://tushare.pro/document/2?doc_id={doc_id}",
            "available_annual_periods": sorted(available_periods, reverse=True), "requested_years": sorted(years),
            "period_instruction": "available_annual_periods来自本次原始响应中截止日前已披露的合并年度，不是程序假定已发布；可能包含未选入的较新年度。用户未指定旧基期时优先较新可得年度，不能把旧基期称为最新。"})


def fetch_statistics(provider, ticker, cutoff, check_cancel):
    check_cancel()
    response = provider.client.query_snapshot("daily_basic", params={"ts_code": ticker,
        "start_date": (cutoff - timedelta(days=30)).strftime("%Y%m%d"), "end_date": cutoff.strftime("%Y%m%d")},
        fields=["ts_code", "trade_date", "total_share", "total_mv", "circ_mv", "close", "pe", "pe_ttm", "ps", "ps_ttm"])
    labels = {"total_share": ("发行人总股本", "万股"), "total_mv": ("总市值", "万元"),
        "circ_mv": ("流通市值", "万元"), "close": ("收盘价", "元/股"),
        "pe": ("市盈率（供应商未在此响应给出分母年度）", "ratio"), "pe_ttm": ("市盈率TTM", "ratio"),
        "ps": ("市销率（供应商未在此响应给出分母年度）", "ratio"), "ps_ttm": ("市销率TTM", "ratio")}
    blocks, catalog = [], []
    for index, record in enumerate(response.records):
        check_cancel()
        period = parse_date(record.get("trade_date"))
        if str(record.get("ts_code", "")).upper() != ticker or not period or not cutoff - timedelta(days=30) <= period <= cutoff:
            continue
        lines = [f"证券代码 {ticker}", "Tushare每日指标（供应商行情，非发行人公告）",
            f"行情日期 {period.isoformat()}；收盘数据可得日，不是财报披露日", "以下标签与单位来自daily_basic接口说明："]
        for field, (label, unit) in labels.items():
            if record.get(field) is not None:
                lines.append(f"{field} | {label} | 单位：{unit} | {record[field]}")
        catalog.append({"record_index": index, "as_of": period.isoformat(), "block_offset": len(blocks)})
        blocks.append({"text": "\n".join(lines), "location": {"source_type": "structured_provider",
            "source_url": "https://api.tushare.pro", "published_at": period.isoformat(), "date_semantics": "market_data_as_of_close",
            "period_end": period.isoformat(), "json_pointer": f"/data/items/{index}", "record_index": index,
            "statement": "daily_basic", "projection": "provider_schema_field_labels_v1",
            "column_pointers": {field: f"/data/items/{index}/{position}" for position, field in enumerate(response.fields)},
            "schema_reference": "https://tushare.pro/document/2?doc_id=32"}})
    return HistorySnapshot(raw=response.raw, source_url="https://api.tushare.pro", blocks=blocks,
        accepted_records=len(blocks), excluded_records=len(response.records) - len(blocks),
        warnings=["行情时点不等于估值日，不自动前推。总股本单位万股，市值万元；多地上市公司的总市值需核对股份类别与定价覆盖。PE/PS分母年度未在响应披露，不能直接当FY倍数。"],
        catalog={"records": catalog, "fields": response.fields, "date_format": "YYYYMMDD", "record_path": "/data/items", "record_columns_path": "/data/fields", "direct_input_fields": direct_input_fields("statistics", response.fields), "schema_reference": "https://tushare.pro/document/2?doc_id=32"})
