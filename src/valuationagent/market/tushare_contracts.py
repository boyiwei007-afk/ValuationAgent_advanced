CONTRACT_VERSION = "tushare-inputs-20261007-v4-raw-operands"
RAW_FIELDS = {
    "income": {
        "total_revenue": "营业总收入", "oper_cost": "营业成本", "operate_profit": "营业利润",
        "n_income": "净利润（含少数股东损益）", "minority_gain": "少数股东损益",
        "int_exp": "利息支出", "fin_exp": "财务费用", "fin_exp_int_exp": "财务费用：利息费用",
        "fin_exp_int_inc": "财务费用：利息收入", "invest_income": "投资净收益",
        "fv_value_chg_gain": "公允价值变动净收益", "non_oper_income": "营业外收入",
        "non_oper_exp": "营业外支出", "oth_income": "其他收益", "asset_disp_income": "资产处置收益",
    },
    "balancesheet": {
        "money_cap": "货币资金", "trad_asset": "交易性金融资产", "st_borr": "短期借款",
        "lt_borr": "长期借款", "bond_payable": "应付债券", "lease_liab": "租赁负债",
        "non_cur_liab_due_1y": "一年内到期的非流动负债", "minority_int": "少数股东权益",
        "inventories": "存货", "accounts_receiv": "应收账款", "acct_payable": "应付账款",
        "notes_receiv": "应收票据", "notes_payable": "应付票据", "prepayment": "预付款项",
        "contract_assets": "合同资产", "contract_liab": "合同负债", "oth_payable": "其他应付款",
        "total_assets": "资产总计", "total_liab": "负债合计",
    },
    "cashflow": {
        "n_cashflow_act": "经营活动产生的现金流量净额",
        "depr_fa_coga_dpba": "固定资产折旧、油气资产折耗、生产性生物资产折旧",
        "amort_intang_assets": "无形资产摊销", "lt_amort_deferred_exp": "长期待摊费用摊销",
        "use_right_asset_dep": "使用权资产折旧",
    },
}
CONTRACTS = {
    "income": {"revenue": ("revenue", "元", "1", "consolidated", "33"),
        "n_income_attr_p": ("net_income_parent", "元", "1", "consolidated", "33"),
        "total_profit": ("profit_before_tax", "元", "1", "consolidated", "33"),
        "income_tax": ("income_tax_expense", "元", "1", "consolidated", "33")},
    "cashflow": {"c_pay_acq_const_fiolta": ("cash_paid_for_ppe_intangibles", "元", "1", "consolidated", "44")},
    "balancesheet": {"total_share": ("common_shares", "股", "1", "issuer", "36")},
    "statistics": {"total_share": ("common_shares", "万股", "10000", "issuer", "32"),
        "total_mv": ("market_cap", "万元", "10000", "issuer", "32")},
}
for statement, labels in RAW_FIELDS.items():
    doc_id = {"income": "33", "balancesheet": "36", "cashflow": "44"}[statement]
    CONTRACTS[statement].update({field: (f"raw.tushare_{statement}_{field}", "元", "1", "consolidated", doc_id)
        for field in labels})


def direct_input_fields(statement, fields):
    return {field: {"metric": metric, "unit": unit, "normalization_factor": factor, "scope": scope,
        "admission_kind": "raw_operand" if metric.startswith("raw.") else "standard_input",
        "label": RAW_FIELDS.get(statement, {}).get(field, field),
        "period_kind": "instant" if statement in {"balancesheet", "statistics"} else "annual",
        "contract_version": CONTRACT_VERSION, "schema_reference": f"https://tushare.pro/document/2?doc_id={doc_id}"}
        for field, (metric, unit, factor, scope, doc_id) in CONTRACTS.get(statement, {}).items() if field in fields}
