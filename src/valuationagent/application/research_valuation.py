"""Convert confirmed research facts into the strict valuation contract."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from valuationagent.finance.integrity import (
    EQUITY_BRIDGE_REVIEW_LABELS,
    equity_bridge_review_findings,
)
from valuationagent.application.source_risks import source_risk_inventory

from valuationagent.schemas.models import (
    AssumptionInputs,
    CompanyInput,
    EvidenceRef,
    FinancialSnapshot,
    ValuationRequest,
    required_financial_metrics,
)

D = Decimal
MIN_AUTOMATIC_HISTORY_YEARS = 4
MAX_SHARE_EVIDENCE_AGE_DAYS = 120
MATERIAL_SHARE_CHANGE = D("0.01")


def _information_cutoff(session):
    return session.information_cutoff_date or session.draft.valuation_date


METRIC_ALIASES = {
    # These statement lines can coexist with different amounts. They must
    # never collapse into one model field or manufacture an approval conflict.
    "revenue": {"revenue", "营业收入"},
    "total_revenue": {"total_revenue", "营业总收入"},
    "main_business_revenue": {"main_business_revenue", "主营业务收入"},
    "ebit": {"ebit", "息税前利润"},
    "ebit_margin": {"ebit_margin", "ebit率", "息税前利润率"},
    "tax_rate": {"tax_rate", "所得税率", "实际税率"},
    "depreciation_amortization": {
        "depreciation_amortization", "折旧摊销", "折旧与摊销", "折旧及摊销",
    },
    "capital_expenditure": {"capital_expenditure", "capex", "资本性支出", "资本开支"},
    "change_operating_nwc": {
        "change_operating_nwc", "经营性营运资本变动", "营运资本变动", "Δnwc", "delta_nwc",
    },
    "cash_and_non_operating_assets": {
        "cash_and_non_operating_assets", "现金及非经营性资产", "货币资金",
    },
    # Keep liquid investments separate from monetary funds.  Mapping both
    # rows to the aggregate cash field silently overwrote one of them and
    # understated the enterprise-to-equity bridge.
    "trading_financial_assets": {
        "trading_financial_assets", "交易性金融资产",
    },
    "interest_bearing_debt": {"interest_bearing_debt", "有息负债", "带息债务"},
    "common_shares": {
        "common_shares", "总股本", "公司总股本", "普通股股数", "普通股股份总数",
        "普通股股份总额", "股份总数",
    },
    "diluted_shares": {
        "diluted_shares", "稀释后股份总数", "稀释后普通股股数", "完全摊薄股数",
    },
    "net_income_parent": {"net_income_parent", "归母净利润", "归属于母公司股东的净利润", "归属于上市公司股东的净利润"},
    "ebitda": {"ebitda", "息税折旧摊销前利润"},
    # Raw statement lines accepted as deterministic derivation inputs.  The
    # LLM extracts and cites them; this assembler, rather than the LLM, performs
    # every calculation below.
    "profit_before_tax": {"profit_before_tax", "利润总额", "税前利润"},
    "operating_profit": {"operating_profit", "营业利润"},
    "income_tax_expense": {"income_tax_expense", "所得税费用"},
    # Bare “利息支出” can be operating cost of a consolidated finance
    # subsidiary (paired with interest income), not a financing add-back for
    # EBIT.  Only an explicitly labelled finance-cost interest amount is
    # normalized here; ambiguous rows remain visible evidence but cannot enter
    # the deterministic calculation.
    "interest_expense": {
        "interest_expense", "利息费用", "财务费用中的利息费用",
        "财务费用-利息支出", "财务费用中的利息支出",
    },
    # Kept as an auditable source fact, but intentionally excluded from the
    # ordinary industrial-company EBIT add-back.  A finance subsidiary's
    # deposit/interbank interest is an operating cost of that business, even
    # though the printed row may simply say “利息支出”.
    "financial_subsidiary_interest_expense": {
        "financial_subsidiary_interest_expense", "金融子公司经营利息支出",
        "财务公司经营利息支出",
    },
    "depreciation_fixed_assets": {
        "depreciation_fixed_assets", "固定资产折旧", "固定资产折旧费",
        "固定资产折旧油气资产折耗生产性生物资产折旧",
    },
    "amortization_intangible_assets": {
        "amortization_intangible_assets", "无形资产摊销",
    },
    "amortization_long_term_deferred_expenses": {
        "amortization_long_term_deferred_expenses", "长期待摊费用摊销",
    },
    "depreciation_right_of_use": {
        "depreciation_right_of_use", "amortization_right_of_use_assets",
        "使用权资产折旧", "使用权资产摊销", "使用权资产折旧摊销",
    },
    "cash_paid_for_ppe_intangibles": {
        "cash_paid_for_ppe_intangibles",
        "购建固定资产无形资产和其他长期资产支付的现金",
        "购建固定资产、无形资产和其他长期资产支付的现金",
    },
    "short_term_borrowings": {"short_term_borrowings", "短期借款"},
    "current_portion_non_current_liabilities": {
        "current_portion_non_current_liabilities", "一年内到期的非流动负债",
    },
    "long_term_borrowings": {"long_term_borrowings", "长期借款"},
    "bonds_payable": {"bonds_payable", "应付债券"},
    "lease_liabilities": {"lease_liabilities", "租赁负债"},
    "minority_interest": {"minority_interest", "noncontrolling_interest", "少数股东权益"},
    "preferred_equity": {"preferred_equity", "优先股权益", "优先股"},
    "associates_and_non_operating_investments": {
        "associates_and_non_operating_investments", "联营及非经营性投资", "长期股权投资",
    },
    "unfunded_pension": {"unfunded_pension", "未弥补养老金缺口"},
    "non_operating_provisions": {
        "non_operating_provisions", "非经营性预计负债", "预计负债",
    },
    "restricted_cash": {
        "restricted_cash", "受限货币资金", "使用受到限制的货币资金",
        "存放中央银行法定存款准备金", "法定存款准备金",
    },
    "financial_institution_deposits": {
        "financial_institution_deposits", "吸收存款及同业存放",
    },
    "interbank_lending": {"interbank_lending", "拆出资金"},
    "restricted_interbank_deposits": {
        "restricted_interbank_deposits", "不能随时支取的同业存款", "受限拆出资金",
    },
    "operating_nwc": {"operating_nwc", "经营性营运资本", "经营营运资本"},
    "inventory_decrease": {"inventory_decrease", "存货的减少", "存货减少"},
    "operating_receivables_decrease": {
        "operating_receivables_decrease", "经营性应收项目的减少", "经营性应收项目减少",
    },
    "operating_payables_increase": {
        "operating_payables_increase", "经营性应付项目的增加", "经营性应付项目增加",
    },
    "accounts_receivable": {"accounts_receivable", "应收账款"},
    "notes_receivable": {"notes_receivable", "应收票据"},
    "receivables_financing": {"receivables_financing", "应收款项融资"},
    "contract_assets": {"contract_assets", "合同资产"},
    "prepayments": {"prepayments", "预付款项", "预付账款"},
    "inventory": {"inventory", "存货"},
    "accounts_payable": {"accounts_payable", "应付账款"},
    "notes_payable": {"notes_payable", "应付票据"},
    "contract_liabilities": {"contract_liabilities", "合同负债"},
}

METRIC_LABELS = {
    **EQUITY_BRIDGE_REVIEW_LABELS,
    "revenue": "营业收入",
    "total_revenue": "营业总收入（独立科目）",
    "main_business_revenue": "主营业务收入（独立科目）",
    "ebit_margin": "EBIT 利润率",
    "tax_rate": "所得税率",
    "depreciation_amortization": "折旧与摊销",
    "capital_expenditure": "资本开支",
    "change_operating_nwc": "经营性营运资本变动",
    "cash_and_non_operating_assets": "现金及非经营性资产",
    "interest_bearing_debt": "有息负债",
    "common_shares": "普通股股数",
    "diluted_shares": "稀释后普通股股数",
    "net_income_parent": "归母净利润",
    "ebitda": "EBITDA",
    "financial_subsidiary_interest_expense": "金融子公司经营利息支出（模型外保留）",
}

REQUIRED_METRICS = {
    "revenue", "tax_rate", "depreciation_amortization", "capital_expenditure",
    "change_operating_nwc", "cash_and_non_operating_assets", "interest_bearing_debt",
    "common_shares", "net_income_parent", "ebitda", "ebit_margin",
}

OPTIONAL_BRIDGE_METRICS = {
    "diluted_shares",
    "lease_liabilities",
    "minority_interest",
    "preferred_equity",
    "associates_and_non_operating_investments",
    "unfunded_pension",
    "non_operating_provisions",
}

DEPRECIATION_COMPONENTS = (
    "depreciation_fixed_assets",
    "amortization_intangible_assets",
    "amortization_long_term_deferred_expenses",
)

DEBT_COMPONENTS = (
    "short_term_borrowings",
    "current_portion_non_current_liabilities",
    "long_term_borrowings",
    "bonds_payable",
    "lease_liabilities",
)

NWC_CASH_FLOW_COMPONENTS = (
    "inventory_decrease",
    "operating_receivables_decrease",
    "operating_payables_increase",
)

NWC_ASSET_COMPONENTS = (
    "accounts_receivable", "notes_receivable", "receivables_financing",
    "contract_assets", "prepayments", "inventory",
)

NWC_LIABILITY_COMPONENTS = (
    "accounts_payable", "notes_payable", "contract_liabilities",
)

DERIVATION_HINTS = {
    "ebit_margin": "利润总额、利息支出（系统推导 EBIT 与利润率）",
    "tax_rate": "所得税费用、利润总额（系统推导实际税率）",
    "depreciation_amortization": "固定资产折旧、无形资产摊销、长期待摊费用摊销；存在租赁时另核对独立披露的使用权资产折旧/摊销",
    "capital_expenditure": "购建固定资产、无形资产和其他长期资产支付的现金",
    "change_operating_nwc": "现金流量补充资料中的存货减少、经营性应收减少、经营性应付增加",
    "interest_bearing_debt": "短期借款、一年内到期非流动负债、长期借款、应付债券、租赁负债",
    "ebitda": "利润总额、利息支出和折旧摊销科目（系统推导）",
}

ASSUMPTION_ALIASES = {
    "wacc": {"wacc", "加权平均资本成本"},
    "terminal_growth": {"terminal_growth", "永续增长率", "终值增长率"},
    "terminal_tax_rate": {"terminal_tax_rate", "永续期税率"},
    "risk_free_rate": {"risk_free_rate", "无风险利率"},
    "equity_risk_premium": {"equity_risk_premium", "股权风险溢价", "市场风险溢价"},
    "beta": {"beta", "贝塔", "贝塔系数"},
    "debt_cost": {"debt_cost", "债务成本"},
    "market_cap": {"market_cap", "市值", "总市值"},
    "quarterly_average_market_cap": {
        "quarterly_average_market_cap", "估值日前四季度平均市值", "四季度平均市值",
    },
    "annual_average_market_cap": {
        "annual_average_market_cap", "年度平均市值", "年均市值",
    },
    "stable_roic": {"stable_roic", "稳定期roic", "永续期roic"},
}


def _normalized_metric(value: str, aliases: dict[str, set[str]]) -> str | None:
    # Cash-flow labels commonly append the printed sign convention, sometimes
    # across lines. Removing that suffix does not change the signed value.
    value = re.sub(r"[（(](?:增加|减少)以[“\"']?[－−-][”\"']?号填列[）)]", "", value)
    value = re.sub(r"^\s*(?:\d+|[一二三四五六七八九十百]+)[.．、]\s*", "", value)
    value = re.sub(r"^\s*其中[:：]\s*", "", value)
    value = re.sub(r"[（(](?:净亏损|亏损总额|亏损|损失)以[“\"']?[－−-][”\"']?号填列[）)]", "", value)
    key = re.sub(r"[\s/／、·（）()_-]+", "", value).casefold()
    for canonical, names in aliases.items():
        if key in {re.sub(r"[\s/／、·（）()_-]+", "", item).casefold() for item in names}:
            return canonical
    # A display label may append a synonymous name, e.g. 普通股股数（总股本）.
    # Require both complete labels to resolve identically; stripping arbitrary
    # brackets would silently conflate 营业收入（营业总收入） and other bases.
    paired = re.fullmatch(r"\s*(.+?)[（(]([^（）()]+)[）)]\s*", value)
    if paired:
        primary = _normalized_metric(paired[1], aliases)
        display_unit = re.sub(r"\s+", "", paired[2]).removeprefix("人民币")
        if display_unit in {"元", "千元", "万元", "百万元", "亿元", "股", "千股", "万股", "百万股", "亿股", "元/股", "元／股", "%", "％"}:
            return primary
        alternate = _normalized_metric(paired[2], aliases)
        if primary is not None and primary == alternate:
            return primary
    return None


def normalize_financial_metric(value: str) -> str | None:
    """Public canonicalizer shared by extraction and valuation handoff."""
    return _normalized_metric(value, METRIC_ALIASES)


def mapped_financial_metric(fact) -> str | None:
    """Resolve contextual mappings or exact accounting aliases, never fuzzy matches."""
    proposed = str(getattr(fact, "standard_metric", "") or "").strip()
    if proposed:
        return normalize_financial_metric(proposed)
    return normalize_financial_metric(fact.metric)


def financial_mapping_issue(fact) -> str:
    from valuationagent.application.observation_consistency import period_readback_issue

    if fact.status == "confirmed" and (issue := period_readback_issue(fact)):
        return issue
    if fact.role != "historical":
        return ""
    source_metric = normalize_financial_metric(fact.metric)
    target = mapped_financial_metric(fact)
    if target in {"ebit", "ebitda", "ebit_margin"} and (
        source_metric in {"operating_profit", "profit_before_tax", "net_income_parent"}
        or re.fullmatch(r"(?:净利润|net_income|net_income_total)", fact.metric.strip(), re.I)
    ):
        return "利润口径越级：营业利润、利润总额或净利润不能直接映射为EBIT/EBITDA/EBIT利润率；保留原始科目，补齐利润总额及核验融资利息后由程序推导。"
    revenue_metrics = {"revenue", "total_revenue", "main_business_revenue"}
    if source_metric in revenue_metrics and target in revenue_metrics and source_metric != target:
        return f"收入口径冲突：原文标签对应{source_metric}，却提交为{target}；营业收入、营业总收入、主营业务收入须保留各自原始指标，不能用语义映射互换。"
    return ""


def _period(value: str) -> date | None:
    raw = value.strip()
    if raw.startswith(("interim:", "ttm:")):
        return None
    if re.search(
        r"(?i)(?:Q[1-4]|第?[一二三四1234]季度|季报|半年度?|半年报|H1)",
        raw,
    ):
        return None
    annual_range = re.search(
        r"(?<!\d)(20\d{2})年?\s*1\s*(?:[-—至到~～])\s*12\s*月",
        raw,
    )
    if annual_range:
        return date(int(annual_range.group(1)), 12, 31)
    matches = list(re.finditer(
        r"(?<!\d)(20\d{2})(?:[-年/.](\d{1,2}))?(?:[-月/.](\d{1,2}))?",
        raw,
    ))
    if not matches:
        return None
    # A date range such as 2024-01-01 to 2024-12-31 represents the period end.
    match = matches[-1]
    year = int(match.group(1))
    month = int(match.group(2) or 12)
    day = int(match.group(3) or (31 if month == 12 else 1))
    try:
        return date(year, month, day)
    except ValueError:
        return None


class ResearchValuationAssembler:
    """Strict handoff: only confirmed, scoped and normalized facts are accepted."""

    def __init__(self, *, block_loader=None):
        self._block_loader = block_loader

    def source_risks(self, session, period_end):
        blocks = self._block_loader(session) if self._block_loader else []
        if isinstance(blocks, dict):
            blocks = list(blocks.values())
        return source_risk_inventory(session, blocks, period_end)

    @staticmethod
    def _verified_dated_issuer_shares(fact):
        """Use an explicitly dated issuer total as well as a dated report total.

        Both bindings have already checked issuer identity, count, unit and
        cutoff against the source.  The old handoff recognized only the
        relative report-disclosure wording and discarded an equally verified
        explicit calendar date.
        """
        checked = fact.verification
        if proof := checked.get("reading_proof"):
            from valuationagent.application.observation_extraction import FACTORS, digest, observation_amount
            from valuationagent.application.observation_consistency import period_readback_issue

            row, basis = proof.get("row", {}), proof.get("basis", {})
            body = {key: value for key, value in proof.items() if key != "proof_id"}
            return (
                fact.status == "confirmed" and not fact.warnings
                and not period_readback_issue(fact)
                and checked.get("observation", {}).get("status") == "verified"
                and checked.get("semantic_review", {}).get("status") == "supported"
                and proof.get("proof_id") == "proof_" + digest(body)[:32]
                and fact.scope == basis.get("scope") == "issuer"
                and fact.unit == basis.get("unit") and fact.unit in {"股", "千股", "万股", "百万股", "亿股"}
                and row.get("period_kind") == "instant"
                and fact.period == row.get("period_end")
                and mapped_financial_metric(fact) == row.get("standard_metric") == "common_shares"
                and fact.raw_value == row.get("raw_value")
                and D(fact.normalized_value) == observation_amount(row["raw_value"], basis["unit"]) * D(FACTORS[basis["unit"]])
            )
        return (
            fact.scope == "issuer"
            and checked.get("scope") == "issuer"
            and checked.get("binding") in {
                "issuer_common_shares", "issuer_report_disclosure_shares",
                "issuer_share_capital_note", "issuer_share_change_table",
                "issuer_listing_issued_shares",
            }
            and (as_of := _period(fact.period)) is not None
            and checked.get("period_end") == as_of.isoformat()
        )

    @staticmethod
    def _latest_statement_period(session):
        periods = [
            period
            for fact in session.facts
            if fact.status == "confirmed"
            and not fact.warnings
            and fact.role == "historical"
            and fact.scope == "consolidated"
            and fact.normalized_value is not None
            and (period := _period(fact.period)) is not None
            and (period.month, period.day) == (12, 31)
            and (
                not session.draft.valuation_date
                or period <= session.draft.valuation_date
            )
        ]
        return max(periods, default=None)

    def _target_source_blocks(self, session):
        """Return original blocks belonging to the target issuer only."""
        if not self._block_loader:
            return []
        blocks = self._block_loader(session)
        if isinstance(blocks, dict):
            blocks = list(blocks.values())
        else:
            blocks = list(blocks)
        grouped = defaultdict(list)
        for block in blocks:
            location = block.get("location") or {}
            if location.get("source_type") == "web_search":
                continue
            file_id = str(
                block.get("file_id")
                or block.get("block_id", "").split(":", 1)[0]
            )
            grouped[file_id].append(block)
        company = re.sub(r"\s+", "", session.draft.company or "")
        short_company = re.sub(
            r"(?:股份有限公司|有限责任公司|有限公司)$", "", company
        )
        ticker = (session.draft.ticker or "").split(".")[0]
        selected = []
        for file_blocks in grouped.values():
            identity = re.sub(
                r"\s+", "", "\n".join(str(item.get("text") or "") for item in file_blocks)
            )
            matched = (
                (bool(company) and company in identity)
                or (len(short_company) >= 2 and short_company in identity)
                or (
                    bool(ticker)
                    and re.search(r"(?<!\d)" + re.escape(ticker) + r"(?!\d)", identity)
                )
            )
            if matched:
                selected.extend(file_blocks)
        return selected

    def capital_structure_timing_issue(self, session):
        """Enforce point-in-time shares and post-baseline capital actions.

        A valuation can use an older income-statement baseline, but it cannot
        silently combine that balance sheet with a materially different share
        count after an IPO, placement or rights issue.  The first stage asks
        for a bounded official status check; once completion is verified, the
        second stage safely stops unsupported bridge arithmetic instead of
        producing a deceptively precise per-share value.
        """
        valuation_date = session.draft.valuation_date
        baseline = self._latest_statement_period(session)
        if not valuation_date or not baseline or valuation_date <= baseline:
            return None

        verified = []
        for fact in session.facts:
            if (
                fact.status == "confirmed"
                and not fact.warnings
                and fact.role == "historical"
                and mapped_financial_metric(fact) == "common_shares"
                and fact.normalized_value is not None
                and self._verified_dated_issuer_shares(fact)
                and (as_of := _period(fact.period)) is not None
                and as_of <= valuation_date
            ):
                verified.append((as_of, D(fact.normalized_value), fact))

        annual = [entry for entry in verified if entry[0] == baseline]
        later = [entry for entry in verified if baseline < entry[0] <= valuation_date]
        latest = max(verified, key=lambda entry: entry[0], default=None)

        # A denominator older than one reporting quarter is not point-in-time
        # evidence.  This is a targeted corporate-action check, not a request
        # to rebuild every financial statement at the valuation date.
        if latest is not None and (valuation_date - latest[0]).days > MAX_SHARE_EVIDENCE_AGE_DAYS:
            latest_label = latest[0].isoformat()
            return {
                "kind": "research_required",
                "code": "STALE_POINT_IN_TIME_SHARES",
                "latest_verified_date": latest_label,
                "age_days": (valuation_date - latest[0]).days,
                "required_since": (valuation_date - timedelta(days=MAX_SHARE_EVIDENCE_AGE_DAYS)).isoformat(),
                "baseline_policy": "保留完整年度利润表基期；股数按独立截止日核验，不以季报或中报替换年度收入。",
                "message": (
                    f"普通股股数最新可核验截止日为{latest_label}，距估值日"
                    f"{valuation_date.isoformat()}超过{MAX_SHARE_EVIDENCE_AGE_DAYS}天。"
                    "先检查已有股数候选的字段映射、发行人口径与精确日期；仍缺时定向核验正式公告中的"
                    "增发、配股、H股上市、可转债转股、回购注销及最新发行人总股数；"
                    "同一来源和目标避免重复检索；该缺口不禁止处理其他年度、可比公司或修复已有证据。"
                ),
            }

        information_cutoff = session.information_cutoff_date or valuation_date
        compact_blocks = []
        for block in self._target_source_blocks(session):
            published = _period(
                str((block.get("location") or {}).get("published_at") or "")[:10]
            )
            if published and published > information_cutoff:
                continue
            compact_blocks.append((
                block,
                re.sub(r"\s+", "", str(block.get("text") or "")),
            ))
        plan_pattern = re.compile(
            r"(?:关于|有關).{0,24}(?:发行H股|發行H股|境外公开发行H股|"
            r"向特定对象发行|向特定對象發行|非公开发行|非公開發行|配股|公开增发|公開增發|"
            r"发行股份购买资产|發行股份購買資產).{0,50}(?:议案|議案|方案|上市|申请|申請)"
        )
        planned_blocks = [
            block for block, text in compact_blocks if plan_pattern.search(text)
        ]
        # PDF page boundaries frequently split “approved/proposed” from the
        # very next paragraph saying that the same issue was listed, completed
        # or terminated.  Inspect a narrow same-document neighbourhood before
        # labelling the action pending; never let a page break turn completed
        # H-share issuance into a false blocker.
        completion_pattern = re.compile(
            r"(?:H股股票|H股股份|境外上市股份).{0,100}(?:挂牌并上市交易|"
            r"获准上市|发行完成|上市完成)|"
            r"超额配售权.{0,40}(?:悉数行使|已获(?:悉数)?行使)|"
            r"(?:发行|配股|增发|配售).{0,60}(?:已完成|完成登记|挂牌上市|上市交易)|"
            r"(?:终止|撤回).{0,40}(?:发行|配股|上市|申请)"
        )

        def source_and_index(block):
            block_id = str(block.get("block_id") or "")
            source = str(block.get("file_id") or block_id.split(":", 1)[0])
            try:
                index = int(block_id.rsplit(":", 1)[1])
            except (IndexError, ValueError):
                index = None
            return source, index

        unresolved_plans = []
        for planned in planned_blocks:
            source, index = source_and_index(planned)
            neighbours = []
            for candidate, text in compact_blocks:
                other_source, other_index = source_and_index(candidate)
                if other_source != source:
                    continue
                if index is None or other_index is None or abs(other_index - index) <= 2:
                    neighbours.append(text)
            if not completion_pattern.search("\n".join(neighbours)):
                unresolved_plans.append(planned)
        planned_blocks = unresolved_plans

        if later and annual:
            latest_later = max(later, key=lambda entry: entry[0])
            annual_value = annual[-1][1]
            if annual_value > 0:
                change = abs(latest_later[1] - annual_value) / annual_value
                if change >= MATERIAL_SHARE_CHANGE:
                    direction = "增加" if latest_later[1] > annual_value else "减少"
                    return {
                        "kind": "unsupported_model_scope",
                        "code": "MATERIAL_POST_BALANCE_SHEET_CAPITAL_ACTION",
                        "message": (
                            f"估值基准日{baseline.isoformat()}后、估值日前，发行人股数已由"
                            f"{annual_value}股{direction}至{latest_later[1]}股（变动"
                            f"{(change * D('100')):.2f}%）。当前基准现金、债务和股本来自变动前报表；"
                            "在没有估值日同日资产负债表，或经核验的融资/回购现金净额及资金使用情况时，"
                            "不能只替换每股分母后继续计算。请保存现有证据并交付说明报告；"
                            "后续可用覆盖该事项的中期/季度报表重新估值。"
                        ),
                    }

        # A formal pending issuance in the issuer's own filing requires one
        # follow-up result/status check.  A verified issuer total within 30
        # days of valuation is sufficient only when it shows no material
        # post-baseline change; a completed material change was handled above.
        if planned_blocks:
            recent_later = [
                entry for entry in later if (valuation_date - entry[0]).days <= 30
            ]
            if not recent_later:
                source_ids = ", ".join(
                    str(block.get("block_id") or "") for block in planned_blocks[:2]
                )
                return {
                    "kind": "research_required",
                    "code": "PENDING_CAPITAL_ACTION_STATUS",
                    "message": (
                        "发行人正式资料披露了可能改变股数和现金桥接的发行/配股/上市计划"
                        f"（来源块：{source_ids}），但估值日前尚无足够接近估值日的正式完成、终止"
                        "或最新总股数证据。请定向检索交易所最终配发结果、股本变动或终止公告；"
                        "不得继续补无关财务字段，也不得让用户确认一个未经核验的股数。"
                    ),
                }
        return None

    def unsupported_model_scope_issue(self, session):
        timing = self.capital_structure_timing_issue(session)
        if timing and timing["kind"] == "unsupported_model_scope":
            return timing["message"]
        return self.model_scope_issue(session)

    def later_issuer_shares_issue(self, session):
        """Flag an annual-report share count contradicted by its later disclosure.

        The dividend-eligibility base is ignored: only the issuer's explicitly
        named total *before* buyback deductions is compared. A later figure is
        a research lead, not a verified substitute or an automatic adjustment.
        """
        if not self._block_loader:
            return ""
        annual = []
        for fact in session.facts:
            period = _period(fact.period)
            if (fact.status == "confirmed" and not fact.warnings and fact.role == "historical"
                    and fact.scope == "issuer" and mapped_financial_metric(fact) == "common_shares"
                    and period and (period.month, period.day) == (12, 31)
                    and fact.normalized_value is not None):
                annual.append((fact.block_id.split(":", 1)[0], period,
                               D(fact.normalized_value), fact.source_sha256))
        if not annual:
            return ""
        blocks = self._block_loader(session)
        if isinstance(blocks, dict):
            blocks = blocks.values()
        source_text = {}
        for block in blocks:
            file_id = str(block.get("file_id") or block.get("block_id", "").split(":", 1)[0])
            if any(source_id == file_id for source_id, _, _, _ in annual):
                source_text.setdefault(file_id, []).append(block.get("text", ""))
        for file_id, period, counted, source_hash in annual:
            if session.draft.valuation_date and session.draft.valuation_date <= period:
                continue
            for text in source_text.get(file_id, []):
                compacted = re.sub(r"\s+", "", text)
                for match in re.finditer(
                    r"截至本(?:年度)?报告披露之日(?:本公司|公司)(?:的)?总股本(?:为)?"
                    r"(\d{1,3}(?:[,，]\d{3})+|\d+)股", compacted,
                ):
                    later = D(match[1].replace(",", "").replace("，", ""))
                    if later != counted:
                        corroborated = any(
                            fact.status == "confirmed" and not fact.warnings
                            and fact.role == "historical" and fact.scope == "issuer"
                            and mapped_financial_metric(fact) == "common_shares"
                            and (fact.block_id.split(":", 1)[0] == file_id
                                 or bool(source_hash and source_hash == fact.source_sha256))
                            and self._verified_dated_issuer_shares(fact)
                            and (as_of := _period(fact.period)) and period < as_of
                            and (not session.draft.valuation_date or as_of <= session.draft.valuation_date)
                            and fact.normalized_value is not None
                            and D(fact.normalized_value) == later
                            for fact in session.facts
                        )
                        if corroborated:
                            continue
                        return (f"{period.isoformat()}年末普通股股数{counted}股与同份年报披露日的"
                                f"公司总股本{later}股不同。不能将年末数直接用作较晚估值日的"
                                "每股价值分母；请取得并核验估值日之前最新股数与期间变动。"
                                "分红基数不能替代总股数。")
        return ""

    def model_scope_issue(self, session):
        """Known unsupported economics should surface before unrelated gaps.

        This uses only confirmed clean historical amounts for the latest
        observed annual period. It never interprets missing risk fields as zero.
        """
        rows = {}
        for fact in session.facts:
            period = _period(fact.period)
            if (fact.status != "confirmed" or fact.warnings or fact.role != "historical"
                    or fact.scope != "consolidated" or fact.normalized_value is None
                    or not period or period.month != 12 or period.day != 31
                    or session.draft.valuation_date and period > session.draft.valuation_date
                    or fact.published_at and _information_cutoff(session)
                    and fact.published_at > _information_cutoff(session)):
                continue
            metrics = rows.setdefault(period, {})
            metric = mapped_financial_metric(fact)
            if metric in EQUITY_BRIDGE_REVIEW_LABELS:
                # Preserve a disclosed non-zero even when another conflicting
                # candidate says zero; ambiguity must not bypass scope review.
                value = D(fact.normalized_value)
                if metric not in metrics or value != 0:
                    metrics[metric] = value
        if not rows:
            return ""
        findings = equity_bridge_review_findings(self._selected_methods(session), rows[max(rows)])
        blocking = next((item for item in findings if item.severity == "blocking"), None)
        return blocking.message if blocking else ""

    @staticmethod
    def _selected_methods(session):
        """Use a reduced method set only after the formal plan was accepted."""
        requested = list(session.draft.methods or [])
        override = list(getattr(session, "valuation_methods_override", []) or [])
        if override and set(override) <= set(requested):
            return override
        return requested or ["dcf", "pe", "ps", "ev_ebitda"]

    @staticmethod
    def _manual_forecast(session):
        proposal = session.forecast_proposal
        return bool(proposal and proposal.status == "confirmed"
                    and (proposal.inputs.revenue_growth_scenarios or proposal.inputs.revenue_growth))

    @staticmethod
    def pending_blockers(session):
        """Pending research notes are not all inputs to the selected model.

        Relevant unresolved values and proposed corrections still block;
        unrelated metrics and duplicate evidence for an accepted value do not.
        Required-input and reconciliation checks remain authoritative below.
        """
        selected = ResearchValuationAssembler._selected_methods(session)
        required = set(required_financial_metrics(selected))
        if "dcf" in selected:
            # These drivers can be deterministically degraded by the formal
            # finance model.  Unresolved candidates remain visible evidence,
            # but they must not keep the agent in a research/confirmation loop.
            required -= {
                "depreciation_amortization",
                "capital_expenditure",
                "change_operating_nwc",
            }
        if {"dcf", "ev_ebitda"} & set(selected):
            required.update(EQUITY_BRIDGE_REVIEW_LABELS)
        dependencies = {
            "ebit_margin": {"ebit", "revenue"},
            "ebit": {"profit_before_tax", "interest_expense"},
            "tax_rate": {"income_tax_expense", "profit_before_tax"},
            "ebitda": {"ebit", "depreciation_amortization"},
            "depreciation_amortization": {*DEPRECIATION_COMPONENTS, "depreciation_right_of_use"},
            "capital_expenditure": {"cash_paid_for_ppe_intangibles"},
            "interest_bearing_debt": set(DEBT_COMPONENTS),
            "change_operating_nwc": {*NWC_CASH_FLOW_COMPONENTS, "operating_nwc"},
            "operating_nwc": {*NWC_ASSET_COMPONENTS, *NWC_LIABILITY_COMPONENTS},
        }
        while True:
            expanded = required | set().union(*(dependencies.get(key, set()) for key in required))
            if expanded == required:
                break
            required = expanded
        confirmed = [f for f in session.facts if f.status == "confirmed" and not f.warnings]
        periods = [_period(f.period) for f in confirmed if f.role == "historical"
                   and mapped_financial_metric(f) in required
                   and f.scope == "consolidated" and (period := _period(f.period))
                   and (period.month, period.day) == (12, 31)]
        latest = max((p.year for p in periods if p), default=None)
        earliest = latest - (MIN_AUTOMATIC_HISTORY_YEARS - 1 if "dcf" in selected and not ResearchValuationAssembler._manual_forecast(session) else 0) if latest else None
        blockers = []
        for fact in session.facts:
            if fact.status != "proposed":
                continue
            if fact.role == "historical":
                metric = mapped_financial_metric(fact)
                period = _period(fact.period)
                if metric not in required or fact.scope == "parent":
                    continue
                if earliest and period and period.year < earliest and metric != "operating_nwc":
                    continue
                if any(mapped_financial_metric(old) == metric and _period(old.period) == period
                       and old.role == fact.role and old.scope == fact.scope
                       and old.normalized_value is not None and fact.normalized_value is not None
                       and D(old.normalized_value) == D(fact.normalized_value) for old in confirmed):
                    continue
                # A weak/incorrectly dated share-count lead is not a blocker
                # when a later, fully bound issuer disclosure proves the exact
                # same total. A differing total still blocks because it may
                # represent a real issuance, cancellation or buyback change.
                if metric == "common_shares" and fact.normalized_value is not None and any(
                    mapped_financial_metric(old) == "common_shares"
                    and old.scope == "issuer"
                    and old.normalized_value is not None
                    and D(old.normalized_value) == D(fact.normalized_value)
                    and ResearchValuationAssembler._verified_dated_issuer_shares(old)
                    and (verified_date := _period(old.period)) is not None
                    and (period is None or verified_date >= period)
                    for old in confirmed
                ):
                    continue
            elif fact.role == "comparable":
                relevant = set(selected)
                if {"pe", "ps"} & relevant:
                    relevant.add("market_cap")
                if "pe" in relevant:
                    relevant.add("net_income_parent")
                if "ps" in relevant:
                    relevant.add("revenue")
                if (fact.standard_metric or fact.metric) not in relevant:
                    continue
            elif fact.role == "assumption":
                if "dcf" not in selected or not _normalized_metric(fact.metric, ASSUMPTION_ALIASES):
                    continue
            else:
                continue
            blockers.append(fact)
        return blockers

    @staticmethod
    def recoverable_driver_repairs(session, financials):
        """Identify already-sourced drivers needing only semantic correction.

        DCF may deliberately fall back when a cash-flow driver is genuinely
        unavailable.  It must not present that fallback as review-ready when
        the filing amount is already bound and the only remaining defects are
        LLM mapping metadata.  The agent gets one bounded correction path;
        source/period/unit failures remain eligible for the normal degraded
        model rather than creating another retrieval loop.
        """
        if financials is None or financials.capital_expenditure is not None:
            return []
        semantic_markers = (
            "语义映射缺少",
            "资本开支基础科目模型处理冲突",
        )
        repairs = []
        for fact in session.facts:
            if (
                fact.status != "proposed"
                or fact.role != "historical"
                or fact.standard_metric != "cash_paid_for_ppe_intangibles"
                or not fact.warnings
                or not all(any(marker in warning for marker in semantic_markers)
                           for warning in fact.warnings)
                or fact.normalized_value is None
            ):
                continue
            repairs.append({
                "fact_id": fact.fact_id,
                "metric": fact.metric,
                "period": fact.period,
                "warnings": list(fact.warnings),
                "required_mapping": {
                    "standard_metric": "cash_paid_for_ppe_intangibles",
                    "semantic_role": "investing",
                    "ebit_treatment": "exclude",
                    "fcff_treatment": "include",
                    "equity_bridge_treatment": "exclude",
                },
            })
        return repairs

    def _evidence(self, session, fact, method: str = "") -> EvidenceRef:
        document = next(
            (doc for doc in session.documents if fact.block_id.startswith(doc.file_id + ":")),
            None,
        )
        note = f"block_id={fact.block_id}; raw_metric={fact.metric}; quote={fact.quote[:500]}"
        mapping = (getattr(fact, "verification", {}) or {}).get("semantic_mapping", {})
        if mapping.get("standard_metric"):
            note += (
                f"; semantic_mapping={mapping.get('standard_metric')}"
                f"; semantic_role={mapping.get('semantic_role', 'unknown')}"
                f"; model_treatment=EBIT:{mapping.get('ebit_treatment', 'review')},"
                f"FCFF:{mapping.get('fcff_treatment', 'review')},"
                f"equity_bridge:{mapping.get('equity_bridge_treatment', 'review')}"
                f"; mapping_rationale={str(mapping.get('rationale', ''))[:500]}"
            )
            if "confidence" in mapping:
                note += f"; mapping_confidence={mapping['confidence']}"
            if proof := fact.verification.get("reading_proof"):
                note += f"; reading_proof={proof['proof_id']}; currency={proof['basis'].get('currency')}"
                note += f"; semantic_review={fact.verification.get('semantic_review', {}).get('status')}; independent_audit=False"
            alternatives = mapping.get("alternative_interpretations") or []
            if alternatives:
                note += "; alternative_interpretations=" + str(alternatives)[:500]
        if method and method != "direct_confirmed_fact":
            note += f"; deterministic_formula={method}"
        assessment = fact.verification.get("source_assessment", {})
        if assessment:
            note += f"; source_tier={assessment.get('source_tier')}; admission={assessment.get('admission')}"
            note += "; source_limitations=" + "；".join(assessment.get("limitations", []))[:500]
        return EvidenceRef(
            evidence_id=fact.fact_id,
            source=document.name if document else "用户在研究会话中确认",
            file_id=document.file_id if document else None,
            page=fact.source_location.get("page"),
            sheet=fact.source_location.get("sheet"),
            cell=fact.source_location.get("cell") or (str(fact.source_location["row"]) if fact.source_location.get("row") else None),
            published_at=fact.published_at,
            source_url=fact.source_url,
            source_sha256=fact.source_sha256 or (document.sha256 if document else ""),
            note=note,
        )

    @staticmethod
    def _derive(values, methods, metric, value, inputs, formula):
        """Add a value calculated solely from already confirmed source facts."""
        if metric in values:
            return
        source_facts = []
        seen = set()
        for input_metric in inputs:
            for fact in values[input_metric][1]:
                if fact.fact_id not in seen:
                    seen.add(fact.fact_id)
                    source_facts.append(fact)
        values[metric] = (value, source_facts)
        methods[metric] = formula

    @staticmethod
    def _depreciation_components(values):
        """Only add a separately disclosed right-of-use charge, never a subtotal.

        A confirmed total is not incremented.  When calculating from components,
        the underlying row labels must distinguish fixed-asset depreciation
        from the lease charge: renaming an inclusive total is not enough.
        """
        if "depreciation_right_of_use" not in values:
            if values.get("lease_liabilities", (D(0),))[0] > 0:
                if "depreciation_amortization" in values:
                    # A reviewed complete total is usable; the partial list of
                    # components is not evidence for a competing subtotal.
                    return None
                raise ValueError(
                    "已确认正租赁负债，但尚缺独立披露的使用权资产折旧/摊销，"
                    "也没有完整折旧摊销总额；不能把缺失项当作零推导D&A。"
                )
            return DEPRECIATION_COMPONENTS
        if values["depreciation_right_of_use"][0] < 0:
            raise ValueError("使用权资产折旧/摊销为负，需核对冲回及口径，不能直接衍生D&A。")
        if "depreciation_fixed_assets" in values:
            fixed_facts = values["depreciation_fixed_assets"][1]
            lease_facts = values["depreciation_right_of_use"][1]
            fixed_ids = {fact.fact_id for fact in fixed_facts}
            lease_ids = {fact.fact_id for fact in lease_facts}
            def source_rows(facts):
                return [
                    (getattr(fact, "verification", {}) or {}).get("source_row")
                    or getattr(fact, "quote", "")
                    for fact in facts
                ]

            def contains_label(rows, metric):
                compact = lambda text: re.sub(r"[\s、,，_-]+", "", text).casefold()
                labels = [compact(name) for name in METRIC_ALIASES[metric]]
                return bool(rows) and all(
                    any(label in compact(row) for label in labels) for row in rows
                )

            rows = source_rows(fixed_facts)
            ambiguous = fixed_ids & lease_ids or any(
                re.search(r"使用权|right[\s_-]*of[\s_-]*use", row, re.I)
                for row in rows
            ) or not contains_label(rows, "depreciation_fixed_assets") or not contains_label(
                source_rows(lease_facts), "depreciation_right_of_use"
            )
            if ambiguous:
                if "depreciation_amortization" in values:
                    return None
                raise ValueError(
                    "固定资产折旧原文可能已含使用权资产折旧/摊销，不能重复相加。"
                    "请核对独立科目行或提供完整折旧摊销总额及组成勾稽。"
                )
        return (*DEPRECIATION_COMPONENTS, "depreciation_right_of_use")

    def _derive_period(self, values, methods, *, require_da=True):
        try:
            da_components = self._depreciation_components(values)
        except ValueError:
            if require_da:
                raise
            # PE/PS do not consume D&A; retain the raw lease evidence without
            # inventing a charge or blocking a different, complete method.
            da_components = None
        if (
            "depreciation_amortization" not in values
            and da_components is not None
            and all(metric in values for metric in da_components)
            and all(values[metric][0] >= 0 for metric in da_components)
        ):
            self._derive(
                values,
                methods,
                "depreciation_amortization",
                sum((values[metric][0] for metric in da_components), D("0")),
                da_components,
                " + ".join(da_components),
            )

        if "capital_expenditure" not in values and "cash_paid_for_ppe_intangibles" in values:
            self._derive(
                values,
                methods,
                "capital_expenditure",
                abs(values["cash_paid_for_ppe_intangibles"][0]),
                ("cash_paid_for_ppe_intangibles",),
                "abs(cash_paid_for_ppe_intangibles)",
            )

        if (
            "ebit" not in values
            and "profit_before_tax" in values
            and "interest_expense" in values
            and values["interest_expense"][0] >= 0
        ):
            self._derive(
                values,
                methods,
                "ebit",
                values["profit_before_tax"][0] + values["interest_expense"][0],
                ("profit_before_tax", "interest_expense"),
                "profit_before_tax + interest_expense",
            )

        if (
            "ebit_margin" not in values
            and "ebit" in values
            and "revenue" in values
            and values["revenue"][0] > 0
        ):
            self._derive(
                values,
                methods,
                "ebit_margin",
                values["ebit"][0] / values["revenue"][0],
                ("ebit", "revenue"),
                "ebit / revenue",
            )

        if (
            "tax_rate" not in values
            and "income_tax_expense" in values
            and "profit_before_tax" in values
            and values["profit_before_tax"][0] > 0
        ):
            tax_rate = values["income_tax_expense"][0] / values["profit_before_tax"][0]
            # Do not clamp an anomalous effective tax rate into a plausible
            # range: leave it missing so the agent must investigate the source.
            if D("0") <= tax_rate <= D("0.6"):
                self._derive(
                    values,
                    methods,
                    "tax_rate",
                    tax_rate,
                    ("income_tax_expense", "profit_before_tax"),
                    "income_tax_expense / profit_before_tax",
                )

        if (
            "ebitda" not in values
            and "ebit" in values
            and "depreciation_amortization" in values
        ):
            self._derive(
                values,
                methods,
                "ebitda",
                values["ebit"][0] + values["depreciation_amortization"][0],
                ("ebit", "depreciation_amortization"),
                "ebit + depreciation_amortization",
            )

        if (
            "interest_bearing_debt" not in values
            and all(metric in values for metric in DEBT_COMPONENTS)
            and all(values[metric][0] >= 0 for metric in DEBT_COMPONENTS)
        ):
            self._derive(
                values,
                methods,
                "interest_bearing_debt",
                sum((values[metric][0] for metric in DEBT_COMPONENTS), D("0")),
                DEBT_COMPONENTS,
                " + ".join(DEBT_COMPONENTS),
            )

        balance_components = NWC_ASSET_COMPONENTS + NWC_LIABILITY_COMPONENTS
        if "operating_nwc" not in values and all(
            metric in values for metric in balance_components
        ):
            operating_nwc = sum(
                (values[metric][0] for metric in NWC_ASSET_COMPONENTS), D("0")
            ) - sum(
                (values[metric][0] for metric in NWC_LIABILITY_COMPONENTS), D("0")
            )
            self._derive(
                values,
                methods,
                "operating_nwc",
                operating_nwc,
                balance_components,
                " + ".join(NWC_ASSET_COMPONENTS)
                + " - ("
                + " + ".join(NWC_LIABILITY_COMPONENTS)
                + ")",
            )

        if "change_operating_nwc" not in values and all(
            metric in values for metric in NWC_CASH_FLOW_COMPONENTS
        ):
            # The indirect cash-flow statement reports decreases in operating
            # assets and increases in operating liabilities, i.e. -Delta NWC.
            change_nwc = -sum(
                (values[metric][0] for metric in NWC_CASH_FLOW_COMPONENTS), D("0")
            )
            self._derive(
                values,
                methods,
                "change_operating_nwc",
                change_nwc,
                NWC_CASH_FLOW_COMPONENTS,
                "-(inventory_decrease + operating_receivables_decrease + "
                "operating_payables_increase)",
            )

    @staticmethod
    def _reconcile_identity(
        period,
        values,
        methods,
        target,
        expected,
        formula,
        *,
        rate=False,
    ):
        """Block material disagreements between a direct metric and its inputs."""
        if target not in values or methods.get(target) != "direct_confirmed_fact":
            return
        actual = values[target][0]
        tolerance = D("0.005") if rate else max(
            D("1"), max(abs(actual), abs(expected), D("1")) * D("0.005")
        )
        if abs(actual - expected) > tolerance:
            raise ValueError(
                "财务勾稽冲突，系统未选择任一数值继续计算："
                f"{period.isoformat()} {METRIC_LABELS.get(target, target)}直接值为 {actual}，"
                f"但按 {formula} 复算为 {expected}（容差 {tolerance}）。"
                "请核对字段定义、期间、单位和合并口径，并拒绝或更正不一致的候选。"
            )
        methods[f"reconciliation.{target}"] = (
            f"passed: direct={actual}; implied={expected}; formula={formula}; "
            f"tolerance={tolerance}"
        )

    def _reconcile_period(self, period, values, methods, *, require_da=True):
        if "ebit" in values and "revenue" in values and values["revenue"][0] > 0:
            self._reconcile_identity(
                period, values, methods, "ebit_margin",
                values["ebit"][0] / values["revenue"][0],
                "ebit / revenue", rate=True,
            )
        if (
            "income_tax_expense" in values
            and "profit_before_tax" in values
            and values["profit_before_tax"][0] > 0
        ):
            self._reconcile_identity(
                period, values, methods, "tax_rate",
                values["income_tax_expense"][0] / values["profit_before_tax"][0],
                "income_tax_expense / profit_before_tax", rate=True,
            )
        try:
            da_components = self._depreciation_components(values)
        except ValueError:
            if require_da:
                raise
            da_components = None
        if da_components is not None and all(metric in values for metric in da_components):
            self._reconcile_identity(
                period, values, methods, "depreciation_amortization",
                sum((values[metric][0] for metric in da_components), D("0")),
                " + ".join(da_components),
            )
        if "ebit" in values and "depreciation_amortization" in values:
            self._reconcile_identity(
                period, values, methods, "ebitda",
                values["ebit"][0] + values["depreciation_amortization"][0],
                "ebit + depreciation_amortization",
            )
        if all(metric in values for metric in DEBT_COMPONENTS):
            self._reconcile_identity(
                period, values, methods, "interest_bearing_debt",
                sum((values[metric][0] for metric in DEBT_COMPONENTS), D("0")),
                " + ".join(DEBT_COMPONENTS),
            )
        balance_components = NWC_ASSET_COMPONENTS + NWC_LIABILITY_COMPONENTS
        if all(metric in values for metric in balance_components):
            implied_nwc = sum(
                (values[metric][0] for metric in NWC_ASSET_COMPONENTS), D("0")
            ) - sum(
                (values[metric][0] for metric in NWC_LIABILITY_COMPONENTS), D("0")
            )
            self._reconcile_identity(
                period, values, methods, "operating_nwc", implied_nwc,
                "operating assets - operating liabilities",
            )
        if all(metric in values for metric in NWC_CASH_FLOW_COMPONENTS):
            implied_change = -sum(
                (values[metric][0] for metric in NWC_CASH_FLOW_COMPONENTS), D("0")
            )
            self._reconcile_identity(
                period, values, methods, "change_operating_nwc", implied_change,
                "-(inventory_decrease + operating_receivables_decrease + "
                "operating_payables_increase)",
            )

    def _structured_financials(self, session) -> list[FinancialSnapshot]:
        from valuationagent.application.observation_consistency import PERIOD_CELL_CONFLICT, period_cell_conflicts

        collisions = period_cell_conflicts([fact for fact in session.facts if fact.role == "historical"])
        if collisions:
            raise ValueError(PERIOD_CELL_CONFLICT + "：" + ", ".join(list(collisions)[:6]))
        share_issue = self.later_issuer_shares_issue(session)
        if share_issue:
            raise ValueError(share_issue)
        timing_issue = self.capital_structure_timing_issue(session)
        if timing_issue:
            raise ValueError(timing_issue["message"])
        selected_methods = self._selected_methods(session)
        required_latest = required_financial_metrics(selected_methods)
        if "dcf" in selected_methods:
            required_latest -= {
                "depreciation_amortization",
                "capital_expenditure",
                "change_operating_nwc",
            }
            required_history = {"revenue", "ebit_margin"}
        else:
            required_history = set(required_latest)
        rows: dict[date, dict[str, tuple[Decimal, list[object]]]] = defaultdict(dict)
        methods: dict[date, dict[str, str]] = defaultdict(dict)
        share_dates: dict[date, date] = {}
        conflicts = []
        dated_issuer_shares = []
        for fact in session.facts:
            if fact.status != "confirmed" or fact.role != "historical":
                continue
            if fact.warnings:
                raise ValueError(f"已确认字段 {fact.metric} 仍有未解决的取证警告，请重新核对来源。")
            cutoff = _information_cutoff(session)
            if fact.published_at and cutoff and fact.published_at > cutoff:
                raise ValueError(f"字段 {fact.metric} 的披露日晚于信息截止日，不能使用未来信息。")
            metric = mapped_financial_metric(fact)
            period = _period(fact.period)
            if not metric or not period or fact.normalized_value is None:
                continue
            if metric == "common_shares" and fact.metric.strip() == "股份总数" and fact.scope != "issuer":
                # A generic quantity label can also describe a share class or
                # a shareholder's holdings; only issuer-scoped proof is enough.
                continue
            if fact.scope != "consolidated" and not (
                fact.scope == "issuer" and metric == "common_shares"
            ):
                continue
            value = D(fact.normalized_value)
            if fact.scope == "issuer" and metric == "common_shares":
                dated_issuer_shares.append((period, value, fact))
                if (period.month, period.day) != (12, 31):
                    # A disclosure-date total is the current per-share
                    # denominator, not a new annual financial statement.
                    continue
            existing = rows[period].get(metric)
            if existing and existing[0] != value:
                conflicts.append(
                    f"{period.isoformat()} {METRIC_LABELS.get(metric, metric)}: "
                    f"{existing[0]} 与 {value}"
                )
                continue
            if existing:
                existing[1].append(fact)
            else:
                rows[period][metric] = (value, [fact])
                methods[period][metric] = "direct_confirmed_fact"

        if conflicts:
            raise ValueError(
                "已确认字段存在同期间同口径冲突，系统未静默覆盖："
                + "；".join(conflicts[:5])
                + "。请拒绝错误字段或提交带 replaces 的更正后再估值。"
            )

        statement_periods = [period for period, values in rows.items()
                             if (period.month, period.day) == (12, 31)
                             and any(metric != "common_shares" for metric in values)]
        if statement_periods:
            latest_statement = max(statement_periods)
            latest_shares = [entry for entry in dated_issuer_shares
                             if latest_statement < entry[0]
                             and (not session.draft.valuation_date or entry[0] <= session.draft.valuation_date)
                             and self._verified_dated_issuer_shares(entry[2])]
            if latest_shares:
                as_of = max(entry[0] for entry in latest_shares)
                matches = [entry for entry in latest_shares if entry[0] == as_of]
                if len({entry[1] for entry in matches}) != 1:
                    raise ValueError("最新发行人股数存在同日冲突，不能静默选择每股价值分母。")
                rows[latest_statement]["common_shares"] = (matches[0][1], [entry[2] for entry in matches])
                methods[latest_statement]["common_shares"] = f"issuer_shares_as_of[{as_of.isoformat()}]"
                share_dates[latest_statement] = as_of

        for period, values in rows.items():
            bridge_findings = equity_bridge_review_findings(
                selected_methods, {key: value[0] for key, value in values.items()},
            )
            blocking_bridge = [
                item for item in bridge_findings if item.severity == "blocking"
            ]
            if blocking_bridge:
                raise ValueError(f"{period.isoformat()}：" + blocking_bridge[0].message)
            require_da = "ev_ebitda" in selected_methods and "ebitda" not in values
            self._derive_period(values, methods[period], require_da=require_da)
            self._reconcile_period(period, values, methods[period], require_da=require_da)

        # If the cash-flow supplement is unavailable, two consecutive confirmed
        # operating-NWC balances provide a second deterministic route.
        ordered_periods = sorted(rows)
        for index, period in enumerate(ordered_periods[1:], start=1):
            previous = ordered_periods[index - 1]
            values = rows[period]
            previous_values = rows[previous]
            consecutive = (
                period.year == previous.year + 1
                and (period.month, period.day) == (previous.month, previous.day)
            )
            if (
                "change_operating_nwc" not in values
                and consecutive
                and "operating_nwc" in values
                and "operating_nwc" in previous_values
            ):
                current_facts = values["operating_nwc"][1]
                previous_facts = previous_values["operating_nwc"][1]
                values["change_operating_nwc"] = (
                    values["operating_nwc"][0] - previous_values["operating_nwc"][0],
                    [*previous_facts, *current_facts],
                )
                methods[period]["change_operating_nwc"] = (
                    f"operating_nwc[{period.isoformat()}] - "
                    f"operating_nwc[{previous.isoformat()}]"
                )
            elif (
                consecutive
                and "change_operating_nwc" in values
                and methods[period].get("change_operating_nwc") == "direct_confirmed_fact"
                and "operating_nwc" in values
                and "operating_nwc" in previous_values
            ):
                self._reconcile_identity(
                    period,
                    values,
                    methods[period],
                    "change_operating_nwc",
                    values["operating_nwc"][0] - previous_values["operating_nwc"][0],
                    f"operating_nwc[{period.isoformat()}] - "
                    f"operating_nwc[{previous.isoformat()}]",
                )

        snapshots = []
        incomplete = []
        latest_statement = max(statement_periods) if statement_periods else None
        for period, values in sorted(rows.items()):
            period_required = (
                required_latest if period == latest_statement else required_history
            )
            if "diluted_shares" in values:
                period_required = set(period_required) - {"common_shares"}
            missing = sorted(period_required - set(values))
            if missing:
                incomplete.append((period, missing))
                continue
            evidence = {
                metric: [
                    self._evidence(session, fact, methods[period].get(metric, ""))
                    for fact in values[metric][1]
                ]
                for metric in sorted(values)
            }
            calculation_methods = {
                metric: methods[period].get(metric, "direct_confirmed_fact")
                for metric in sorted(values)
            }
            calculation_methods.update({
                metric: method
                for metric, method in sorted(methods[period].items())
                if metric.startswith("reconciliation.")
            })
            has_derivation = any(
                method != "direct_confirmed_fact"
                for method in calculation_methods.values()
            )
            snapshots.append(
                FinancialSnapshot(
                    period_end=period,
                    common_shares_as_of=share_dates.get(period, period if "common_shares" in values else None),
                    diluted_shares_as_of=period if "diluted_shares" in values else None,
                    interest_bearing_debt_includes_leases=(
                        True
                        if "lease_liabilities" in methods[period].get(
                            "interest_bearing_debt", ""
                        )
                        else None
                    ),
                    **{metric: (abs(values[metric][0]) if metric == "capital_expenditure" else values[metric][0])
                       for metric in (REQUIRED_METRICS | OPTIONAL_BRIDGE_METRICS)
                       if metric in values},
                    source_label=(
                        "研究会话已确认原始科目及确定性推导"
                        if has_derivation else "研究会话已确认字段"
                    ),
                    evidence=evidence,
                    statement_items={
                        metric: values[metric][0] for metric in sorted(values)
                    },
                    calculation_methods=calculation_methods,
                    published_at=max((f.published_at for _, facts in values.values() for f in facts if f.published_at), default=None),
                )
            )
        latest_row_year = max(period.year for period in rows) if rows else None
        target_start_year = (
            latest_row_year - MIN_AUTOMATIC_HISTORY_YEARS + 1
            if latest_row_year is not None else None
        )
        if "dcf" not in selected_methods or self._manual_forecast(session):
            target_start_year = latest_row_year
        blocking_incomplete = [
            item for item in incomplete
            if target_start_year is None or item[0].year >= target_start_year
        ]
        if rows and (not snapshots or blocking_incomplete):
            target_period, missing = max(
                blocking_incomplete or incomplete, key=lambda item: item[0]
            )
            labels = "、".join(METRIC_LABELS.get(metric, metric) for metric in missing)
            hints = list(dict.fromkeys(
                DERIVATION_HINTS[metric] for metric in missing if metric in DERIVATION_HINTS
            ))
            older = len(blocking_incomplete or incomplete) - 1
            raise ValueError(
                "已确认字段尚不能组成完整年度快照："
                f"优先补齐最近年度 {target_period.isoformat()}，仍缺 {labels}。"
                + ("可从年报原始科目提取并由系统计算：" + "；".join(hints) + "。" if hints else "")
                + ("先补齐最近年度，再继续形成至少4个连续完整年度，才能进入自动收入预测" if "dcf" in selected_methods else "相对估值只要求所选方法的最近年度指标及同期可比样本。")
                + (f"；另有 {older} 个不完整比较期可在后续补强趋势分析" if older else "")
            )
        # A reviewed, explicit forecast needs a complete latest baseline, not
        # invented historical inputs. Older evidence remains in the research
        # record, but is not presented as a complete history to the calculator.
        return snapshots[-1:] if self._manual_forecast(session) else snapshots

    @staticmethod
    def _history_readiness_error(snapshots, assumptions=None):
        if assumptions is not None and (
            assumptions.revenue_growth
            or (
                assumptions.revenue_growth_scenarios
                and assumptions.revenue_growth_scenarios.get("base")
            )
        ):
            return None
        years = sorted({snapshot.period_end.year for snapshot in snapshots})
        if not years:
            return None
        latest = years[-1]
        target_years = list(range(latest - MIN_AUTOMATIC_HISTORY_YEARS + 1, latest + 1))
        missing = [year for year in target_years if year not in years]
        consecutive = years == list(range(years[0], years[-1] + 1))
        if len(years) >= MIN_AUTOMATIC_HISTORY_YEARS and consecutive:
            return None
        detail = (
            "；优先补齐 " + "、".join(map(str, missing))
            if missing else "；现有完整年度之间存在断档"
        )
        return (
            "正式自动收入预测至少需要4个连续年度完整快照，目标为近10年；"
            f"当前只有 {len(years)} 个完整年度（{', '.join(map(str, years))}）"
            + detail
            + "。已完成年度会保留，不要求重新提取；也可提供经确认的手工收入增长路径。"
        )

    def structured_readiness_error(self, session) -> str | None:
        """Explain whether confirmed non-Tushare facts form a calculable snapshot."""
        try:
            snapshots = self._structured_financials(session)
            assumptions, _ = self._assumptions(session)
        except ValueError as exc:
            return str(exc)
        if not snapshots:
            return "已确认字段中没有可识别的完整年度财务快照；请补齐估值所需字段、期间、单位和合并口径"
        return (
            self._history_readiness_error(snapshots, assumptions)
            if "dcf" in self._selected_methods(session)
            else None
        )

    def _peers(self, session, baseline_end=None):
        from valuationagent.application.peer_inputs import assemble_peers

        return assemble_peers(session, self._evidence, baseline_end)

    def _assumptions(self, session) -> tuple[AssumptionInputs, dict[str, list[EvidenceRef]]]:
        values, evidence = {}, {}
        wacc_market_dates = []
        wacc_market_sources = []
        confirmed_wacc_metrics = set()
        wacc_market_metrics = {
            "wacc", "risk_free_rate", "equity_risk_premium", "beta", "debt_cost",
        }
        for fact in session.facts:
            if fact.status != "confirmed" or fact.role != "assumption":
                continue
            metric = _normalized_metric(fact.metric, ASSUMPTION_ALIASES)
            if metric and fact.normalized_value is not None:
                values[metric] = D(fact.normalized_value)
                ref = self._evidence(session, fact)
                evidence[metric] = [ref]
                if metric in wacc_market_metrics:
                    confirmed_wacc_metrics.add(metric)
                    try:
                        wacc_market_dates.append(date.fromisoformat(fact.period))
                    except (TypeError, ValueError):
                        pass
                    wacc_market_sources.append(
                        ref.source_url or ref.source or ref.evidence_id
                    )
        wacc_bundle_confirmed = (
            "wacc" in confirmed_wacc_metrics
            or {
                "risk_free_rate", "equity_risk_premium", "beta", "debt_cost",
            } <= confirmed_wacc_metrics
        )
        if wacc_bundle_confirmed and wacc_market_dates:
            # Freshness is measured from the oldest component in the WACC
            # snapshot, so a current share price cannot hide a stale ERP/Beta.
            values["market_inputs_as_of"] = min(wacc_market_dates)
        if wacc_bundle_confirmed and wacc_market_sources:
            values["market_inputs_source"] = "；".join(
                dict.fromkeys(wacc_market_sources)
            )[:500]
        proposal = session.forecast_proposal
        if proposal is not None:
            from valuationagent.application.forecast_inputs import forecast_input_evidence
            from valuationagent.application.valuation_plan import scope_key
            if proposal.scope_key != scope_key(session):
                raise ValueError("估值范围已改变，原预测方案已失效；请重新提出并确认预测假设")
            if proposal.status != "confirmed":
                raise ValueError("预测方案尚未确认，请集中复核估值方案后再计算")
            input_refs = forecast_input_evidence(session, proposal.evidence_ids)
            for metric, value in proposal.inputs.model_dump(
                exclude_none=True, exclude_defaults=True
            ).items():
                if metric in values and values[metric] != value:
                    raise ValueError(f"预测方案与已确认假设 {metric} 冲突；请更正旧假设，不能静默覆盖")
                values[metric] = value
                evidence[metric] = [EvidenceRef(
                    evidence_id=proposal.proposal_id, source="forecast_assumption",
                    note="预测假设，不是历史事实；自动预览及自动计算不代表用户逐项批准。依据：" + proposal.rationale
                         + "；关联证据：" + ", ".join(proposal.evidence_ids)
                         + "；风险：" + "；".join(proposal.risks),
                ), *input_refs]
        return AssumptionInputs.model_validate(values), evidence

    def build(self, session) -> ValuationRequest:
        if session.input_dataset is not None:
            from valuationagent.application.input_workspace import prepare_dataset

            records = session.input_dataset.active_records()
            has_sources = any(row.source.kind != "user" and row.role == "historical" for row in records)
            if has_sources and (scope_issue := self.model_scope_issue(session)):
                raise ValueError(scope_issue)
            assumptions, evidence = {}, {}
            if session.forecast_proposal:
                forecast_session = session.model_copy(deep=True)
                forecast_session.facts = []
                proposal, evidence = self._assumptions(forecast_session)
                assumptions = proposal.model_dump(exclude_none=True, exclude_defaults=True)
            return prepare_dataset(session, self._selected_methods(session),
                                   assumptions=assumptions, assumption_evidence=evidence)
        if not (session.draft.company or session.draft.ticker):
            raise ValueError("请先确认公司名称或A股代码。")
        if session.draft.valuation_date is None:
            raise ValueError("请先确认估值基准日。")
        if not session.data_source_preference:
            raise ValueError("请先确认资料来源：联网获取或自行上传。")

        methods = self._selected_methods(session)
        for fact in session.facts:
            if fact.status != "confirmed" or not (issue := financial_mapping_issue(fact)):
                continue
            target = mapped_financial_metric(fact)
            affected = {"dcf", "ev_ebitda"} if target in {"ebit", "ebitda", "ebit_margin"} else {"dcf", "ps"}
            if affected.intersection(methods):
                raise ValueError(f"已确认事实 {fact.fact_id} 的语义映射须更正：{issue}")
        use_ticker = bool(
            session.data_source_preference == "online" and session.draft.ticker
        )
        if not use_ticker and (scope_issue := self.model_scope_issue(session)):
            raise ValueError(scope_issue)
        if not use_ticker and self.pending_blockers(session):
            raise ValueError("仍有待确认候选字段，请先确认、拒绝或更正。")
        # An explicit online+ticker choice uses the point-in-time structured
        # market provider. Search snippets remain research evidence and cannot
        # silently override Tushare statement data, whether proposed or confirmed.
        snapshots = [] if use_ticker else self._structured_financials(session)
        assumptions, assumption_evidence = self._assumptions(session)
        if not use_ticker and not snapshots:
            if session.data_source_preference == "upload":
                raise ValueError("已选择自行上传，但尚未形成可提交的完整财务快照。")
            if session.data_source_preference == "web":
                raise ValueError("已选择联网检索，但经来源核验和用户确认的字段尚未形成完整年度财务快照。")
            raise ValueError("没有可提交的完整财务快照；请补齐确认字段，或使用A股代码联网取数。")
        if not use_ticker and {"dcf", "ev_ebitda"} & set(methods):
            inventory = self.source_risks(session, snapshots[-1].period_end)
            if inventory["unresolved"]:
                detail = "；".join(f"{r['label']}（{r['block_id']}，第{r['page'] or '?'}页）"
                                  for r in inventory["unresolved"][:5])
                raise ValueError("已加载原文存在尚未提取核验的估值桥接风险科目：" + detail
                                 + "。须绑定当前基期、合并口径并复核，不能把漏提取当零。此检查仅覆盖已加载原文，不代表全文件排查。")
        history_error = self._history_readiness_error(snapshots, assumptions)
        if not use_ticker and "dcf" in methods and history_error:
            raise ValueError(history_error)
        if not use_ticker and not session.draft.industry:
            raise ValueError("结构化资料估值需要确认非金融行业，以匹配金融小组参数库。")

        peers = self._peers(session, snapshots[-1].period_end if snapshots else None) if not use_ticker else []
        relative_methods = [
            method for method in methods if method in {"pe", "ps", "ev_ebitda"}
        ]
        if not use_ticker and relative_methods:
            insufficient = {
                method: sum(getattr(peer, method) is not None for peer in peers)
                for method in relative_methods
                if sum(getattr(peer, method) is not None for peer in peers) < 3
            }
            if insufficient:
                detail = "、".join(
                    f"{method.upper()} 当前{count}家" for method, count in insufficient.items()
                )
                raise ValueError(
                    "相对估值尚缺与明确选择的统一行情日一致、FY同口径的可比公司倍数："
                    f"{detail}；每个已选择的相对估值方法至少需要3家，优先5家。"
                    "目标公司自身收盘价只用于结果交叉核验，不能替代可比样本。"
                )
        return ValuationRequest(
            company=CompanyInput(
                ticker=session.draft.ticker or None,
                name=session.draft.company or None,
                industry=session.draft.industry or None,
            ),
            valuation_date=session.draft.valuation_date,
            language=session.language,
            data_source="ticker" if use_ticker else "structured",
            assumption_source=(
                "manual"
                if assumptions.model_dump(exclude_none=True, exclude_defaults=True)
                else "automatic"
            ),
            mode="snapshot",
            forecast_years=10,
            methods=methods,
            requested_methods=session.draft.methods or methods,
            excluded_methods=dict(getattr(session, "valuation_method_exclusions", {}) or {}),
            financials=snapshots[-1] if snapshots else None,
            historical_financials=snapshots[:-1],
            assumptions=assumptions,
            peers=peers,
            assumption_evidence=assumption_evidence,
            discount_policy="year_end",
            user_goal=session.draft.objective or "完成可追溯的企业估值并解释关键假设",
        )
