from __future__ import annotations

import math
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx

from valuationagent.core.data import DataBundle, LocalDataProvider
from valuationagent.finance.industry import (
    FINANCIAL_KEYWORDS,
    IndustryParameterRegistry,
)
from valuationagent.schemas.models import (
    CompanyInput,
    EvidenceRef,
    FinancialSnapshot,
    PeerCompany,
    ValuationRequest,
)

D = Decimal
TUSHARE_DOC = "https://tushare.pro/document/2"


class TushareApiError(ValueError):
    """Safe provider error that keeps the failing API name."""

    def __init__(self, api_name: str, message: str):
        self.api_name = api_name
        super().__init__(f"Tushare接口 {api_name} 失败：{message}")


class TusharePermissionError(TushareApiError):
    """The token is valid but cannot call one optional or required API."""

TUSHARE_INDUSTRY_MAP = (
    (("软件", "IT", "互联网", "元器件", "半导体", "通信设备", "电脑设备"), "电子 / 计算机 / 半导体"),
    (("医药", "医疗", "生物", "制药"), "医药 / 生物"),
    (("航空", "船舶", "军工"), "军工 / 航空航天"),
    (("电气设备", "光伏", "电池", "新能源", "储能"), "电力设备 / 新能源"),
    (("机械", "专用设备", "通用设备", "工程机械"), "机械设备"),
    (("汽车", "汽配", "摩托车"), "汽车"),
    (("化工", "化纤", "塑料", "橡胶", "农药化肥"), "化工"),
    (("造纸", "家具", "文教", "家居用品", "日用化工", "家用电器", "食品", "饮料", "白酒", "乳制品"), "轻工制造"),
    (("纺织", "服饰", "服装", "鞋"), "纺织服饰制鞋"),
    (("建筑", "建材", "水泥", "陶瓷", "玻璃"), "建筑业"),
    (("铝", "铜", "铅锌", "小金属", "黄金", "有色", "金属新材料"), "有色金属"),
    (("钢", "普钢", "特种钢"), "钢铁"),
    (("农业", "种植", "饲料", "畜牧", "渔业", "林业", "农林牧渔"), "农林牧渔"),
    (("煤", "石油", "天然气", "油气"), "煤炭 / 石油"),
    (("电力", "供气供热", "水务", "环境保护"), "公用事业"),
    (("百货", "零售", "超市", "商贸", "批发", "商品城"), "商贸零售"),
    (("旅游", "酒店", "餐饮", "景点"), "餐饮旅游"),
    (("房地产", "房产"), "房地产"),
    (("运输", "物流", "港口", "航运", "机场", "公路", "铁路", "仓储"), "交通运输"),
    (("传媒", "广告", "影视", "教育", "专业服务", "通信服务"), "服务业（兜底）"),
)


def map_tushare_industry(label: str) -> tuple[str, str | None]:
    """Map detailed vendor labels to the versioned finance-team registry."""
    raw = (label or "").strip()
    if any(word in raw for word in FINANCIAL_KEYWORDS):
        raise ValueError("当前项目不研究金融行业，不能为该公司运行通用FCFF估值。")
    registry = IndustryParameterRegistry()
    try:
        return registry.resolve(raw).name, None
    except ValueError:
        pass
    for keywords, target in TUSHARE_INDUSTRY_MAP:
        if any(keyword.casefold() in raw.casefold() for keyword in keywords):
            return target, f"Tushare行业“{raw}”按显式映射进入参数库“{target}”，请复核主营业务。"
    return "工业（兜底）", f"Tushare行业“{raw or '空'}”未命中细分类，暂用工业兜底参数，必须人工复核。"


def normalize_a_share_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if ticker.endswith((".SH", ".SZ", ".BJ")):
        return ticker
    symbol = ticker.split(".")[0]
    if not (len(symbol) == 6 and symbol.isdigit()):
        raise ValueError("A股代码应为6位数字，或带.SH/.SZ/.BJ后缀。")
    if symbol.startswith(("4", "8", "9")):
        return symbol + ".BJ"
    if symbol.startswith(("5", "6", "9")):
        return symbol + ".SH"
    return symbol + ".SZ"


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        number = D(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _record_date(value: Any) -> date | None:
    raw = str(value or "").replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        return None
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))


class TushareApiClient:
    """Small dependency-free client for the official Tushare Pro JSON API."""

    version = "tushare-pro-json-v1"

    def __init__(
        self,
        token: str,
        *,
        endpoint: str = "https://api.tushare.pro",
        timeout_seconds: float = 40,
        transport=None,
    ):
        if not token.strip():
            raise ValueError("TUSHARE_TOKEN不能为空。")
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Tushare接口必须是无账号信息的HTTPS地址。")
        self._token = token.strip()
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self._transport = transport

    def query(
        self, api_name: str, *, params: dict[str, Any] | None = None, fields: list[str] | None = None
    ) -> list[dict[str, Any]]:
        payload = {
            "api_name": api_name,
            "token": self._token,
            "params": params or {},
            "fields": ",".join(fields or []),
        }
        try:
            with httpx.Client(
                timeout=self.timeout_seconds,
                transport=self._transport,
                follow_redirects=False,
            ) as client:
                response = client.post(self.endpoint, json=payload)
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            suffix = f"（HTTP {code}）" if code else ""
            raise ValueError(f"Tushare数据请求失败{suffix}，请检查网络与数据源配置。") from None
        if int(body.get("code", -1)) != 0:
            raw = str(body.get("msg") or "").lower()
            if "token" in raw:
                raise TushareApiError(api_name, "Token无效或权限不足")
            if "权限" in raw or "permission" in raw:
                raise TusharePermissionError(api_name, "接口权限不足")
            raise TushareApiError(api_name, f"供应商错误代码 {body.get('code')}")
        data = body.get("data") or {}
        names = data.get("fields") or []
        return [dict(zip(names, row)) for row in data.get("items") or []]


class TushareDataProvider:
    """A-share facts and comparable companies with point-in-time controls."""

    version = "tushare-a-share-2026-09"

    INCOME_FIELDS: ClassVar[list[str]] = [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
        "revenue", "total_revenue", "operate_profit", "total_profit", "n_income_attr_p",
        "income_tax", "oper_cost", "biz_tax_surchg", "sell_exp", "admin_exp", "rd_exp",
        "assets_impair_loss", "oth_income",
    ]
    BALANCE_FIELDS: ClassVar[list[str]] = [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
        "money_cap", "total_hldr_eqy_exc_min_int", "total_assets", "total_liab",
        "fix_assets", "intan_assets", "lt_amor_exp", "use_right_assets", "lease_liab",
        "minority_int", "accounts_receiv", "inventories", "acct_payable",
    ]
    CASHFLOW_FIELDS: ClassVar[list[str]] = [
        "ts_code", "ann_date", "f_ann_date", "end_date", "report_type", "comp_type",
        "c_pay_acq_const_fiolta", "depr_fa_coga_dpba", "amort_intang_assets",
        "lt_amort_deferred_exp", "use_right_asset_dep",
    ]
    INDICATOR_FIELDS: ClassVar[list[str]] = [
        "ts_code", "ann_date", "end_date", "ebit", "ebitda", "daa", "interestdebt",
        "networking_capital", "working_capital", "fixed_assets", "or_yoy", "ebit_of_gr", "roe",
    ]

    def __init__(self, client: TushareApiClient, *, peer_limit: int = 8):
        self.client = client
        self.peer_limit = max(3, min(peer_limit, 12))
        self.local = LocalDataProvider()

    @staticmethod
    def _available_annual(rows: list[dict[str, Any]], cutoff: date) -> dict[str, dict[str, Any]]:
        selected: dict[str, dict[str, Any]] = {}
        for row in rows:
            end = str(row.get("end_date") or "")
            if not end.endswith("1231"):
                continue
            period_end = _record_date(end)
            if period_end is None or period_end > cutoff:
                continue
            if row.get("report_type") not in (None, "", "1", 1):
                continue
            announced = _record_date(row.get("f_ann_date") or row.get("ann_date"))
            if announced and announced > cutoff:
                continue
            old = selected.get(end)
            old_date = _record_date((old or {}).get("f_ann_date") or (old or {}).get("ann_date"))
            if old is None or (announced or date.min) >= (old_date or date.min):
                selected[end] = row
        return selected

    def _query_financial_rows(self, ticker: str, cutoff: date):
        start = date(cutoff.year - 12, 1, 1).strftime("%Y%m%d")
        end = cutoff.strftime("%Y%m%d")
        params = {"ts_code": ticker, "start_date": start, "end_date": end}
        income = self._available_annual(
            self.client.query("income", params=params, fields=self.INCOME_FIELDS), cutoff
        )
        balance = self._available_annual(
            self.client.query("balancesheet", params=params, fields=self.BALANCE_FIELDS), cutoff
        )
        cashflow = self._available_annual(
            self.client.query("cashflow", params=params, fields=self.CASHFLOW_FIELDS), cutoff
        )
        indicator = self._available_annual(
            self.client.query("fina_indicator", params=params, fields=self.INDICATOR_FIELDS), cutoff
        )
        return income, balance, cashflow, indicator

    def _latest_daily(self, ticker: str, cutoff: date) -> dict[str, Any]:
        rows = self.client.query(
            "daily_basic",
            params={
                "ts_code": ticker,
                "start_date": (cutoff - timedelta(days=20)).strftime("%Y%m%d"),
                "end_date": cutoff.strftime("%Y%m%d"),
            },
            fields=[
                "ts_code", "trade_date", "close", "pe_ttm", "ps_ttm", "total_share", "total_mv"
            ],
        )
        eligible = [row for row in rows if (_record_date(row.get("trade_date")) or date.max) <= cutoff]
        if not eligible:
            raise ValueError(f"{ticker}在估值日前没有可用每日指标。")
        return max(eligible, key=lambda row: str(row.get("trade_date") or ""))

    def _market_cap_statistics(
        self, ticker: str, cutoff: date
    ) -> dict[str, Decimal]:
        rows = self.client.query(
            "daily_basic",
            params={
                "ts_code": ticker,
                "start_date": (cutoff - timedelta(days=370)).strftime("%Y%m%d"),
                "end_date": cutoff.strftime("%Y%m%d"),
            },
            fields=["ts_code", "trade_date", "total_mv"],
        )
        observations = [
            (_record_date(row.get("trade_date")), _decimal(row.get("total_mv")))
            for row in rows
            if (_record_date(row.get("trade_date")) or date.max) <= cutoff
            and _decimal(row.get("total_mv")) is not None
        ]
        observations = [
            (trade_date, value * D("10000"))
            for trade_date, value in observations
            if trade_date is not None and value is not None and value > 0
        ]
        values = [value for _, value in observations]
        if len(values) < 60:
            return {}
        quarterly: dict[tuple[int, int], list[Decimal]] = {}
        for trade_date, value in observations:
            key = (trade_date.year, (trade_date.month - 1) // 3 + 1)
            quarterly.setdefault(key, []).append(value)
        quarterly_means = [
            sum(group, D(0)) / D(len(group))
            for _, group in sorted(quarterly.items())[-4:]
        ]
        return {
            "quarterly_average_market_cap": (
                sum(quarterly_means, D(0)) / D(len(quarterly_means))
            ),
            "annual_average_market_cap": sum(values, D(0)) / D(len(values)),
            "market_cap_period_low": min(values),
            "market_cap_period_high": max(values),
        }

    def _quarterly_average_market_cap(
        self, ticker: str, cutoff: date
    ) -> Decimal | None:
        """Compatibility wrapper for integrations that still consume one value."""

        return self._market_cap_statistics(ticker, cutoff).get(
            "quarterly_average_market_cap"
        )

    @staticmethod
    def _evidence(ticker: str, period: str, api: str, field: str, ann_date=None) -> list[EvidenceRef]:
        return [
            EvidenceRef(
                evidence_id=f"tushare:{ticker}:{period}:{api}:{field}",
                source=f"Tushare Pro · {api}",
                published_at=_record_date(ann_date),
                note=f"field={field}; ts_code={ticker}; period={period}; docs={TUSHARE_DOC}",
            )
        ]

    def _financial_snapshots(
        self,
        ticker: str,
        cutoff: date,
        daily: dict[str, Any],
        market_cap_statistics: dict[str, Decimal] | None = None,
    ) -> tuple[list[FinancialSnapshot], list[str]]:
        income, balance, cashflow, indicator = self._query_financial_rows(ticker, cutoff)
        periods = sorted(set(income) & set(balance) & set(cashflow) & set(indicator))[-10:]
        if len(periods) < 4:
            raise ValueError(f"{ticker}仅取得{len(periods)}个完整年报期，无法运行自动收入模型。")
        warnings: list[str] = []
        rows = []
        previous_nwc: Decimal | None = None
        previous_nwc_evidence: list[EvidenceRef] = []
        previous_nwc_period: str | None = None
        latest_shares = (_decimal(daily.get("total_share")) or D(0)) * D("10000")
        latest_market_cap = (_decimal(daily.get("total_mv")) or D(0)) * D("10000")
        if latest_shares <= 0:
            raise ValueError("每日指标缺少总股本，无法形成每股估值。")
        for period in periods:
            inc, bal, cash, ind = income[period], balance[period], cashflow[period], indicator[period]
            revenue_field = "revenue" if _decimal(inc.get("revenue")) is not None else "total_revenue"
            revenue = _decimal(inc.get(revenue_field))
            ebit_field = "ebit" if _decimal(ind.get("ebit")) is not None else "operate_profit"
            ebit = (
                _decimal(ind.get("ebit"))
                if ebit_field == "ebit"
                else _decimal(inc.get("operate_profit"))
            )
            net_income = _decimal(inc.get("n_income_attr_p"))
            if not revenue or revenue <= 0 or ebit is None or net_income is None:
                raise ValueError(f"{ticker} {period} 缺少收入、EBIT或归母净利润。")
            da = _decimal(ind.get("daa"))
            ebitda = _decimal(ind.get("ebitda"))
            da_method = "tushare.fina_indicator.daa" if da is not None else ""
            ebitda_method = "tushare.fina_indicator.ebitda" if ebitda is not None else ""
            if da is None and ebitda is not None:
                da = ebitda - ebit
                da_method = "fina_indicator.ebitda - ebit"
            if da is None:
                component_fields = (
                    "depr_fa_coga_dpba",
                    "amort_intang_assets",
                    "lt_amort_deferred_exp",
                    "use_right_asset_dep",
                )
                component_values = {
                    field_name: _decimal(cash.get(field_name))
                    for field_name in component_fields
                }
                expected_component_fields = (
                    component_fields
                    if int(period[:4]) >= 2019
                    else component_fields[:3]
                )
                if all(
                    component_values[field_name] is not None
                    for field_name in expected_component_fields
                ):
                    da = sum(
                        (
                            component_values[field_name]
                            for field_name in expected_component_fields
                        ),
                        D(0),
                    )
                    da_method = "sum(disclosed_cashflow_depreciation_amortization_components)"
                elif any(
                    component_values[field_name] is not None
                    for field_name in expected_component_fields
                ):
                    warnings.append(
                        f"{period}折旧摊销明细仅部分可得，未将缺失分项当作0；"
                        "D&A保留为未知并由估值模型降级。"
                    )
            if da is not None and da < 0:
                warnings.append(
                    f"{period} D&A为负数，未截断为0；已作为异常值留空并降级。"
                )
                da = None
                da_method = "invalid_negative_source_value"
            if ebitda is None and da is not None:
                ebitda = ebit + da
                ebitda_method = "ebit + depreciation_amortization"
            raw_capex = _decimal(cash.get("c_pay_acq_const_fiolta"))
            capex = abs(raw_capex) if raw_capex is not None else None
            receivable = _decimal(bal.get("accounts_receiv"))
            inventory = _decimal(bal.get("inventories"))
            payable = _decimal(bal.get("acct_payable"))
            direct_nwc = (
                receivable + inventory - payable
                if receivable is not None and inventory is not None and payable is not None
                else None
            )
            # Generic working capital includes financing/cash balances and is
            # not interchangeable with operating NWC in FCFF.  Prefer the
            # vendor's net operating capital; otherwise use disclosed trade
            # receivables, inventory and payables or leave the driver unknown.
            nwc_field = "networking_capital"
            nwc = _decimal(ind.get(nwc_field))
            if nwc is None:
                nwc = direct_nwc
                nwc_method = (
                    "accounts_receivable + inventory - accounts_payable"
                    if direct_nwc is not None
                    else ""
                )
                nwc_evidence = []
                for field_name in ("accounts_receiv", "inventories", "acct_payable"):
                    if _decimal(bal.get(field_name)) is not None:
                        nwc_evidence.extend(
                            self._evidence(ticker, period, "balancesheet", field_name)
                        )
                if nwc is None and _decimal(ind.get("working_capital")) is not None:
                    warnings.append(
                        f"{period}仅有一般营运资金working_capital；其口径不能直接用于"
                        "FCFF的经营营运资本，ΔNWC保持未知。"
                    )
            else:
                nwc_method = f"tushare.fina_indicator.{nwc_field}"
                nwc_evidence = self._evidence(
                    ticker, period, "fina_indicator", nwc_field
                )
            change_nwc = (
                nwc - previous_nwc
                if nwc is not None and previous_nwc is not None
                else None
            )
            change_nwc_method = (
                f"operating_nwc[{period}] - operating_nwc[{previous_nwc_period}]"
                if change_nwc is not None
                else "unavailable_without_consecutive_operating_nwc"
            )
            if nwc is None:
                warnings.append(
                    f"{period}缺少经营营运资本，ΔNWC保留为未知并由估值模型显式降级。"
                )
            elif previous_nwc is None:
                warnings.append(
                    f"{period}为首个可用营运资本期，无法计算同口径ΔNWC；已保留为未知。"
                )
            total_profit = _decimal(inc.get("total_profit"))
            tax_expense = _decimal(inc.get("income_tax"))
            if total_profit is not None and total_profit > 0 and tax_expense is not None:
                raw_tax_rate = tax_expense / total_profit
                tax_rate = max(D(0), min(D("0.6"), raw_tax_rate))
                tax_method = "income_tax / total_profit (clamped_to_0_60_percent)"
                if tax_rate != raw_tax_rate:
                    warnings.append(
                        f"{period}由所得税费用/利润总额得到的税率"
                        f"{raw_tax_rate:.2%}超出0%–60%校验区间，已保留原始证据并"
                        f"按{tax_rate:.2%}建模，需人工复核一次性税项。"
                    )
            else:
                tax_rate = D("0.25")
                tax_method = "policy_default_25_percent_due_missing_tax_base"
                warnings.append(f"{period}无法由所得税费用/利润总额计算税率，暂用25%并要求复核。")
            period_date = _record_date(period)
            announced = inc.get("f_ann_date") or inc.get("ann_date")
            evidence = {
                "revenue": self._evidence(ticker, period, "income", revenue_field, announced),
                "ebit_margin": self._evidence(
                    ticker,
                    period,
                    "fina_indicator" if ebit_field == "ebit" else "income",
                    ebit_field,
                    announced,
                ),
                "net_income_parent": self._evidence(ticker, period, "income", "n_income_attr_p", announced),
            }
            if da is not None:
                if _decimal(ind.get("daa")) is not None:
                    evidence["depreciation_amortization"] = self._evidence(
                        ticker, period, "fina_indicator", "daa", announced
                    )
                elif _decimal(ind.get("ebitda")) is not None:
                    evidence["depreciation_amortization"] = [
                        *self._evidence(ticker, period, "fina_indicator", "ebitda", announced),
                        *evidence["ebit_margin"],
                    ]
                else:
                    evidence["depreciation_amortization"] = []
                    for field_name in (
                        "depr_fa_coga_dpba",
                        "amort_intang_assets",
                        "lt_amort_deferred_exp",
                        "use_right_asset_dep",
                    ):
                        if _decimal(cash.get(field_name)) is not None:
                            evidence["depreciation_amortization"].extend(
                                self._evidence(
                                    ticker, period, "cashflow", field_name, announced
                                )
                            )
            if capex is not None:
                evidence["capital_expenditure"] = self._evidence(
                    ticker, period, "cashflow", "c_pay_acq_const_fiolta", announced
                )
            if change_nwc is not None:
                evidence["change_operating_nwc"] = [
                    *previous_nwc_evidence,
                    *nwc_evidence,
                ]
            cash_value = _decimal(bal.get("money_cap"))
            debt_value = _decimal(ind.get("interestdebt"))
            lease_value = _decimal(bal.get("lease_liab"))
            minority_value = _decimal(bal.get("minority_int"))
            for label, value in (
                ("货币资金", cash_value),
                ("有息负债", debt_value),
                ("租赁负债", lease_value),
            ):
                if value is not None and value < 0:
                    raise ValueError(
                        f"{ticker} {period} {label}为负数，不能静默截断为0。"
                    )
            if cash_value is not None:
                evidence["cash_and_non_operating_assets"] = self._evidence(
                    ticker, period, "balancesheet", "money_cap", announced
                )
            if debt_value is not None:
                evidence["interest_bearing_debt"] = self._evidence(
                    ticker, period, "fina_indicator", "interestdebt", announced
                )
            if period == periods[-1]:
                evidence["common_shares"] = self._evidence(
                    ticker,
                    str(daily.get("trade_date") or cutoff.strftime("%Y%m%d")),
                    "daily_basic",
                    "total_share",
                )
            if lease_value is not None:
                evidence["lease_liabilities"] = self._evidence(
                    ticker, period, "balancesheet", "lease_liab", announced
                )
            if minority_value is not None:
                evidence["minority_interest"] = self._evidence(
                    ticker, period, "balancesheet", "minority_int", announced
                )
            if total_profit is not None and total_profit > 0 and tax_expense is not None:
                evidence["tax_rate"] = [
                    *self._evidence(ticker, period, "income", "income_tax", announced),
                    *self._evidence(ticker, period, "income", "total_profit", announced),
                ]
            if ebitda is not None:
                evidence["ebitda"] = (
                    self._evidence(ticker, period, "fina_indicator", "ebitda", announced)
                    if _decimal(ind.get("ebitda")) is not None
                    else [*evidence["ebit_margin"], *evidence.get("depreciation_amortization", [])]
                )
            statement_items = {
                key: value
                for key, value in {
                    "market_cap": latest_market_cap if period == periods[-1] else None,
                    **(
                        market_cap_statistics
                        if period == periods[-1] and market_cap_statistics
                        else {}
                    ),
                    "total_equity": _decimal(bal.get("total_hldr_eqy_exc_min_int")),
                    "operating_nwc": nwc,
                    "fixed_assets_net": _decimal(bal.get("fix_assets") or ind.get("fixed_assets")),
                    "intangible_assets": _decimal(bal.get("intan_assets")),
                    "long_term_deferred_expenses": _decimal(bal.get("lt_amor_exp")),
                    "right_of_use_assets": _decimal(bal.get("use_right_assets")),
                    "lease_liabilities": _decimal(bal.get("lease_liab")),
                    "depreciation_fixed_assets": _decimal(cash.get("depr_fa_coga_dpba")),
                    "amortization_intangibles": _decimal(cash.get("amort_intang_assets")),
                    "amortization_long_term_deferred": _decimal(cash.get("lt_amort_deferred_exp")),
                    "depreciation_right_of_use": _decimal(cash.get("use_right_asset_dep")),
                    "accounts_receivable": receivable,
                    "inventory": inventory,
                    "accounts_payable": payable,
                    "operating_cost": _decimal(inc.get("oper_cost")),
                    "taxes_and_surcharges": _decimal(inc.get("biz_tax_surchg")),
                    "selling_expense": _decimal(inc.get("sell_exp")),
                    "administrative_expense": _decimal(inc.get("admin_exp")),
                    "research_expense": _decimal(inc.get("rd_exp")),
                    "impairment_loss": _decimal(inc.get("assets_impair_loss")),
                    "other_income": _decimal(inc.get("oth_income")),
                }.items()
                if value is not None
            }
            rows.append(
                FinancialSnapshot(
                    period_end=period_date,
                    published_at=_record_date(announced),
                    revenue=revenue,
                    ebit_margin=ebit / revenue,
                    tax_rate=tax_rate,
                    depreciation_amortization=da,
                    capital_expenditure=capex,
                    change_operating_nwc=change_nwc,
                    cash_and_non_operating_assets=cash_value,
                    interest_bearing_debt=debt_value,
                    lease_liabilities=(
                        lease_value
                        if lease_value is not None
                        else None
                    ),
                    minority_interest=minority_value,
                    common_shares=(latest_shares if period == periods[-1] else None),
                    common_shares_as_of=(
                        _record_date(daily.get("trade_date"))
                        if period == periods[-1]
                        else None
                    ),
                    net_income_parent=net_income,
                    ebitda=ebitda,
                    source_label=f"Tushare Pro · {ticker} · annual statements",
                    evidence=evidence,
                    statement_items=statement_items,
                    calculation_methods={
                        "ebit_margin": f"{ebit_field} / {revenue_field}",
                        "tax_rate": tax_method,
                        "depreciation_amortization": da_method or "unavailable",
                        "capital_expenditure": (
                            "abs(cashflow.c_pay_acq_const_fiolta)"
                            if capex is not None
                            else "unavailable"
                        ),
                        "change_operating_nwc": change_nwc_method,
                        "operating_nwc": nwc_method or "unavailable",
                        "ebitda": ebitda_method or "unavailable",
                        "interest_bearing_debt": (
                            "tushare.fina_indicator.interestdebt"
                            if debt_value is not None
                            else "unavailable"
                        ),
                    },
                )
            )
            previous_nwc = nwc
            previous_nwc_evidence = nwc_evidence if nwc is not None else []
            previous_nwc_period = period if nwc is not None else None
        return rows, list(dict.fromkeys(warnings))

    def _company(self, ticker: str) -> CompanyInput:
        rows = self.client.query(
            "stock_basic",
            params={"ts_code": ticker, "list_status": "L"},
            fields=["ts_code", "symbol", "name", "industry", "market", "exchange", "curr_type", "list_status"],
        )
        if not rows:
            raise ValueError(f"未找到正常上市的A股代码 {ticker}。")
        row = rows[0]
        if any(word in str(row.get("industry") or "") for word in FINANCIAL_KEYWORDS):
            raise ValueError("当前项目不研究金融行业，不能为该公司运行通用FCFF估值。")
        return CompanyInput(
            ticker=ticker,
            name=row.get("name"),
            exchange=row.get("exchange"),
            industry=row.get("industry"),
            currency="CNY",
        )

    def _peer_indicator(self, ticker: str, cutoff: date) -> dict[str, Any] | None:
        start = date(cutoff.year - 2, 1, 1).strftime("%Y%m%d")
        rows = self.client.query(
            "fina_indicator",
            params={"ts_code": ticker, "start_date": start, "end_date": cutoff.strftime("%Y%m%d")},
            fields=self.INDICATOR_FIELDS,
        )
        annual = self._available_annual(rows, cutoff)
        return annual[max(annual)] if annual else None

    def _select_peers(
        self,
        company: CompanyInput,
        target_daily: dict[str, Any],
        cutoff: date,
        target_period: date | None = None,
    ) -> tuple[list[PeerCompany], list[str]]:
        universe = self.client.query(
            "stock_basic",
            params={"list_status": "L"},
            fields=["ts_code", "name", "industry", "market", "exchange", "list_status"],
        )
        same = [
            row for row in universe
            if row.get("industry") == company.industry
            and row.get("ts_code") != company.ticker
            and "ST" not in str(row.get("name") or "").upper()
            and not any(word in str(row.get("industry") or "") for word in FINANCIAL_KEYWORDS)
        ]
        if not same:
            return [], ["同一Tushare行业下没有可用非金融候选公司。"]
        pricing_date = _record_date(target_daily.get("trade_date"))
        if pricing_date is None or pricing_date > cutoff:
            return [], ["缺少估值日前可核验的统一交易日，同业倍数不可计算。"]
        market_rows = self.client.query(
            "daily_basic",
            params={"trade_date": str(target_daily["trade_date"])},
            fields=["ts_code", "trade_date", "total_mv"],
        )
        market = {
            row["ts_code"]: row for row in market_rows
            if _record_date(row.get("trade_date")) == pricing_date
        }
        target_mv = (_decimal(target_daily.get("total_mv")) or D(0)) * D("10000")
        target_indicator = self._peer_indicator(company.ticker, cutoff) or {}
        target_period = target_period or _record_date(target_indicator.get("end_date"))
        if target_period is None:
            return [], ["目标公司FY财务期间未确定，不能拼接同业年度倍数。"]
        period_key = target_period.strftime("%Y%m%d")
        target_growth = _decimal(target_indicator.get("or_yoy")) or D(0)
        target_margin = _decimal(target_indicator.get("ebit_of_gr")) or D(0)
        ranked = []
        for row in same:
            daily = market.get(row["ts_code"])
            mv = (_decimal((daily or {}).get("total_mv")) or D(0)) * D("10000")
            if not daily or mv <= 0 or target_mv <= 0:
                continue
            size_score = abs(math.log(float(mv / target_mv)))
            ranked.append((size_score, row, daily, mv))
        ranked.sort(key=lambda item: item[0])
        peers = []
        warnings = []
        for size_score, row, daily, market_cap in ranked[: max(self.peer_limit * 2, 12)]:
            indicator = self._peer_indicator(row["ts_code"], cutoff)
            if not indicator or indicator.get("end_date") != period_key:
                continue
            income_rows = self.client.query(
                "income",
                params={
                    "ts_code": row["ts_code"],
                    "start_date": f"{target_period.year}0101",
                    "end_date": cutoff.strftime("%Y%m%d"),
                },
                fields=[
                    "ts_code", "ann_date", "f_ann_date", "end_date",
                    "report_type", "revenue", "total_revenue", "n_income_attr_p",
                ],
            )
            income = self._available_annual(income_rows, cutoff).get(period_key)
            if income is None:
                continue
            growth = _decimal(indicator.get("or_yoy")) or D(0)
            margin = _decimal(indicator.get("ebit_of_gr")) or D(0)
            quality_score = size_score + abs(float(growth - target_growth)) / 20 + abs(float(margin - target_margin)) / 10
            profit = _decimal(income.get("n_income_attr_p"))
            revenue_field = "revenue" if _decimal(income.get("revenue")) is not None else "total_revenue"
            sales = _decimal(income.get(revenue_field))
            pe = market_cap / profit if profit is not None and profit > 0 else None
            ps = market_cap / sales if sales is not None and sales > 0 else None
            if pe is None and ps is None:
                continue
            market_evidence = self._evidence(
                row["ts_code"], pricing_date.isoformat(), "daily_basic", "total_mv"
            )
            announced = income.get("f_ann_date") or income.get("ann_date")
            peer_evidence = {}
            if pe is not None:
                peer_evidence["pe"] = [
                    *market_evidence,
                    *self._evidence(row["ts_code"], period_key, "income", "n_income_attr_p", announced),
                ]
            if ps is not None:
                peer_evidence["ps"] = [
                    *market_evidence,
                    *self._evidence(row["ts_code"], period_key, "income", revenue_field, announced),
                ]
            peers.append((quality_score, PeerCompany(
                ticker=row["ts_code"],
                name=str(row.get("name") or row["ts_code"]),
                pe=pe if pe and pe > 0 else None,
                ps=ps if ps and ps > 0 else None,
                ev_ebitda=None,
                market_cap=market_cap,
                revenue_growth=growth / D("100"),
                ebit_margin=margin / D("100"),
                selection_score=D(str(quality_score)),
                peer_tier="broad",
                rationale=(
                    f"Tushare同一行业={company.industry}；规模/增长/EBIT率距离得分={quality_score:.4f}；"
                    f"市场数据日={pricing_date}；同业FY期间={target_period}；"
                    "PE/PS由同日市值除以已披露同年度利润/收入复算"
                ),
                as_of_date=pricing_date,
                financial_period_end=target_period,
                multiple_basis="FY",
                evidence=peer_evidence,
            )))
        peers.sort(key=lambda item: item[0])
        selected = [
            item[1].model_copy(update={"peer_tier": "core" if index < 5 else "broad"})
            for index, item in enumerate(peers[: self.peer_limit])
        ]
        if len(selected) < 3:
            warnings.append(f"生产筛选后仅{len(selected)}家可比公司，倍数结果将标记样本不足。")
        if selected:
            warnings.append(
                "自动同业EV/EBITDA未生成：同行债务、受限现金、租赁及少数股东"
                "桥接尚未完整核验；需要已核验的同口径企业价值倍数。"
            )
        return selected, warnings

    def resolve(self, request: ValuationRequest, store) -> DataBundle:
        if request.mode == "demo" or request.data_source != "ticker":
            return self.local.resolve(request, store)
        ticker = normalize_a_share_ticker(request.company.ticker or "")
        market_warnings: list[str] = []
        try:
            company = self._company(ticker)
        except TusharePermissionError as exc:
            if not request.company.industry:
                raise ValueError(
                    f"{exc}。Token 可以连接 Tushare，但无法读取公司行业；"
                    "为避免把金融企业或错误行业套入通用FCFF模型，请回到研究对话确认该公司的非金融行业，"
                    "例如“行业为医药 / 生物，然后继续估值”，或为该Token开通 stock_basic 权限。"
                ) from None
            exchange = request.company.exchange or {
                "SH": "SSE", "SZ": "SZSE", "BJ": "BSE",
            }.get(ticker.rsplit(".", 1)[-1])
            company = request.company.model_copy(update={
                "ticker": ticker,
                "name": request.company.name or ticker,
                "exchange": exchange,
                "currency": "CNY",
            })
            market_warnings.append(
                "Tushare stock_basic 权限不足；公司名称与行业采用研究会话中已确认的信息，"
                "并保留为数据质量限制。"
            )
        daily = self._latest_daily(ticker, request.valuation_date)
        try:
            market_cap_statistics = self._market_cap_statistics(
                ticker, request.valuation_date
            )
        except ValueError as exc:
            market_cap_statistics = {}
            market_warnings.append(
                f"四季度平均市值取数失败（{exc}），WACC将降级使用基准日前最近市值。"
            )
        if not market_cap_statistics:
            market_warnings.append(
                "估值日前一年有效市值观测不足60个交易日，WACC降级使用基准日前最近市值。"
            )
        financials, warnings = self._financial_snapshots(
            ticker, request.valuation_date, daily, market_cap_statistics
        )
        warnings = market_warnings + warnings
        peers = list(request.peers)
        if any(method != "dcf" for method in request.methods) and not peers:
            try:
                peers, peer_warnings = self._select_peers(
                    company, daily, request.valuation_date, financials[-1].period_end
                )
                warnings.extend(peer_warnings)
            except ValueError as exc:
                peers = []
                warnings.append(
                    f"自动可比公司筛选未完成（{exc}）。DCF继续执行；PE、PS和EV/EBITDA"
                    "将标记为样本不足，补充至少3家经确认的可比公司后可创建更正版本。"
                )
        model_industry, mapping_warning = map_tushare_industry(company.industry or "")
        company = company.model_copy(update={"industry": model_industry})
        if mapping_warning:
            warnings.append(mapping_warning)
        assumption_updates = {}
        if (
            request.assumptions.quarterly_average_market_cap is None
            and market_cap_statistics.get("quarterly_average_market_cap") is not None
        ):
            assumption_updates["quarterly_average_market_cap"] = (
                market_cap_statistics["quarterly_average_market_cap"]
            )
        effective_assumptions = request.assumptions.model_copy(
            update=assumption_updates
        )
        effective_assumption_evidence = dict(request.assumption_evidence)
        market_date = _record_date(daily.get("trade_date"))
        if market_date is not None:
            period = market_date.isoformat()
            for key in (
                "quarterly_average_market_cap",
                "annual_average_market_cap",
                "market_cap_period_low",
                "market_cap_period_high",
            ):
                if market_cap_statistics.get(key) is not None:
                    effective_assumption_evidence[key] = self._evidence(
                        ticker, period, "daily_basic", key
                    )
        return DataBundle(
            company=company,
            financials=financials[-1],
            historical_financials=financials[:-1],
            peers=peers,
            assumptions=effective_assumptions,
            assumption_evidence=effective_assumption_evidence,
            warnings=list(dict.fromkeys(warnings)),
        )
