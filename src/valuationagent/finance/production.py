"""Production valuation policies shared by calculation and independent review.

The helpers in this module deliberately contain no orchestration logic.  They
turn already-confirmed inputs into deterministic terminal-value and
enterprise-to-equity bridge calculations so the live model, spreadsheet export
and independent verifier can use exactly the same policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


D = Decimal
ZERO = D(0)

BRIDGE_ITEM_LABELS = {
    "lease_liabilities": "租赁负债",
    "trading_financial_assets": "交易性金融资产",
    "associates_and_non_operating_investments": "联营及非经营性投资",
    "minority_interest": "少数股东权益",
    "preferred_equity": "优先股权益",
    "unfunded_pension": "未弥补养老金缺口",
    "non_operating_provisions": "非经营性预计负债",
}


@dataclass(frozen=True)
class TerminalCashFlow:
    nopat: Decimal
    reinvestment_rate: Decimal
    reinvestment: Decimal
    fcff: Decimal


def normalized_terminal_cash_flow(
    final_nopat: Decimal,
    terminal_growth: Decimal,
    stable_roic: Decimal,
) -> TerminalCashFlow:
    """Return N+1 FCFF with growth supported by economically required reinvestment.

    In a stable state, growth equals reinvestment rate multiplied by ROIC.  The
    old implementation grew the final FCFF while forcing CapEx to D&A, which
    combined positive perpetual growth with zero net reinvestment.
    """

    if stable_roic <= ZERO:
        raise ValueError("稳定期ROIC必须为正。")
    if terminal_growth >= stable_roic:
        raise ValueError("永续增长率必须低于稳定期ROIC，否则再投资率不成立。")
    nopat = final_nopat * (D(1) + terminal_growth)
    if nopat <= ZERO:
        raise ValueError("终值期NOPAT非正，Gordon稳定增长模型不适用。")
    # A shrinking perpetuity releases invested capital.  Keep the same
    # g/ROIC identity for negative growth in the engine and exported workbook.
    reinvestment_rate = terminal_growth / stable_roic
    reinvestment = nopat * reinvestment_rate
    return TerminalCashFlow(
        nopat=nopat,
        reinvestment_rate=reinvestment_rate,
        reinvestment=reinvestment,
        fcff=nopat - reinvestment,
    )


def _explicit_value(financials, field_name: str, *statement_names: str):
    field_value = getattr(financials, field_name, None)
    normalized_field = D(str(field_value)) if field_value is not None else None
    for name in statement_names:
        value = financials.statement_items.get(name)
        if value is not None:
            normalized_statement = D(str(value))
            if (
                normalized_field is not None
                and normalized_statement != normalized_field
            ):
                raise ValueError(
                    f"{field_name}的结构化字段与statement_items不一致，"
                    "不能静默选边。"
                )
            return normalized_statement
    return normalized_field


def _truthy_statement_flag(financials, name: str) -> bool:
    value = financials.statement_items.get(name)
    if value is None:
        return False
    try:
        return D(str(value)) != ZERO
    except Exception:
        return str(value).strip().lower() in {"true", "yes", "y"}


def effective_capitalized_debt(financials) -> tuple[Decimal, bool, tuple[str, ...]]:
    """Return debt for WACC and EV bridge with leases included exactly once."""

    debt = D(str(financials.interest_bearing_debt or ZERO))
    if not debt.is_finite() or debt < ZERO:
        raise ValueError("有息负债必须为有限非负数。")
    lease_value = _explicit_value(financials, "lease_liabilities", "lease_liabilities")
    lease = lease_value or ZERO
    if not lease.is_finite() or lease < ZERO:
        raise ValueError("租赁负债必须为有限非负数。")
    debt_method = financials.calculation_methods.get("interest_bearing_debt", "")
    includes_lease = financials.interest_bearing_debt_includes_leases
    if includes_lease is None and "lease_liabilities" in debt_method:
        includes_lease = True
    warnings = []
    if lease > ZERO and includes_lease is not True:
        debt += lease
        if includes_lease is None:
            warnings.append(
                "有息负债是否包含租赁负债未明确；按资本化租赁政策将租赁负债单列计入净债务。"
            )
    return debt, lease_value is None and includes_lease is not True, tuple(warnings)


@dataclass(frozen=True)
class EquityBridgeInputs:
    distributable_cash: Decimal
    total_debt: Decimal
    common_equity_additions: dict[str, Decimal] = field(default_factory=dict)
    common_equity_deductions: dict[str, Decimal] = field(default_factory=dict)
    share_count: Decimal = ZERO
    unmeasured_items: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def equity_value(self, enterprise_value: Decimal) -> Decimal:
        return (
            enterprise_value
            + self.distributable_cash
            + sum(self.common_equity_additions.values(), ZERO)
            - self.total_debt
            - sum(self.common_equity_deductions.values(), ZERO)
        )

    def entries(self, enterprise_value: Decimal) -> dict[str, Decimal]:
        equity = self.equity_value(enterprise_value)
        rows = {
            "operating_enterprise_value": enterprise_value,
            "distributable_cash_and_non_operating_assets": self.distributable_cash,
            "total_debt_including_incremental_leases": -self.total_debt,
        }
        rows.update({key: value for key, value in self.common_equity_additions.items()})
        rows.update({key: -value for key, value in self.common_equity_deductions.items()})
        rows["common_equity_value"] = equity
        rows["diluted_or_common_shares"] = self.share_count
        return rows


def resolve_equity_bridge(
    financials,
    operating_cash: Decimal = ZERO,
    *,
    policy: str = "use_disclosed_book_values",
) -> EquityBridgeInputs:
    """Resolve a transparent EV-to-common-equity bridge.

    Lease liabilities are capitalized.  They are added to debt only when the
    confirmed interest-bearing-debt total does not already include them.  For
    other debt-like/equity claims, market-value statement items take priority;
    disclosed book values are explicit proxies unless the caller requests the
    strict market-value policy.
    """

    warnings: list[str] = []
    operating_cash = D(str(operating_cash))
    if not operating_cash.is_finite() or operating_cash < ZERO:
        raise ValueError("经营必需现金必须为有限非负数。")
    cash = D(str(financials.cash_and_non_operating_assets or ZERO))
    if not cash.is_finite() or cash < ZERO:
        raise ValueError("现金及非经营性资产必须为有限非负数。")
    distributable_cash = max(ZERO, cash - operating_cash)
    unmeasured_items: list[str] = []
    debt, lease_unmeasured, debt_warnings = effective_capitalized_debt(financials)
    warnings.extend(debt_warnings)
    if lease_unmeasured:
        unmeasured_items.append("lease_liabilities")

    additions: dict[str, Decimal] = {}
    if not _truthy_statement_flag(
        financials, "cash_and_non_operating_assets_includes_trading_financial_assets"
    ):
        trading_market = financials.statement_items.get(
            "trading_financial_assets_market_value"
        )
        trading_assets = (
            D(str(trading_market))
            if trading_market is not None
            else financials.statement_items.get("trading_financial_assets")
        )
        if trading_assets is not None:
            trading_assets = D(str(trading_assets))
            if not trading_assets.is_finite() or trading_assets < ZERO:
                raise ValueError("交易性金融资产桥接值必须为有限非负数。")
        if trading_assets not in (None, ZERO) and trading_market is None:
            if policy == "require_market_values":
                raise ValueError(
                    "股权价值桥接缺少trading_financial_assets_market_value，"
                    "严格政策不允许以报表列示值替代。"
                )
            warnings.append(
                "交易性金融资产使用报表列示的公允价值计量账面金额作为桥接代理。"
            )
        if trading_assets:
            additions["trading_financial_assets"] = trading_assets
        elif trading_assets is None:
            unmeasured_items.append("trading_financial_assets")

    if not _truthy_statement_flag(
        financials, "cash_and_non_operating_assets_includes_associates"
    ):
        associates_market = financials.statement_items.get("associates_market_value")
        associates = (
            D(str(associates_market))
            if associates_market is not None
            else _explicit_value(
                financials,
                "associates_and_non_operating_investments",
                "associates_and_non_operating_investments",
            )
        )
        if associates is not None and (
            not associates.is_finite() or associates < ZERO
        ):
            raise ValueError("联营及非经营性投资桥接值必须为有限非负数。")
        if associates not in (None, ZERO) and associates_market is None:
            if policy == "require_market_values":
                raise ValueError(
                    "股权价值桥接缺少associates_market_value，"
                    "严格政策不允许以账面值替代。"
                )
            warnings.append(
                "联营及非经营性投资使用已披露账面值作为市场价值代理。"
            )
        if associates:
            additions["associates_and_non_operating_investments"] = associates
        elif associates is None:
            unmeasured_items.append("associates_and_non_operating_investments")

    deductions: dict[str, Decimal] = {}
    specs = (
        ("minority_interest", "minority_interest_market_value", "minority_interest"),
        ("preferred_equity", "preferred_equity_market_value", "preferred_equity"),
        ("unfunded_pension", "unfunded_pension_market_value", "unfunded_pension"),
        (
            "non_operating_provisions",
            "non_operating_provisions_market_value",
            "non_operating_provisions",
        ),
    )
    for output_name, market_key, field_name in specs:
        market_value = financials.statement_items.get(market_key)
        disclosed = _explicit_value(financials, field_name, field_name)
        if market_value is not None:
            value = D(str(market_value))
        else:
            value = disclosed
            if value not in (None, ZERO):
                if policy == "require_market_values":
                    raise ValueError(
                        f"股权价值桥接缺少{market_key}，严格政策不允许以账面值替代。"
                    )
                warnings.append(
                    f"{BRIDGE_ITEM_LABELS[output_name]}"
                    "使用已披露账面值作为市场价值代理。"
                )
        if value is not None and (not value.is_finite() or value < ZERO):
            raise ValueError(f"{output_name}桥接值必须为有限非负数。")
        if value not in (None, ZERO):
            deductions[output_name] = value
        elif value is None:
            unmeasured_items.append(output_name)

    share_count = _explicit_value(financials, "diluted_shares", "diluted_shares")
    if share_count is None and financials.common_shares is not None:
        share_count = D(str(financials.common_shares))
    if share_count is None:
        raise ValueError("缺少稀释后或普通股股数，无法计算每股价值。")
    if not share_count.is_finite() or share_count <= ZERO:
        raise ValueError("普通股/稀释后股数必须为正。")

    if unmeasured_items:
        warnings.append(
            "以下股权桥接项未建立，不等同于已确认为0："
            + "、".join(BRIDGE_ITEM_LABELS[item] for item in unmeasured_items)
            + "；本次未做相应加减项，结果将降级披露。"
        )

    return EquityBridgeInputs(
        distributable_cash=distributable_cash,
        total_debt=debt,
        common_equity_additions=additions,
        common_equity_deductions=deductions,
        share_count=share_count,
        unmeasured_items=tuple(unmeasured_items),
        warnings=tuple(warnings),
    )
