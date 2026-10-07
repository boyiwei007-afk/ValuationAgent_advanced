"""Formal, traceable XLSX and PDF exports for completed valuation runs."""

from __future__ import annotations

from decimal import Decimal
from io import BytesIO
from pathlib import Path

from valuationagent.schemas.models import RunRecord


BASELINE_FIELDS = (
    ("revenue", "营业收入"),
    ("ebit_margin", "EBIT利润率"),
    ("tax_rate", "有效所得税率"),
    ("depreciation_amortization", "折旧与摊销"),
    ("capital_expenditure", "资本开支"),
    ("change_operating_nwc", "经营性营运资本变动"),
    ("cash_and_non_operating_assets", "现金及非经营性资产"),
    ("interest_bearing_debt", "有息负债"),
    ("lease_liabilities", "租赁负债"),
    ("minority_interest", "少数股东权益"),
    ("preferred_equity", "优先股权益"),
    ("associates_and_non_operating_investments", "联营及非经营性投资"),
    ("common_shares", "普通股股数"),
    ("diluted_shares", "稀释后普通股股数"),
    ("net_income_parent", "归母净利润"),
    ("ebitda", "EBITDA"),
)


def _number(value):
    return float(value) if isinstance(value, Decimal) else value


def _label(value):
    return {
        "high": "高", "medium": "中", "low": "低", "unknown": "未评估",
        "adequate": "充足", "limited": "有限", "insufficient": "不足",
        "success": "成功", "completed": "完成", "not_available": "不可用",
        "not_applicable": "不适用", "not_applicable_relative_only": "仅相对估值，不使用DCF假设",
        "direct_confirmed_fact": "已确认原始字段", "user": "用户确认", "core": "核心同业",
    }.get(str(value), value)


def _completed(record: RunRecord):
    if record.result is None:
        raise ValueError("估值尚未完成，不能导出正式报告。")
    return record.result


def _baseline_method(snapshot, field):
    if getattr(snapshot, field, snapshot.statement_items.get(field)) is None:
        return "未提供；不用于本次所选估值方法"
    method = snapshot.calculation_methods.get(field, "direct_confirmed_fact")
    reconciliation = snapshot.calculation_methods.get(f"reconciliation.{field}")
    return "；".join(str(_label(part)) for part in (method, reconciliation) if part)


def _source_note(ref):
    return "；".join(part for part in (ref.note, ref.source_url,
        f"SHA-256 {ref.source_sha256}" if ref.source_sha256 else "") if part)


class ValuationReportExporter:
    """Generate reviewer-friendly reports without changing model results."""

    def xlsx(self, record: RunRecord, audit_context=None) -> bytes:
        try:
            from openpyxl import Workbook
            from openpyxl.chart import BarChart, Reference
            from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
            from openpyxl.utils import get_column_letter
        except ImportError as exc:
            raise ValueError("Excel报告组件未安装，请安装 valuationagent[reports]。") from exc

        result = _completed(record)
        audit_context = audit_context or {}
        decision_meta = ((audit_context.get("decisions") or [{}])[-1] or {})
        decision_labels = {
            "accepted": "已接受，可按披露范围使用",
            "accepted_with_warnings": "附带警示接受",
            "review_required": "需复核，不得作为正式数值结论",
            "rejected": "已拒绝，不得作为估值结论",
        }
        conclusion_status = decision_labels.get(
            decision_meta.get("outcome"),
            "计算完成（未绑定工作区决策账本）",
        )
        wb = Workbook()
        wb.remove(wb.active)
        navy, teal, pale, amber = "14273D", "0F8B8D", "EAF4F4", "FFF3CD"
        thin = Side(style="thin", color="D7DEE7")

        def sheet(name, widths=(24, 24, 48)):
            ws = wb.create_sheet(name)
            ws.sheet_view.showGridLines = False
            ws.freeze_panes = "A2"
            for index, width in enumerate(widths, 1):
                ws.column_dimensions[get_column_letter(index)].width = width
            return ws

        def header(ws, row=1):
            for cell in ws[row]:
                cell.fill = PatternFill("solid", fgColor=navy)
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = Border(bottom=thin)
            ws.row_dimensions[row].height = 28

        def rows(ws, values, *, formula_columns=()):
            for value in values:
                ws.append([_number(item) for item in value])
            for row in ws.iter_rows():
                for cell in row:
                    # Names and source text are data, never spreadsheet code.
                    if cell.data_type == "f" and cell.column not in formula_columns:
                        cell.data_type = "s"
                    cell.alignment = Alignment(vertical="top", wrap_text=True)
                    cell.border = Border(bottom=thin)

        summary = sheet("估值摘要", (24, 28, 55))
        rows(summary, [
            ["项目", "结果", "说明"],
            ["公司", result.company.name or result.company.ticker, result.company.ticker or ""],
            ["估值基准日", str(result.valuation_date), "所有市场与公开信息不得晚于该日"],
            ["模型版本", result.model_version, f"运行ID：{record.run_id} · 修订v{record.revision}"],
            ["数据质量置信度", result.data_quality.confidence,
             f"结果等级 {result.data_quality.result_grade} · 可比历史 {result.data_quality.comparable_years}/{result.data_quality.historical_years} 年"],
            ["市场参数质量", result.data_quality.market_input_quality,
             "降级字段：" + ("、".join(result.data_quality.degraded_fields) or "无")],
            ["证据覆盖率", result.data_quality.evidence_coverage,
             "核心字段中带逐项证据的比例"],
            ["可比样本质量", result.data_quality.peer_sample_quality,
             f"行业参数等级 {result.data_quality.industry_parameter_quality} · 元数据 {result.data_quality.industry_metadata_completeness}"],
            ["DCF每股价值", result.dcf.per_share_value if result.dcf else None,
             f"区间 {result.dcf.range_low} – {result.dcf.range_high}" if result.dcf else "未采用"],
            ["退出倍数交叉校验", result.dcf.exit_multiple_per_share if result.dcf else None,
             (
                 f"相对Gordon差异 {result.dcf.terminal_method_gap:.2%}；仅作校验"
                 if result.dcf and result.dcf.terminal_method_gap is not None
                 else "无文档支持的行业倍数或未采用DCF"
             )],
            ["相对估值区间", (
                f"{result.reconciliation.relative_range[0]:.2f} – "
                f"{result.reconciliation.relative_range[1]:.2f}"
                if result.reconciliation.relative_range
                else None
             ), "成功方法的区间并列展示，不强行平均" if result.reconciliation.relative_range else "没有成功方法"],
            ["交叉验证", result.reconciliation.conclusion, "不同方法保持独立，不强行平均"],
            ["决策层结论状态", conclusion_status,
             decision_meta.get("selected_action", "正式使用前应核验适用范围与关键假设")],
            ["结论", result.executive_summary, ""],
            ["原请求估值方法", " / ".join(str(method).upper() for method in record.request.requested_methods),
             "用户在最终方案确认前选择的方法"],
            ["本次实际估值方法", " / ".join(str(method).upper() for method in record.request.methods),
             "仅包含具备可靠输入的方法"],
            *[
                ["未采用方法", method.upper(), reason]
                for method, reason in record.request.excluded_methods.items()
            ],
        ])
        header(summary)
        summary["B7"].number_format = "0.00%"
        summary["B9"].number_format = '¥#,##0.00'

        assumptions = sheet("关键假设", (26, 22, 70))
        assumption_rows = [
            ["参数", "数值", "依据/口径"],
            ["WACC", result.assumptions.wacc if result.dcf else "不适用", result.assumptions.rationale.get("wacc", "")],
            ["永续增长率", result.assumptions.terminal_growth if result.dcf else "不适用", result.assumptions.rationale.get("terminal_growth", "")],
            ["稳定期ROIC", result.dcf.stable_roic if result.dcf and result.dcf.stable_roic is not None else "不适用", result.assumptions.rationale.get("stable_roic", "")],
            ["假设来源", result.assumptions.source, ""],
        ]
        assumption_rows += [[f"WACC组成 · {key}", value, ""] for key, value in result.assumptions.wacc_components.items()]
        assumption_rows += [[f"行业参数 · {key}", str(value), ""] for key, value in result.assumptions.industry_parameters.items()]
        assumption_rows += [[f"经营驱动 · {key}", value, ""] for key, value in result.assumptions.operating_drivers.items()]
        assumption_rows += [[f"计算方法 · {key}", value, ""] for key, value in result.assumptions.calculation_methods.items()]
        assumption_rows += [[f"指定倍数 · {key}", value, "显式假设，不是可比样本统计"] for key, value in record.request.assumptions.relative_multiples.items()]
        assumption_rows += [["模型决定", item, ""] for item in result.assumptions.model_decisions]
        rows(assumptions, assumption_rows)
        header(assumptions)
        for cell in (assumptions["B2"], assumptions["B3"], assumptions["B4"]):
            cell.number_format = "0.00%"
            cell.font = Font(color="1F4E78")

        historical = sheet("历史财务", (16, 20, 20, 20, 20, 20, 18, 45, 32))
        rows(historical, [["报告期", "营业收入", "EBIT率", "归母净利润", "EBITDA", "资本开支", "可比状态", "可比说明", "来源"]] + [
            [item.period_end, item.revenue, item.ebit_margin, item.net_income_parent,
             item.ebitda, item.capital_expenditure, item.comparability_status,
             item.comparability_note, item.source_label]
            for item in [*record.request.historical_financials,
                         *([result.effective_financials] if result.effective_financials else [])]
        ])
        header(historical)
        for row in range(2, historical.max_row + 1):
            historical.cell(row, 3).number_format = "0.00%"
            for col in (2, 4, 5, 6):
                historical.cell(row, col).number_format = '#,##0.00'

        derivation = sheet("基期推导", (18, 32, 20, 74, 45, 68))
        derivation_rows = [["性质", "字段", "数值", "计算/取数口径", "证据ID", "原文定位与说明"]]
        base = result.effective_financials
        if base:
            baseline_names = {field for field, _ in BASELINE_FIELDS}
            for field, label in BASELINE_FIELDS:
                refs = base.evidence.get(field, [])
                locations = []
                for ref in refs:
                    location = " · ".join(part for part in [
                        ref.file_id,
                        f"第{ref.page}页" if ref.page else None,
                        f"{ref.sheet}!{ref.cell or ''}" if ref.sheet else ref.cell,
                    ] if part)
                    locations.append("；".join(part for part in [location, ref.note] if part))
                derivation_rows.append([
                    "估值输入",
                    f"{label} ({field})",
                    getattr(base, field),
                    _baseline_method(base, field),
                    "；".join(ref.evidence_id for ref in refs),
                    "\n".join(locations),
                ])
            for field, value in sorted(base.statement_items.items()):
                if field in baseline_names:
                    continue
                derivation_rows.append([
                    "原始/中间科目",
                    field,
                    value,
                    _baseline_method(base, field),
                    "",
                    "用于上述估值输入的确定性复算；逐项证据见对应估值输入。",
                ])
        else:
            derivation_rows.append(["状态", "无有效基期财务", "", "", "", ""])
        rows(derivation, derivation_rows)
        header(derivation)
        for row_index in range(2, derivation.max_row + 1):
            derivation.cell(row_index, 3).number_format = '#,##0.0000'
            derivation.row_dimensions[row_index].height = 66

        forecast = sheet("预测与FCFF", (12, 16, 16, 16, 16, 16, 16, 16, 42, 16, 16, 48))
        rows(forecast, [["年度", "收入增长率", "营业收入", "EBIT率", "EBIT", "NOPAT", "折旧摊销", "FCFF", "审计复算公式", "资本开支", "Δ经营营运资本", "计算口径"]] + [
            [item.year, item.revenue_growth, item.revenue, item.ebit_margin, item.ebit,
             item.nopat, item.depreciation_amortization, item.fcff,
             f"=F{row}+G{row}-J{row}-K{row}", item.capital_expenditure,
             item.change_operating_nwc,
             "；".join(f"{key}={value}" for key, value in item.calculation_methods.items())]
            for row, item in enumerate(result.forecast, 2)
        ], formula_columns=(9,))
        header(forecast)
        for row in range(2, forecast.max_row + 1):
            forecast.cell(row, 2).number_format = "0.00%"
            forecast.cell(row, 4).number_format = "0.00%"
            for col in (3, 5, 6, 7, 8, 10, 11):
                forecast.cell(row, col).number_format = '#,##0.00'
            forecast.cell(row, 9).font = Font(color="008000")
            # Keep the full calculation-method trail visible for audit and replay.
            forecast.row_dimensions[row].height = 76

        dcf_recalc = None
        dcf_data_start = None
        if result.dcf and result.effective_financials and result.forecast:
            dcf_recalc = sheet(
                "DCF复算",
                (12, 16, 20, 16, 20, 14, 20, 20, 20, 20, 20, 16, 16, 20, 16, 22, 22, 28, 28),
            )
            total_debt = -result.dcf.bridge.get(
                "total_debt_including_incremental_leases",
                -result.effective_financials.interest_bearing_debt,
            )
            bridge_shares = result.dcf.bridge.get(
                "diluted_or_common_shares",
                result.effective_financials.diluted_shares
                or result.effective_financials.common_shares,
            )
            surplus_cash = result.dcf.bridge.get(
                "surplus_cash",
                result.effective_financials.cash_and_non_operating_assets,
            )
            other_bridge_adjustment = (
                result.dcf.equity_value
                - result.dcf.enterprise_value
                - surplus_cash
                + total_debt
            )
            normalized_terminal = (
                result.assumptions.calculation_methods.get("terminal_value")
                == "gordon_growth_normalized_reinvestment"
                and result.dcf.stable_roic is not None
            )
            stable_roic = result.dcf.stable_roic if normalized_terminal else None
            rows(dcf_recalc, [
                ["参数", "数值", "说明"],
                ["WACC", result.assumptions.wacc, "可编辑；修改后公式输出与敏感性表联动"],
                ["永续增长率", result.assumptions.terminal_growth, "可编辑；必须低于WACC"],
                ["现金及非经营性资产", result.effective_financials.cash_and_non_operating_assets, "企业价值到股权价值桥接"],
                ["总债务（含增量租赁负债）", total_debt, "资本化租赁政策下的完整债务扣减"],
                ["稀释后或普通股股数", bridge_shares,
                 "每股价值分母；优先使用稀释后股数；披露截止日：" + str(result.effective_financials.diluted_shares_as_of or result.effective_financials.common_shares_as_of or "未记录")],
                ["基期营业收入", result.effective_financials.revenue, "最近一期已确认财务事实"],
                ["贴现政策", record.request.discount_policy, "贴现期沿用系统本次运行结果"],
                ["稳定期ROIC", stable_roic, (
                    "终值再投资率 = g / 稳定期ROIC"
                    if normalized_terminal else "本次计算未使用稳定期再投资公式"
                )],
                ["经营必需现金", -result.dcf.bridge.get("operating_cash_requirement", Decimal(0)), "从现金中扣除，仅剩余现金参与股权价值桥接"],
                ["其他股权桥接净调整", other_bridge_adjustment, "联营投资等为加项；少数股东权益、优先股、养老金与预计负债等为扣项"],
            ])
            header(dcf_recalc)
            for row_index in (2, 3, 4, 5, 6, 7, 9, 10, 11):
                cell = dcf_recalc.cell(row_index, 2)
                cell.fill = PatternFill("solid", fgColor="FFF2CC")
                cell.font = Font(color="0000FF")
            for row_index in (2, 3, 9):
                dcf_recalc.cell(row_index, 2).number_format = "0.00%"
            for row_index in (4, 5, 6, 7, 10, 11):
                dcf_recalc.cell(row_index, 2).number_format = '#,##0.00'

            dcf_recalc.cell(13, 1, "基准情景DCF复算")
            dcf_recalc.cell(13, 1).font = Font(bold=True, color=navy)
            dcf_headers = [
                "年度", "收入增长率", "营业收入", "EBIT率", "EBIT", "税率", "NOPAT",
                "折旧摊销", "资本开支", "Δ经营营运资本", "FCFF", "贴现期", "贴现因子", "FCFF现值", "现金流比例",
            ]
            dcf_recalc.append(dcf_headers)
            header(dcf_recalc, 14)
            dcf_data_start = 15
            for offset, item in enumerate(result.forecast):
                row_index = dcf_data_start + offset
                previous_revenue = "$B$7" if offset == 0 else f"C{row_index - 1}"
                tax_rate = item.tax_rate
                if tax_rate is None and offset < len(result.assumptions.tax_rate_path):
                    tax_rate = result.assumptions.tax_rate_path[offset]
                if tax_rate is None:
                    tax_rate = result.effective_financials.tax_rate
                dcf_recalc.append([
                    item.year,
                    _number(item.revenue_growth),
                    f"={previous_revenue}*(1+B{row_index})",
                    _number(item.ebit_margin),
                    f"=C{row_index}*D{row_index}",
                    _number(tax_rate),
                    (
                        f"=E{row_index}-MAX(0,E{row_index})*F{row_index}"
                        if normalized_terminal
                        else f"=E{row_index}*(1-F{row_index})"
                    ),
                    _number(item.depreciation_amortization),
                    _number(item.capital_expenditure),
                    _number(item.change_operating_nwc),
                    f"=G{row_index}+H{row_index}-I{row_index}-J{row_index}",
                    _number(item.discount_period),
                    f"=1/(1+$B$2)^L{row_index}",
                    f"=K{row_index}*O{row_index}*M{row_index}",
                    _number(item.cash_flow_fraction),
                ])
                for column in (2, 4, 6, 8, 9, 10, 12):
                    input_cell = dcf_recalc.cell(row_index, column)
                    input_cell.fill = PatternFill("solid", fgColor="FFF2CC")
                    input_cell.font = Font(color="0000FF")
                for column in (2, 4, 6, 13):
                    dcf_recalc.cell(row_index, column).number_format = "0.00%"
                for column in (3, 5, 7, 8, 9, 10, 11, 14):
                    dcf_recalc.cell(row_index, column).number_format = '#,##0.00'
                dcf_recalc.cell(row_index, 12).number_format = "0.00"

            last_row = dcf_data_start + len(result.forecast) - 1
            terminal_period = f"L{last_row}" if record.request.discount_policy == "year_end" else f"(L{last_row}+O{last_row}/2)"
            cash_bridge = "MAX(0,$B$4-$B$10)"
            if normalized_terminal:
                summary_rows = [
                    ["公式输出", "数值"],
                    ["显性期FCFF现值", f"=SUM(N{dcf_data_start}:N{last_row})"],
                    ["终值期NOPAT", f"=G{last_row}*(1+$B$3)"],
                    ["稳定期再投资率", "=$B$3/$B$9"],
                    ["终值期再投资", "=Q3*Q4"],
                    ["终值期FCFF", "=Q3-Q5"],
                    ["终值", "=Q6/($B$2-$B$3)"],
                    ["终值现值", f"=Q7/(1+$B$2)^{terminal_period}"],
                    ["企业价值", "=Q2+Q8"],
                    ["加：可分配现金及非经营性资产", f"={cash_bridge}"],
                    ["减：总债务", "=$B$5"],
                    ["加/减：其他股权桥接净调整", "=$B$11"],
                    ["普通股股权价值", "=Q9+Q10-Q11+Q12"],
                    ["每股价值", "=Q13/$B$6"],
                    ["系统本次结果", _number(result.dcf.per_share_value)],
                    ["复算差异", "=Q14-Q15"],
                    ["复算检查", '=IF(ABS(Q16)<=0.01,"一致","需复核")'],
                ]
            else:
                summary_rows = [
                    ["公式输出", "数值"],
                    ["显性期FCFF现值", f"=SUM(N{dcf_data_start}:N{last_row})"],
                    ["终年FCFF", f"=K{last_row}"],
                    ["终值", "=Q3*(1+$B$3)/($B$2-$B$3)"],
                    ["终值现值", f"=Q4/(1+$B$2)^{terminal_period}"],
                    ["企业价值", "=Q2+Q5"],
                    ["加：可分配现金及非经营性资产", f"={cash_bridge}"],
                    ["减：总债务", "=$B$5"],
                    ["加/减：其他股权桥接净调整", "=$B$11"],
                    ["普通股股权价值", "=Q6+Q7-Q8+Q9"],
                    ["每股价值", "=Q10/$B$6"],
                    ["系统本次结果", _number(result.dcf.per_share_value)],
                    ["复算差异", "=Q11-Q12"],
                    ["复算检查", '=IF(ABS(Q13)<=0.01,"一致","需复核")'],
                ]
            for row_index, values in enumerate(summary_rows, 1):
                dcf_recalc.cell(row_index, 16, values[0])
                dcf_recalc.cell(row_index, 17, values[1])
            header(dcf_recalc, 1)
            for row_index in range(2, len(summary_rows) + 1):
                dcf_recalc.cell(row_index, 17).number_format = '#,##0.00'
            if normalized_terminal:
                dcf_recalc.cell(4, 17).number_format = "0.00%"
            dcf_recalc.freeze_panes = "A15"

            scenario_row = 20
            for column, value in enumerate(["情景", "系统每股价值", "收入增长路径", "EBIT率路径"], 16):
                dcf_recalc.cell(scenario_row, column, value)
            header(dcf_recalc, scenario_row)
            for offset, scenario in enumerate(("pessimistic", "base", "optimistic"), 1):
                output_row = scenario_row + offset
                dcf_recalc.cell(output_row, 16, scenario)
                dcf_recalc.cell(
                    output_row, 17,
                    _number(result.dcf.scenario_values.get(scenario)),
                )
                dcf_recalc.cell(
                    output_row, 18,
                    ", ".join(f"{value:.2%}" for value in result.assumptions.revenue_growth_scenarios.get(scenario, [])),
                )
                dcf_recalc.cell(
                    output_row, 19,
                    ", ".join(f"{value:.2%}" for value in result.assumptions.ebit_margin_scenarios.get(scenario, [])),
                )
                for column in range(16, 20):
                    dcf_recalc.cell(output_row, column).alignment = Alignment(vertical="top", wrap_text=True)
                dcf_recalc.cell(output_row, 17).number_format = '¥#,##0.00'
                dcf_recalc.row_dimensions[output_row].height = 42
            dcf_recalc.column_dimensions["P"].width = 18
            dcf_recalc.column_dimensions["Q"].width = 18
            dcf_recalc.column_dimensions["R"].width = 42
            dcf_recalc.column_dimensions["S"].width = 42

        requested_relative = any(str(method) != "dcf" for method in record.request.methods)
        if requested_relative or result.effective_peers or result.relative:
            relative = sheet("可比公司", (16, 24, 14, 14, 18, 16, 16, 16, 16, 52))
            rows(relative, [["代码", "公司", "P/E", "P/S", "EV/EBITDA", "层级", "筛选得分", "收入增长", "EBIT率", "筛选依据"]] + [
                [peer.ticker, peer.name, peer.pe, peer.ps, peer.ev_ebitda,
                 peer.peer_tier, peer.selection_score, peer.revenue_growth,
                 peer.ebit_margin, peer.rationale + f"；选样依据={peer.selection_basis}；市值口径={peer.pricing_basis}"]
                for peer in result.effective_peers
            ])
            header(relative)
            for row_index in range(2, relative.max_row + 1):
                for column in (3, 4, 5, 7):
                    relative.cell(row_index, column).number_format = "0.00"
                for column in (8, 9):
                    relative.cell(row_index, column).number_format = "0.00%"
                relative.row_dimensions[row_index].height = 28
            if result.relative:
                relative.append([])
                relative.append(["相对估值结果"])
                title_row = relative.max_row
                relative.cell(title_row, 1).font = Font(bold=True, color=navy)
                relative.append([
                    "方法", "每股价值", "估值区间", "样本数", "样本质量",
                    "异常值数", "统计口径", "纳入公司", "失败原因",
                ])
                result_header = relative.max_row
                header(relative, result_header)
                for item in result.relative:
                    relative.append([
                        item.method.upper(), item.per_share_value,
                        f"{item.range_low:.2f} - {item.range_high:.2f}" if item.range_low is not None and item.range_high is not None else "",
                        item.sample_size, item.sample_quality, item.outlier_count,
                        item.statistic, ", ".join(item.peer_tickers), item.reason or "",
                    ])
                    output_row = relative.max_row
                    relative.cell(output_row, 2).number_format = '¥#,##0.00'
                    relative.row_dimensions[output_row].height = 38

        sensitivity = sheet("敏感性分析", (20, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 70))
        growths = sorted({cell.terminal_growth for cell in result.sensitivity})
        waccs = sorted({cell.wacc for cell in result.sensitivity})
        table = [["WACC / 永续增长率", *growths]] if result.sensitivity else [["相对估值敏感性", "见下表：其余条件不变的单因素压力测试"]]
        lookup = {(cell.wacc, cell.terminal_growth): cell for cell in result.sensitivity}
        for wacc in waccs:
            table.append([wacc, *[
                lookup[(wacc, growth)].per_share_value if lookup[(wacc, growth)].valid else "无效"
                for growth in growths
            ]])
        rows(sensitivity, table)
        header(sensitivity)
        if dcf_recalc is not None and dcf_data_start is not None:
            dcf_last_row = dcf_data_start + len(result.forecast) - 1
            for row_index in range(2, 2 + len(waccs)):
                for column_index in range(2, 2 + len(growths)):
                    wacc_ref = f"$A{row_index}"
                    growth_ref = f"{get_column_letter(column_index)}$1"
                    explicit_terms = "+".join(
                        f"'DCF复算'!$K${forecast_row}*'DCF复算'!$O${forecast_row}/(1+{wacc_ref})^'DCF复算'!$L${forecast_row}"
                        for forecast_row in range(dcf_data_start, dcf_last_row + 1)
                    )
                    terminal_period_ref = f"'DCF复算'!$L${dcf_last_row}" if record.request.discount_policy == "year_end" else f"('DCF复算'!$L${dcf_last_row}+'DCF复算'!$O${dcf_last_row}/2)"
                    cash_ref = "MAX(0,'DCF复算'!$B$4-'DCF复算'!$B$10)"
                    terminal_term = (
                        (
                            f"'DCF复算'!$G${dcf_last_row}*(1+{growth_ref})*"
                            f"(1-{growth_ref}/'DCF复算'!$B$9)/"
                            f"({wacc_ref}-{growth_ref})/(1+{wacc_ref})^{terminal_period_ref}"
                        )
                        if normalized_terminal
                        else (
                            f"'DCF复算'!$K${dcf_last_row}*(1+{growth_ref})/"
                            f"({wacc_ref}-{growth_ref})/(1+{wacc_ref})^{terminal_period_ref}"
                        )
                    )
                    invalid_condition = f"{wacc_ref}<={growth_ref}"
                    if normalized_terminal:
                        invalid_condition = (
                            f"OR({invalid_condition},"
                            f"{growth_ref}>='DCF复算'!$B$9)"
                        )
                    sensitivity.cell(row_index, column_index).value = (
                        f'=IF({invalid_condition},"无效",'
                        f"({explicit_terms}+{terminal_term}+{cash_ref}-"
                        f"'DCF复算'!$B$5+'DCF复算'!$B$11)/'DCF复算'!$B$6)"
                    )
        for row in range(2, sensitivity.max_row + 1):
            sensitivity.cell(row, 1).number_format = "0.00%"
        for col in range(2, sensitivity.max_column + 1):
            sensitivity.cell(1, col).number_format = "0.00%"
            for row in range(2, sensitivity.max_row + 1):
                sensitivity.cell(row, col).number_format = '¥#,##0.00'

        if result.sensitivity_studies:
            sensitivity.append([])
            sensitivity.append([
                "编号", "参数", "基准输入", "低值输入", "高值输入", "基准每股",
                "低值每股", "高值每股", "最大相对变动", "影响等级", "状态", "说明",
            ])
            detail_header = sensitivity.max_row
            for study in result.sensitivity_studies:
                sensitivity.append([
                    study.study_id, study.parameter, study.baseline_input,
                    study.low_input, study.high_input, study.baseline_per_share,
                    study.low_per_share, study.high_per_share,
                    study.max_relative_change, study.classification,
                    study.status, study.rationale,
                ])
            header(sensitivity, detail_header)
            for row in range(detail_header + 1, sensitivity.max_row + 1):
                for col in (6, 7, 8):
                    sensitivity.cell(row, col).number_format = '¥#,##0.00'
                sensitivity.cell(row, 9).number_format = "0.00%"
            sensitivity.column_dimensions["L"].width = 70

            completed_rows = [
                (study.study_id + " " + study.parameter, study.max_relative_change)
                for study in result.sensitivity_studies
                if study.status == "completed" and study.max_relative_change is not None
            ]
            completed_rows.sort(key=lambda item: item[1], reverse=True)
            if completed_rows:
                chart_start = sensitivity.max_row + 3
                sensitivity.cell(chart_start, 1, "敏感性因素")
                sensitivity.cell(chart_start, 2, "最大相对变动")
                for offset, (label, impact) in enumerate(completed_rows, 1):
                    sensitivity.cell(chart_start + offset, 1, label)
                    sensitivity.cell(chart_start + offset, 2, _number(impact))
                    sensitivity.cell(chart_start + offset, 2).number_format = "0.00%"
                chart = BarChart()
                chart.type = "bar"
                chart.style = 10
                chart.title = "敏感性因素排序"
                chart.y_axis.title = "参数"
                chart.x_axis.title = "相对估值变动"
                chart.x_axis.numFmt = "0%"
                chart.legend = None
                chart.height = 8
                chart.width = 15
                chart.add_data(
                    Reference(sensitivity, min_col=2, min_row=chart_start,
                              max_row=chart_start + len(completed_rows)),
                    titles_from_data=True,
                )
                chart.set_categories(
                    Reference(sensitivity, min_col=1, min_row=chart_start + 1,
                              max_row=chart_start + len(completed_rows))
                )
                sensitivity.add_chart(chart, "N2")

        sources = sheet("来源与风险", (14, 24, 36, 32, 16, 68))
        source_rows = [["性质", "字段/编号", "来源", "定位", "发布日期", "说明"]]
        for field, refs in result.effective_financials.evidence.items() if result.effective_financials else []:
            for ref in refs:
                location = " · ".join(part for part in [
                    ref.file_id,
                    f"第{ref.page}页" if ref.page else None,
                    f"{ref.sheet}!{ref.cell or ''}" if ref.sheet else ref.cell,
                ] if part)
                source_rows.append([
                    "用户输入（未经外部核验）" if ref.source == "user_input" else "事实", field, ref.source, location,
                    str(ref.published_at) if ref.published_at else "", _source_note(ref),
                ])
        for field, refs in result.assumption_evidence.items():
            for ref in refs:
                location = " · ".join(part for part in [
                    ref.file_id,
                    f"第{ref.page}页" if ref.page else None,
                    f"{ref.sheet}!{ref.cell or ''}" if ref.sheet else ref.cell,
                ] if part)
                source_rows.append([
                    "假设", field, ref.source, location,
                    str(ref.published_at) if ref.published_at else "", _source_note(ref),
                ])
        for peer in result.effective_peers:
            for metric, refs in peer.evidence.items():
                for ref in refs:
                    source_rows.append(["可比事实", f"{peer.ticker} {metric}", ref.source,
                        f"{peer.name} · {peer.as_of_date} · {peer.multiple_basis}", str(ref.published_at or ""), _source_note(ref)])
        source_rows += [
            ["推论", f"Q{index}", "系统质量评估", "", "", note]
            for index, note in enumerate(result.data_quality.notes, 1)
        ]
        source_rows += [["风险", f"W{index}", "系统校验", "", "", warning] for index, warning in enumerate(result.warnings, 1)]
        if not result.warnings:
            source_rows.append(["风险", "", "系统校验", "", "", "本次运行未产生系统警告；仍需人工复核关键假设与同业口径。"])
        rows(sources, source_rows)
        header(sources)
        for row_index in range(2, sources.max_row + 1):
            description = str(sources.cell(row_index, 6).value or "")
            sources.row_dimensions[row_index].height = min(240, max(34, 15 * (2 + len(description) // 40)))

        if not result.forecast:
            wb.remove(forecast)

        if audit_context:
            disposition_by_finding = {
                item["finding_id"]: item
                for item in audit_context.get("finding_dispositions", [])
            }
            challenge = sheet("挑战与处置", (22, 14, 30, 54, 22, 54, 54))
            challenge_rows = [[
                "异议ID", "严重性", "类别", "异议", "处置", "处置依据", "后续动作"
            ]]
            for finding in audit_context.get("findings", []):
                disposition = disposition_by_finding.get(finding["finding_id"], {})
                challenge_rows.append([
                    finding["finding_id"], finding["severity"], finding["category"],
                    finding["title"] + "\n" + finding["analysis"],
                    disposition.get("decision", "待处理"),
                    disposition.get("rationale", ""),
                    disposition.get("resulting_action", finding.get("recommendation", "")),
                ])
            if len(challenge_rows) == 1:
                challenge_rows.append(["-", "-", "-", "未形成重大异议", "-", "-", "-"])
            rows(challenge, challenge_rows)
            header(challenge)

            adopted_evidence = {
                item["evidence_id"] for item in audit_context.get("evidence_usage", [])
                if item.get("adopted")
            }
            evidence_book = sheet("证据账本", (22, 12, 24, 34, 50, 16, 16, 14, 26))
            evidence_rows = [[
                "证据ID", "等级", "来源类型", "发布机构/来源", "URL/定位", "截止日合规",
                "综合评分", "本版本采用", "状态/限制",
            ]]
            for item in audit_context.get("evidence_ledger", []):
                location = " · ".join(
                    f"{key}={value}" for key, value in (item.get("locator") or {}).items()
                    if value not in (None, "")
                )
                evidence_rows.append([
                    item["evidence_id"], item["authority_tier"], item["source_type"],
                    item.get("publisher") or item.get("provider") or item.get("title"),
                    "\n".join(part for part in (item.get("url"), location) if part),
                    item.get("information_cutoff_ok"), item.get("confidence"),
                    "是" if item["evidence_id"] in adopted_evidence else "否",
                    item.get("status", "") + ("；" + "、".join(item.get("limitations") or []) if item.get("limitations") else ""),
                ])
            if len(evidence_rows) == 1:
                evidence_rows.append(["-", "-", "-", "无工作区证据记录", "", "", "", "", ""])
            rows(evidence_book, evidence_rows)
            header(evidence_book)

            actions_book = sheet("行动账本", (23, 18, 24, 52, 24, 18, 25))
            action_rows = [["行动ID", "执行者", "动作", "可核验摘要", "工具", "耗时ms", "状态"]]
            for item in audit_context.get("action_ledger", []):
                action_rows.append([
                    item["action_id"], item["actor"], item["action_type"], item["summary"],
                    "@".join(part for part in (item.get("tool_name"), item.get("tool_version")) if part),
                    item.get("duration_ms"), item["status"],
                ])
            rows(actions_book, action_rows)
            header(actions_book)

            version_book = sheet("版本与复现", (30, 72, 64))
            workspace = audit_context.get("workspace") or {}
            version = audit_context.get("version") or {}
            spec = audit_context.get("model_spec") or {}
            calculation = audit_context.get("calculation") or {}
            version_rows = [
                ["项目", "值", "说明"],
                ["工作区", workspace.get("workspace_id", "-"), workspace.get("title", "")],
                ["运行模式", workspace.get("run_policy", "-"), "review=审阅模式；automatic=自动模式"],
                ["估值版本", f"V{version.get('number', record.revision)}", version.get("reason", "")],
                ["ModelSpec", spec.get("model_spec_id", "-"), spec.get("snapshot_hash", "")],
                ["估值日", spec.get("valuation_date", str(result.valuation_date)), ""],
                ["信息截止日", spec.get("information_cutoff_date", "-"), "估值不得静默使用该日之后发布的信息"],
                ["实际执行日", spec.get("execution_date", "-"), ""],
                ["计算记录", calculation.get("calculation_id", "-"), calculation.get("status", "")],
                ["输入哈希", calculation.get("input_hash", record.input_hash), ""],
                ["结果哈希", calculation.get("result_hash", "-"), ""],
                ["金融模型", spec.get("financial_model_name", result.model_version), spec.get("financial_model_version", result.model_version)],
                ["计算复现", (audit_context.get("reproducibility") or {}).get("calculation", "not_packaged"), "计算不需要重新联网或重新调用LLM"],
                ["数据获取复现", (audit_context.get("reproducibility") or {}).get("data_acquisition", "not_packaged"), "网页可能变化，以快照、引文和哈希为准"],
            ]
            rows(version_book, version_rows)
            header(version_book)

        for ws in wb.worksheets:
            if ws.title == "DCF复算" and dcf_data_start is not None:
                ws.auto_filter.ref = f"A11:O{dcf_data_start + len(result.forecast) - 1}"
            else:
                ws.auto_filter.ref = ws.dimensions
            for row in range(2, ws.max_row + 1):
                if row % 2 == 0:
                    for cell in ws[row]:
                        if cell.fill.fill_type is None:
                            cell.fill = PatternFill("solid", fgColor=pale)
            ws.sheet_properties.pageSetUpPr.fitToPage = True
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 0
        summary["A1"].fill = PatternFill("solid", fgColor=teal)
        sources["A2"].fill = PatternFill("solid", fgColor=amber)
        wb.calculation.fullCalcOnLoad = True
        wb.calculation.forceFullCalc = True
        output = BytesIO()
        wb.save(output)
        return output.getvalue()

    def pdf(self, record: RunRecord, audit_context=None) -> bytes:
        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFError, TTFont
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            from reportlab.platypus import (
                KeepTogether,
                PageBreak,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise ValueError("PDF报告组件未安装，请安装 valuationagent[reports]。") from exc

        result = _completed(record)
        font_path = next((path for path in [
            Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")
        ] if path.is_file()), None)
        font = "Helvetica"
        if font_path:
            try:
                pdfmetrics.registerFont(TTFont("ValuationCN", str(font_path), subfontIndex=0))
                font = "ValuationCN"
            except (OSError, TTFError, ValueError):
                font = "Helvetica"
        if font == "Helvetica":
            # Portable CJK fallback; never silently render Chinese as squares.
            pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            font = "STSong-Light"
        styles = getSampleStyleSheet()
        title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName=font,
                               fontSize=22, leading=30, textColor=colors.HexColor("#14273D"),
                               alignment=TA_CENTER, spaceAfter=12)
        h2 = ParagraphStyle("H2CN", parent=styles["Heading2"], fontName=font,
                            fontSize=14, textColor=colors.HexColor("#0F6F72"), spaceBefore=10, spaceAfter=7)
        body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName=font,
                              fontSize=9, leading=14, textColor=colors.HexColor("#24364B"))
        small = ParagraphStyle("SmallCN", parent=body, fontSize=7.5, leading=11)
        table_header = ParagraphStyle(
            "TableHeaderCN", parent=small, textColor=colors.white
        )

        def p(value, style=body):
            text = str(_label(value) if value is not None else "-")
            text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return Paragraph(text, style)

        def table(data, widths=None, header=True):
            rendered = []
            for row_index, row in enumerate(data):
                style = table_header if header and row_index == 0 else small
                rendered.append([p(cell, style) for cell in row])
            t = Table(rendered, colWidths=widths, repeatRows=1 if header else 0)
            commands = [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D7DEE7")),
                ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F2F7F8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            if header:
                commands += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14273D")),
                             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
            t.setStyle(TableStyle(commands))
            return t

        output = BytesIO()
        doc = SimpleDocTemplate(output, pagesize=A4, rightMargin=16*mm, leftMargin=16*mm,
                                topMargin=16*mm, bottomMargin=16*mm,
                                title=f"{result.company.name or result.company.ticker}估值报告")

        def page_decor(canvas, document):
            canvas.saveState()
            canvas.setFont(font, 7.5)
            canvas.setFillColor(colors.HexColor("#65758B"))
            canvas.drawString(16*mm, 9*mm, f"ValuationAgent | {result.company.name or result.company.ticker or ''}")
            canvas.drawRightString(A4[0] - 16*mm, 9*mm, f"Page {document.page}")
            canvas.restoreState()

        audit_context = audit_context or {}
        workspace_meta = audit_context.get("workspace") or {}
        version_meta = audit_context.get("version") or {}
        spec_meta = audit_context.get("model_spec") or {}
        calculation_meta = audit_context.get("calculation") or {}
        checkpoint_meta = audit_context.get("checkpoint") or {}
        decision_meta = ((audit_context.get("decisions") or [{}])[-1] or {})
        decision_labels = {
            "accepted": "已接受，可按披露范围使用",
            "accepted_with_warnings": "附带警示接受",
            "review_required": "需复核，不得作为正式数值结论",
            "rejected": "已拒绝，不得作为估值结论",
        }
        conclusion_status = decision_labels.get(
            decision_meta.get("outcome"),
            "计算完成（未绑定工作区决策账本）",
        )

        story = [p("ValuationAgent 正式估值报告", title),
                 p(f"{result.company.name or ''} · {result.company.ticker or ''} · 基准日 {result.valuation_date}"),
                 Spacer(1, 5*mm), p("1. 估值摘要与结论", h2)]
        story.append(table([
            ["方法", "每股价值", "估值区间/状态"],
            ["DCF", result.dcf.per_share_value if result.dcf else "—",
             f"{result.dcf.range_low} – {result.dcf.range_high}" if result.dcf else "未采用"],
            *([["退出倍数校验", result.dcf.exit_multiple_per_share,
                f"相对Gordon差异 {result.dcf.terminal_method_gap:.2%}；不替代主估值"]]
              if result.dcf and result.dcf.exit_multiple_per_share is not None else []),
            *[[item.method.upper(), item.per_share_value or "—",
               f"{item.range_low} – {item.range_high}" if item.status == "success" else item.reason]
              for item in result.relative],
        ], [35*mm, 38*mm, 85*mm]))
        story.append(table([
            ["决策层结论状态", conclusion_status],
            ["处置要求", decision_meta.get("selected_action", "正式使用前应核验适用范围与关键假设")],
        ], [42*mm, 116*mm], header=False))
        if decision_meta.get("outcome") in {"review_required", "rejected"}:
            story.append(p(
                "重要：本报告保留计算结果仅用于诊断和复核，不构成可直接采用的正式数值结论。",
                h2,
            ))
        if record.request.excluded_methods:
            story += [p("数据缺失与方法降级披露", h2), table([
                ["原选方法", "处理", "原因"],
                *[
                    [method.upper(), "未进入本次计算", reason]
                    for method, reason in record.request.excluded_methods.items()
                ],
            ], [30*mm, 40*mm, 88*mm])]
        story += [Spacer(1, 3*mm), p(result.executive_summary)]
        story += [p("2. 研究对象、估值日和信息截止日", h2), table([
            ["项目", "记录值", "审计说明"],
            ["研究对象", result.company.name or "-", result.company.ticker or "未记录证券代码"],
            ["估值日", result.valuation_date, "市场参数与股权价值口径的时点"],
            ["信息截止日", spec_meta.get("information_cutoff_date") or workspace_meta.get("information_cutoff_date") or result.valuation_date,
             "不得静默采用该日之后发布的信息"],
            ["实际执行日", spec_meta.get("execution_date") or workspace_meta.get("execution_date") or "未记录",
             "与估值日、信息截止日分别记录"],
            ["币种", result.company.currency, "除非表格另有说明"],
        ], [37*mm, 43*mm, 78*mm])]
        story += [p("3. 公司类型、行业和模型选择", h2), table([
            ["项目", "结论", "依据/范围"],
            ["公司类型", checkpoint_meta.get("company_type") or workspace_meta.get("company_type") or "未单独记录",
             "公司类型决定行业化适配与可用模型"],
            ["行业", result.company.industry or checkpoint_meta.get("industry") or "未确认",
             workspace_meta.get("industry_strategy") or "沿用已冻结金融模型的行业适配"],
            ["请求方法", "、".join(str(item).upper() for item in record.request.requested_methods),
             "用户目标或自动基准方案"],
            ["实际方法", "、".join(str(item).upper() for item in record.request.methods),
             "仅执行通过确定性输入检查的方法"],
        ], [37*mm, 43*mm, 78*mm])]
        story += [p("4. 数据来源与证据覆盖", h2), table([
            ["指标", "结论", "说明"],
            ["总体置信度", result.data_quality.confidence,
             f"结果等级 {result.data_quality.result_grade} · 可比历史 {result.data_quality.comparable_years}/{result.data_quality.historical_years} 年"],
            ["市场参数", result.data_quality.market_input_quality,
             "降级字段：" + ("、".join(result.data_quality.degraded_fields) or "无")],
            ["证据覆盖率", f"{result.data_quality.evidence_coverage:.2%}",
             (f"行业参数 {result.data_quality.industry_parameter_quality} · 元数据 {result.data_quality.industry_metadata_completeness}"
              if result.dcf else "仅统计本次相对估值所需的基期财务字段")],
            ["可比样本", result.data_quality.peer_sample_quality,
             "；".join(result.data_quality.notes) or "未产生额外质量提示"],
            ["证据账本", len(audit_context.get("evidence_ledger", [])),
             "每条事实保留发布机构、URL/文件、定位、引文、哈希、等级与截止日判断"],
        ], [38*mm, 34*mm, 86*mm])]
        if result.effective_financials:
            base = result.effective_financials
            baseline_rows = [["基期字段", "数值", "计算/取数口径", "证据ID"]]
            for field, label in BASELINE_FIELDS:
                if getattr(base, field) is None:
                    continue
                refs = base.evidence.get(field, [])
                baseline_rows.append([
                    label,
                    getattr(base, field),
                    _baseline_method(base, field),
                    "；".join(ref.evidence_id for ref in refs) or "-",
                ])
            story += [p("5. 历史财务标准化", h2), table(
                baseline_rows,
                [31*mm, 31*mm, 65*mm, 31*mm],
            )]
        else:
            story += [p("5. 历史财务标准化", h2), p("本版本没有可用于正式计算的基期财务快照。", small)]

        adjustment_rows = [["调整项目", "处理方式", "理由/影响"]]
        if result.effective_financials:
            base = result.effective_financials
            for field, method in base.calculation_methods.items():
                if field.startswith("reconciliation.") or any(
                    token in str(method).lower() for token in ("adjust", "derive", "reclass", "exclude")
                ):
                    adjustment_rows.append([field, method, "按冻结输入的可核验计算口径处理"])
            if base.comparability_note:
                adjustment_rows.append(["可比性", base.comparability_status, base.comparability_note])
        for method, reason in record.request.excluded_methods.items():
            adjustment_rows.append([method.upper(), "方法降级/排除", reason])
        if len(adjustment_rows) == 1:
            adjustment_rows.append(["无单列调整", "保持披露口径", "未识别需要单列披露的非经常性或重分类调整"])
        story += [p("6. 非经常性及口径调整", h2), table(
            adjustment_rows, [40*mm, 43*mm, 75*mm]
        )]

        story += [p("7. 经营预测及全部假设", h2), table([
            ["参数", "基准值", "依据"],
            ["收入增速路径", "、".join(f"{value:.2%}" for value in result.assumptions.revenue_growth), result.assumptions.rationale.get("revenue_growth", "")],
            ["EBIT率路径", "、".join(f"{value:.2%}" for value in result.assumptions.ebit_margin), result.assumptions.rationale.get("ebit_margin", "")],
            ["永续增长率", f"{result.assumptions.terminal_growth:.2%}" if result.dcf else "不适用", result.assumptions.rationale.get("terminal_growth", "")],
            ["稳定期ROIC", f"{result.dcf.stable_roic:.2%}" if result.dcf and result.dcf.stable_roic is not None else "不适用", result.assumptions.rationale.get("stable_roic", "")],
            ["假设来源", result.assumptions.source, "详细证据和用户覆盖见第15节及导出账本"],
        ], [38*mm, 45*mm, 75*mm])]
        if result.forecast:
            story += [PageBreak(), p(f"7.1 {len(result.forecast)}年预测与FCFF", h2)]
            story.append(table([["年", "收入增长", "EBIT率", "收入", "FCFF"]] + [
                [item.year, f"{item.revenue_growth:.2%}", f"{item.ebit_margin:.2%}",
                 f"{item.revenue:,.0f}", f"{item.fcff:,.0f}"] for item in result.forecast
            ], [19*mm, 29*mm, 26*mm, 43*mm, 41*mm]))
        story += [p("8. WACC 及资本成本依据", h2), table([
            ["参数", "数值", "依据"],
            ["WACC", f"{result.assumptions.wacc:.2%}" if result.dcf else "不适用", result.assumptions.rationale.get("wacc", "")],
            *[[key, value, result.assumptions.rationale.get(key, "WACC组成参数")]
              for key, value in result.assumptions.wacc_components.items()],
            ["市场参数时点", record.request.assumptions.market_inputs_as_of or "未单列记录", record.request.assumptions.market_inputs_source or "见证据索引"],
        ], [38*mm, 34*mm, 86*mm])]
        if result.dcf:
            story += [p("9. DCF 估值", h2), table([
                ["项目", "数值", "项目", "数值"],
                ["显性期FCFF现值", (
                    f"{result.dcf.present_value_explicit:,.0f}"
                    if result.dcf.present_value_explicit is not None else "-"
                 ), "终值现值", (
                    f"{result.dcf.present_value_terminal:,.0f}"
                    if result.dcf.present_value_terminal is not None else "-"
                 )],
                ["企业价值", f"{result.dcf.enterprise_value:,.0f}",
                 "股权价值", f"{result.dcf.equity_value:,.0f}"],
                ["终值占企业价值", f"{result.dcf.terminal_value_share:.2%}",
                 "基准每股价值", f"{result.dcf.per_share_value:.2f}"],
                ["终值期FCFF", (
                    f"{result.dcf.terminal_fcff:,.0f}"
                    if result.dcf.terminal_fcff is not None else "-"
                 ), "稳定期再投资率", (
                    f"{result.dcf.terminal_reinvestment_rate:.2%}"
                    if result.dcf.terminal_reinvestment_rate is not None else "-"
                 )],
            ], [42*mm, 37*mm, 42*mm, 37*mm])]
            scenario_labels = {
                "pessimistic": "悲观", "base": "基准", "optimistic": "乐观",
            }
            story.append(table(
                [["情景", "每股价值"]] + [
                    [scenario_labels.get(name, name), f"{value:.2f}"]
                    for name, value in result.dcf.scenario_values.items()
                ],
                [42*mm, 37*mm],
            ))
            story.append(p(
                "公式口径：终值期FCFF = N+1期NOPAT × (1 − g/稳定期ROIC)；企业价值 = "
                "显性期FCFF现值 + 终值现值。股权价值按可分配现金、完整债务、租赁、"
                "少数股东权益及其他非经营项目完成桥接，再除以稀释后或普通股股数。",
                small,
            ))
            if result.dcf.bridge_unmeasured_items:
                story.append(p(
                    "未建立桥接项："
                    + "、".join(result.dcf.bridge_unmeasured_items)
                    + "。这些项目未被视为已确认0，本次未做相应调整，详见风险与质量说明。",
                    small,
                ))
        else:
            story += [p("9. DCF 估值", h2), p("本版本未采用DCF；不展示WACC-g矩阵或终值结果。", small)]
        requested_relative = any(str(method) != "dcf" for method in record.request.methods)
        if requested_relative or result.effective_peers or result.relative:
            story += [PageBreak(), p("10. 相对估值", h2)]
            if record.request.peer_screening:
                story.append(p("可比选样为Agent判断，未独立核验业务、规模与盈利质量可比性。"))
                for entry in record.request.peer_screening:
                    story.append(p(f"{entry['name']}（{entry['ticker']}）：{entry['status']}；{entry['rationale']}；"
                        + "；".join(entry['reasons'])))
            if any(peer.pricing_basis == "a_share_equivalent" for peer in result.effective_peers):
                story.append(p("部分可比使用总股本乘A股价格的等值市值，多类别股份时不等于各市场实际市值之和；估值须按此条件解读。"))
            if result.effective_peers:
                story.append(table([["代码", "公司", "P/E", "P/S", "EV/EBITDA", "层级/得分"]] + [
                    [peer.ticker, peer.name, peer.pe or "-", peer.ps or "-", peer.ev_ebitda or "-",
                     f"{_label(peer.peer_tier)} / {peer.selection_score if peer.selection_score is not None else '-'}"]
                    for peer in result.effective_peers
                ], [24*mm, 38*mm, 22*mm, 22*mm, 27*mm, 31*mm]))
            story.append(table(
                [["方法", "状态", "每股价值", "区间/原因"]] + [
                    [
                        item.method.upper(), item.status,
                        f"{item.per_share_value:.2f}" if item.per_share_value is not None else "-",
                        (
                            f"{item.range_low:.2f} - {item.range_high:.2f}"
                            if item.range_low is not None and item.range_high is not None else
                            f"指定倍数 {item.selected_multiple}；股权价值 {item.equity_value}；非统计区间" if item.valuation_basis == "explicit_multiple" else item.reason
                        ),
                    ]
                    for item in result.relative
                ],
                [30*mm, 28*mm, 30*mm, 70*mm],
            ))
        else:
            story += [PageBreak(), p("10. 相对估值", h2), p("本版本未采用相对估值方法。", small)]
        story += [p("11. 股权价值桥接", h2)]
        if result.dcf:
            story.append(table(
                [["桥接项目", "数值"]]
                + [[name, value] for name, value in result.dcf.bridge.items()]
                + [["企业价值", result.dcf.enterprise_value], ["普通股股权价值", result.dcf.equity_value]],
                [65*mm, 60*mm],
            ))
            if result.dcf.bridge_unmeasured_items:
                story.append(p(
                    "未建立且未按0处理的桥接项目：" + "、".join(result.dcf.bridge_unmeasured_items),
                    small,
                ))
        else:
            story.append(p("本版本未采用企业价值到股权价值的DCF桥接。", small))

        story += [p("12. 敏感性分析", h2)]
        growths = sorted({cell.terminal_growth for cell in result.sensitivity})
        waccs = sorted({cell.wacc for cell in result.sensitivity})
        lookup = {(cell.wacc, cell.terminal_growth): cell for cell in result.sensitivity}
        sensitivity_rows = [["WACC / g", *[f"{g:.2%}" for g in growths]]]
        for wacc in waccs:
            sensitivity_rows.append([f"{wacc:.2%}", *[
                f"{lookup[(wacc, growth)].per_share_value:.2f}" if lookup[(wacc, growth)].valid else "无效"
                for growth in growths]])
        if result.sensitivity:
            story.append(table(sensitivity_rows))
        if result.sensitivity_studies:
            story += [p("12.1 敏感性项目与结果", h2)]
            if not result.dcf:
                story.append(p("各项分别测试基期指标或同业倍数上下变动10%，其他条件不变；属于压力测试，不是概率置信区间。", small))
            impact_rows = [["编号", "参数", "低值", "基准", "高值", "影响", "状态"]]
            for study in result.sensitivity_studies:
                impact_rows.append([
                    study.study_id,
                    study.parameter,
                    f"{study.low_per_share:.2f}" if study.low_per_share is not None else "—",
                    f"{study.baseline_per_share:.2f}" if study.baseline_per_share is not None else "—",
                    f"{study.high_per_share:.2f}" if study.high_per_share is not None else "—",
                    (
                        f"{study.max_relative_change:.2%} / {_label(study.classification)}"
                        if study.max_relative_change is not None else "未测试"
                    ),
                    study.status,
                ])
            story.append(table(
                impact_rows,
                [14*mm, 37*mm, 21*mm, 21*mm, 21*mm, 28*mm, 22*mm],
            ))
            unavailable_notes = [
                f"{study.study_id} {study.parameter}：{study.rationale}"
                for study in result.sensitivity_studies
                if study.status == "not_available"
            ]
            if unavailable_notes:
                story.append(KeepTogether([
                    p("未完成敏感性项及原因", h2),
                    p(f"• {unavailable_notes[0]}", small),
                ]))
                story += [p(f"• {note}", small) for note in unavailable_notes[1:]]

        method_values = []
        if result.dcf:
            method_values.append(("DCF", result.dcf.per_share_value))
        method_values += [
            (item.method.upper(), item.per_share_value)
            for item in result.relative
            if item.status == "success" and item.per_share_value is not None
        ]
        story += [p("13. 方法交叉验证", h2)]
        if method_values:
            numeric_values = [value for _, value in method_values]
            spread = max(numeric_values) - min(numeric_values) if len(numeric_values) > 1 else 0
            story.append(table(
                [["方法", "每股价值"], *method_values, ["方法间极差", spread]],
                [65*mm, 60*mm],
            ))
            story.append(p(
                "不同方法衡量的经济口径并不完全相同；差异用于风险判断，不以简单平均替代专业判断。",
                small,
            ))
        else:
            story.append(p("没有可交叉验证的数值方法。", small))

        story += [PageBreak(), p("14. 挑战 Agent 发现与处理", h2)]
        dispositions = {
            item["finding_id"]: item
            for item in audit_context.get("finding_dispositions", [])
        }
        challenge_rows = [["严重性", "异议", "处置", "依据与后续动作"]]
        for finding in audit_context.get("findings", []):
            disposition = dispositions.get(finding["finding_id"], {})
            challenge_rows.append([
                finding["severity"], finding["title"],
                disposition.get("decision", "待处理"),
                "；".join(part for part in (
                    disposition.get("rationale"),
                    disposition.get("resulting_action") or finding.get("recommendation"),
                ) if part),
            ])
        if len(challenge_rows) == 1:
            challenge_rows.append(["-", "未形成需单列披露的挑战记录", "-", "直接运行未绑定工作区挑战账本"])
        story.append(table(challenge_rows, [22*mm, 48*mm, 31*mm, 57*mm]))

        story += [p("15. 用户提供数据及用户覆盖", h2)]
        overrides = checkpoint_meta.get("user_overrides") or []
        user_evidence = [
            item for item in audit_context.get("evidence_ledger", [])
            if item.get("source_type") in {"user_document", "user_statement"}
        ]
        user_rows = [["类别", "项目", "值/说明"]]
        user_rows += [["用户覆盖", item.get("field") or item.get("parameter") or "未命名", item.get("value") or item] for item in overrides]
        user_rows += [["用户证据", item.get("evidence_id"), item.get("title")] for item in user_evidence]
        if len(user_rows) == 1:
            user_rows.append(["无", "-", "本版本未记录用户覆盖值或用户文件证据"])
        story.append(table(user_rows, [31*mm, 45*mm, 82*mm]))

        story += [p("16. 事实、推论和观点区分", h2), table([
            ["类型", "本报告中的定义", "示例"],
            ["事实", "可回指原始文件、网页、引文和定位的披露数据", "历史收入、股数、净债务项目"],
            ["转换/计算", "由已披露事实按明确公式和单位规则得到", "万元转亿元、EBIT与FCFF推导"],
            ["模型推论", "由冻结 ModelSpec 和金融公式产生", "DCF每股价值、敏感性结果"],
            ["观点/判断", "选择方法、同业和风险解释", "可比公司适配度、结果合理性判断"],
        ], [28*mm, 78*mm, 52*mm])]

        story += [p("17. 风险、适用范围和局限", h2)]
        story.append(p(
            "口径说明：已确认财务及其原文定位属于事实；经营路径与资本成本属于假设；"
            "估值区间、敏感性和结论属于模型推论，不构成投资建议。",
            small,
        ))
        notes = result.warnings or ["系统未产生运行警告；关键假设、同业口径与业务判断仍需人工复核。"]
        story += [p(f"• {note}") for note in notes]

        story += [p("18. 数据缺失与代理方法", h2)]
        gap_rows = [["项目", "处理", "影响"]]
        for method, reason in record.request.excluded_methods.items():
            gap_rows.append([method.upper(), "排除该方法", reason])
        for field in result.data_quality.degraded_fields:
            gap_rows.append([field, "降级或代理", "已降低结果质量等级并在敏感性/风险中披露"])
        if result.dcf:
            for item in result.dcf.bridge_unmeasured_items:
                gap_rows.append([item, "保持未知，不按0填充", "股权价值桥可能不完整"])
        if len(gap_rows) == 1:
            gap_rows.append(["无重大缺失", "无额外代理", "仍需结合适用范围审阅"])
        story.append(table(gap_rows, [44*mm, 46*mm, 68*mm]))

        story += [p("19. 版本变更记录", h2)]
        versions = audit_context.get("versions") or []
        if versions:
            story.append(table(
                [["版本", "原因", "状态", "变更归因"]] + [
                    [f"V{item['number']}", item["reason"], item["status"], item.get("change_attribution") or item.get("changes") or "-"]
                    for item in versions
                ],
                [18*mm, 53*mm, 28*mm, 59*mm],
            ))
        else:
            story.append(p(f"独立运行修订号 V{record.revision}；未绑定工作区版本账本。", small))

        evidence_rows = [["性质", "字段", "来源", "定位"]]
        if result.effective_financials:
            for field, refs in result.effective_financials.evidence.items():
                for ref in refs:
                    location = " · ".join(part for part in [
                        ref.file_id,
                        f"第{ref.page}页" if ref.page else None,
                        f"{ref.sheet}!{ref.cell or ''}" if ref.sheet else ref.cell,
                    ] if part)
                    evidence_rows.append([
                        "用户输入（未经外部核验）" if ref.source == "user_input" else "事实", field, ref.source,
                        "；".join(part for part in [location, _source_note(ref)] if part),
                    ])
        for field, refs in result.assumption_evidence.items():
            for ref in refs:
                location = " · ".join(part for part in [
                    ref.file_id,
                    f"第{ref.page}页" if ref.page else None,
                    f"{ref.sheet}!{ref.cell or ''}" if ref.sheet else ref.cell,
                ] if part)
                evidence_rows.append([
                    "假设", field, ref.source,
                    "；".join(part for part in [location, _source_note(ref)] if part),
                ])
        for peer in result.effective_peers:
            for metric, refs in peer.evidence.items():
                for ref in refs:
                    evidence_rows.append(["可比事实", f"{peer.ticker} {metric}", ref.source,
                        f"{peer.name} · {peer.as_of_date} · {peer.multiple_basis}；{_source_note(ref)}"])
        workspace_evidence_ids = {row[1] for row in evidence_rows[1:]}
        for item in audit_context.get("evidence_ledger", []):
            if item.get("evidence_id") in workspace_evidence_ids:
                continue
            locator = " · ".join(
                f"{key}={value}" for key, value in (item.get("locator") or {}).items()
                if value not in (None, "")
            )
            evidence_rows.append([
                item.get("source_type", "证据"), item.get("evidence_id", "-"),
                f"[{item.get('authority_tier', '-')}] {item.get('publisher') or item.get('title')}",
                "；".join(part for part in (item.get("url"), locator, item.get("source_sha256")) if part),
            ])
        story += [p("20. 来源索引", h2)]
        if len(evidence_rows) > 1:
            story += [table(
                evidence_rows,
                [18*mm, 32*mm, 40*mm, 74*mm],
            )]
        else:
            story.append(p("本版本未携带可展示的来源索引。", small))

        story += [p("21. 复现清单和模型版本", h2), table([
            ["项目", "记录值", "复现说明"],
            ["运行ID", record.run_id, "定位确定性计算运行"],
            ["工作区/版本", workspace_meta.get("workspace_id", "未绑定"), f"V{version_meta.get('number', record.revision)}"],
            ["ModelSpec", spec_meta.get("model_spec_id", "未绑定"), spec_meta.get("snapshot_hash", "")],
            ["金融模型版本", result.model_version, spec_meta.get("financial_model_version", "")],
            ["输入哈希", calculation_meta.get("input_hash") or result.effective_input_hash or record.input_hash, "校验冻结输入"],
            ["结果哈希", calculation_meta.get("result_hash") or "未绑定", "校验完整数值结果"],
            ["计算复现", (audit_context.get("reproducibility") or {}).get("calculation", "运行包可离线复算"), "无需重新搜索或调用LLM"],
            ["数据获取复现", (audit_context.get("reproducibility") or {}).get("data_acquisition", "依赖保存的来源引用"), "网页变化时以快照、引文和哈希为准"],
            ["舍入规则", spec_meta.get("rounding_policy", "Decimal精度；展示时按字段舍入"), "报告显示精度不改变底层计算"],
        ], [38*mm, 54*mm, 66*mm])]
        doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
        return output.getvalue()

    def export(self, record: RunRecord, format: str, *, store=None) -> tuple[bytes, str, str]:
        selected = format.lower().lstrip(".")
        if record.result is None and selected in {"pdf", "json"}:
            import json
            from valuationagent.application.result_document import build_run_diagnostic, render_run_diagnostic
            diagnostic = build_run_diagnostic(record, store)
            if selected == "pdf":
                return render_run_diagnostic(diagnostic), "application/pdf", "pdf"
            return json.dumps(diagnostic, ensure_ascii=False, indent=2).encode("utf-8"), "application/json", "json"
        from valuationagent.application.workspace_reporting import (
            build_workspace_report_context,
        )
        audit_context = build_workspace_report_context(store, record.run_id)
        if selected == "xlsx":
            return self.xlsx(record, audit_context), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
        if selected == "pdf":
            return self.pdf(record, audit_context), "application/pdf", "pdf"
        if selected == "json":
            import json
            from valuationagent.application.reproducibility import build_valuation_bundle
            return json.dumps(build_valuation_bundle(store, record), ensure_ascii=False, indent=2).encode("utf-8"), "application/json", "json"
        raise ValueError("导出格式仅支持 json、xlsx 或 pdf。")
