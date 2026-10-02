"""Independent challenge and decision layers for completed valuations.

These checks do not alter a financial result.  They turn model-risk signals
already produced by the deterministic engine into explicit, reviewable
findings.  A language model may later explain them, but cannot suppress them.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal

from valuationagent.schemas.workspace import (
    ChallengeFinding,
    DecisionRecord,
    FindingDisposition,
)


def _id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}{digest}"


class ValuationChallengeService:
    """Run reproducible red-team checks against one persisted run."""

    def review(self, workspace_id: str, record) -> list[ChallengeFinding]:
        result = record.result
        if result is None:
            return [
                self._finding(
                    workspace_id,
                    record.run_id,
                    "scope",
                    "blocking",
                    "尚无可挑战的数值结果",
                    "该运行没有完成的估值输出，不能将诊断或缺数说明当作价格结论。",
                    "先解决运行中的复核或错误，再生成新版本。",
                )
            ]

        findings: list[ChallengeFinding] = []
        quality = result.data_quality
        coverage = Decimal(str(quality.evidence_coverage))
        if coverage < Decimal("0.5"):
            findings.append(self._finding(
                workspace_id, record.run_id, "data_quality", "high",
                "核心字段证据覆盖偏低",
                f"确定性质量评估的证据覆盖率为 {coverage:.1%}；部分模型输入缺少逐项来源链。",
                "补充高权威来源并重跑，报告中维持低置信度标识。",
            ))
        elif coverage < Decimal("0.8"):
            findings.append(self._finding(
                workspace_id, record.run_id, "data_quality", "warning",
                "证据覆盖仍可提高",
                f"核心字段证据覆盖率为 {coverage:.1%}，足以运行但不足以视为全面核验。",
                "优先补证对估值最敏感的收入、利润率、净债务和股数。",
            ))

        if quality.confidence == "low" or quality.result_grade in {"C", "D"}:
            findings.append(self._finding(
                workspace_id, record.run_id, "data_quality", "high",
                "结果置信等级较低",
                f"系统质量等级为 {quality.result_grade}，置信度为 {quality.confidence}。",
                "把本次区间作为初步判断，并在投资决策前进行人工财务复核。",
            ))

        if result.dcf:
            spread = Decimal(str(result.assumptions.wacc)) - Decimal(str(result.assumptions.terminal_growth))
            if spread <= Decimal("0.02"):
                findings.append(self._finding(
                    workspace_id, record.run_id, "assumption", "high",
                    "WACC 与永续增长率间距过窄",
                    f"WACC 与永续增长率仅相差 {spread:.2%}，终值对微小参数变化高度敏感。",
                    "复核长期名义增长约束，并重点查看二维敏感性矩阵。",
                ))
            terminal_share = Decimal(str(result.dcf.terminal_value_share))
            if terminal_share >= Decimal("0.75"):
                findings.append(self._finding(
                    workspace_id, record.run_id, "model_risk", "high",
                    "DCF 主要由终值驱动",
                    f"终值现值占企业价值 {terminal_share:.1%}，显式预测期解释力有限。",
                    "延长可验证预测期或使用退出倍数及相对估值进行交叉核验。",
                ))
            elif terminal_share >= Decimal("0.60"):
                findings.append(self._finding(
                    workspace_id, record.run_id, "model_risk", "warning",
                    "终值占比较高",
                    f"终值现值占企业价值 {terminal_share:.1%}。",
                    "报告中突出终值假设，并保留 WACC/永续增长率敏感性结果。",
                ))
            if result.dcf.bridge_unmeasured_items:
                findings.append(self._finding(
                    workspace_id, record.run_id, "scope", "high",
                    "权益桥仍有未计量项目",
                    "企业价值到股权价值的桥接存在未量化事项："
                    + "、".join(result.dcf.bridge_unmeasured_items),
                    "取得相关项目金额后创建新版本，或明确说明区间不含该调整。",
                ))

        for relative in result.relative:
            if relative.status != "success":
                continue
            if relative.sample_quality != "adequate" or relative.sample_size < 5:
                findings.append(self._finding(
                    workspace_id, record.run_id, "method", "warning",
                    f"{relative.method.upper()} 可比样本有限",
                    f"有效样本 {relative.sample_size} 家，样本质量为 {relative.sample_quality}。",
                    "复核同业业务可比性、口径时点与异常值剔除，不机械依赖中位数。",
                ))

        high_sensitivity = [
            item.parameter for item in result.sensitivity_studies
            if item.status == "completed" and item.classification == "high"
        ]
        if high_sensitivity:
            findings.append(self._finding(
                workspace_id, record.run_id, "sensitivity", "warning",
                "关键参数敏感性较高",
                "高敏感参数包括：" + "、".join(high_sensitivity) + "。",
                "决策时同时使用上下行情景，不把单点估值当作精确价格。",
            ))

        if record.request.excluded_methods:
            findings.append(self._finding(
                workspace_id, record.run_id, "method", "info",
                "部分原选方法已降级排除",
                "；".join(
                    f"{method.upper()}：{reason}"
                    for method, reason in record.request.excluded_methods.items()
                ),
                "结论只适用于实际执行的方法，不能声称已完成全部方法交叉验证。",
            ))

        for index, warning in enumerate(result.warnings):
            findings.append(self._finding(
                workspace_id, record.run_id, "model_risk", "warning",
                f"模型警示 {index + 1}", str(warning), "在报告限制条件中保留该警示。",
            ))
        return findings

    @staticmethod
    def _finding(workspace_id, run_id, category, severity, title, analysis, recommendation):
        return ChallengeFinding(
            finding_id=_id("finding_", run_id, category, title, analysis),
            workspace_id=workspace_id,
            run_id=run_id,
            category=category,
            severity=severity,
            title=title,
            analysis=analysis,
            recommendation=recommendation,
        )


class ValuationDecisionService:
    """Integrate engine output and challenge findings without averaging opinions."""

    def dispositions(
        self,
        workspace_id: str,
        record,
        findings: list[ChallengeFinding],
    ) -> list[FindingDisposition]:
        """Give every challenge an explicit, reportable resolution."""

        rows = []
        for finding in findings:
            if finding.severity == "blocking":
                decision = "blocking_recalculation"
                action = "阻断当前版本作为结论；修复后以新 ModelSpec 重算。"
            elif finding.severity == "high":
                decision = "blocking_recalculation"
                action = "高风险尚未解决，保留草案并要求人工复核或补证重算。"
            elif finding.severity == "warning":
                decision = "accepted"
                action = "保留为未解决警示；披露和敏感性分析不代表风险已经缓释。"
            else:
                decision = "accepted"
                action = "作为适用范围说明写入报告。"
            rows.append(FindingDisposition(
                disposition_id=_id(
                    "disposition_", record.run_id, finding.finding_id, decision
                ),
                workspace_id=workspace_id,
                run_id=record.run_id,
                finding_id=finding.finding_id,
                decision=decision,
                rationale=(
                    f"挑战严重性为 {finding.severity}。决策层依据确定性结果、"
                    "证据覆盖与该异议的建议动作进行处置；未重新心算估值。"
                ),
                resulting_action=action,
            ))
        return rows

    def decide(
        self,
        workspace_id: str,
        record,
        findings: list[ChallengeFinding],
        dispositions: list[FindingDisposition] | None = None,
    ) -> DecisionRecord:
        dispositions = dispositions or self.dispositions(
            workspace_id, record, findings
        )
        severities = {item.severity for item in findings if item.status == "open"}
        if record.result is None or "blocking" in severities:
            outcome = "review_required"
            selected = "暂停将该运行作为数值结论，解决阻塞事项后创建新版本。"
        elif "high" in severities:
            outcome = "review_required"
            selected = "保留估值区间，但显著披露高风险发现并要求人工复核。"
        elif "warning" in severities or str(record.status) == "completed_with_warnings":
            outcome = "accepted_with_warnings"
            selected = "接受确定性计算结果，同时保留全部限制和敏感性提示。"
        else:
            outcome = "accepted"
            selected = "接受本版本结果，并按报告所列适用范围使用。"
        rationale = (
            f"确定性估值运行状态为 {record.status}；挑战层形成 {len(findings)} 项发现，"
            f"其中高风险 {sum(f.severity == 'high' for f in findings)} 项、"
            f"阻塞 {sum(f.severity == 'blocking' for f in findings)} 项。"
        )
        return DecisionRecord(
            decision_id=_id("decision_", record.run_id, outcome, *[f.finding_id for f in findings]),
            workspace_id=workspace_id,
            run_id=record.run_id,
            outcome=outcome,
            rationale=rationale,
            finding_ids=[item.finding_id for item in findings],
            disposition_ids=[item.disposition_id for item in dispositions],
            alternatives=["补充证据后重算", "调整假设并创建版本", "更换或排除估值方法"],
            selected_action=selected,
        )
