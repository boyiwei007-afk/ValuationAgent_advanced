from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import TypeAdapter

from valuationagent.core.data import DataBundle, DataProvider, LocalDataProvider
from valuationagent.core.i18n import translator
from valuationagent.core.plugins import FinancialModelPlugin
from valuationagent.core.tools import canonical
from valuationagent.finance.integrity import validate_peer_inputs, verify_calculations
from valuationagent.schemas.models import AssumptionSet, DataQualityAssessment, DcfResult, ForecastYear, MultipleResult, ReconciliationResult, SensitivityCell, SensitivityStudy, ValidationFinding, ValuationOutput, ValuationRequest
from valuationagent.storage.sqlite import SQLiteRunStore

STAGES = [
    ("data_intake", "资料与来源"),
    ("financial_validation", "财务审核"),
    ("industry_parameters", "行业识别与参数"),
    ("assumption_resolution", "经营假设"),
    ("financial_forecast", "现金流预测"),
    ("dcf_valuation", "DCF 估值"),
    ("relative_valuation", "相对估值"),
    ("sensitivity", "敏感性分析"),
    ("reconciliation", "区间验证"),
    ("reporting", "结果与依据"),
]


class WorkflowState(TypedDict, total=False):
    run_id: str
    request: ValuationRequest
    bundle: DataBundle
    assumptions: AssumptionSet
    forecast: list[ForecastYear]
    dcf: DcfResult | None
    relative: list[MultipleResult]
    sensitivity: list[SensitivityCell]
    sensitivity_studies: list[SensitivityStudy]
    reconciliation: ReconciliationResult
    data_quality: DataQualityAssessment
    warnings: list[str]
    industry_parameters: dict[str, Any]
    blocked: bool
    review: dict
    result: ValuationOutput


@dataclass
class WorkflowServices:
    store: SQLiteRunStore
    finance: FinancialModelPlugin
    data: DataProvider | None = None
    should_pause: Any = None

    def event(self, run_id, **kwargs):
        return self.store.append_event(run_id, **kwargs)

    def say(self, run_id, content, stage=None):
        content = translator(self.store.get_run(run_id).request.language)(content)
        msg = self.store.add_message(run_id, "assistant", content, stage)
        self.event(
            run_id,
            type="conversation.message",
            stage=stage,
            status="completed",
            summary=content,
            payload={"message_id": msg.message_id, "role": "assistant"},
        )

    def tool(self, run_id, stage, name, inputs, fn, output_type=Any, *, cache=True):
        if not getattr(self.finance, "supports_incremental_inputs", False):
            inputs = {
                "tool_inputs": inputs,
                "full_request": self.store.get_run(run_id).request,
            }
        raw = canonical(inputs)
        key = hashlib.sha256(
            canonical(
                {
                    "workflow": "0.5.0",
                    "plugin": self.finance.plugin_id,
                    "version": self.finance.version,
                    "tool": name,
                    "inputs": inputs,
                }
            ).encode()
        ).hexdigest()
        adapter = TypeAdapter(output_type)
        previous = self.store.checkpoint(run_id, key) if cache else None
        if previous:
            result = adapter.validate_json(previous["output_json"])
            self.event(
                run_id,
                type="tool.cached",
                stage=stage,
                status="cached",
                tool=name,
                summary="复用相同输入的已完成结果",
                payload={"artifact_id": key, "source_run_id": previous["run_id"]},
            )
            return result
        call_id = "call_" + uuid.uuid4().hex
        self.event(
            run_id,
            type="tool.started",
            stage=stage,
            status="running",
            tool=name,
            tool_call_id=call_id,
            summary=name,
            payload={"input_hash": key},
        )
        started = time.perf_counter()
        try:
            result = adapter.validate_python(fn())
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, (ValueError, NotImplementedError))
                else "工具执行异常，请检查插件或恢复任务。"
            )
            self.event(
                run_id,
                type="tool.failed",
                stage=stage,
                status="failed",
                tool=name,
                tool_call_id=call_id,
                summary=message,
                duration_ms=int((time.perf_counter() - started) * 1000),
                payload={"error_type": type(exc).__name__},
            )
            raise
        encoded = adapter.dump_json(result).decode()
        # Noncached Agent tools still receive unique evidence artifacts.
        artifact = key if cache else key + "-" + call_id
        self.store.save_checkpoint(run_id, artifact, name, raw, encoded)
        self.event(
            run_id,
            type="tool.completed",
            stage=stage,
            status="completed",
            tool=name,
            tool_call_id=call_id,
            summary=f"{name} 完成",
            duration_ms=int((time.perf_counter() - started) * 1000),
            payload={"artifact_id": artifact},
        )
        return result

    def block(self, state, stage, code, message):
        review = {"code": code, "stage": stage, "message": message}
        self.event(
            state["run_id"],
            type="review.required",
            stage=stage,
            status="waiting_review",
            summary=message,
            payload=review,
        )
        self.say(state["run_id"], message, stage)
        return {"blocked": True, "review": review}


def build_workflow(services: WorkflowServices):
    finance = services.finance

    def intake(s):
        req = s["request"]
        provider = services.data or LocalDataProvider()
        refs = [
            services.store.get_file(fid)
            for fid in req.file_ids + req.assumption_file_ids
        ]
        bundle = services.tool(
            s["run_id"],
            "data_intake",
            "resolve_financial_input",
            {
                "request": req,
                "files": [
                    {k: m[k] for k in ("file_id", "sha256", "role")} for m in refs
                ],
                "provider": provider.version,
            },
            lambda: provider.resolve(req, services.store),
            DataBundle,
        )
        effective = req.model_copy(
            update={
                "company": bundle.company or req.company,
                "financials": bundle.financials,
                "historical_financials": bundle.historical_financials,
                "peers": bundle.peers,
                "assumptions": bundle.assumptions,
                "assumption_evidence": bundle.assumption_evidence,
            }
        )
        services.say(
            s["run_id"],
            "已载入合成演示数据。"
            if req.mode == "demo"
            else "资料快照已建立，开始核对来源与口径。",
            "data_intake",
        )
        method_warnings = [
            f"{method.upper()}因可靠数据不足未进入本次计算：{reason}"
            for method, reason in req.excluded_methods.items()
        ]
        return {
            "bundle": bundle,
            "request": effective,
            "warnings": [*bundle.warnings, *method_warnings],
        }


    def validate(s):
        req = s["request"]
        findings = services.tool(
            s["run_id"],
            "financial_validation",
            "validate_financials",
            {
                "financials": s["bundle"].financials,
                "historical_financials": req.historical_financials,
                "date": req.valuation_date,
                "company": req.company,
                "methods": req.methods,
                "forecast_years": req.forecast_years,
                "discount_policy": req.discount_policy,
            },
            lambda: finance.validate(req, s["bundle"].financials),
            list[ValidationFinding],
        )
        findings += validate_peer_inputs(req, s["bundle"].peers)
        services.event(
            s["run_id"],
            type="validation.summary",
            stage="financial_validation",
            status="completed",
            summary=f"发现 {len(findings)} 个提示或阻断项",
            payload={"findings": [f.model_dump(mode="json") for f in findings]},
        )
        blocking = [f for f in findings if f.severity == "blocking"]
        if blocking:
            return services.block(
                s,
                "financial_validation",
                "FINANCIAL_VALIDATION_FAILED",
                blocking[0].message,
            )
        return {
            "warnings": s["warnings"]
            + [f.message for f in findings if f.severity == "warning"]
        }

    def assumptions(s):
        req = s["request"]
        result = services.tool(
            s["run_id"],
            "assumption_resolution",
            "resolve_assumptions",
            {
                "financials": s["bundle"].financials,
                "historical_financials": req.historical_financials,
                "assumptions": req.assumptions,
                "years": req.forecast_years,
                "methods": req.methods,
                "evidence": req.assumption_evidence,
                "industry_parameters": s.get("industry_parameters", {}),
            },
            lambda: finance.resolve_assumptions(req, s["bundle"].financials),
            AssumptionSet,
        )
        services.say(s["run_id"],
            translator(req.language)("假设已固定：WACC {wacc:.2%}，永续增长率 {growth:.2%}。").format(wacc=result.wacc, growth=result.terminal_growth)
            if "dcf" in req.methods else "相对估值：使用已确认可比样本，不构建DCF预测或折现率假设。",
            "assumption_resolution")
        return {"assumptions": result}

    def industry_parameters(s):
        if "dcf" not in s["request"].methods:
            return {"industry_parameters": {"mode": "not_applicable_relative_only"}}
        resolver = getattr(finance, "resolve_industry_parameters", None)
        if resolver is None:
            return {"industry_parameters": {}}
        result = services.tool(
            s["run_id"],
            "industry_parameters",
            "lookup_industry_parameters",
            {
                "industry": s["request"].company.industry,
                "model_version": finance.version,
            },
            lambda: resolver(s["request"]),
            dict[str, Any],
        )
        services.event(
            s["run_id"],
            type="industry.parameters.resolved",
            stage="industry_parameters",
            status="completed",
            summary=result.get("name", result.get("mode", "行业参数已建立")),
            payload={"parameters": result},
        )
        return {"industry_parameters": result}

    def forecast(s):
        if "dcf" not in s["request"].methods:
            return {"forecast": []}
        req = s["request"]
        inputs = {
            "financials": s["bundle"].financials,
            "historical_financials": req.historical_financials,
            "growth": s["assumptions"].revenue_growth,
            "margin": s["assumptions"].ebit_margin,
            "years": req.forecast_years,
            "date": req.valuation_date,
            "policy": req.discount_policy,
        }
        return {
            "forecast": services.tool(
                s["run_id"],
                "financial_forecast",
                "forecast_financials",
                inputs,
                lambda: finance.forecast(req, s["bundle"].financials, s["assumptions"]),
                list[ForecastYear],
            )
        }

    def dcf(s):
        if "dcf" not in s["request"].methods:
            return {"dcf": None}
        inputs = {
            "financials": s["bundle"].financials,
            "historical_financials": s["request"].historical_financials,
            "assumptions": s["assumptions"],
            "forecast": s["forecast"],
            "years": s["request"].forecast_years,
            "date": s["request"].valuation_date,
            "policy": s["request"].discount_policy,
        }
        result = services.tool(
            s["run_id"],
            "dcf_valuation",
            "calculate_dcf",
            inputs,
            lambda: finance.dcf(
                s["request"], s["bundle"].financials, s["assumptions"], s["forecast"]
            ),
            DcfResult,
        )
        return {"dcf": result, "warnings": s["warnings"] + result.scenario_warnings}

    def relative(s):
        if not any(m != "dcf" for m in s["request"].methods):
            return {"relative": []}
        result = services.tool(
            s["run_id"],
            "relative_valuation",
            "calculate_relative_valuation",
            {
                "financials": s["bundle"].financials,
                "peers": s["bundle"].peers,
                "methods": s["request"].methods,
            },
            lambda: finance.relative(
                s["request"], s["bundle"].financials, s["bundle"].peers
            ),
            list[MultipleResult],
        )
        return {
            "relative": result,
            "warnings": s["warnings"]
            + [f"{r.method}: {r.reason}" for r in result if r.reason],
        }

    def sensitivity(s):
        from valuationagent.finance.relative_sensitivity import relative_sensitivity
        relative_studies = []
        if any(m != "dcf" for m in s["request"].methods):
            relative_studies = services.tool(s["run_id"], "sensitivity", "run_relative_sensitivity",
                {"financials": s["bundle"].financials, "peers": s["bundle"].peers, "methods": s["request"].methods},
                lambda: relative_sensitivity(finance, s["request"], s["bundle"].financials, s["bundle"].peers), list[SensitivityStudy])
        if s.get("dcf") is None:
            return {"sensitivity": [], "sensitivity_studies": relative_studies}
        grid = services.tool(
                s["run_id"],
                "sensitivity",
                "run_sensitivity",
                {
                    "financials": s["bundle"].financials,
                    "historical_financials": s["request"].historical_financials,
                    "assumptions": s["assumptions"],
                    "years": s["request"].forecast_years,
                    "date": s["request"].valuation_date,
                    "policy": s["request"].discount_policy,
                },
                lambda: finance.sensitivity(
                    s["request"], s["bundle"].financials, s["assumptions"]
                ),
                list[SensitivityCell],
            )
        studies = []
        if hasattr(finance, "sensitivity_studies"):
            studies = services.tool(
                s["run_id"],
                "sensitivity",
                "run_sensitivity_catalogue",
                {
                    "financials": s["bundle"].financials,
                    "historical_financials": s["request"].historical_financials,
                    "assumptions": s["assumptions"],
                    "catalogue": "S1-S20",
                },
                lambda: finance.sensitivity_studies(
                    s["request"], s["bundle"].financials, s["assumptions"]
                ),
                list[SensitivityStudy],
            )
        return {"sensitivity": grid, "sensitivity_studies": studies + relative_studies}

    def reconcile(s):
        if s.get("dcf") is None and not any(
            r.status == "success" for r in s.get("relative", [])
        ):
            return services.block(
                s,
                "reconciliation",
                "NO_VALID_VALUATION",
                "所选方法均未形成有效估值，请补充同业或复核模型适用性。",
            )
        result = services.tool(
            s["run_id"],
            "reconciliation",
            "reconcile_valuations",
            {"dcf": s.get("dcf"), "relative": s.get("relative", [])},
            lambda: finance.reconcile(s.get("dcf"), s.get("relative", [])),
            ReconciliationResult,
        )
        warnings = list(s["warnings"])
        if result.method_comparison.get("status") == "conflict_review_required":
            warnings.append("DCF与相对估值差异超过复核阈值，报告不合并区间并要求复核关键假设与同业口径。")
        return {"reconciliation": result, "warnings": warnings}

    def report(s):
        record = services.store.get_run(s["run_id"])
        req = s["request"]
        dcf = s.get("dcf")
        checks = verify_calculations(req, s["bundle"].financials, s["assumptions"], s["forecast"], dcf, s["relative"], s["bundle"].peers)
        services.event(s["run_id"], type="calculation.verified", stage="reporting", status="completed",
                       summary="报告前独立算术对账通过", payload={"checks": checks})
        _ = translator(req.language)
        reconciliation = s["reconciliation"].model_copy(
            update={"conclusion": _(s["reconciliation"].conclusion)}
        )
        model_version = (
            finance.model_version_for(req)
            if hasattr(finance, "model_version_for")
            else finance.version
        )
        summary = f"{req.company.name or req.company.ticker or _('目标企业')} · {req.valuation_date} · {req.company.currency}. "
        summary += (
            _("DCF 基准 {base:.2f}/股，区间 {low:.2f}—{high:.2f}/股。").format(
                base=dcf.per_share_value, low=dcf.range_low, high=dcf.range_high
            )
            if dcf
            else ""
        )
        model_note = (
            _("数据模式 {mode}；模型 {version}，尚待金融团队核准。")
            if model_version.endswith("-reference")
            else _(
                "数据模式 {mode}；模型 {version}。正式模型按金融小组规则执行，降级和待复核项见警告。"
            )
        )
        summary += reconciliation.conclusion + " " + model_note.format(
            mode=req.mode, version=model_version
        )
        quality = (
            finance.assess_quality(
                req,
                s["bundle"].financials,
                s["bundle"].peers,
                s["relative"],
            )
            if hasattr(finance, "assess_quality")
            else DataQualityAssessment(
                historical_years=len(req.historical_financials) + 1,
                comparable_years=len(req.historical_financials) + 1,
                confidence="medium",
                notes=["当前金融插件未提供分项数据质量评分。"],
            )
        )
        summary += _(" 数据质量置信度：{confidence}；结果等级：{grade}。").format(
            confidence=quality.confidence,
            grade=quality.result_grade,
        )
        effective = {
            "request": req,
            "financials": s["bundle"].financials,
            "peers": s["bundle"].peers,
            "assumptions": s["assumptions"],
        }
        result = ValuationOutput(
            run_id=s["run_id"],
            revision=record.revision,
            company=req.company,
            valuation_date=req.valuation_date,
            currency=req.company.currency,
            mode=req.mode,
            language=req.language,
            input_hash=record.input_hash,
            effective_input_hash=hashlib.sha256(
                canonical(effective).encode()
            ).hexdigest(),
            effective_financials=s["bundle"].financials,
            effective_peers=s["bundle"].peers,
            assumption_evidence=req.assumption_evidence,
            discount_policy=req.discount_policy,
            model_version=model_version,
            assumptions=s["assumptions"],
            forecast=s["forecast"],
            dcf=dcf,
            relative=s["relative"],
            sensitivity=s["sensitivity"],
            sensitivity_studies=s.get("sensitivity_studies", []),
            reconciliation=reconciliation,
            data_quality=quality,
            executive_summary=summary,
            warnings=s["warnings"],
            calculation_checks=checks,
        )
        services.say(s["run_id"], summary, "reporting")
        services.event(
            s["run_id"],
            type="artifact.created",
            stage="reporting",
            status="completed",
            summary="结构化估值结果已生成",
            payload={"artifact_type": "valuation_result_json", "run_id": s["run_id"]},
        )
        return {"result": result}

    graph = StateGraph(WorkflowState)
    for (stage, label), fn in zip(
        STAGES,
        [
            intake,
            validate,
            industry_parameters,
            assumptions,
            forecast,
            dcf,
            relative,
            sensitivity,
            reconcile,
            report,
        ],
    ):

        def wrapped(s, fn=fn, stage=stage, label=label):
            if services.should_pause and services.should_pause():
                return services.block(
                    s,
                    stage,
                    "USER_PAUSED",
                    "已按你的请求暂停。当前工具已结束，可以恢复任务。",
                )
            services.event(
                s["run_id"],
                type="stage.started",
                stage=stage,
                status="running",
                summary=label,
            )
            try:
                update = fn(s)
            except (ValueError, NotImplementedError) as exc:
                code = (
                    "INVALID_ASSUMPTION"
                    if stage == "assumption_resolution"
                    else "DATA_INPUT_UNAVAILABLE"
                    if stage == "data_intake"
                    else "TOOL_REVIEW_REQUIRED"
                )
                update = services.block(s, stage, code, str(exc))
            status = "waiting_review" if update.get("blocked") else "completed"
            services.event(
                s["run_id"],
                type="stage.completed",
                stage=stage,
                status=status,
                summary=label,
            )
            return update

        graph.add_node(stage, wrapped)
    graph.add_edge(START, STAGES[0][0])
    for index, (stage, _) in enumerate(STAGES):
        following = STAGES[index + 1][0] if index + 1 < len(STAGES) else END
        graph.add_conditional_edges(
            stage,
            lambda s: "stop" if s.get("blocked") else "continue",
            {"stop": END, "continue": following},
        )
    return graph.compile()
