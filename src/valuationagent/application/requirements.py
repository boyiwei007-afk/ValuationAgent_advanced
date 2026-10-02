"""Build the valuation requirement graph from methods, not from search results."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from valuationagent.schemas.models import required_financial_metrics
from valuationagent.application.research_valuation import ResearchValuationAssembler, mapped_financial_metric, _period
from valuationagent.schemas.workspace import DataRequirement


METRIC_LABELS = {
    "revenue": "营业收入",
    "ebit_margin": "EBIT 利润率",
    "tax_rate": "有效税率",
    "depreciation_amortization": "折旧与摊销",
    "capital_expenditure": "资本开支",
    "change_operating_nwc": "经营性营运资本变动",
    "cash_and_non_operating_assets": "现金及非经营资产",
    "interest_bearing_debt": "有息债务",
    "common_shares": "估值日可用普通股股数",
    "net_income_parent": "归母净利润",
    "ebitda": "EBITDA",
}

METHOD_REQUIREMENTS = {
    method: set(required_financial_metrics([method]))
    for method in ("dcf", "pe", "ps", "ev_ebitda")
}

FORMAL_SOURCES = [
    "交易所或监管机构正式披露",
    "公司公告或投资者关系网站",
    "用户上传的原始文件",
    "有许可的结构化金融数据",
]

FALLBACKS = [
    "正式来源直接披露",
    "其他正式来源交叉验证",
    "由已核验事实确定性推导",
    "公司历史比例（仅适用时）",
    "可比公司或行业参数",
    "用户明确假设",
    "情景区间并降低结果等级",
]


def _requirement_id(workspace_id: str, key: str) -> str:
    digest = hashlib.sha256(f"{workspace_id}|{key}".encode()).hexdigest()[:24]
    return "req_" + digest


def _fact_metric(fact) -> str:
    return mapped_financial_metric(fact) or fact.metric


def build_requirement_graph(workspace, session, progress=None) -> list[DataRequirement]:
    """Return a deterministic dependency graph for the current scope.

    Search is an implementation detail of unresolved nodes.  A method change
    therefore adds/removes only its own nodes and does not invalidate unrelated
    evidence or force the annual report to be crawled again.
    """

    methods = list(session.draft.methods or [])
    by_metric: dict[str, set[str]] = defaultdict(set)
    for method in methods:
        for metric in METHOD_REQUIREMENTS.get(str(method), set()):
            by_metric[metric].add(str(method))

    clean_facts = [
        fact for fact in session.facts
        if fact.status in {"confirmed", "proposed"} and not fact.warnings
    ]
    cutoff = session.information_cutoff_date or session.draft.valuation_date
    annual_dates = [_period(fact.period) for fact in clean_facts if fact.role == "historical" and fact.scope == "consolidated"
                    and _period(fact.period) is not None and (_period(fact.period).month, _period(fact.period).day) == (12, 31)
                    and (not cutoff or _period(fact.period) <= cutoff)]
    latest_annual = max(annual_dates, default=None)
    facts_by_metric: dict[str, list] = defaultdict(list)
    for fact in clean_facts:
        if fact.role != "historical" or fact.normalized_value is None:
            continue
        if cutoff and (fact.published_at and fact.published_at > cutoff or _period(fact.period) and _period(fact.period) > cutoff):
            continue
        if _fact_metric(fact) == "common_shares":
            if fact.scope != "issuer":
                continue
        elif fact.scope != "consolidated" or _period(fact.period) != latest_annual:
            continue
        facts_by_metric[_fact_metric(fact)].append(fact)

    searches = list(session.search_history)
    financial_attempts = 0
    comparable_attempts = sum(
        1 for item in searches if item.get("purpose") == "comparables"
    )

    rows: list[DataRequirement] = []

    def add(
        key,
        category,
        label,
        done,
        *,
        metric="",
        linked_methods=(),
        priority="normal",
        weight=0.5,
        attempts=0,
        depends_on=(),
        acceptable_sources=(),
        fallbacks=(),
        fact_ids=(),
        evidence_ids=(),
        resolution="",
        resolution_type="",
    ):
        status = "satisfied" if done else "pending"
        rows.append(DataRequirement(
            requirement_id=_requirement_id(workspace.workspace_id, key),
            workspace_id=workspace.workspace_id,
            category=category,
            label=label,
            metric=metric,
            methods=list(linked_methods),
            priority=priority,
            importance_weight=weight,
            status=status,
            attempt_count=attempts,
            acceptable_sources=list(acceptable_sources),
            fallback_chain=list(fallbacks),
            depends_on=[_requirement_id(workspace.workspace_id, item) for item in depends_on],
            fact_ids=list(fact_ids),
            evidence_ids=list(evidence_ids),
            resolution=("已由权威状态或有效证据满足" if done and not resolution else resolution),
            resolution_type=("direct" if done and not resolution_type else resolution_type),
        ))

    add("scope.company", "scope", "确认研究对象与证券代码（明确时自动完成）",
        bool(session.draft.company or session.draft.ticker), priority="critical", weight=1)
    add("scope.date", "scope", "确定估值日、信息截止日与实际执行日",
        bool(session.draft.valuation_date), priority="critical", weight=1)
    add("scope.methods", "scope", "完成公司类型/行业路由并选择估值方法",
        bool(methods), priority="critical", weight=1)
    add("scope.sources", "scope", "确定用户资料与外部来源的组合策略",
        bool(session.data_source_preference), priority="high", weight=.8)

    metric_ids = []
    for metric, linked_methods in sorted(by_metric.items()):
        key = f"fact.{metric}"
        metric_ids.append(key)
        matches = facts_by_metric.get(metric, [])
        evidence_ids = [fact.block_id for fact in matches]
        timing_issue = (progress or {}).get("capital_structure") if metric == "common_shares" else None
        add(
            key,
            "capital_structure" if metric in {
                "common_shares", "cash_and_non_operating_assets", "interest_bearing_debt"
            } else "historical_financial",
            f"取得并核验{METRIC_LABELS.get(metric, metric)}",
            bool(matches) and not timing_issue,
            metric=metric,
            linked_methods=sorted(linked_methods),
            priority="critical" if metric in {"common_shares", "revenue"} else "high",
            weight=1 if metric in {"common_shares", "revenue"} else .8,
            attempts=financial_attempts,
            depends_on=("scope.company", "scope.date", "scope.methods"),
            acceptable_sources=FORMAL_SOURCES,
            fallbacks=FALLBACKS,
            fact_ids=[fact.fact_id for fact in matches],
            evidence_ids=evidence_ids,
            resolution=timing_issue["message"] if timing_issue else "",
        )

    if "dcf" in methods:
        proposal = session.forecast_proposal
        # ``ready_for_review`` means the deterministic finance adapter has all
        # required facts and can resolve the exact assumptions in the single
        # pre-valuation checkpoint.  Do not make the graph look stalled merely
        # because those assumptions have not yet been frozen by that checkpoint.
        deterministic_assumptions_ready = bool(
            progress and progress.get("ready_for_review")
        )
        add(
            "assumption.dcf",
            "forecast_assumption",
            "形成三情景经营预测、WACC与永续增长假设",
            bool(proposal) or deterministic_assumptions_ready,
            metric="dcf_assumptions",
            linked_methods=("dcf",),
            priority="critical",
            weight=1,
            depends_on=tuple(
                f"fact.{metric}" for metric in sorted(METHOD_REQUIREMENTS["dcf"])
            ),
            acceptable_sources=("历史财务推导", "市场参数", "管理层指引", "用户明确假设"),
            fallbacks=("历史驱动预测", "显式三情景假设", "用户锁定假设"),
            resolution=(
                "已形成显式三情景预测方案"
                if proposal
                else "已由历史财务和行业/市场参数形成确定性假设"
                if deterministic_assumptions_ready
                else ""
            ),
            resolution_type="assumption" if proposal or deterministic_assumptions_ready else "",
        )

    relative_methods = [method for method in methods if method != "dcf"]
    if relative_methods:
        peers = [fact for fact in clean_facts if fact.role == "comparable"]
        try:
            samples = ResearchValuationAssembler()._peers(session, latest_annual)
        except ValueError:
            samples = []
        add(
            "comparable.sample",
            "comparable",
            "取得同日、同口径且业务可比的样本与倍数",
            all(sum(getattr(peer, method) is not None for peer in samples) >= 3 for method in relative_methods),
            metric="comparable_multiples",
            linked_methods=relative_methods,
            priority="high",
            weight=.8,
            attempts=comparable_attempts,
            depends_on=("scope.company", "scope.date", "scope.methods"),
            acceptable_sources=("交易所行情", "有许可的结构化金融数据", "公司正式财务披露"),
            fallbacks=("扩大核心同业", "使用宽口径同业并扩大区间", "排除该相对估值方法"),
            fact_ids=[fact.fact_id for fact in peers],
            evidence_ids=[fact.block_id for fact in peers],
        )

    model_ready = bool(progress and progress.get("ready_for_review"))
    add(
        "model.spec",
        "deliverable",
        "冻结可复算的 ModelSpec",
        model_ready,
        priority="critical",
        weight=1,
        depends_on=tuple(["scope.company", "scope.date", "scope.methods", *metric_ids]),
        resolution_type="derived" if model_ready else "",
    )
    add(
        "review.prevaluation",
        "deliverable",
        "完成唯一一次估值前集中审阅",
        bool(workspace.active_checkpoint_id),
        priority="critical",
        weight=1,
        depends_on=("model.spec",),
    )
    return rows


def requirement_edges(requirements: list[DataRequirement]) -> list[dict[str, str]]:
    return [
        {"from": dependency, "to": item.requirement_id}
        for item in requirements
        for dependency in item.depends_on
    ]
