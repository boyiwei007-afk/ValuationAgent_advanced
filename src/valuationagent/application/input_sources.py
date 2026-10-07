from datetime import date
from decimal import Decimal

from valuationagent.core.tools import canonical
from valuationagent.market.tushare import normalize_a_share_ticker
from valuationagent.schemas.inputs import InputDataset, InputRecord, InputSource


def dataset_basis(records):
    return "user_scenario" if any(row.assertion == "user_input" for row in records) or all(row.source.kind == "user" for row in records) else "research"


def interpretation_digest(fact):
    from valuationagent.application.input_workspace import digest

    return digest(canonical(fact.model_dump(mode="json", exclude={"status"})))


def source_fact(session, fact_id, *, as_raw=False):
    from valuationagent.application.input_workspace import FINANCIAL_FIELDS
    from valuationagent.application.input_derivations import RAW_INPUT_FIELDS
    from valuationagent.application.research_valuation import financial_mapping_issue, mapped_financial_metric

    fact = next((entry for entry in session.facts if entry.fact_id == fact_id), None)
    if fact is None or fact.status == "rejected" or fact.role not in {"historical", "comparable"}:
        raise ValueError("INPUT_SOURCE: 只能选择本工作区未撤回的历史事实或可比数据。")
    if fact.warnings or fact.verification.get("observation", {}).get("status") != "verified" or fact.verification.get("semantic_review", {}).get("status") != "supported":
        raise ValueError("INPUT_ADMISSION: 来源数据尚未完成原文绑定与语义复核；先处理该fact_id的警告，不重抄数值。")
    comparable = fact.role == "comparable"
    metric = (fact.standard_metric or fact.metric) if comparable else mapped_financial_metric(fact)
    if not as_raw and metric not in FINANCIAL_FIELDS | RAW_INPUT_FIELDS:
        raise ValueError("INPUT_METRIC: 该原始科目尚不能直接作为模型字段；保留原始口径，不把利润科目重命名为EBIT。")
    shares = metric in {"common_shares", "diluted_shares"}
    instant = shares or metric == "market_cap"
    if fact.scope != ("issuer" if instant else "consolidated"):
        raise ValueError("INPUT_SCOPE: 模型使用合并经营数据和发行人股数，不能混入母公司或股份类别小计。")
    proof = fact.verification.get("reading_proof", {})
    row = proof.get("row", {})
    try:
        period = date.fromisoformat(row.get("period_end", ""))
    except ValueError:
        raise ValueError("INPUT_PERIOD: 原始期间不是明确的年度或存量时点，不能猜测或截断季度期间。") from None
    if row.get("period_kind") not in {"annual", "instant"} or fact.period != (str(period.year) if row["period_kind"] == "annual" else period.isoformat()):
        raise ValueError("INPUT_PERIOD: 原始期间与已复核解释不一致；中期流量不替代完整年度。")
    from valuationagent.application.input_calculations import INSTANT_FIELDS

    if comparable and not as_raw and row["period_kind"] != ("instant" if metric in INSTANT_FIELDS else "annual"):
        raise ValueError("INPUT_PERIOD: 可比分母必须为完整年度，桥接余额及市值必须是独立时点。")
    if metric in RAW_INPUT_FIELDS - INSTANT_FIELDS and row["period_kind"] != "annual":
        raise ValueError("INPUT_PERIOD: 原始流量科目须有完整年度依据；不把季度或时点金额拼入年度推导。")
    if mapping_issue := financial_mapping_issue(fact.model_copy(update={"status": "confirmed", "role": "historical"})):
        raise ValueError("INPUT_METRIC: " + mapping_issue)
    document = next((entry for entry in session.documents if fact.block_id.startswith(entry.file_id + ":")), None)
    if document is None or not document.sha256 or fact.source_sha256 != document.sha256:
        raise ValueError("INPUT_SOURCE_HASH: 原始文件归属或哈希不匹配。")
    basis = proof.get("basis", {})
    ticker = fact.peer_ticker if comparable else session.draft.ticker
    if ticker and normalize_a_share_ticker(basis.get("entity_ticker", "")) != normalize_a_share_ticker(ticker):
        raise ValueError("INPUT_ENTITY: 原文复核主体与当前任务代码不一致。")
    if comparable and (not ticker or normalize_a_share_ticker(ticker) == normalize_a_share_ticker(session.draft.ticker)):
        raise ValueError("INPUT_ENTITY: 可比必须有明确的其他公司代码，目标公司不能作自身可比。")
    if not ticker and basis.get("entity_name") != session.draft.company:
        raise ValueError("INPUT_ENTITY: 原文复核主体与当前任务名称不一致。")
    currency = proof.get("basis", {}).get("currency")
    neutral = fact.unit in {"股", "千股", "万股", "百万股", "亿股", "%", "ratio"}
    if not neutral and (not currency or currency == "unknown"):
        raise ValueError("INPUT_CURRENCY: 复核后的来源仍未明确币种，不因A股或接口默认值补造币种。")
    if neutral:
        currency = None
    cutoff = session.information_cutoff_date or session.draft.valuation_date
    if cutoff and (period > cutoff or fact.published_at and fact.published_at > cutoff):
        raise ValueError("INPUT_DATE: 数据时点或披露日晚于信息截止日。")
    if not fact.normalized_value:
        raise ValueError("INPUT_AMOUNT: 来源解释没有已核验数值。")
    if as_raw:
        if fact.semantic_role == "financial_subsidiary":
            raise ValueError("INPUT_RAW_SCOPE: 金融子公司科目不能通过通用加减公式绕过专项模型范围；保留原始事实，使用经验证的分部模型或另行评估可用方法。")
        if fact.unit not in {"元", "千元", "万元", "百万元", "亿元"} or fact.scope != "consolidated":
            raise ValueError("INPUT_RAW_DIMENSION: 原始计算科目仅接收合并口径金额，不把股数或比率伪装成金额。")
        metric = "raw." + fact.fact_id
    return fact, document, metric, period, currency


def build_source_inputs(runtime, values, dataset=None):
    from valuationagent.application.file_workspace import source_bytes
    from valuationagent.application.input_workspace import digest

    session = runtime.session
    entity = session.draft.company or session.draft.ticker
    if not entity:
        raise ValueError("INPUT_ENTITY: 先明确当前研究公司，不从文件名推定主体。")
    existing = {row.input_id: row for row in dataset.active_records()} if dataset else {}
    saved, checked_files = [], set()
    selected_ids = [value.fact_id for value in values]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("INPUT_DUPLICATE: 同批不能重复选择同一个来源事实。")
    for selection in values:
        fact, document, metric, period, currency = source_fact(session, selection.fact_id, as_raw=selection.as_raw)
        if document.file_id not in checked_files:
            source_bytes(runtime.service.store, session, document.file_id)
            checked_files.add(document.file_id)
        if dataset is None:
            dataset = InputDataset(entity=entity, currency=currency or "CNY", analysis_basis="research")
        if dataset.entity not in {entity, session.draft.ticker} or currency and dataset.currency != currency:
            raise ValueError("INPUT_SCOPE: 来源主体或币种与当前输入工作区不一致。")
        ticker = fact.peer_ticker if fact.role == "comparable" else session.draft.ticker
        ticker = normalize_a_share_ticker(ticker) if ticker else ""
        if any(key not in existing or existing[key].metric != metric or existing[key].role != fact.role
               or fact.role == "comparable" and existing[key].entity_ticker != ticker for key in selection.replaces):
            raise ValueError("INPUT_REPLACEMENT: 只能替换当前有效的同指标输入，不撤回原始披露。")
        source = InputSource(kind="provider" if document.provenance_type == "structured_provider" else "document",
            source_id=fact.fact_id, sha256=document.sha256, quote=fact.quote, file_id=document.file_id,
            locator=canonical({"block_id": fact.block_id, **fact.source_location}), source_url=fact.source_url,
            published_at=fact.published_at, authority_tier=document.authority_tier,
            limitations=fact.verification.get("source_assessment", {}).get("limitations", []),
            interpretation_sha256=interpretation_digest(fact))
        instant = metric in {"common_shares", "diluted_shares", "market_cap"}
        payload = dict(entity=dataset.entity if fact.role == "historical" else ticker, entity_ticker=ticker, role=fact.role,
            metric=metric, label=fact.metric, period_kind=fact.verification["reading_proof"]["row"]["period_kind"],
            value=fact.normalized_value, original_amount=fact.raw_value,
            unit=fact.unit, currency=currency, scope=fact.scope, period_end=None if instant else period,
            as_of=period if instant else None, assertion="reported", source=source, supersedes=selection.replaces)
        row = InputRecord(input_id="input_" + digest(canonical(payload))[:24], **payload)
        if row.input_id not in {entry.input_id for entry in dataset.records}:
            dataset.records.append(row)
            saved.append(row.input_id)
    dataset.analysis_basis = dataset_basis(dataset.active_records())
    return dataset, saved


def validate_source_inputs(session, records):
    from valuationagent.application.observation_consistency import period_cell_conflicts

    facts = []
    for row in records:
        if row.source.kind == "user":
            continue
        if row.source.provider_binding:
            from valuationagent.application.provider_inputs import validate_provider_input

            validate_provider_input(session, row)
            continue
        if row.source.kind not in {"document", "provider"}:
            raise ValueError("INPUT_DERIVATION: 推导输入需要已注册公式及依赖，不接收无依据的计算结果。")
        fact, document, metric, period, currency = source_fact(session, row.source.source_id, as_raw=row.metric.startswith("raw."))
        ticker = fact.peer_ticker if fact.role == "comparable" else session.draft.ticker
        ticker = normalize_a_share_ticker(ticker) if ticker else ""
        if (interpretation_digest(fact) != row.source.interpretation_sha256 or document.sha256 != row.source.sha256
                or fact.role != row.role or ticker != row.entity_ticker
                or metric != row.metric or currency != row.currency or Decimal(fact.normalized_value) != row.value
                or fact.metric != row.label or fact.verification["reading_proof"]["row"]["period_kind"] != row.period_kind
                or fact.scope != row.scope or fact.unit != row.unit or fact.raw_value != row.original_amount
                or fact.quote != row.source.quote or row.assertion != "reported"
                or fact.published_at != row.source.published_at
                or period != (row.as_of if metric in {"common_shares", "diluted_shares", "market_cap"} else row.period_end)):
            raise ValueError("INPUT_SOURCE_CHANGED: 来源解释已更正或输入与原始选择不一致；重新选择当前事实，不覆盖已冻结估值。")
        facts.append(fact)
    if period_cell_conflicts(facts):
        raise ValueError("INPUT_PERIOD_COLLISION: 同一原文数值位置被赋给多个年度，须核对后再计算。")
