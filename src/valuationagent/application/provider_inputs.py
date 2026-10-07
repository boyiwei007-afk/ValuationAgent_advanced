from datetime import date
from decimal import Decimal, InvalidOperation
import re

from pydantic import Field

from valuationagent.application.file_workspace import source_bytes, source_document
from valuationagent.application.input_workspace import digest
from valuationagent.application.json_records import decode_json, pointer_value
from valuationagent.core.tools import canonical
from valuationagent.market.tushare import normalize_a_share_ticker
from valuationagent.market.tushare_contracts import CONTRACT_VERSION, CONTRACTS, RAW_FIELDS
from valuationagent.schemas.inputs import InputDataset, InputRecord, InputSource, ProviderInputValue
from valuationagent.schemas.models import ApiModel


class InputCandidateQuery(ApiModel):
    file_id: str = Field(min_length=1)
    metrics: list[str] = Field(default_factory=list, max_length=12, description="可选标准字段过滤；不修改字段含义。")
    period_end: date | None = Field(default=None, description="只查看该财报截止日；期间类型由契约保留，流量为annual、资产负债表为instant，不是行情日。")
    as_of: date | None = Field(default=None, description="只查看这个真实时点的股数或市值；不是财报年度。")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=6, ge=1, le=12)


def provider_record(raw, pointer):
    payload = decode_json(raw)
    fields = pointer_value(payload, "/data/fields")
    values = pointer_value(payload, pointer)
    if (not isinstance(fields, list) or not all(isinstance(field, str) and field for field in fields)
            or len(fields) != len(set(fields)) or not isinstance(values, list) or len(values) != len(fields)):
        raise ValueError("INPUT_PROVIDER_SHAPE: 原始记录列名或行宽无效，不猜测字段位置。")
    return dict(zip(fields, values))


def target_ticker(session):
    try:
        return normalize_a_share_ticker(session.draft.ticker)
    except ValueError:
        raise ValueError("INPUT_TARGET_REQUIRED: 当前任务尚未保存有效目标证券代码；这不是本次请求代码格式错误。原始来源可读，先用update_task(draft={ticker:已核对的目标代码})确认研究主体，再list_input_candidates选择现有来源，无需重新取数。") from None


def resolve_provider_candidate(session, candidate_id, replaces=None):
    match = re.fullmatch(r"(file_[a-f0-9]+)@(0|[1-9]\d*):([a-z][a-z0-9_]*)", candidate_id)
    if not match:
        raise ValueError("INPUT_PROVIDER_REFERENCE: 选择本工作区input_candidates的真实candidate_id，不自行编造字段映射。")
    file_id, index, field = match.groups()
    document = source_document(session, file_id)
    parts = document.provider.split(":")
    if document.provenance_type != "structured_provider" or len(parts) != 4 or parts[0] != "tushare":
        raise ValueError("INPUT_PROVIDER_CONTRACT: 此文件没有可直接选择的字段契约。")
    contract = CONTRACTS.get(parts[2], {}).get(field)
    if contract is None:
        raise ValueError("INPUT_PROVIDER_FIELD: 该字段不是契约允许的直接模型输入，须先解释与复核。")
    role = "historical" if parts[1] == target_ticker(session) else "comparable"
    return ProviderInputValue(file_id=file_id, record_pointer=f"/data/items/{index}", field=field,
        metric=contract[0], role=role, replaces=replaces or [])


def provider_inventory(session, document, raw):
    parts = document.provider.split(":")
    if document.provenance_type != "structured_provider" or len(parts) != 4 or parts[0] != "tushare":
        return [], {"INPUT_PROVIDER_CONTRACT": 1}
    try:
        target_ticker(session)
    except ValueError:
        return [], {"INPUT_TARGET_REQUIRED": 1}
    payload = decode_json(raw)
    records = pointer_value(payload, "/data/items")
    fields = pointer_value(payload, "/data/fields")
    if not isinstance(records, list) or not isinstance(fields, list):
        raise ValueError("INPUT_PROVIDER_SHAPE: 原始响应不是合法记录数组。")
    candidates, excluded = [], {}
    for index, values in enumerate(records):
        if (not all(isinstance(field, str) and field for field in fields) or len(set(fields)) != len(fields)
                or not isinstance(values, list) or len(values) != len(fields)):
            raise ValueError("INPUT_PROVIDER_SHAPE: 原始记录列名或行宽无效，不猜测字段位置。")
        record = dict(zip(fields, values))
        for field in CONTRACTS.get(parts[2], {}):
            if field not in record:
                continue
            candidate_id = f"{document.file_id}@{index}:{field}"
            selection = resolve_provider_candidate(session, candidate_id)
            try:
                row = build_provider_record(session, document, selection, record)
            except ValueError as exc:
                code = str(exc).split(":", 1)[0]
                excluded[code] = excluded.get(code, 0) + 1
                continue
            candidates.append((candidate_id, row))
    return candidates, excluded


def provider_candidates(session, document, raw, query=None):
    query = query or InputCandidateQuery(file_id=document.file_id)
    records, excluded = provider_inventory(session, document, raw)
    candidates = [{"candidate_id": candidate_id, "ticker": row.entity_ticker, "role": row.role,
                "metric": row.metric, "original_amount": row.original_amount, "unit": row.unit,
                "label": row.label, "period_kind": row.period_kind,
                "admission_kind": "raw_operand" if row.metric.startswith("raw.") else "standard_input",
                "normalized_value": str(row.value), "currency": row.currency, "scope": row.scope,
                "period_end": row.period_end.isoformat() if row.period_end else None,
                "as_of": row.as_of.isoformat() if row.as_of else None,
                "published_at": row.source.published_at.isoformat(),
                "pricing_basis": row.source.provider_binding.get("pricing_basis")} for candidate_id, row in records]
    available_periods = sorted({item["period_end"] or item["as_of"] for item in candidates}, reverse=True)
    candidates = [item for item in candidates if (not query.metrics or item["metric"] in query.metrics)
        and (query.period_end is None or item["period_end"] == query.period_end.isoformat())
        and (query.as_of is None or item["as_of"] == query.as_of.isoformat())]
    candidates.sort(key=lambda item: item["period_end"] or item["as_of"], reverse=True)
    items = candidates[query.offset:query.offset + query.limit]
    next_offset = query.offset + len(items)
    next_action = {"tool": "list_input_candidates", "arguments": query.model_copy(update={"offset": next_offset}).model_dump(mode="json", exclude_none=True)} if next_offset < len(candidates) else None
    instruction = "这是有界候选页，不是全部数据已入模。record_inputs(provider_values=[{candidate_id:实际ID}])选择所需期间和指标。其他日期或字段用list_input_candidates筛选/翻页，不重新下载。raw.*仅为可引用的原始科目，不是估值字段；由LLM用record_inputs(calculations=...)声明财务解释，程序计算并保留判断限制。原值重读与角色检查由程序负责；契约校验不是独立审计。"
    if "INPUT_TARGET_REQUIRED" in excluded:
        instruction = "INPUT_TARGET_REQUIRED: 原始响应已保存，但当前任务没有目标证券代码，暂不判断目标/可比角色。先update_task(draft={ticker:已核对的目标代码})，再list_input_candidates读取本文件；不要重新取数，不把此状态当作接口或文档解析失败。"
    return {"items": items, "total": len(candidates), "offset": query.offset, "next_action": next_action,
        "available_periods": available_periods, "excluded_by_reason": excluded, "contract_version": CONTRACT_VERSION,
        "instruction": instruction}


def list_input_candidates(runtime, args):
    document = source_document(runtime.session, args.file_id)
    _, raw = source_bytes(runtime.service.store, runtime.session, args.file_id)
    return {"file_id": args.file_id, **provider_candidates(runtime.session, document, raw, args)}


def build_provider_record(session, document, selection, record):
    parts = document.provider.split(":")
    if document.provenance_type != "structured_provider" or len(parts) != 4 or parts[0] != "tushare":
        raise ValueError("INPUT_PROVIDER_CONTRACT: 该来源没有已注册的直接字段契约；用extract_observations解释原文并复核，再以source_values选择。不能把来源数值写入user_values。")
    statement = parts[2]
    contract = CONTRACTS.get(statement, {}).get(selection.field)
    if contract is None or selection.metric != contract[0]:
        supported = {field: definition[0] for field, definition in CONTRACTS.get(statement, {}).items()}
        raise ValueError("INPUT_PROVIDER_FIELD: 字段与直接模型输入契约不符；可直接使用：" + canonical(supported)
            + "。其他字段保留原始含义，经LLM解释和确定性推导，不把营业利润当EBIT或货币资金直接当可分配现金。")
    target = target_ticker(session)
    ticker = parts[1]
    if (record.get("ts_code") != ticker or normalize_a_share_ticker(ticker) != ticker
            or (selection.role == "historical") != (ticker == target)):
        raise ValueError("INPUT_PROVIDER_ENTITY: API记录不是当前目标公司，不改写代码或挪用可比公司数值。")
    from valuationagent.application.issuer_identity import require_identity

    identity = require_identity(session, ticker, session.draft.company if selection.role == "historical" else "")
    cutoff = min(day for day in (session.draft.valuation_date, session.information_cutoff_date) if day is not None) if session.draft.valuation_date or session.information_cutoff_date else None
    if cutoff is None:
        raise ValueError("INPUT_PROVIDER_DATE: 先明确估值日与信息截止日。")
    try:
        period = date.fromisoformat(str(record.get("trade_date") if statement == "statistics" else record.get("end_date")))
        published = period if statement == "statistics" else date.fromisoformat(str(record.get("f_ann_date") or record.get("ann_date")))
    except ValueError:
        raise ValueError("INPUT_PROVIDER_DATE: 原始记录缺少有效期间或可得日，不能补造。") from None
    if not period <= published <= cutoff:
        raise ValueError("INPUT_PROVIDER_DATE: 期间或可得日晚于信息截止日，或者披露早于期间。")
    if statement != "statistics" and (str(record.get("report_type")) != "1" or (period.month, period.day) != (12, 31)):
        raise ValueError("INPUT_PROVIDER_PERIOD: 仅接收供应商report_type=1的完整年度，不把季度、母公司或调整表冒充本期合并年报。")
    company_type = record.get("comp_type")
    if statement != "statistics" and (isinstance(company_type, bool) or company_type not in (None, "", "1", 1)
            or contract[0].startswith("raw.") and str(company_type) != "1"):
        raise ValueError("INPUT_PROVIDER_SCOPE: 供应商声明该主体不是一般工商业，不能通过通用原始科目公式绕过金融企业模型范围。")
    raw_value = record.get(selection.field)
    try:
        number = Decimal(str(raw_value))
        if not number.is_finite() or isinstance(raw_value, bool):
            raise InvalidOperation()
    except InvalidOperation:
        raise ValueError("INPUT_PROVIDER_AMOUNT: 原始字段为空或非有限数值；不补零。") from None
    metric, unit, factor, scope, doc_id = contract
    if metric == "market_cap":
        try:
            total_shares = Decimal(str(record.get("total_share")))
            close = Decimal(str(record.get("close")))
            if not total_shares.is_finite() or not close.is_finite() or total_shares <= 0 or close <= 0:
                raise InvalidOperation()
        except InvalidOperation:
            raise ValueError("INPUT_PROVIDER_MARKET_CAP: 需同条记录的正值total_share和close核对A股价格等值口径，不把流通市值或其他价格口径直接套用。") from None
        expected = total_shares * close
        if number <= 0 or abs(number - expected) > max(Decimal("0.01"), expected * Decimal("0.0001")):
            raise ValueError("INPUT_PROVIDER_MARKET_CAP: total_mv与同日总股本乘收盘价不一致，须核对股份类别或接口定义，不自动调整数值。")
    shares = metric == "common_shares"
    instant = shares or metric == "market_cap"
    binding = {"contract_version": CONTRACT_VERSION, "statement": statement, "record_pointer": selection.record_pointer,
        "field": selection.field, "record": record, "issuer_identity": identity,
        "schema_reference": f"https://tushare.pro/document/2?doc_id={doc_id}",
        "date_semantics": "market_data_as_of_close" if statement == "statistics" else "financial_announcement",
        "admission_kind": "raw_operand" if metric.startswith("raw.") else "standard_input",
        "period_kind": "instant" if statement in {"statistics", "balancesheet"} else "annual",
        "pricing_basis": "a_share_equivalent" if metric == "market_cap" else None}
    source = InputSource(kind="provider", source_id="provider_" + digest(canonical([document.sha256, statement, selection.record_pointer, selection.field]))[:24],
        sha256=document.sha256, quote=canonical({selection.field: raw_value}), file_id=document.file_id,
        locator=canonical({"json_pointer": selection.record_pointer, "provider_field": selection.field, "statement": statement}),
        source_url=document.source_url, published_at=published, authority_tier=document.authority_tier,
        limitations=["供应商字段契约校验，不是发行人原件或独立审计；单位来自版本化接口适配契约。", *document.warnings],
        interpretation_sha256=digest(canonical(binding)), provider_binding=binding)
    payload = dict(entity=(session.draft.company or session.draft.ticker) if selection.role == "historical" else ticker,
        entity_ticker=ticker, role=selection.role, metric=metric, value=number * Decimal(factor),
        original_amount=str(raw_value), unit=unit, currency=None if shares else "CNY", scope=scope,
        label=RAW_FIELDS.get(statement, {}).get(selection.field, selection.field),
        period_kind="instant" if statement in {"statistics", "balancesheet"} else "annual",
        period_end=None if instant else period, as_of=period if instant else None, assertion="reported", source=source, supersedes=selection.replaces)
    return InputRecord(input_id="input_" + digest(canonical(payload))[:24], **payload)


def build_provider_inputs(runtime, values, dataset=None):
    saved, raw_files = [], {}
    existing = {row.input_id: row for row in dataset.active_records()} if dataset else {}
    for chosen in values:
        selection = resolve_provider_candidate(runtime.session, chosen.candidate_id, chosen.replaces)
        document = source_document(runtime.session, selection.file_id)
        if selection.file_id not in raw_files:
            _, raw_files[selection.file_id] = source_bytes(runtime.service.store, runtime.session, selection.file_id)
        record = provider_record(raw_files[selection.file_id], selection.record_pointer)
        row = build_provider_record(runtime.session, document, selection, record)
        dataset = dataset or InputDataset(entity=runtime.session.draft.company or runtime.session.draft.ticker,
            currency=row.currency or "CNY", analysis_basis="research")
        if (row.role == "historical" and dataset.entity != row.entity) or row.currency and dataset.currency != row.currency:
            raise ValueError("INPUT_SCOPE: API主体或币种与已有输入不一致。")
        if any(key not in existing or (existing[key].metric, existing[key].role) != (row.metric, row.role)
               or row.role == "comparable" and existing[key].entity_ticker != row.entity_ticker for key in selection.replaces):
            raise ValueError("INPUT_REPLACEMENT: 只能更正当前有效的同指标输入。")
        if row.input_id not in {item.input_id for item in dataset.records}:
            dataset.records.append(row)
            saved.append(row.input_id)
    return dataset, saved


def validate_provider_input(session, row, raw=None):
    document = source_document(session, row.source.file_id)
    binding = row.source.provider_binding
    if binding.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("INPUT_PROVIDER_CONTRACT: 供应商解释契约已改变，请重新选入；旧冻结估值不变。")
    selection = ProviderInputValue(file_id=document.file_id, record_pointer=binding["record_pointer"],
        field=binding["field"], metric=row.metric, role=row.role, replaces=row.supersedes)
    record = provider_record(raw, selection.record_pointer) if raw is not None else binding["record"]
    expected = build_provider_record(session, document, selection, record)
    if expected != row:
        raise ValueError("INPUT_SOURCE_CHANGED: API输入、来源字节或字段契约与选定记录不一致，不能复用。")
