from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from decimal import Decimal

from valuationagent.core.tools import canonical
from valuationagent.application.input_derivations import RAW_INPUT_FIELDS, derivation_dependencies, derive_snapshot_inputs
from valuationagent.application.input_calculations import INSTANT_FIELDS, apply_calculations
from valuationagent.application.input_conflicts import conflict_error
from valuationagent.schemas.inputs import ComparableSelection, InputDataset, InputRecord, InputSource
from valuationagent.schemas.models import AssumptionInputs, CompanyInput, EvidenceRef, FinancialSnapshot, ValuationRequest, required_financial_metrics


UNIT_FACTORS = {"元": Decimal(1), "千元": Decimal(1000), "万元": Decimal(10000), "亿元": Decimal(100000000),
                "股": Decimal(1), "万股": Decimal(10000), "亿股": Decimal(100000000), "ratio": Decimal(1), "%": Decimal("0.01")}
MULTIPLES = {"pe_multiple": "pe", "ps_multiple": "ps", "ev_ebitda_multiple": "ev_ebitda"}
FINANCIAL_FIELDS = {
    "revenue", "net_income_parent", "ebit_margin", "tax_rate", "ebitda", "depreciation_amortization",
    "capital_expenditure", "change_operating_nwc", "cash_and_non_operating_assets", "interest_bearing_debt",
    "common_shares", "diluted_shares", "lease_liabilities", "minority_interest", "preferred_equity",
    "associates_and_non_operating_investments", "unfunded_pension", "non_operating_provisions", "market_price", "market_cap",
}
RATIO_FIELDS = {"ebit_margin", "tax_rate", "wacc", "terminal_growth", *MULTIPLES}


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_amount(text, unit, quote):
    normalized = text.strip().replace("，", ",").replace("％", "%").replace("−", "-")
    if unit == "元" and normalized.endswith("元/股"):
        normalized = normalized[:-2]
    match = re.fullmatch(r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(亿元|万元|千元|元|亿股|万股|股|倍|%)?", normalized)
    if not match:
        raise ValueError("INPUT_AMOUNT: 引用用户给出的一个完整数值及单位；不要提交整行或计算结果。")
    suffix = match[2]
    expected_suffix = "倍" if unit == "ratio" else unit
    if suffix and suffix != expected_suffix:
        raise ValueError("INPUT_UNIT: 金额字面单位与unit不一致，不允许重新标单位。")
    if not suffix and unit != "ratio" and unit not in re.findall(r"亿元|万元|千元|元|亿股|万股|股|%", quote.replace("％", "%")):
        raise ValueError(f"INPUT_UNIT: 裸数所选单位={unit}，单位依据={quote!r}，不包含该单位。用unit_ref或共享user_basis.unit_ref选择明确单位行；兼容文本接口可用unit_quote。不是再问用户、不用联网；不要把单位拼进原值。")
    return Decimal(match[1].replace(",", "")) * UNIT_FACTORS[unit]


def user_amount_location(value, message, *, selected_start=None):
    from valuationagent.llm.context_manager import sensitive_spans

    if value.unit_quote and value.unit_quote not in message:
        raise ValueError("INPUT_UNIT: unit_quote必须是该用户消息中明确单位的原话，不能改写。")
    if sensitive_spans(value.unit_quote):
        raise ValueError("INPUT_UNIT: 单位依据包含凭据，不将凭据保存为财务证据。")
    protected = sensitive_spans(message)
    if not value.amount_text:
        raise ValueError("INPUT_SOURCE: 提交amount_ref或原话amount_text；优先read_user_input按行号选择，不编造数字。")
    matches = [match for match in re.finditer(re.escape(value.amount_text), message)
        if not any(start < match.end() and end > match.start() for start, end in protected)]
    if not matches:
        raise ValueError("INPUT_SOURCE: 数值不在所选用户原话中；不把API、文档或助手估算冒充用户输入。")
    complete = []
    unit_suffix = re.search(r"(?:亿元|万元|千元|元|亿股|万股|股|倍|[%％])\s*$", value.amount_text)
    for match in matches:
        before, after = message[:match.start()], message[match.end():]
        if (before and before[-1] in "0123456789.,，" or before.rstrip().endswith(("-", "−", "－", "+", "(", "（"))
                or not unit_suffix and (after and after[0] in "0123456789%％)）" or re.match(r"[.,，]\d|[eE][+-]?\d", after))
                or re.search(r"\d[eE][+-]?$", before)
                or re.match(r"\s*(?:亿元|万元|千元|元|亿股|万股|股|倍|[%％])", after)):
            continue
        complete.append(match)
    if not complete:
        raise ValueError("INPUT_AMOUNT_BOUNDARY: 数值截断了相邻数字、符号、百分号或单位；按用户原文选择完整数值，不取子串或擅自换算。")

    def excerpt_bounds(match, radius):
        start, end = max(0, match.start() - radius), min(len(message), match.end() + radius)
        for secret_start, secret_end in protected:
            if start < secret_end <= match.start():
                start = secret_end
            if match.end() <= secret_start < end:
                end = secret_start
        return start, end

    occurrence = value.amount_occurrence
    if selected_start is not None:
        indices = [index for index, match in enumerate(complete) if match.start() == selected_start]
        if len(indices) != 1:
            raise ValueError("USER_INPUT_REF: 所选数字存在符号/边界问题，不按截断数值入模。")
        if occurrence is not None and occurrence != indices[0]:
            raise ValueError("USER_INPUT_REF: 数值引用与手写匹配序号冲突。")
        occurrence = indices[0]
    if value.value_context:
        if sensitive_spans(value.value_context):
            raise ValueError("INPUT_SOURCE: 数值定位片段包含凭据，请只选择财务原话。")
        contexts = list(re.finditer(re.escape(value.value_context), message))
        candidates = [index for index, match in enumerate(complete)
            if any(context.start() <= match.start() and match.end() <= context.end() for context in contexts)]
        if len(candidates) != 1:
            if contexts and not candidates:
                raise ValueError("INPUT_AMOUNT_CONTEXT: value_context在原话中，但amount_text没有逐字出现在该片段中。原话是裸数就提交裸数，用unit_quote或user_basis.unit_quote单独绑定单位，不能把单位拼进amount_text。")
            options = [{"amount_occurrence": index, "context": message[slice(*excerpt_bounds(complete[index], 60))]}
                for index in candidates[:4]]
            raise ValueError("INPUT_AMOUNT_CONTEXT: value_context必须逐字匹配，不能用省略号或拼接；重复样本请包含公司名称的更长原话，或用amount_occurrence。候选：" + canonical(options))
        if occurrence is not None and occurrence != candidates[0]:
            raise ValueError("INPUT_AMOUNT_CONTEXT: 原话片段与amount_occurrence指向不同位置。")
        occurrence = candidates[0]
    if occurrence is None and len(complete) > 1:
        options = [{"amount_occurrence": index, "context": message[slice(*excerpt_bounds(match, 45))]}
            for index, match in enumerate(complete[:8])]
        raise ValueError("INPUT_AMOUNT_AMBIGUOUS: 同一原话出现多个相同数值，请根据科目选择amount_occurrence；" + canonical(options))
    occurrence = occurrence or 0
    if occurrence >= len(complete):
        raise ValueError("INPUT_AMOUNT_OCCURRENCE: 序号超出所选消息中完整数值的匹配数量。")
    match = complete[occurrence]
    start, end = excerpt_bounds(match, 160)
    return message[start:end], {"amount_start": match.start(), "amount_end": match.end(),
        "amount_occurrence": occurrence, "quote_start": start, "quote_end": end,
        "unit_quote": value.unit_quote, "value_context": value.value_context, "offset_unit": "unicode_codepoint"}


def supports_user_date(value, quote, *, annual=False):
    normalized = re.sub(r"\s+", "", quote)
    exact = re.search(rf"(?<!\d){value.year}(?:年|[-/])0?{value.month}(?:月|[-/])0?{value.day}(?:日)?(?!\d)", normalized)
    year_only = annual and value.month == 12 and value.day == 31 and re.fullmatch(rf"{value.year}(?:年|年度)", normalized)
    annual_list = (annual and value.month == 12 and value.day == 31
        and re.search(rf"(?<!\d){value.year}(?!\d)", normalized)
        and re.search(r"完整年度|年度|年末", normalized)
        and not re.search(r"季度|半年|中期|[1-9]月", normalized))
    return bool(exact or year_only or annual_list)


def bind_user_date(value, quote, message, *, annual=False):
    from valuationagent.llm.context_manager import sensitive_spans

    if sensitive_spans(quote):
        raise ValueError("INPUT_DATE_SOURCE: 日期依据包含凭据，请仅引用用户明确的日期，不复制密钥。")
    if value is None:
        if quote:
            raise ValueError("INPUT_DATE_SOURCE: 未填写日期却提交了日期依据。原话已有日期/完整年度时应保留已知日期，引用原文后填写period_end/as_of；不要为了修复引用错误而清空已知日期。真正未知时日期和依据均为空。")
        return
    if quote and quote in message and supports_user_date(value, quote, annual=annual):
        return
    candidates = [{"line": number, "text": line[:240]} for number, line in enumerate(message.splitlines(), 1)
        if not sensitive_spans(line) and supports_user_date(value, line, annual=annual)]
    field = "periods[].line / period_ref" if annual else "as_of.line / as_of_ref"
    detail = {"date": value.isoformat(), "reference_field": field,
        "date_candidates": candidates[:4], "candidates_omitted": max(0, len(candidates) - 4)}
    raise ValueError(f"INPUT_DATE_SOURCE: 日期={value.isoformat()}与所选日期依据={quote!r}不一致。"
        "line必须指向日期/年度声明，不是金额所在行；共享日期可在多批复用同一真实行。"
        "以下仅为原文位置候选，仍需判断是否适用于本主体/科目，不自动选择、不改日期、不把季度或旧日期当完整年度。"
        + canonical(detail))


def build_user_inputs(runtime, args, dataset=None):
    session = runtime.session
    messages = {message.message_id: message for message in runtime.service.store.list_messages(session.session_id) if message.role == "user"}
    current_id = session.turn_control.message_id if session.turn_control else next(reversed(messages), "")
    entity = args.company or (session.input_dataset.entity if session.input_dataset else None) or session.draft.company or session.draft.ticker or "用户情景"
    dataset = dataset or InputDataset(entity=entity, currency=args.currency, analysis_basis="user_scenario")
    if dataset.entity != entity or dataset.currency != args.currency:
        raise ValueError(f"INPUT_SCOPE: 当前输入主体={dataset.entity!r}、币种={dataset.currency}；本次提交主体={entity!r}、币种={args.currency}。更正同一任务时省略company沿用当前主体，并使用replaces；不要通过update_task改公司来修复这个错误。真正切换公司请新建任务。")
    existing = {row.input_id: row for row in dataset.active_records()}
    saved = []
    for reference, field in (("period_ref", "period_end"), ("as_of_ref", "as_of")):
        if getattr(args.user_basis, reference) and not getattr(args.user_basis, field) and not any(getattr(value, field) for value in args.user_values):
            raise ValueError(f"INPUT_DATE_REQUIRED: user_basis.{reference}仅定位原文，不会自动选择日期。请同时填写user_basis.{field}或逐项{field}，依据原话判断，不清空已知年度或时点。")
    for index, value in enumerate(args.user_values):
        value = value.model_copy(update={name: getattr(args.user_basis, name) for name in ("period_end", "as_of")
            if getattr(args.user_basis, name) is not None and name not in value.model_fields_set})
        value = value.model_copy(update={name: getattr(value, name) or getattr(args.user_basis, name)
            for name in ("unit_quote", "period_quote", "as_of_quote")
            if name == "unit_quote" or name == "period_quote" and value.period_end or name == "as_of_quote" and value.as_of})
        comparable = value.role == "comparable"
        peer_fields = {"market_price", "market_cap"} if comparable else set()
        raw_metric = re.fullmatch(r"raw\.[a-z][a-z0-9_]{0,79}", value.metric) is not None
        if value.metric not in FINANCIAL_FIELDS | RATIO_FIELDS | RAW_INPUT_FIELDS | peer_fields and not raw_metric:
            raise ValueError("INPUT_METRIC: 未支持的模型字段：" + value.metric)
        if raw_metric and value.period_kind == "unknown":
            raise ValueError("INPUT_PERIOD_KIND: 自定义原始科目须说明annual流量或instant存量；不补造用户没提供的日期。")
        message = messages.get(value.message_id or current_id)
        if message is None:
            raise ValueError("INPUT_SOURCE: 只能选择当前工作区真实用户消息；API值用provider_values，已复核文档用source_values。")
        if comparable and (not value.entity or value.entity not in message.content or value.entity == entity):
            raise ValueError("INPUT_PEER_ENTITY: 用户可比的entity须为原消息中的样本名称，不使用目标自身、不虚构证券代码。")
        if not comparable and value.entity and value.entity != entity:
            raise ValueError("INPUT_ENTITY: 目标输入不得绑定其他主体；可比数值使用role=comparable。")
        from valuationagent.application.user_message import resolve_user_references

        value, selected_start = resolve_user_references(value, args.user_basis, message.content)
        quote, locator = user_amount_location(value, message.content, selected_start=selected_start)
        locator.update({name: getattr(value, name) for name in ("amount_ref", "unit_ref", "period_ref", "as_of_ref") if getattr(value, name)})
        if "/股" in value.amount_text and value.metric != "market_price":
            raise ValueError("INPUT_DIMENSION: 元/股属于每股价格，不能作为营业收入等总金额。")
        try:
            numeric = parse_amount(value.amount_text, value.unit, value.unit_quote)
            bind_user_date(value.period_end, value.period_quote, message.content, annual=True)
            bind_user_date(value.as_of, value.as_of_quote, message.content)
        except ValueError as exc:
            raise ValueError(f"{exc}；user_values[{index}]，metric={value.metric}，entity={value.entity or entity}，amount_ref={value.amount_ref}") from None
        if value.metric == "market_price" and value.unit != "元":
            raise ValueError("INPUT_DIMENSION: 每股价格使用元，不能套用报表金额的万元/亿元单位。")
        if value.metric == "market_cap" and value.unit not in {"元", "千元", "万元", "亿元"}:
            raise ValueError("INPUT_DIMENSION: 市值使用金额单位。")
        if value.metric in RATIO_FIELDS and value.unit not in {"ratio", "%"}:
            raise ValueError("INPUT_DIMENSION: 比率和倍数不能使用金额或股数单位。")
        if value.metric in {"common_shares", "diluted_shares"} and value.unit not in {"股", "万股", "亿股"}:
            raise ValueError("INPUT_DIMENSION: 股数必须使用股数单位，不能使用股本面值金额。")
        if (raw_metric or value.metric in (FINANCIAL_FIELDS | RAW_INPUT_FIELDS) - RATIO_FIELDS - {"common_shares", "diluted_shares"}) and value.unit not in {"元", "千元", "万元", "亿元"}:
            raise ValueError("INPUT_DIMENSION: 金额必须使用金额单位。")
        if value.metric in MULTIPLES and value.unit != "ratio":
            raise ValueError("INPUT_DIMENSION: 倍数使用ratio而不是百分比。")
        peer_key = "user:" + value.entity if comparable else ""
        if any(key not in existing or existing[key].metric != value.metric or existing[key].role != value.role
               or comparable and existing[key].entity_ticker != peer_key for key in value.replaces):
            raise ValueError("INPUT_REPLACEMENT: 只能更正当前有效的同指标输入。")
        payload = dict(entity=value.entity if comparable else entity, role=value.role, entity_ticker=peer_key,
                       metric=value.metric, label=value.metric,
                       period_kind=value.period_kind if raw_metric else "instant" if value.metric in INSTANT_FIELDS | peer_fields else "annual",
                       value=numeric, original_amount=value.amount_text, unit=value.unit,
                       currency=args.currency, scope=value.scope, period_end=value.period_end, as_of=value.as_of,
                       assertion="assumption" if value.metric in MULTIPLES or value.metric in {"wacc", "terminal_growth"} else "user_input",
                       source=InputSource(kind="user", source_id=message.message_id, sha256=digest(message.content), quote=quote,
                                          locator=canonical({**locator, "period_quote": value.period_quote, "as_of_quote": value.as_of_quote})),
                       supersedes=value.replaces)
        row = InputRecord(input_id="input_" + digest(canonical(payload))[:24], **payload)
        if row.input_id not in {entry.input_id for entry in dataset.records}:
            dataset.records.append(row)
            saved.append(row.input_id)
        if comparable and peer_key not in dataset.comparables:
            dataset.comparables[peer_key] = ComparableSelection(ticker=peer_key, name=value.entity,
                rationale="用户指定的情景可比样本；不是自动选股、真实证券身份核验或市场观测。")
    from valuationagent.application.input_sources import dataset_basis

    dataset.analysis_basis = dataset_basis(dataset.active_records())
    return dataset, saved


def input_evidence(row):
    source = row.source
    if source.kind == "user":
        return EvidenceRef(evidence_id=row.input_id, source="user_input", source_sha256=source.sha256,
            note=f"用户消息 {source.source_id}：{source.quote}；用户提供的数值或假设，不是外部核验。")
    location = json.loads(source.locator)
    assessment = "供应商字段契约校验，不是LLM原文复核或独立审计。" if source.provider_binding else "同一LLM语义复核不等于独立审计。"
    return EvidenceRef(evidence_id=row.input_id, source=source.kind, file_id=source.file_id,
        source_sha256=source.sha256, source_url=source.source_url, published_at=source.published_at,
        page=location.get("page"), sheet=location.get("sheet"), cell=location.get("cell"),
        note=f"原始解释 {source.source_id}；定位 {source.locator}；引文 {source.quote}；来源等级 {source.authority_tier}；"
             + assessment + "；".join(source.limitations))


def prepare_dataset(session, methods, *, assumptions=None, assumption_evidence=None):
    from valuationagent.application.input_sources import dataset_basis, validate_source_inputs

    dataset = session.input_dataset
    rows = [row for row in dataset.active_records() if row.role == "historical"]
    if not rows:
        raise ValueError("INPUTS_MISSING: 数据集没有有效输入。")
    company = session.draft.company or session.draft.ticker or dataset.entity
    if company != dataset.entity and session.draft.ticker != dataset.entity:
        raise ValueError("INPUT_ENTITY: 数据集主体与当前任务不一致，不能用于另一家公司。")
    equity_methods_only = set(methods) <= {"pe", "ps"}
    if equity_methods_only:
        relevant_metrics = required_financial_metrics(methods) | {"diluted_shares", "market_price", "market_cap"}
        relevant_metrics.update(metric for metric, method in MULTIPLES.items() if method in methods)
        rows = [row for row in rows if row.metric in relevant_metrics]
        missing_denominators = (required_financial_metrics(methods) - {"common_shares"}) - {row.metric for row in rows}
        if missing_denominators:
            raise ValueError("INPUTS_MISSING: 目标公司缺少所选方法的年度分母：" + ",".join(sorted(missing_denominators))
                + "；可比公司输入不能替代目标公司数据。")
    validate_source_inputs(session, rows)
    analysis_basis = dataset_basis(dataset.active_records())
    cutoff = session.draft.valuation_date or date.today()
    multiples, assumption_values = {}, dict(assumptions or {})
    assumption_evidence = dict(assumption_evidence or {})
    if equity_methods_only:
        relative_assumptions = {"relative_multiples", "market_cap", "annual_average_market_cap",
            "quarterly_average_market_cap", "market_cap_period_low", "market_cap_period_high",
            "market_inputs_as_of", "market_inputs_source", "market_inputs_stale_after_days"}
        assumption_values = {key: value for key, value in assumption_values.items() if key in relative_assumptions}
        assumption_evidence = {key: value for key, value in assumption_evidence.items()
            if key in relative_assumptions or MULTIPLES.get(key) in methods}
    share_fields = {"common_shares", "diluted_shares"}
    calculation_metrics = derivation_dependencies(required_financial_metrics(methods)) | ({"operating_nwc"} if "dcf" in methods else set())
    if not equity_methods_only:
        calculation_metrics.update({"lease_liabilities", "minority_interest"})
    periods = {row.period_end for row in rows if row.period_end and row.metric not in RATIO_FIELDS and row.metric not in {"common_shares", "diluted_shares"}}
    from valuationagent.application.input_baseline import selected_period

    explicit_baseline = selected_period(dataset)
    if explicit_baseline:
        if explicit_baseline not in periods:
            raise ValueError(f"INPUT_BASELINE_MISSING: 用户指定{explicit_baseline}基期尚无年度输入，不能静默改用另一年。")
        periods = {period for period in periods if period <= explicit_baseline}
    baseline = explicit_baseline or (max(periods) if periods else None)
    known_latest = max(dataset.available_target_annual_periods, default=None)
    if not explicit_baseline and known_latest and (not baseline or known_latest > baseline):
        raise ValueError(f"INPUT_BASELINE_NEWER_AVAILABLE: 已保存来源显示{known_latest}年度可得，但尚未选入；复用已有候选或acquire_financial_inputs补该年度，不重复解释旧年为最新。用户明确要求旧基期才用record_inputs(baseline)引用其原话。")
    selected = []
    for row in rows:
        if row.as_of and row.as_of > cutoff or row.period_end and row.period_end > cutoff:
            raise ValueError("INPUT_DATE: 数据时点晚于估值日。")
        if row.metric in MULTIPLES or row.metric in {"wacc", "terminal_growth"}:
            target = multiples if row.metric in MULTIPLES else assumption_values
            key = MULTIPLES.get(row.metric, row.metric)
            if key in target and target[key] != row.value:
                raise conflict_error(row.metric, rows)
            target[key] = row.value
            assumption_evidence.setdefault(row.metric, []).append(input_evidence(row))
            selected.append(row)
    existing_multiples = assumption_values.pop("relative_multiples", {})
    for method, value in existing_multiples.items():
        if method not in methods:
            continue
        if method in multiples and multiples[method] != value:
            raise ValueError("INPUT_CONFLICT: 预测方案与显式倍数冲突。")
        multiples[method] = value
    from valuationagent.application.input_peers import prepare_input_peers

    peers, peer_records, peer_screening = prepare_input_peers(session,
        [str(method) for method in methods if method != "dcf" and method not in multiples], baseline)
    dated_shares = {}
    for metric in share_fields:
        candidates = [row for row in rows if row.metric == metric]
        dates = [row.as_of for row in candidates if row.as_of]
        newest = max(dates) if dates else None
        dated_shares[metric] = [row for row in candidates if not row.as_of or row.as_of == newest]
    snapshots = []
    snapshot_periods = (sorted(periods) or [None]) if "dcf" in methods else [baseline]
    for period in snapshot_periods:
        current = [row for row in rows if (row.metric in (FINANCIAL_FIELDS | RAW_INPUT_FIELDS) - share_fields or row.metric.startswith("raw."))
                   and (row.period_end == period or row.period_end is None and period == baseline)]
        if period == baseline:
            current += [row for group in dated_shares.values() for row in group]
        if not current:
            continue
        financials, evidence = {}, {}
        for row in current:
            if row.metric in financials and financials[row.metric] != row.value:
                raise conflict_error(row.metric, current)
            financials[row.metric] = row.value
            evidence.setdefault(row.metric, []).append(input_evidence(row))
            if row.metric in share_fields and row.as_of:
                financials[row.metric + "_as_of"] = row.as_of
        financials, evidence, declared = apply_calculations(dataset, period, financials, evidence,
            calculation_metrics)
        financials, evidence, derivations = derive_snapshot_inputs(financials, evidence, required_financial_metrics(methods))
        raw_fields = RAW_INPUT_FIELDS | {key for key in financials if key.startswith("raw.")}
        snapshots.append(FinancialSnapshot(period_end=period, source_label="统一输入工作区；逐项保留来源性质及确定性推导",
            currency=dataset.currency, evidence=evidence, calculation_methods={**declared, **derivations},
            statement_items={key: value for key, value in financials.items() if key in raw_fields},
            **{key: value for key, value in financials.items() if key not in raw_fields}))
        selected.extend(current)
    if not snapshots:
        raise ValueError("INPUTS_MISSING: 没有当前方法的财务数据；假设和倍数不替代财务指标。")
    if "dcf" in methods:
        from valuationagent.application.input_balances import derive_balance_changes

        snapshots = derive_balance_changes(snapshots)
    snapshot = snapshots[-1]
    missing = required_financial_metrics(methods) - {key for key, value in snapshot.model_dump().items() if value is not None}
    if snapshot.diluted_shares is not None:
        missing.discard("common_shares")
    missing.update(method + "_multiple_or_peers" for method in methods if method != "dcf" and method not in multiples
                   and sum(getattr(peer, method, None) is not None for peer in peers) < 3)
    if missing:
        detail = "；".join(f"{item['ticker']}：{'、'.join(item['reasons'])}" for item in peer_screening if item["reasons"])
        raise ValueError("INPUTS_MISSING: 当前方法缺少模型输入：" + "、".join(sorted(missing))
            + ("；可比筛选：" + detail if detail else "；自主相对估值请record_inputs保存可比选样及comparable角色的同日市值、同年度分母；用户未给倍数时不能虚构用户假设。"))
    if analysis_basis == "research":
        shares_date = snapshot.diluted_shares_as_of if snapshot.diluted_shares is not None else snapshot.common_shares_as_of
        if shares_date is None or (cutoff - shares_date).days > 120:
            raise ValueError("INPUT_SHARES_DATE: 研究估值的股数需独立时点，当前模型要求距估值日不超过120天；不能用财务基期冒充。")
    forecast_ids = {ref.evidence_id for refs in assumption_evidence.values() for ref in refs if ref.evidence_id.startswith("input_")}
    frozen_rows = {row.input_id: row for row in selected
        if "dcf" in methods or row.period_end in {None, baseline} or row.metric in share_fields}
    frozen_rows.update({row.input_id: row for row in peer_records})
    frozen_rows.update({row.input_id: row for row in dataset.active_records() if row.input_id in forecast_ids})
    return ValuationRequest(company=CompanyInput(name=company, ticker=session.draft.ticker or None, industry=session.draft.industry or None, currency=dataset.currency),
        valuation_date=cutoff, language=session.language, data_source="structured", assumption_source="manual" if assumption_values or multiples else "automatic", analysis_basis=analysis_basis,
        forecast_years=10, discount_policy="year_end",
        financials=snapshot, historical_financials=snapshots[:-1] if "dcf" in methods else [], peers=peers, peer_screening=peer_screening,
        assumptions=AssumptionInputs(relative_multiples=multiples, **assumption_values), assumption_evidence=assumption_evidence, methods=methods,
        requested_methods=session.draft.methods or methods, excluded_methods=dict(session.valuation_method_exclusions),
        input_records=[row.model_dump(mode="json") for row in frozen_rows.values()],
        input_calculations=[calculation.model_dump(mode="json") for calculation in dataset.active_calculations()
            if (not calculation.entity_ticker and calculation.metric in calculation_metrics
                and ("dcf" in methods or calculation.period_end == baseline))
            or calculation.calculation_id in {key for item in peer_screening for key in item.get("calculation_ids", [])}],
        baseline_selection={**dataset.baseline_selection, "policy": dataset.baseline_selection.get("policy", "latest_available"),
            "period_end": baseline.isoformat() if baseline else None,
            "observed_available_target_periods": [period.isoformat() for period in dataset.available_target_annual_periods]},
        user_goal=session.draft.objective or "按明确来源与假设计算估值")
