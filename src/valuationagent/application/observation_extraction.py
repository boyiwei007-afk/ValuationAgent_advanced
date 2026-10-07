"""LLM-owned document interpretation with mechanically anchored observations."""
from __future__ import annotations

import hashlib
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import Field, model_validator

from valuationagent.application.file_workspace import source_bytes, source_document
from valuationagent.application.observation_consistency import PERIOD_CELL_CONFLICT, PERIOD_READBACK_REQUIRED, period_cell_conflicts, retire_contradicted_entities
from valuationagent.application.research_valuation import (
    METRIC_ALIASES, DEBT_COMPONENTS, _period, financial_mapping_issue, mapped_financial_metric,
)
from valuationagent.core.tools import canonical
from valuationagent.schemas.models import ApiModel
from valuationagent.schemas.research import FactCandidate


REVIEW_PENDING = "SEMANTIC_REVIEW_PENDING: 原文定位通过，LLM尚未复核主体/期间/单位/口径/字段映射"
REVIEW_UNRESOLVED = "SEMANTIC_REVIEW_NEEDS_EVIDENCE: 已复核但仍有歧义或矛盾，需要补证或更正观察"
PUBLICATION_PENDING = "SOURCE_PUBLICATION_PENDING: LLM从原文提出的披露日期尚未复核，不以报告期或抓取日代替"
FACT_CONFLICT = "同主体同口径存在数值冲突；解决之前不进入计算"
FACTORS = {"元": "1", "千元": "1000", "万元": "10000", "百万元": "1000000", "亿元": "100000000",
           "股": "1", "千股": "1000", "万股": "10000", "百万股": "1000000", "亿股": "100000000", "%": "0.01", "ratio": "1"}
STOCK_METRICS = {"common_shares", "diluted_shares", "cash_and_non_operating_assets", "trading_financial_assets",
    "interest_bearing_debt", "minority_interest", "preferred_equity", "associates_and_non_operating_investments",
    "unfunded_pension", "non_operating_provisions", "restricted_cash", "financial_institution_deposits",
    "interbank_lending", "restricted_interbank_deposits", "operating_nwc", "accounts_receivable", "notes_receivable",
    "receivables_financing", "contract_assets", "prepayments", "inventory", "accounts_payable", "notes_payable", "contract_liabilities", *DEBT_COMPONENTS}
ROLE_INDEPENDENT_METRICS = {"revenue", "total_revenue", "main_business_revenue", "net_income_parent",
    "profit_before_tax", "income_tax_expense", "common_shares", "diluted_shares"}


class EvidenceSpan(ApiModel):
    block_id: str = Field(min_length=1, max_length=200)
    start_line: int = Field(ge=1, description="工具返回的块内原始行号，1起始；不是PDF页码。")
    end_line: int | None = Field(default=None, ge=1)
    quote: str = Field(default="", max_length=1800, description="可省略：程序直接读取指定连续行作为证据，避免重抄空格。填写时必须是原文逐字子串，不能改写。")
    occurrence: int = Field(default=0, ge=0, le=100, description="同一范围内重复引文的0起始序号；例如重述前后同名年度列。")


class ReadingBasis(ApiModel):
    entity_name: str = Field(min_length=1, max_length=200)
    entity_ticker: str = Field(max_length=24, description="必填。当前任务有证券代码时填写与其一致的代码，并用entity_refs证明原文主体；可比公司填写自身代码。仅没有证券代码的纯文件讨论可填空字符串。")
    entity_refs: list[str] = Field(min_length=1, max_length=4)
    scope: Literal["consolidated", "parent", "issuer"] = Field(description="consolidated=合并财务报表；parent=母公司单体财务报表；issuer=上市发行人股份结构/总市值。股份变动表可由公司标题与表格语境证明issuer，不要求原文印出英文issuer或合并二字；不能把母公司报表股本金额当股数。")
    scope_refs: list[str] = Field(min_length=1, max_length=4)
    unit: Literal["元", "千元", "万元", "百万元", "亿元", "股", "千股", "万股", "百万股", "亿股", "%", "ratio"]
    unit_refs: list[str] = Field(min_length=1, max_length=4)
    currency: str | None = Field(default=None, pattern="^[A-Z]{3}$", description="金额必须声明原文币种，使用ISO代码；股数和比率不填。当前计算器仅支持CNY，不隐式换汇。")
    source_published_at: date | None = Field(default=None, description="来源元数据缺失时，可解释本文件原文明确披露/发布的日期，须source_publication_refs和单独复核。不是报表截止日、董事会批准日、下载日或父目录日期；不知道则省略。不能覆盖冲突的已有日期。")
    source_publication_refs: list[str] = Field(default_factory=list, max_length=4, description="支持本文件发布日期的anchors键名，不接受其他文件或搜索摘要的日期。")

    @model_validator(mode="after")
    def publication_shape(self):
        if bool(self.source_published_at) != bool(self.source_publication_refs):
            raise ValueError("来源披露日期和source_publication_refs须同时提供，不能无依据指定日期")
        return self


class Observation(ApiModel):
    metric: str = Field(min_length=1, max_length=120, description="原文科目名称，和standard_metric分开；修正映射不需要改写原文名称。")
    standard_metric: str = Field(min_length=1, max_length=120, description="标准模型字段使用inspect_requirements.metric_catalog的ID。无法直接映射但需保留的原始金额可用raw.前缀及英文小写标识，例如raw.operating_cost；不是最终模型字段，需原文复核后as_raw选择并由声明计算引用。不得把原始利润直接改名EBIT。")
    raw_value: str = Field(min_length=1, max_length=100, description="原文选定列的完整数值，保留负号/小数/百分号，不换算。可保留与basis.unit一致的明确单位后缀，如7.71亿元且unit=亿元；单位仍须原文依据及复核。不填整句、区间、约数推算或缺失占位符。")
    value_ref: str = Field(min_length=1, max_length=80, description="anchors字典中一个键名，例如rev_row；绝不是数值或引文正文。该键对应的原文可以是整行，工具从行内定位raw_value。")
    value_occurrence: int | None = Field(default=None, ge=0, le=100, description="仅当同一片段中有多个相同数值时，指定匹配数值的0起始序号；否则缩小引文范围。")
    value_segments: list[str] = Field(default_factory=list, max_length=8, description="仅文本层数字粘连时：列出同一原文行范围内完整连续数串的所有分段anchor ID（含value_ref），不能省略字符；分列语义仍须LLM复核。")
    label_refs: list[str] = Field(min_length=1, max_length=4)
    period_kind: Literal["annual", "instant", "interim", "ttm"] = Field(description="按字段经济含义选择，不按文件标题选择。股数/资产/负债/市值为instant并省略period_start，即使来自年报也不是annual；年度收入/利润是annual。期末日期仍须原文证据。")
    period_start: date | None = Field(default=None, description="annual且period_end为12-31时可省略，完整日历年定义规范为同年01-01，不补造其他年度。interim/ttm必须填真实开始日；instant不填。")
    period_end: date
    period_refs: list[str] = Field(min_length=1, max_length=4)
    basis: ReadingBasis | None = Field(default=None, description="通常省略并使用顶层共享basis。仅该行口径不同才提供完整ReadingBasis，含独立entity_refs/scope_refs/unit_refs；不重复填写缺引用的半份basis。")
    revision: Literal["reported", "restated", "unknown"] = "reported"
    revision_refs: list[str] = Field(default_factory=list, max_length=4)
    semantic_role: Literal["operating", "financing", "financial_subsidiary", "investing", "tax", "equity", "non_operating", "unknown"] = "unknown"
    ebit_treatment: Literal["include", "exclude", "review"] = "review"
    fcff_treatment: Literal["include", "exclude", "review"] = "review"
    equity_bridge_treatment: Literal["include", "exclude", "review"] = "review"
    role: Literal["historical", "comparable"] = Field(default="historical", description="historical仅用于当前估值目标；其他可比公司的财务、市值或倍数必须显式填写comparable，并在basis中保留可比公司自身名称/代码，不能改写成目标主体。")
    multiple_basis: Literal["FY", "TTM", "forward", "unknown"] = "unknown"
    denominator_period_end: date | None = Field(default=None, description="直接披露的可比倍数所用财务分母截止日；FY须为明确年度12-31，不是定价日。基础市值/年度财务不填。")
    denominator_refs: list[str] = Field(default_factory=list, max_length=4)
    rationale: str = Field(min_length=20, max_length=1000)
    uncertainties: list[str] = Field(default_factory=list, max_length=6)
    replaces: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def period_shape(self):
        if self.period_kind == "annual" and self.period_start is None and (self.period_end.month, self.period_end.day) == (12, 31):
            self.period_start = date(self.period_end.year, 1, 1)
        if self.period_kind == "instant":
            if self.period_start is not None:
                raise ValueError("时点字段不填写period_start")
        elif not self.period_start or self.period_start > self.period_end:
            raise ValueError("期间字段需要有效起止日期")
        if self.period_kind == "annual" and not (self.period_start == date(self.period_end.year, 1, 1) and self.period_end == date(self.period_end.year, 12, 31)):
            raise ValueError("当前年度模型只接收完整日历年度；其他期间明确标记interim，不扩大成全年")
        if self.revision == "restated" and not self.revision_refs:
            raise ValueError("重述列必须引用重述依据，不得仅凭较新报告推断")
        return self


class ExtractObservations(ApiModel):
    file_id: str = Field(min_length=1, max_length=100)
    anchors: dict[str, EvidenceSpan] = Field(min_length=1, max_length=64, description="自定义短ID到原文位置的字典。例如rev_row:{block_id:已读取块ID,start_line:12}。各类*_refs和value_ref均填写这些短ID，不填正文。")
    basis: ReadingBasis
    rows: list[Observation] = Field(min_length=1, max_length=12)


class ObservationSelection(ApiModel):
    fact_ids: list[str] = Field(min_length=1, max_length=12)


class ReviewChecks(ApiModel):
    entity: Literal["supported", "ambiguous", "contradicted"]
    amount: Literal["supported", "ambiguous", "contradicted"]
    period: Literal["supported", "ambiguous", "contradicted"]
    unit: Literal["supported", "ambiguous", "contradicted"]
    scope: Literal["supported", "ambiguous", "contradicted"]
    mapping: Literal["supported", "ambiguous", "contradicted"]
    publication: Literal["supported", "ambiguous", "contradicted"] | None = Field(default=None, description="复核包含LLM披露日期主张时必填：原文是否支持本文件的发布日期，而非报告期/批准日/抓取日；其他包可省略。")


class ObservationReview(ApiModel):
    fact_id: str
    packet_id: str = Field(min_length=1, max_length=100)
    rationale: str = Field(min_length=20, max_length=1200)
    checks: ReviewChecks
    source_period_end: date | None = Field(default=None, description="requires_period_readback=true时必须显式填写：仅根据当前原文独立读出该数值真实截止日/变动生效日，不照抄候选日期；无法确定填null并将period标ambiguous。年报中的某次资本变动只支持实际生效日，不自动支持年末。")


class ReviewObservations(ApiModel):
    reviews: list[ObservationReview] = Field(min_length=1, max_length=12)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def source_identity(document):
    return {key: getattr(document, key) for key in ("authority_tier", "provenance_type", "source_url", "acquisition_ref")}


def resolve_span(span, blocks, file_id):
    block = blocks.get(span.block_id)
    if not block or span.block_id.rsplit(":", 1)[0] != file_id:
        raise ValueError("ANCHOR_SCOPE: 证据须来自当前文件的已读取原文块")
    lines = block["text"].splitlines()
    last = span.end_line or span.start_line
    if not 1 <= span.start_line <= last <= len(lines) or last - span.start_line >= 20:
        raise ValueError("ANCHOR_RANGE: 每个证据片段须在原始块内连续1至20行")
    excerpt = "\n".join(lines[span.start_line - 1:last])
    quote = span.quote or excerpt
    if not quote.strip():
        raise ValueError("ANCHOR_EMPTY: 所选行没有文本；读取相邻行或其他视图，不默认空白为0")
    positions = [match.start() for match in re.finditer(re.escape(quote), excerpt)]
    if span.occurrence >= len(positions):
        raise ValueError("ANCHOR_TEXT: 逐字引文或重复序号与指定原文行不一致；不要拼接或补字")
    return {**span.model_dump(mode="json"), "quote": quote, "end_line": last, "char_offset": positions[span.occurrence],
            "block_sha256": hashlib.sha256(block["text"].encode()).hexdigest(), "location": block.get("location", {})}


def scalar(text):
    value = text.strip().replace("，", ",").replace("−", "-").replace("－", "-").replace("％", "%")
    if value.startswith(("(", "（")) and value.endswith((")", "）")):
        value = "-" + value[1:-1].strip()
    value = value.removesuffix("%").strip()
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", value):
        raise ValueError("AMOUNT_NOT_EXPLICIT: 金额证据必须是一个完整数值；空白/横线不自动当0，不推算缺失数值")
    try:
        result = Decimal(value.replace(",", ""))
    except InvalidOperation:
        raise ValueError("AMOUNT_NOT_EXPLICIT: 数值不可解析") from None
    if not result.is_finite():
        raise ValueError("AMOUNT_NOT_FINITE: 数值必须有限")
    return result


def check_amount_boundary(anchor, blocks):
    lines = blocks[anchor["block_id"]]["text"].splitlines()
    excerpt = "\n".join(lines[anchor["start_line"] - 1:anchor["end_line"]])
    start = anchor["char_offset"]
    end = start + len(anchor["quote"])
    before, after = excerpt[:start], excerpt[end:]
    if (before and before[-1] in "0123456789.,，" or before.rstrip().endswith(("-", "−", "－", "+", "(", "（"))
            or after and after[0] in "0123456789%％)）" or re.match(r"[.,，]\d|[eE][+-]?\d", after)
            or after.lstrip().startswith(("%", "％", ")", "）"))
            or re.search(r"\d[eE][+-]?$", before)):
        raise ValueError("AMOUNT_BOUNDARY: 金额片段截断了符号、括号、百分号或相邻数字；引用完整数值，粘连列请换layout/sheet/页面视图核对")


def observation_amount(raw_value, unit):
    value = raw_value.strip()
    for suffix in sorted({*FACTORS, "倍"} - {"%", "ratio"}, key=len, reverse=True):
        if value.endswith(suffix):
            expected_unit = "ratio" if suffix == "倍" else suffix
            if unit != expected_unit:
                raise ValueError(f"AMOUNT_UNIT: raw_value后缀{suffix}与声明单位{unit}不一致；核对原文，不自动换算或改单位")
            value = value[:-len(suffix)].strip()
            break
    return scalar(value)


def check_amount_segments(row, anchors, blocks):
    if not row.value_segments:
        check_amount_boundary(anchors[row.value_ref], blocks)
        return
    if row.value_ref not in row.value_segments or len(set(row.value_segments)) != len(row.value_segments) or len(row.value_segments) < 2:
        raise ValueError("AMOUNT_SEGMENTS: 显式分段必须包含目标片段及相邻数值，不能重复或只报目标数")
    segments = sorted((anchors[key] for key in row.value_segments), key=lambda anchor: anchor["char_offset"])
    ranges = {(anchor["block_id"], anchor["start_line"], anchor["end_line"]) for anchor in segments}
    if len(ranges) != 1:
        raise ValueError("AMOUNT_SEGMENTS: 粘连数值的分段必须引用同一原文行范围")
    for index, anchor in enumerate(segments):
        scalar(anchor["quote"])
        if index and segments[index - 1]["char_offset"] + len(segments[index - 1]["quote"]) != anchor["char_offset"]:
            raise ValueError("AMOUNT_SEGMENTS: 相邻分段必须完整连续、不重叠、不跳字符")
    check_amount_boundary({**segments[0], "quote": "".join(anchor["quote"] for anchor in segments)}, blocks)


def resolve_amount(row, anchors, blocks, unit=None):
    anchor = anchors[row.value_ref]
    expected = observation_amount(row.raw_value, unit or (row.basis.unit if row.basis else None))
    if row.value_segments:
        check_amount_segments(row, anchors, blocks)
        if scalar(anchor["quote"]) != expected:
            raise ValueError("AMOUNT_MISMATCH: raw_value与金额片段不一致")
        if row.value_occurrence not in (None, 0):
            raise ValueError("AMOUNT_OCCURRENCE: 独立金额片段只有一个数值")
        return {**anchor, "source_ref": row.value_ref, "resolution_method": "explicit_segments"}
    number = r"\d[\d,，]*(?:\.\d+)?(?:[eE][+-]?\d+)?"
    pattern = rf"[（(]\s*{number}\s*[%％]?\s*[)）]|[+\-−－]?\s*{number}(?:\s*[%％])?"
    matches = []
    for match in re.finditer(pattern, anchor["quote"]):
        literal = match.group().strip()
        offset = match.start() + len(match.group()) - len(match.group().lstrip())
        candidate = {**anchor, "quote": literal, "char_offset": anchor["char_offset"] + offset}
        try:
            amount = scalar(literal)
            check_amount_boundary(candidate, blocks)
        except ValueError:
            continue
        if amount == expected:
            matches.append(candidate)
    if not matches:
        try:
            scalar(anchor["quote"])
        except ValueError:
            raise ValueError("AMOUNT_NOT_FOUND: 原文片段内没有与raw_value一致的完整数值；保留符号、小数和百分号，不推算缺失数值。") from None
        check_amount_boundary(anchor, blocks)
        raise ValueError("AMOUNT_MISMATCH: raw_value与金额片段不一致")
    if len(matches) > 1 and row.value_occurrence is None:
        offsets = [candidate["char_offset"] - anchor["char_offset"] for candidate in matches]
        raise ValueError(f"AMOUNT_AMBIGUOUS: 原文有{len(matches)}个相同数值（片段内字符偏移{offsets}）；根据年度列指定value_occurrence或缩小引文，不自动选第一列。")
    occurrence = row.value_occurrence or 0
    if occurrence >= len(matches):
        raise ValueError(f"AMOUNT_OCCURRENCE: 只有{len(matches)}个匹配数值，value_occurrence超出范围")
    selected = matches[occurrence]
    lines = blocks[selected["block_id"]]["text"].splitlines()
    excerpt = "\n".join(lines[selected["start_line"] - 1:selected["end_line"]])
    positions = [match.start() for match in re.finditer(re.escape(selected["quote"]), excerpt)]
    selected["occurrence"] = positions.index(selected["char_offset"])
    return {**selected, "source_ref": row.value_ref, "resolution_method": "exact_numeric_match",
            "text_offset": sum(len(line) + 1 for line in lines[:selected["start_line"] - 1]) + selected["char_offset"]}


def period_text(row):
    if row.period_kind == "annual":
        return str(row.period_end.year)
    if row.period_kind == "instant":
        return row.period_end.isoformat()
    return f"{row.period_kind}:{row.period_start.isoformat()}/{row.period_end.isoformat()}"


def amount_locations(row, span, blocks, unit=None):
    if not span or span.block_id not in blocks:
        return []
    lines = blocks[span.block_id]["text"].splitlines()
    locations = []
    for number, line in enumerate(lines, 1):
        anchor = {"block_id": span.block_id, "start_line": number, "end_line": number,
                  "quote": line, "char_offset": 0}
        try:
            resolve_amount(row.model_copy(update={"value_segments": [], "value_occurrence": None}), {row.value_ref: anchor}, blocks, unit)
        except ValueError as exc:
            if not str(exc).startswith("AMOUNT_AMBIGUOUS"):
                continue
        locations.append({"block_id": span.block_id, "start_line": number, "text": line,
                          "context": [{"line": offset + 1, "text": lines[offset]} for offset in range(max(0, number - 2), min(len(lines), number + 1))]})
        if len(locations) == 4:
            break
    return locations


def identity(fact):
    return (mapped_financial_metric(fact) or fact.standard_metric, _period(fact.period) or fact.period,
            fact.scope, fact.role, fact.peer_ticker or fact.peer_name, fact.multiple_basis, fact.denominator_period_end)


def same_observation(previous, current, proof):
    if identity(previous) == identity(current):
        return True
    old_proof = previous.verification.get("reading_proof")
    if old_proof:
        old_anchor = old_proof.get("resolved_value", old_proof["anchors"][old_proof["row"]["value_ref"]])
        new_anchor = proof.get("resolved_value", proof["anchors"][proof["row"]["value_ref"]])
        source_keys = ("block_id", "start_line", "end_line", "char_offset", "quote", "block_sha256")
        return (all(old_anchor[key] == new_anchor[key] for key in source_keys)
                and all(old_proof["basis"][key] == proof["basis"][key] for key in ("entity_name", "entity_ticker"))
                and previous.role == current.role)
    return (identity(previous)[1:] == identity(current)[1:] and previous.metric == current.metric
            and previous.block_id == current.block_id and previous.normalized_value == current.normalized_value)


def model_issues(fact, row, basis):
    issues = []
    target = fact.standard_metric
    raw_metric = re.fullmatch(r"raw\.[a-z][a-z0-9_]{0,79}", target) is not None
    if fact.unit in {"元", "千元", "万元", "百万元", "亿元"} and basis.currency != "CNY":
        issues.append("MODEL_CURRENCY: 金额须有原文币种依据；当前计算器仅接收CNY，不能默认为人民币或自动换汇")
    if fact.role == "comparable":
        if target in {"pe", "ps", "ev_ebitda"}:
            if fact.multiple_basis != "FY" or not row.denominator_period_end or not row.denominator_refs:
                issues.append("MODEL_MULTIPLE: 直接可比倍数须有FY分母年度及引用；TTM或预测倍数不能替代年度口径")
            elif (row.denominator_period_end.month, row.denominator_period_end.day) != (12, 31) or row.denominator_period_end >= row.period_end:
                issues.append("MODEL_MULTIPLE_PERIOD: FY分母须是定价日前的完整日历年度")
            if fact.unit != "ratio":
                issues.append("MODEL_DIMENSION: 可比倍数使用ratio，不使用货币单位")
            if row.period_kind != "instant":
                issues.append("MODEL_PERIOD_KIND: 可比倍数必须标记确切定价时点")
        elif target in {"market_cap", "revenue", "net_income_parent"}:
            if fact.unit not in {"元", "千元", "万元", "百万元", "亿元"}:
                issues.append("MODEL_DIMENSION: 可比基础数据须为金额，不能以倍数、股数代替")
            if target == "market_cap":
                if row.period_kind != "instant" or basis.scope != "issuer":
                    issues.append("MODEL_MARKET_CAP: 使用发行人全部普通股总市值及确切定价日，不使用流通市值或单一股份类别市值")
            elif row.period_kind != "annual" or basis.scope != "consolidated":
                issues.append("MODEL_PEER_ANNUAL: 可比分母使用完整年度合并收入/归母净利润，不使用季度、TTM、母公司利润或预测值")
        if target in {"pe", "ps", "ev_ebitda", "market_cap", "revenue", "net_income_parent"}:
            return issues
        if basis.scope != "consolidated":
            issues.append("MODEL_SCOPE: 可比经营分母与资本桥接必须保持合并口径")
        if row.period_kind not in {"annual", "instant"}:
            issues.append("MODEL_PEER_ANNUAL: 可比流量使用完整年度，桥接余额使用独立时点，不使用季度或TTM替代年度")
    if target not in METRIC_ALIASES and not raw_metric:
        issues.append("MODEL_METRIC_UNKNOWN: 原文观察已保存，但此科目不在当前计算器字段字典内")
        return issues
    expected_units = ({"股", "千股", "万股", "百万股", "亿股"} if target in {"common_shares", "diluted_shares"}
                      else {"%", "ratio"} if target in {"tax_rate", "ebit_margin"}
                      else {"元", "千元", "万元", "百万元", "亿元"})
    if fact.unit not in expected_units:
        issues.append("MODEL_DIMENSION: 股数、金额、比率维度不能互换")
    if not raw_metric and (target in STOCK_METRICS and row.period_kind != "instant" or target not in STOCK_METRICS and row.period_kind == "instant"):
        issues.append("MODEL_PERIOD_KIND: " + (
            f"{target}为时点存量；用period_kind=instant，省略period_start，period_end填写原文真实截止日并引用依据；年报标题不能把股数/余额变成annual。用replaces更正，复核不能修复字段类型。"
            if target in STOCK_METRICS else
            f"{target}为期间流量；按原文填写annual/interim/ttm及真实起止日，不使用instant。用replaces更正，不扩大季度为全年。"))
    if raw_metric and row.period_kind not in {"annual", "instant"}:
        issues.append("MODEL_PERIOD_KIND: 原始计算科目须明确完整年度流量或时点存量，季度或TTM不拼入年度模型。")
    if target in {"common_shares", "diluted_shares"} and fact.scope != "issuer":
        issues.append("MODEL_SHARE_SCOPE: 股数需发行人口径和独立截止日，不以面值金额代替；若原文确为发行人股份总数，用basis.scope=issuer及真实scope_refs重新提交并replaces，不能用corroborate_facts清除此口径错误")
    elif fact.scope == "issuer" and target not in {"common_shares", "diluted_shares"}:
        issues.append("MODEL_SCOPE: 非股数财务字段不能使用issuer口径")
    if fact.semantic_role == "unknown" and target not in ROLE_INDEPENDENT_METRICS:
        issues.append("MODEL_ROLE: 尚未明确科目的经济角色")
    mapping_issue = financial_mapping_issue(fact.model_copy(update={"role": "historical"}))
    if mapping_issue:
        issues.append("MODEL_MAPPING: " + mapping_issue)
    treatments = (fact.ebit_treatment, fact.fcff_treatment, fact.equity_bridge_treatment)
    if target in DEBT_COMPONENTS and treatments != ("exclude", "exclude", "include"):
        issues.append("MODEL_DEBT: 债务只进入权益桥接，不直接作为EBIT/FCFF")
    if target == "interest_expense" and (fact.semantic_role != "financing" or fact.ebit_treatment != "include"):
        issues.append("MODEL_INTEREST: EBIT加回项必须明确是融资利息")
    if fact.semantic_role == "financial_subsidiary" and "include" in treatments:
        issues.append("MODEL_FINANCIAL_SCOPE: 当前通用计算器不直接合入金融子公司业务")
    if target == "cash_paid_for_ppe_intangibles" and (fact.semantic_role != "investing" or treatments != ("exclude", "include", "exclude")):
        issues.append("MODEL_CAPEX: 购建长期资产现金支出须作为投资活动FCFF推导输入")
    if target in {"cash_and_non_operating_assets", "trading_financial_assets"} and (fact.semantic_role != "non_operating" or treatments != ("exclude", "exclude", "include")):
        issues.append("MODEL_CASH: 现金类桥接调整需明确可用范围和非经营性处理")
    return issues


def extract_observations(runtime, args):
    from valuationagent.application.financial_evidence import PUBLIC_REVIEW, source_assessment

    store, session = runtime.service.store, runtime.session
    document = source_document(session, args.file_id)
    meta, _ = source_bytes(store, session, args.file_id)
    blocks = runtime.service._blocks(session)
    results = []
    for index, row in enumerate(args.rows):
        runtime.service._check_execution()
        anchors = {}
        try:
            basis = row.basis or args.basis
            groups = {"entity": basis.entity_refs, "unit": basis.unit_refs, "scope": basis.scope_refs,
                      "period": row.period_refs, "label": row.label_refs, "revision": row.revision_refs,
                      "publication": basis.source_publication_refs,
                      "denominator": row.denominator_refs,
                      "value": [row.value_ref], "numeric_segments": row.value_segments}
            keys = list(dict.fromkeys(key for references in groups.values() for key in references))
            missing = [key for key in keys if key not in args.anchors]
            if missing:
                raise ValueError(f"ANCHOR_MISSING: 未定义的引用ID={missing}；有效ID={list(args.anchors)}。value_ref和其他refs必须填写anchors的键名，不是数值或引文。已有原文，无需换文件。")
            anchors = {key: resolve_span(args.anchors[key], blocks, args.file_id) for key in keys}
            value_anchor = resolve_amount(row, anchors, blocks, basis.unit)
            amount = scalar(value_anchor["quote"])
            if (row.raw_value.rstrip().endswith(("%", "％")) or value_anchor["quote"].rstrip().endswith(("%", "％"))) and basis.unit != "%":
                raise ValueError("AMOUNT_UNIT: 显式百分数不能当作金额或普通小数倍数")
            if row.role == "historical":
                if not session.draft.company and not session.draft.ticker:
                    raise ValueError("ENTITY_TASK_MISSING: 先update_task保存已读原文的公司名称/代码；仅阅读附件时valuation_requested=false，不会启动估值，然后重提这些观察。")
                if session.draft.ticker and basis.entity_ticker != session.draft.ticker:
                    raise ValueError(f"ENTITY_TARGET_MISMATCH: historical行的basis.entity_ticker={basis.entity_ticker!r}与任务代码{session.draft.ticker!r}不一致；若正在提取可比公司，保持其真实主体并显式设置row.role=comparable。若并非可比取证则核对来源，不改写原文主体或任务来绕过校验。")
                if not session.draft.ticker and basis.entity_name != session.draft.company:
                    raise ValueError("ENTITY_TARGET_MISMATCH: 主体与当前任务不一致")
            elif not basis.entity_ticker or not basis.entity_name:
                raise ValueError("PEER_IDENTITY: 可比公司必须有独立名称和证券代码")
            block = blocks[value_anchor["block_id"]]
            location = block.get("location", {})
            published = location.get("published_at")
            if not published and document.acquisition_ref:
                published = store.research_source_location(session.session_id, document.acquisition_ref).get("published_at")
            try:
                metadata_date = date.fromisoformat(str(published)[:10]) if published else None
            except ValueError:
                metadata_date = None
            resolved_publication = metadata_date or basis.source_published_at
            publication = {"resolved_date": resolved_publication.isoformat() if resolved_publication else None,
                           "metadata_date": metadata_date.isoformat() if metadata_date else None,
                           "llm_claim": basis.source_published_at.isoformat() if basis.source_published_at else None}
            proof = {"schema": "llm-observation-v1", "file_id": args.file_id, "source_sha256": meta["sha256"],
                     "source_identity": source_identity(document),
                     "publication": publication,
                     "basis": basis.model_dump(mode="json"), "row": row.model_dump(mode="json"), "groups": groups, "anchors": anchors,
                     "resolved_value": value_anchor}
            proof["proof_id"] = "proof_" + digest(proof)[:32]
            fact_id = "fact_" + digest(proof)[:32]
            previous = next((fact for fact in session.facts if fact.fact_id == fact_id), None)
            if previous:
                results.append({"row": index, "fact_id": fact_id, "status": previous.status, "warnings": previous.warnings,
                                "duplicate": True, "next_action": observation_next_action(previous)})
                continue
            fact = FactCandidate(fact_id=fact_id, metric=row.metric, standard_metric=row.standard_metric,
                raw_value=row.raw_value, normalized_value=str(amount * Decimal(FACTORS[basis.unit])), unit=basis.unit,
                period=period_text(row), scope=basis.scope, role=row.role, multiple_basis=row.multiple_basis,
                denominator_period_end=row.denominator_period_end,
                peer_ticker=basis.entity_ticker if row.role == "comparable" else "", peer_name=basis.entity_name if row.role == "comparable" else "",
                block_id=value_anchor["block_id"], quote=value_anchor["quote"],
                context_block_ids=list(dict.fromkeys(anchor["block_id"] for anchor in anchors.values()))[:8],
                semantic_role=row.semantic_role, ebit_treatment=row.ebit_treatment, fcff_treatment=row.fcff_treatment,
                equity_bridge_treatment=row.equity_bridge_treatment, mapping_rationale=row.rationale,
                source_location=block.get("location", {}), source_sha256=meta["sha256"],
                source_url=document.source_url or block.get("location", {}).get("source_url", ""))
            existing = {item.fact_id: item for item in session.facts}
            if any(key not in existing or not same_observation(existing[key], fact, proof) for key in row.replaces):
                raise ValueError("REPLACEMENT_SCOPE: 更正须同主体/科目/期间/口径，或引用完全相同的原文数值位置来修正原解释；不能撤销无关事实")
            fact.warnings = [REVIEW_PENDING, *model_issues(fact, row, basis)]
            fact.published_at = resolved_publication
            if basis.source_published_at:
                fact.warnings.append(PUBLICATION_PENDING)
                if metadata_date and metadata_date != basis.source_published_at:
                    fact.warnings.append("SOURCE_PUBLICATION_CONFLICT: 原文日期主张与已取得来源元数据冲突，不能覆盖日期绕过信息截止日")
            cutoff = session.information_cutoff_date or session.draft.valuation_date
            if cutoff and (row.period_end > cutoff or fact.published_at and fact.published_at > cutoff):
                fact.warnings.append("来源期间或披露日期晚于信息截止日")
            if document.provenance_type != "user_upload" and not fact.published_at:
                fact.warnings.append("披露日期未知，不能用抓取日替代")
            if document.authority_tier == "C":
                fact.warnings.append(PUBLIC_REVIEW)
            elif document.authority_tier not in {"A", "B"}:
                fact.warnings.append("SOURCE_GRADE: 当前来源仅可作线索，不作为计算输入")
            fact.verification = {"reading_proof": proof,
                "observation": {"status": "verified", "issues": [], "meaning": "仅验证原文片段位置、字符一致和确定性换算；主体、期间、单位与会计口径由LLM解释，尚待语义复核。"},
                "semantic_mapping": {"resolution_method": "llm_document_interpretation", "standard_metric": row.standard_metric,
                    "semantic_role": row.semantic_role, "rationale": row.rationale, "review_status": "pending",
                    "ebit_treatment": row.ebit_treatment, "fcff_treatment": row.fcff_treatment,
                    "equity_bridge_treatment": row.equity_bridge_treatment},
                "semantic_review": {"status": "pending", "uncertainties": row.uncertainties}}
            fact.verification["source_assessment"] = source_assessment(fact, document, cutoff)
            session.facts.append(fact)
            results.append({"row": index, "fact_id": fact_id, "status": fact.status, "warnings": fact.warnings,
                            "next_action": observation_next_action(fact)})
        except ValueError as exc:
            span = args.anchors.get(row.value_ref)
            results.append({"row": index, "error": str(exc)[:800], "raw_value": row.raw_value,
                            "value_ref": row.value_ref, "source_quote": anchors.get(row.value_ref, {}).get("quote", span.quote if span else None),
                            "value_locations": amount_locations(row, span, blocks, basis.unit) if str(exc).startswith(("AMOUNT_", "ANCHOR_")) else [],
                            "repair": observation_repair(str(exc))})
    saved = sum("fact_id" in result and not result.get("duplicate") for result in results)
    duplicates = sum(bool(result.get("duplicate")) for result in results)
    usable_duplicate = any(result.get("duplicate") and result.get("status") != "rejected" for result in results)
    ok = bool(saved or usable_duplicate)
    return {"ok": ok, "rows": results, "saved_count": saved, "duplicate_count": duplicates,
            "error": {"code": "OBSERVATION_BINDING", "message": "未保存新观察；已撤回解释不能复活，重复提交不算进展。按每行next_action/error/repair修正，不把主体/角色错误当作缺文件反复下载。"} if not ok else None,
            "instruction": "定位成功不等于语义核验。按每行next_action推进：MODEL类问题须先更正解释，不能反复复核清除；无此类问题再prepare_observation_review/review_observations。不凭高置信分自动入模。"}


def observation_next_action(fact):
    if fact.status == "rejected":
        return {"tool": "inspect_context", "arguments": {"section": "facts", "query": "proposed"},
                "instruction": "该观察已撤回；不要重提相同解释或复核它。检查现有有效候选，需要更正时依据原文提交不同解释。"}
    model_errors = [warning for warning in fact.warnings if warning.startswith("MODEL_")]
    if model_errors:
        return {"tool": "extract_observations", "replaces": [fact.fact_id], "issues": model_errors,
                "instruction": "先根据原文更正上述字段类型/单位/模型含义，再重新提交；不修改数值来迁就模型，review_observations不能清除MODEL约束。确实不适用当前计算器时保留为资料，不重复尝试入模。"}
    if fact.status == "confirmed" and not fact.warnings:
        return {"tool": "inspect_requirements", "instruction": "该项已准入，不重复提取/复核；继续其他方法缺口。"}
    if fact.verification.get("semantic_review", {}).get("status") in {"pending", "needs_evidence"}:
        return {"tool": "prepare_observation_review", "fact_ids": [fact.fact_id],
                "instruction": "先读取原文复核包；已有语义矛盾须补证并用replaces更正，不能照抄supported。"}
    return {"tool": "inspect_requirements", "instruction": "语义复核不能清除来源、时效或冲突约束；按具体警告补证，不反复复核同一包。"}


def observation_repair(error):
    if error.startswith("MODEL_"):
        return "按具体MODEL错误核对字段类型、单位、时点与模型范围；已有原文时用extract_observations及replaces更正解释。不要换阅读视图来修复参数类型，不把复核supported当作约束豁免。"
    if error.startswith(("ENTITY_TARGET_MISMATCH", "PEER_IDENTITY")):
        return "先区分目标公司与可比公司。可比取证须设置row.role=comparable，basis保留原文公司的名称/代码；目标历史数据用historical。不得改写主体来通过校验，此错误不要求重读金额或下载。"
    if error.startswith("ENTITY_TASK_MISSING"):
        return "先结束单文件任务并返回主循环确认研究主体；仅阅读时保持valuation_requested=false，再用已有片段提交。不自动将当前文件公司设为估值目标。"
    if error.startswith("REPLACEMENT_SCOPE"):
        return "核对replaces中的旧事实是否确实属于同一观察；修正年度/单位/口径时需引用完全相同的原文数值位置。不能撤销无关事实，也不能省略原文依据。"
    return "value_ref须填anchors键名，如rev_row。anchors只须提供block_id、start_line/end_line，可不抄quote。raw_value填选定完整数值；重复金额用value_occurrence，只有粘连列用value_segments。不要重复下载。"


def _select(session, fact_id):
    fact = next((fact for fact in session.facts if fact.fact_id == fact_id), None)
    if fact and fact.status == "rejected":
        raise ValueError(f"OBSERVATION_REJECTED: {fact_id}已经撤回，不能复核或复活；用inspect_context(section=facts)选择仍有效的观察，需要更正时重新extract_observations。")
    if not fact or not fact.verification.get("reading_proof"):
        raise ValueError(f"OBSERVATION_NOT_FOUND: {fact_id}没有可用的原文观察；先inspect_context(section=facts)核对真实ID，必要时extract_observations提交，不编造fact_id或packet_id。")
    return fact


def review_packet(runtime, fact):
    proof = fact.verification["reading_proof"]
    body = {key: value for key, value in proof.items() if key != "proof_id"}
    if "proof_" + digest(body)[:32] != proof["proof_id"]:
        raise ValueError("PROOF_CHANGED: 已保存解释记录完整性校验失败")
    if source_identity(source_document(runtime.session, proof["file_id"])) != proof["source_identity"]:
        raise ValueError("SOURCE_CHANGED: 来源等级或取得方式改变，需重新提取并复核")
    row = Observation.model_validate(proof["row"])
    basis = ReadingBasis.model_validate(proof["basis"])
    expected = {"metric": row.metric, "standard_metric": row.standard_metric, "raw_value": row.raw_value,
                "unit": basis.unit, "period": period_text(row), "scope": basis.scope, "role": row.role,
                "semantic_role": row.semantic_role, "ebit_treatment": row.ebit_treatment,
                "fcff_treatment": row.fcff_treatment, "equity_bridge_treatment": row.equity_bridge_treatment,
                "multiple_basis": row.multiple_basis,
                "denominator_period_end": row.denominator_period_end,
                "peer_ticker": basis.entity_ticker if row.role == "comparable" else "",
                "peer_name": basis.entity_name if row.role == "comparable" else "",
                "normalized_value": str(observation_amount(row.raw_value, basis.unit) * Decimal(FACTORS[basis.unit]))}
    if "publication" in proof:
        published = proof["publication"]["resolved_date"]
        expected["published_at"] = date.fromisoformat(published) if published else None
    if any(getattr(fact, key) != value for key, value in expected.items()):
        raise ValueError("OBSERVATION_CHANGED: 字段与保存的解释不一致，不能复用核验记录")
    meta, _ = source_bytes(runtime.service.store, runtime.session, proof["file_id"])
    if meta["sha256"] != proof["source_sha256"]:
        raise ValueError("SOURCE_CHANGED: 原文文件已改变，不能沿用旧复核包")
    blocks = runtime.service._blocks(runtime.session)
    context = {}
    for anchor in proof["anchors"].values():
        block = blocks.get(anchor["block_id"])
        if not block or hashlib.sha256(block["text"].encode()).hexdigest() != anchor["block_sha256"] or block.get("location", {}) != anchor["location"]:
            raise ValueError("SOURCE_CHANGED: 证据块已改变，需重新提取")
        if anchor["block_id"] not in context:
            context[anchor["block_id"]] = {"location": block.get("location", {}), "text": block["text"]}
    if len(canonical(context)) > 24000:
        raise ValueError("REVIEW_CONTEXT_LIMIT: 单项观察上下文过大，请用更精确的证据片段拆分")
    task = {"company": runtime.session.draft.company, "ticker": runtime.session.draft.ticker,
            "cutoff": str(runtime.session.information_cutoff_date or runtime.session.draft.valuation_date or "")}
    return {"fact_id": fact.fact_id, "packet_id": "review_" + digest([proof["proof_id"], task, "period-readback-v1"])[:32],
            "task": task, "interpretation": {"basis": proof["basis"], "row": proof["row"], "groups": proof["groups"]},
            "publication": proof.get("publication"),
            "requires_period_readback": row.period_kind == "instant",
            "required_checks": ["entity", "amount", "period", "unit", "scope", "mapping"] + (["publication"] if basis.source_published_at else []),
            "anchors": proof["anchors"], "resolved_value": proof.get("resolved_value"), "original_context": context,
            "instruction": "对照原文逐项检查主体、金额边界、期间/重述列、单位、口径和字段映射；有value_segments时尤其检查分列依据，字符覆盖并不证明列边界正确。不要照抄上次判断，歧义标ambiguous；这是同一LLM复核，不是独立审计。"}


def prepare_reviews(runtime, args):
    packets, context = [], {}
    selected = [_select(runtime.session, fact_id) for fact_id in dict.fromkeys(args.fact_ids)]
    for fact in selected:
        runtime.service._check_execution()
        packet = review_packet(runtime, fact)
        originals = packet.pop("original_context")
        packet["context_ids"] = list(originals)
        combined = {**context, **originals}
        size = len(canonical({"packets": [*packets, packet], "original_context": combined}))
        if size > 48000 and not packets:
            raise ValueError("REVIEW_PACKET_LIMIT: 单项复核包超过限额；缩小证据范围并用replaces修正，不重复请求同一包")
        if size > 48000:
            break
        context = combined
        packets.append(packet)
        fact.verification["semantic_review"]["issued_packet_id"] = packet["packet_id"]
    return {"packets": packets, "original_context": context,
            "instruction": "按每项context_ids读取共享original_context，同一原文仅传一次；逐项复核，不把共享原文当作独立来源。",
            "remaining_fact_ids": [fact_id for fact_id in args.fact_ids if fact_id not in {packet["fact_id"] for packet in packets}],
            "untrusted_source_data": True}


def review_observations(runtime, args):
    from valuationagent.application.financial_evidence import source_assessment

    prepared = []
    identities = [review.fact_id for review in args.reviews]
    if len(set(identities)) != len(identities):
        raise ValueError("REVIEW_DUPLICATE: 同一批次每个fact_id只复核一次，不允许后项覆盖前项判断")
    for review in args.reviews:
        runtime.service._check_execution()
        fact = _select(runtime.session, review.fact_id)
        packet = review_packet(runtime, fact)
        semantic = fact.verification["semantic_review"]
        if review.packet_id != packet["packet_id"] or semantic.get("issued_packet_id") != review.packet_id:
            raise ValueError("REVIEW_PACKET_REQUIRED: 先prepare_observation_review读取当前原文复核包；任务或来源变化后旧包失效")
        if "publication" in packet["required_checks"] and review.checks.publication is None:
            raise ValueError("PUBLICATION_REVIEW_REQUIRED: 本项含LLM提出的披露日期，须单独核查publication；不能用期间检查代替")
        if packet["requires_period_readback"]:
            if "source_period_end" not in review.model_fields_set:
                raise ValueError(PERIOD_READBACK_REQUIRED)
            review = review.model_copy(deep=True)
            claimed_date = packet["interpretation"]["row"]["period_end"]
            if review.source_period_end is None:
                review.checks.period = "ambiguous"
            elif review.source_period_end.isoformat() != claimed_date:
                review.checks.period = "contradicted"
        supported = all(value == "supported" for value in review.checks.model_dump(exclude_none=True).values())
        uncertainties = fact.verification["reading_proof"]["row"]["uncertainties"]
        if supported and uncertainties:
            raise ValueError("REVIEW_UNRESOLVED: 提取时声明的歧义尚未消除；补证后用replaces提交修正观察，不可用复核按钮清除")
        prepared.append((review, fact, packet, supported))
    results = []
    for review, fact, packet, supported in prepared:
        semantic = fact.verification["semantic_review"]
        cell_conflicts = period_cell_conflicts(runtime.session.facts).get(fact.fact_id, [])
        if supported and fact.status == "confirmed" and not fact.warnings and not cell_conflicts and semantic.get("status") == "supported" and semantic.get("reviewed_packet_id") == packet["packet_id"]:
            results.append({"fact_id": fact.fact_id, "status": fact.status, "warnings": fact.warnings, "duplicate": True})
            continue
        semantic.update(status="supported" if supported else "needs_evidence", checks=review.checks.model_dump(exclude_none=True),
                        rationale=review.rationale, reviewer="workspace_llm", independent_audit=False,
                        reviewed_packet_id=packet["packet_id"])
        if packet["requires_period_readback"]:
            semantic.update(source_period_end=review.source_period_end.isoformat() if review.source_period_end else None,
                            period_readback_version=1)
        fact.verification["semantic_mapping"]["review_status"] = semantic["status"]
        if supported:
            fact.warnings = [issue for issue in fact.warnings if issue not in {REVIEW_PENDING, REVIEW_UNRESOLVED, PERIOD_CELL_CONFLICT, PERIOD_READBACK_REQUIRED, PUBLICATION_PENDING}]
            if cell_conflicts:
                for target in runtime.session.facts:
                    if target.fact_id in [fact.fact_id, *cell_conflicts]:
                        target.status = "proposed"
                        target.warnings = list(dict.fromkeys([*target.warnings, PERIOD_CELL_CONFLICT]))
                        target.verification.setdefault("source_assessment", {}).update(admission="blocked", consistency="period_collision")
            row = Observation.model_validate(fact.verification["reading_proof"]["row"])
            basis = ReadingBasis.model_validate(fact.verification["reading_proof"]["basis"])
            cutoff = runtime.session.information_cutoff_date or runtime.session.draft.valuation_date
            if cutoff and (row.period_end > cutoff or fact.published_at and fact.published_at > cutoff):
                fact.warnings.append("来源期间或披露日期晚于信息截止日")
            if fact.role == "historical" and (runtime.session.draft.ticker and basis.entity_ticker != runtime.session.draft.ticker or not runtime.session.draft.ticker and basis.entity_name != runtime.session.draft.company):
                fact.warnings.append("ENTITY_TARGET_MISMATCH: 研究主体改变，旧观察不能用于新公司")
            fact.warnings = list(dict.fromkeys(fact.warnings))
            replaced_ids = fact.verification["reading_proof"]["row"]["replaces"]
            conflicts = [other for other in runtime.session.facts if other.fact_id != fact.fact_id and other.status != "rejected"
                         and other.fact_id not in replaced_ids and identity(other) == identity(fact)
                         and (not other.warnings or other.warnings == [FACT_CONFLICT]) and other.normalized_value is not None
                         and Decimal(other.normalized_value) != Decimal(fact.normalized_value)]
            if conflicts:
                for target in [fact, *conflicts]:
                    target.status = "proposed"
                    target.warnings = list(dict.fromkeys([*target.warnings, FACT_CONFLICT]))
                    target.verification.setdefault("source_assessment", {}).update(admission="blocked", consistency="conflict")
            if not fact.warnings:
                fact.status = "confirmed"
                for previous in runtime.session.facts:
                    if previous.fact_id in replaced_ids:
                        previous.status = "rejected"
                if replaced_ids:
                    runtime.session.staged_supersessions[fact.fact_id] = replaced_ids
            else:
                fact.status = "proposed"
        else:
            fact.status = "proposed"
            fact.warnings = list(dict.fromkeys([issue for issue in fact.warnings if issue != REVIEW_PENDING] + [REVIEW_UNRESOLVED]))
        retire_contradicted_entities([fact])
        document = source_document(runtime.session, fact.verification["reading_proof"]["file_id"])
        fact.verification["source_assessment"] = source_assessment(fact, document, runtime.session.information_cutoff_date or runtime.session.draft.valuation_date)
        results.append({"fact_id": fact.fact_id, "status": fact.status, "warnings": fact.warnings,
                        **({"period_readback": {"source_period_end": semantic.get("source_period_end"),
                                                "candidate_period_end": packet["interpretation"]["row"]["period_end"],
                                                "instruction": "日期不一致时按原文更正观察并replaces；不改原文日期来保留旧候选。"}} if packet["requires_period_readback"] else {}),
                        "semantic_review": semantic["status"], "next_action": observation_next_action(fact)})
    return {"reviews": results, "instruction": "只有定位通过、LLM逐维复核及模型/来源约束均通过才为confirmed；这不等于独立审计或用户批准。"}
