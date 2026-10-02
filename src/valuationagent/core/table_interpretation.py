"""Source-grounded table structures proposed by the workspace LLM."""
import hashlib
import re
from typing import Literal

from pydantic import Field

from valuationagent.core.evidence import ANNUAL_HEADER_TOKEN, STATEMENT_TITLE, compact
from valuationagent.core.tools import canonical
from valuationagent.schemas.models import ApiModel


class SourceAnchor(ApiModel):
    block_id: str
    start_line: int = Field(ge=1)
    end_line: int | None = Field(default=None, ge=1)


class InterpretedColumn(ApiModel):
    period: str = Field(pattern=r"^(19|20)\d{2}$")
    label: str = Field(min_length=4, max_length=80, description="年度表头中逐字复制的完整日期/年份标签，按原文从左到右排列。")


class TableInterpretation(ApiModel):
    file_id: str
    data_block_ids: list[str] = Field(min_length=1, max_length=24)
    data_ranges: list[SourceAnchor] = Field(default_factory=list, max_length=24, description="块内含多张表时，逐块指定数据行范围；填写时须覆盖全部data_block_ids。")
    header: SourceAnchor
    columns: list[InterpretedColumn] = Field(min_length=1, max_length=6)
    scope: Literal["consolidated", "parent"]
    scope_evidence: SourceAnchor
    unit: Literal["元", "千元", "万元", "百万元", "亿元", "股", "千股", "万股", "百万股", "亿股", "%", "ratio"]
    unit_evidence: SourceAnchor
    unit_extent: Literal["table", "financial_statements"] = "table"
    rationale: str = Field(min_length=20, max_length=1200)


def source_anchor(anchor, blocks):
    block = blocks.get(anchor.block_id)
    if not block:
        raise ValueError("TABLE_SOURCE_MISSING: 原文块不存在；先读取原文，不是增加候选。")
    lines = block["text"].splitlines()
    end = anchor.end_line or anchor.start_line
    if not 1 <= anchor.start_line <= end <= len(lines) or end - anchor.start_line > 5:
        raise ValueError("TABLE_ANCHOR_INVALID: 表头证据须定位原始块内连续1至6行。")
    return {"block_id": anchor.block_id, "start_line": anchor.start_line, "end_line": end,
            "text": "\n".join(lines[anchor.start_line - 1:end]), "location": block.get("location", {})}


def compile_table(spec, blocks):
    anchors = {name: source_anchor(getattr(spec, name), blocks)
               for name in ("header", "scope_evidence", "unit_evidence")}
    ids = set(spec.data_block_ids) | {entry["block_id"] for entry in anchors.values()}
    if any(key not in blocks or key.rsplit(":", 1)[0] != spec.file_id for key in ids):
        raise ValueError("TABLE_SOURCE_MISMATCH: 表格及表头必须属于同一已读取文件。")
    ordered = [block for block in blocks.values() if block["block_id"].rsplit(":", 1)[0] == spec.file_id]
    ordered.sort(key=lambda block: block.get("location", {}).get("page", 0))
    positions = {block["block_id"]: index for index, block in enumerate(ordered)}
    header_position = positions[spec.header.block_id]
    sheet = blocks[spec.header.block_id].get("location", {}).get("sheet")
    if sheet and any(blocks[key].get("location", {}).get("sheet") not in {None, sheet} for key in ids):
        raise ValueError("TABLE_SHEET_BOUNDARY: 不同工作表的表头不能互相借用。")
    if any(positions[key] < header_position for key in spec.data_block_ids):
        raise ValueError("TABLE_FUTURE_HEADER: 不能用后文表头解释前文数值。")
    for name in ("scope_evidence", "unit_evidence"):
        anchor = anchors[name]
        if (positions[anchor["block_id"]], anchor["end_line"]) > (header_position, anchors["header"]["start_line"]):
            raise ValueError("TABLE_FUTURE_HEADER: 单位和口径声明须在本表年度列之前。")
    scope_text = compact(anchors["scope_evidence"]["text"])
    expected_scope, opposite_scope = ("合并", "母公司") if spec.scope == "consolidated" else ("母公司", "合并")
    if expected_scope not in scope_text or opposite_scope in scope_text:
        raise ValueError("TABLE_SCOPE_AMBIGUOUS: 口径证据不明确；混合合并/母公司表不能使用单口径契约。")
    unit_text = compact(anchors["unit_evidence"]["text"])
    units = re.findall(r"百万元|亿元|万元|千元|元|百万股|亿股|万股|千股|股|%|ratio", unit_text)
    if set(units) != {spec.unit}:
        raise ValueError("TABLE_UNIT_CONFLICT: 单位必须由完整声明支持，不能从多个单位中挑选。")
    if spec.unit_extent == "financial_statements" and not any(word in unit_text for word in ("报表", "财务", "除特别说明", "除另有说明")):
        raise ValueError("TABLE_UNIT_EXTENT: 章节级单位继承需要财务报表统一计量声明。")
    if spec.unit_extent == "table" and (positions[spec.unit_evidence.block_id], spec.unit_evidence.start_line) < (positions[spec.scope_evidence.block_id], spec.scope_evidence.start_line):
        adjacent = (spec.unit_evidence.block_id == spec.scope_evidence.block_id
                    and not "".join(blocks[spec.unit_evidence.block_id]["text"].splitlines()[anchors["unit_evidence"]["end_line"]:spec.scope_evidence.start_line - 1]).strip())
        if not adjacent:
            raise ValueError("TABLE_UNIT_EXTENT: 远于本表标题的单位不能默认继承；统一声明须明确financial_statements范围。")
    header = anchors["header"]["text"]
    remainder = re.sub(ANNUAL_HEADER_TOKEN, "", header)
    if re.search(r"\d\s*月|半年|季度|[-/]\d{1,2}[-/]\d{1,2}", remainder):
        raise ValueError("TABLE_PERIOD_CONFLICT: 原始表头包含非年度日期，不能截取年份冒充全年。")
    previous_end = -1
    for column in spec.columns:
        if header.count(column.label) != 1:
            raise ValueError("TABLE_COLUMN_AMBIGUOUS: 每个年度标签须唯一定位；不得省略日期或重排比较列。")
        match = re.fullmatch(ANNUAL_HEADER_TOKEN, column.label.strip())
        if not match or match[1] != column.period:
            raise ValueError("TABLE_PERIOD_CONFLICT: 年度标签与期间不符；季度或中报不能充作完整年度。")
        start = header.index(column.label)
        if start < previous_end:
            raise ValueError("TABLE_COLUMN_ORDER: 年度列必须遵循原文顺序。")
        previous_end = start + len(column.label)
    source_years = re.findall(ANNUAL_HEADER_TOKEN, header)
    if source_years != [column.period for column in spec.columns] or len(set(source_years)) != len(source_years):
        raise ValueError("TABLE_COLUMN_COVERAGE: 必须保留所有年度列；重复年度/重述前后多列须换明确来源，不能挑列。")
    last_position = max(positions[key] for key in spec.data_block_ids)
    ranges = {}
    for anchor in spec.data_ranges:
        if anchor.block_id not in spec.data_block_ids or anchor.block_id in ranges:
            raise ValueError("TABLE_ROW_RANGE: 数据范围必须唯一对应已声明块。")
        line_count = len(blocks[anchor.block_id]["text"].splitlines())
        end = anchor.end_line or anchor.start_line
        if not 1 <= anchor.start_line <= end <= line_count:
            raise ValueError("TABLE_ROW_RANGE: 数据行范围超出原文。")
        ranges[anchor.block_id] = [anchor.start_line, end]
    if ranges and set(ranges) != set(spec.data_block_ids):
        raise ValueError("TABLE_ROW_RANGE: 显式数据范围须覆盖全部数据块。")
    table_start = blocks[spec.header.block_id].get("location", {}).get("table")
    for index in range(header_position, last_position + 1):
        block = ordered[index]
        if table_start and block.get("location", {}).get("table") not in {None, table_start}:
            raise ValueError("TABLE_BOUNDARY: 不能跨另一张结构化表继承年度列。")
        content = block["text"]
        if index == last_position and block["block_id"] in ranges:
            content = "\n".join(content.splitlines()[:ranges[block["block_id"]][1]])
        if index == header_position:
            content = "\n".join(content.splitlines()[anchors["header"]["end_line"]:])
        if re.search(STATEMENT_TITLE, content):
            raise ValueError("TABLE_BOUNDARY: 数据范围跨越新报表标题；请为下一张表单独解释。")
        declared = re.findall(r"单位\s*(?:为\s*)?[:：]?\s*(?:人民币\s*)?(百万元|亿元|万元|千元|元|百万股|亿股|万股|千股|股)", content)
        if any(unit != spec.unit for unit in declared):
            raise ValueError("TABLE_UNIT_CONFLICT: 表内出现不同计量单位，不能继承章节默认单位。")
    scope_position = positions[spec.scope_evidence.block_id]
    scope_end = anchors["scope_evidence"]["end_line"]
    between = "\n".join(["\n".join(ordered[scope_position]["text"].splitlines()[scope_end:]),
                          *[block["text"] for block in ordered[scope_position + 1:header_position]],
                          "\n".join(ordered[header_position]["text"].splitlines()[:anchors["header"]["start_line"] - 1])]
                         if scope_position != header_position else
                         ordered[header_position]["text"].splitlines()[scope_end:anchors["header"]["start_line"] - 1])
    if re.search(STATEMENT_TITLE, between):
        raise ValueError("TABLE_SCOPE_BOUNDARY: 口径证据属于另一张报表；请定位本表标题。")
    local_units = re.findall(r"单位\s*(?:为\s*)?[:：]?\s*(?:人民币\s*)?(百万元|亿元|万元|千元|元|百万股|亿股|万股|千股|股)", between)
    if any(unit != spec.unit for unit in local_units):
        raise ValueError("TABLE_UNIT_CONFLICT: 本表局部声明覆盖了章节默认单位。")
    source_hashes = {key: hashlib.sha256(blocks[key]["text"].encode()).hexdigest() for key in sorted(ids)}
    payload = {"spec": spec.model_dump(mode="json"), "anchors": anchors, "source_hashes": source_hashes}
    return {"table_id": "table_" + hashlib.sha256(canonical(payload).encode()).hexdigest()[:24], **payload,
            "header": header, "columns": [int(column.period) for column in spec.columns],
            "scope": spec.scope, "unit": spec.unit, "data_ranges": ranges, "status": "bound",
            "limitations": ["结构由LLM解释、定位和数值由程序验证；不等于独立审计或模型准入。"]}


def get_table_binding(session, table_id, block_id, blocks):
    record = next((entry for entry in session.table_interpretations if entry["table_id"] == table_id), None)
    if not record:
        raise ValueError("TABLE_NOT_REGISTERED: 先用interpret_financial_table注册表格结构。")
    current = compile_table(TableInterpretation.model_validate(record["spec"]), blocks)
    if current["table_id"] != table_id:
        raise ValueError("TABLE_SOURCE_CHANGED: 原文快照已改变，需重新解释表格。")
    if block_id not in record["spec"]["data_block_ids"]:
        raise ValueError("TABLE_ROW_OUTSIDE: 候选行不在该表已声明的数据范围内。")
    return current
