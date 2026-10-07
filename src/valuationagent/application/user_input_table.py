from datetime import date
from typing import Literal

from pydantic import Field

from valuationagent.application.record_inputs import record_inputs
from valuationagent.application.input_workspace import FINANCIAL_FIELDS, RATIO_FIELDS, RAW_INPUT_FIELDS
from valuationagent.application.user_message import NUMBER, missing_amount_reference, selected_line
from valuationagent.schemas.inputs import RecordInputs
from valuationagent.schemas.models import ApiModel


Unit = Literal["元", "千元", "万元", "亿元", "股", "万股", "亿股", "ratio", "%"]


class UserDate(ApiModel):
    date: date
    line: int = Field(ge=1, description="支持date的日期/完整年度声明所在行号，不是金额所在行。共享日期可跨批复用同一原文行；不猜行号或把数据行当日期行。")


class UserValueRow(ApiModel):
    metric: str = Field(min_length=1,
        description="按实际含义选择目录字段：税前利润=profit_before_tax，所得税费用=income_tax_expense，折旧摊销=depreciation_amortization。未列出的原始科目用raw.小写名称并明确period_kind，例如raw.interest_expense；不要重命名为不等价的模型字段。缺失不等于0，不录入LLM推算值。",
        json_schema_extra={"anyOf": [{"enum": sorted(FINANCIAL_FIELDS | RATIO_FIELDS | RAW_INPUT_FIELDS)},
            {"pattern": r"^raw\.[a-z][a-z0-9_]{0,79}$"}]})
    amount_ref: str = Field(pattern=r"^[1-9]\d*:[1-9]\d*$", description="read_user_input返回的原数值引用，如8:1。")
    unit: Unit | None = Field(default=None, description="仅裸数单位不同于共享unit时填写。原值自带单位/百分号则程序保留该字面单位；显式覆盖仍须与原文一致。")
    replaces: list[str] = Field(default_factory=list, max_length=8)
    period_kind: Literal["annual", "instant", "unknown"] = Field(default="unknown",
        description="raw.*必须明确annual流量或instant存量；标准字段由程序定义，不推断未知日期。")


class UserInputTable(ApiModel):
    unit: Unit = Field(description="本批共享的单位，逐行可覆盖。不是原始数值的一部分，不能凭空换倍率。")
    unit_line: int | None = Field(ge=1, description="必须填写：裸数使用共享单位时填原单位行号；所有行都自带单位/百分号时填null。不要把金额行当作共享单位声明。")
    periods: list[UserDate] = Field(default_factory=list, max_length=4,
        description="明确年度日期及原文行。用户明确多个完整年度同值时可填多个；程序只扩展这些已声明年度，不猜历史。没有财务年度则留空。")
    as_of: UserDate | None = Field(default=None, description="明确的股数/资本/定价时点及其原文行；未知省略，不用当前日期冒充。")
    role: Literal["historical", "comparable"] = "historical"
    entity: str = Field(default="", max_length=100, description="目标省略；可比填用户给出的真实样本名，不编造证券代码。")
    scope: Literal["consolidated", "issuer", "assumption"] = "consolidated"
    message_id: str = Field(default="", description="默认当前用户消息；引用旧消息填真实用户message_id。")
    rows: list[UserValueRow] = Field(min_length=1, max_length=12)


def record_user_table(runtime, args):
    messages = {item.message_id: item for item in runtime.service.store.list_messages(runtime.session.session_id) if item.role == "user"}
    current = runtime.session.turn_control.message_id if runtime.session.turn_control else next(reversed(messages), "")
    message = messages.get(args.message_id or current)
    if message is None:
        raise ValueError("USER_INPUT_REF: 只能读取本工作区真实用户消息。")
    active = {row.input_id: row for row in runtime.session.input_dataset.active_records()} if runtime.session.input_dataset else {}
    if len(args.periods) > 1:
        periods = {period.date for period in args.periods}
        for row in args.rows:
            if any(key not in active or active[key].metric != row.metric or active[key].period_end not in periods
                    for key in row.replaces):
                raise ValueError("USER_TABLE_REPLACEMENT: 多年度更正只能引用这些明确年度中的同指标有效输入；各年度分别替换，不删除其他年度。")
    values = []
    for period in args.periods or [None]:
        for row in args.rows:
            line, ordinal = map(int, row.amount_ref.split(":"))
            _, _, body = selected_line(message.content, line)
            amounts = list(NUMBER.finditer(body))
            if ordinal > len(amounts):
                raise missing_amount_reference(message.content, row.amount_ref)
            literal = amounts[ordinal - 1][0].strip()
            literal_unit = next((unit for unit in ("亿元", "万元", "千元", "元/股", "元", "亿股", "万股", "股", "倍", "%", "％") if literal.endswith(unit)), None)
            literal_unit = {"元/股": "元", "倍": "ratio", "％": "%"}.get(literal_unit, literal_unit)
            if literal_unit is None and args.unit_line is None and (row.unit or args.unit) != "ratio":
                raise ValueError(f"USER_TABLE_UNIT_LINE: rows中的{row.metric}，amount_ref={row.amount_ref}是裸数（包括显式0），本工具须填顶层unit_line指向已提供的共享单位行。read_user_input读取单位声明后修正unit_line；不重录已保存字段、不联网。")
            values.append({"metric": row.metric, "amount_ref": row.amount_ref,
                "unit": row.unit or literal_unit or args.unit, "unit_ref": args.unit_line,
                "period_end": period.date if period else None, "period_ref": period.line if period else None,
                "as_of": args.as_of.date if args.as_of else None, "as_of_ref": args.as_of.line if args.as_of else None,
                "role": args.role, "entity": args.entity, "scope": args.scope, "period_kind": row.period_kind,
                "message_id": args.message_id, "replaces": [key for key in row.replaces
                    if len(args.periods) <= 1 or active[key].period_end == period.date]})
    return record_inputs(runtime, RecordInputs(user_values=values))
