from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from valuationagent.schemas.models import ApiModel, JsonDecimal


class InputSource(ApiModel):
    kind: Literal["user", "document", "provider", "derivation"]
    source_id: str
    sha256: str
    quote: str
    locator: str = ""
    file_id: str = ""
    source_url: str = ""
    published_at: date | None = None
    authority_tier: str = ""
    limitations: list[str] = Field(default_factory=list)
    interpretation_sha256: str = ""
    provider_binding: dict = Field(default_factory=dict)


class InputRecord(ApiModel):
    input_id: str
    entity: str
    role: Literal["historical", "comparable"] = "historical"
    entity_ticker: str = ""
    metric: str
    label: str = ""
    period_kind: Literal["annual", "instant", "unknown"] = "unknown"
    value: JsonDecimal
    original_amount: str
    unit: str
    currency: str | None
    scope: str
    period_end: date | None = None
    as_of: date | None = None
    assertion: Literal["user_input", "reported", "assumption", "derived"]
    source: InputSource
    supersedes: list[str] = Field(default_factory=list)


class ComparableCapitalBridge(ApiModel):
    balance_date: date = Field(description="独立资本桥接余额日，可采用估值日前已披露的最近报表；不得冒充行情日，所用余额必须同日。")
    debt_includes_leases: bool = Field(description="有息债务是否已经包含租赁负债，租赁只计入一次。")
    cash_includes_associates: bool = Field(description="现金及非经营性资产是否已含另列的联营及非经营性投资，避免重复减除。")
    rationale: str = Field(min_length=30, max_length=2000, description="解释现金可用性、债务覆盖、租赁与EBITDA口径一致性、余额日和市场价值/账面代理的选择；这是建模判断，不是独立审计。")
    limitations: list[str] = Field(min_length=1, max_length=8)


class ComparableSelection(ApiModel):
    ticker: str = Field(min_length=1, max_length=124)
    name: str = Field(min_length=1, max_length=100)
    rationale: str = Field(min_length=12, max_length=1600, description="LLM说明业务可比性、差异与选择或剔除原因；这是判断，不是独立核验。")
    enabled: bool = True
    capital_bridge: ComparableCapitalBridge | None = Field(default=None, description="仅真实样本EV/EBITDA需要；PE/PS不要求。数值从同主体已保存输入读取，不在这里填金额。")


class CalculationTerm(ApiModel):
    input_id: str = Field(min_length=1, description="当前工作区有效input_id；只引用已有数值，不传数字常量或代码。")
    operation: Literal["add", "subtract"]


class InputCalculationDraft(ApiModel):
    metric: Literal["ebit", "depreciation_amortization", "cash_and_non_operating_assets", "interest_bearing_debt", "lease_liabilities", "minority_interest", "operating_nwc"]
    entity_ticker: str = Field(default="", description="目标公司省略；可比公式填已登记的规范A股代码，如000001.SZ。依赖必须全部属于该可比，不能混入目标或其他公司。")
    period_end: date | None = Field(default=None, description="与所有依赖的期间一致；仅全部为未注明年度的用户情景时可省略。")
    terms: list[CalculationTerm] = Field(min_length=1, max_length=24,
        description="从已有原始输入声明口径。单项表示有依据的直接映射，多项表示加减；不为满足项数编造零值，不自动假定未列组成不存在。")
    rationale: str = Field(min_length=30, max_length=2000,
        description="解释每项加减的经济含义、经营/非经营分类、是否覆盖完整口径，以及subtotal/组成是否重复。这是建模判断，不冒充原始披露或独立审计；不能仅说已确认。")
    limitations: list[str] = Field(min_length=1, max_length=8, description="明确遗漏风险、口径判断或假设限制，不声称公式正确就等于财务含义正确。")
    debt_includes_leases: bool | None = Field(default=None, description="有息债务公式必填：是否已包括租赁负债，防止后续重复扣除。其他公式不填写。")
    replaces: list[str] = Field(default_factory=list, max_length=12, description="更正同指标同期间的旧calculation_id；原始输入更正后，公式必须显式改为新input_id。")

    @model_validator(mode="after")
    def meaningful_explanation(self):
        if len(self.rationale.strip()) < 30 or any(not 6 <= len(item.strip()) <= 600 for item in self.limitations):
            raise ValueError("计算口径说明须至少30个非空白字符，每项限制须6至600字符；不能以空白或无限长文本替代说明。")
        return self


class InputCalculation(InputCalculationDraft):
    calculation_id: str


class InputDataset(ApiModel):
    entity: str
    currency: str = "CNY"
    analysis_basis: Literal["research", "user_scenario"]
    records: list[InputRecord] = Field(default_factory=list)
    comparables: dict[str, ComparableSelection] = Field(default_factory=dict)
    calculations: list[InputCalculation] = Field(default_factory=list)
    baseline_selection: dict = Field(default_factory=dict)
    available_target_annual_periods: list[date] = Field(default_factory=list)

    def active_records(self):
        superseded = {key for row in self.records for key in row.supersedes}
        return [row for row in self.records if row.input_id not in superseded]

    def active_calculations(self):
        superseded = {key for row in self.calculations for key in row.replaces}
        return [row for row in self.calculations if row.calculation_id not in superseded]


class UserInputValue(ApiModel):
    metric: str = Field(description="标准字段：revenue、ebitda、ebit、ebit_margin、depreciation_amortization、capital_expenditure、operating_nwc、change_operating_nwc、net_income_parent、tax_rate、common_shares、diluted_shares、cash_and_non_operating_assets、interest_bearing_debt、lease_liabilities、minority_interest、preferred_equity、unfunded_pension；原始科目profit_before_tax、income_tax_expense、cash_paid_for_ppe_intangibles等由程序推导。折旧摊销不是depreciation_and_amortization。可比可另填market_price或market_cap，程序由股价×股数算市值再算倍数，不手抄推导值；倍数假设pe_multiple/ps_multiple/ev_ebitda_multiple。")
    role: Literal["historical", "comparable"] = "historical"
    entity: str = Field(default="", max_length=100, description="仅用户提供的可比样本填原话中的名称，如可比A；不需要虚构上市代码。目标公司省略。")
    amount_ref: str = Field(default="", pattern=r"^$|^[1-9]\d*:[1-9]\d*$", description="优先使用read_user_input返回的行:数值序号。填引用时省略amount_text/value_context，程序直接取原值，不重抄。")
    amount_text: str = Field(default="", max_length=80, description="不用amount_ref时，提供原话中的完整数值及字面单位。不把共享单位拼到裸数后面，不换算。")
    unit_ref: int | None = Field(default=None, ge=1, description="read_user_input中明确单位的行号，替代手写unit_quote。")
    period_ref: int | None = Field(default=None, ge=1, description="明确年度或期间的原消息行号，替代period_quote；period_end仍由LLM判断并填写。")
    as_of_ref: int | None = Field(default=None, ge=1, description="明确时点的原消息行号，替代as_of_quote；as_of仍由LLM判断并填写。")
    value_context: str = Field(default="", max_length=400, description="可选：包含所选数值的原话短片段，如归母净利润：150。相同数值反复出现时优先用它定位，必须逐字匹配，不拼接、不重写；唯一定位后无需数amount_occurrence。")
    amount_occurrence: int | None = Field(default=None, ge=0, le=100, description="仅同一用户消息出现多个相同完整数值时，指定0起始匹配序号；引文和字符位置由程序从原消息保存，不重抄或改写quote。")
    message_id: str = Field(default="", description="默认本轮用户消息；引用更早给数时填真实用户message_id。")
    unit: Literal["元", "千元", "万元", "亿元", "股", "万股", "亿股", "ratio", "%"]
    unit_quote: str = Field(default="", max_length=200, description="仅amount_text不含单位且不是ratio时必填：用户明确单位的原话，如金额单位为亿元。数字后已有单位应直接包含在amount_text，不拆开转录。")
    period_end: date | None = Field(default=None, description="用户明确给出才填写；未知保持null，不伪造年度。")
    as_of: date | None = Field(default=None, description="股数独立时点；用户未给出保持null。")
    period_quote: str = Field(default="", max_length=200, description="period_end非空时必填真实用户日期/年度原话；没给年度就不要填period_end。")
    as_of_quote: str = Field(default="", max_length=200, description="as_of非空时必填用户明确的年月日原话，不用当前日期冒充。")
    scope: Literal["consolidated", "issuer", "assumption"] = "consolidated"
    period_kind: Literal["annual", "instant", "unknown"] = Field(default="unknown",
        description="raw.自定义原始科目需明确年度流量annual或期末存量instant；日期仍必须来自用户原话。标准字段由程序判定。")
    replaces: list[str] = Field(default_factory=list, description="更正现有输入时填真实input_id，不静默覆盖冲突值。")


class SelectedSourceInput(ApiModel):
    fact_id: str = Field(min_length=1, description="本工作区已完成来源绑定和语义复核的真实fact_id；不能填写原文block_id或自行编造数值。")
    as_raw: bool = Field(default=False, description="仅保留原始科目而不直接作为最终模型字段时为true；例如货币资金还需扣除受限及经营现金。原始金额/主体/期间/口径与语义复核仍须通过。")
    replaces: list[str] = Field(default_factory=list, description="显式替换同指标的模型输入input_id；不撤回原始披露或其他情景。")


class ProviderInputValue(ApiModel):
    file_id: str = Field(min_length=1, description="fetch_financial_history保存并读取过的原始API快照ID，不是用户消息。")
    record_pointer: str = Field(pattern=r"^/data/items/(0|[1-9]\d*)$", description="read_file返回的原始记录json_pointer，例如/data/items/2；不是筛选后行号。")
    field: str = Field(min_length=1, description="接口真实列名，如n_income_attr_p或total_share。金额、单位、日期从原始记录读取，不手抄。")
    metric: str = Field(min_length=1, description="你选择的标准模型字段，须符合供应商字段契约，如net_income_parent、revenue、common_shares。")
    role: Literal["historical", "comparable"] = Field(default="historical", description="目标公司用historical；可比公司用comparable，真实证券代码从API记录读取，不能把可比数字放进目标公司。")
    replaces: list[str] = Field(default_factory=list)


class SelectedProviderInput(ApiModel):
    candidate_id: str = Field(min_length=1, max_length=160, description="fetch_financial_history返回的input_candidates中真实candidate_id。直接选择，不重抄field、metric、数值、日期或角色；这些从原始API记录与契约读取。")
    replaces: list[str] = Field(default_factory=list, description="更正时显式替换同公司同指标的旧input_id；否则省略。")


class UserInputBasis(ApiModel):
    period_end: date | None = Field(default=None, description="可选共享完整年度截止日，仍由LLM根据period_ref原话选择；逐项period_end覆盖。")
    as_of: date | None = Field(default=None, description="可选共享明确时点，仍由LLM根据as_of_ref原话选择；逐项as_of覆盖。")
    unit_ref: int | None = Field(default=None, ge=1)
    period_ref: int | None = Field(default=None, ge=1)
    as_of_ref: int | None = Field(default=None, ge=1)
    unit_quote: str = Field(default="", max_length=200, description="该批共享的用户单位原话；不由程序猜单位。逐项unit仍必填，逐项unit_quote可覆盖。")
    period_quote: str = Field(default="", max_length=200, description="该批共享的完整年度/日期原话，可包含多个年度。逐项period_end仍由LLM依据原话选择，不自动扩展年份。")
    as_of_quote: str = Field(default="", max_length=200, description="该批共享的存量时点原话；不能用年报年度冒充股数时点。")


class BaselineInstruction(ApiModel):
    period_end: date | None = Field(description="仅用户明确指定历史估值基期时填写完整年度12月31日；null表示用户要求恢复最新可得年度。历史采集years不是基期指令。")
    user_quote: str = Field(min_length=1, max_length=600, description="用户明确选择基期的原话；不能引用助手计划、搜索片段或自行声称旧年是最新。")
    message_id: str = Field(default="", max_length=120, description="默认当前用户消息；引用之前明确的用户基期指令时填写其真实message_id。")


class RecordInputs(ApiModel):
    company: str = Field(default="", max_length=200, description="省略时沿用当前主体；更正已有输入不要另造名称。")
    currency: str = Field(default="CNY", pattern="^[A-Z]{3}$")
    baseline: BaselineInstruction | None = Field(default=None, description="可选：保存用户明确指定的年度基期；省略沿用当前选择，默认最新可得完整年度。")
    user_basis: UserInputBasis = Field(default_factory=UserInputBasis)
    user_values: list[UserInputValue] = Field(default_factory=list, max_length=64, description="仅用户消息明确给出的数值或假设。API/文件数字不能放这里；用户只给20倍时这里仅提交pe_multiple。")
    provider_values: list[SelectedProviderInput] = Field(default_factory=list, max_length=64, description="选择fetch_financial_history展示的input_candidates，只填candidate_id及可选replaces。候选包含原值、单位、期间、主体与字段契约；不必再read_file或重复映射。")
    source_values: list[SelectedSourceInput] = Field(default_factory=list, max_length=64, description="PDF/Excel等已完成原文绑定与语义复核的fact_id；尚未复核先extract_observations和review_observations。")
    comparables: list[ComparableSelection] = Field(default_factory=list, max_length=20, description="记录或修订可比选样判断；同代码替换选择说明但不覆盖数值。enabled=false剔除，保留来源与旧冻结结果。数值另用comparable角色的provider_values/source_values提交。")
    calculations: list[InputCalculationDraft] = Field(default_factory=list, max_length=12,
        description="可选：LLM声明EBIT或现金/债务桥接如何由已保存原始输入加减构成，程序精确计算并追踪来源。先record_inputs保存原始数据，再引用返回的input_id；不编造结果、不补缺项为0、不提交可执行代码。")

    @model_validator(mode="after")
    def nonempty(self):
        if not self.user_values and not self.provider_values and not self.source_values and not self.comparables and not self.calculations and self.baseline is None:
            raise ValueError("至少提交一个用户输入、API字段引用、已复核来源事实或可比选择")
        return self
