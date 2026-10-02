"""Presentation text only; source evidence and financial values stay intact."""

from __future__ import annotations


ENGLISH = {
    "完成": "Done",
    "资料与来源": "Data & sources",
    "财务审核": "Financial review",
    "经营假设": "Assumptions",
    "现金流预测": "Cash flow forecast",
    "DCF 估值": "DCF valuation",
    "相对估值": "Relative valuation",
    "敏感性分析": "Sensitivity",
    "区间验证": "Cross-validation",
    "结果与依据": "Results & evidence",
    "状态": "Status",
    "任务已创建 · 版本 {revision} · {mode}": "Run created · Version {revision} · {mode}",
    "无": "none",
    "已载入合成演示数据。": "Synthetic demo data loaded.",
    "资料快照已建立，开始核对来源与口径。": "Data snapshot created. Checking sources and definitions.",
    "假设已固定：WACC {wacc:.2%}，永续增长率 {growth:.2%}。": "Assumptions set: WACC {wacc:.2%}; terminal growth {growth:.2%}.",
    "目标企业": "Target company",
    "DCF 基准 {base:.2f}/股，区间 {low:.2f}—{high:.2f}/股。": "DCF base {base:.2f}/share; range {low:.2f}—{high:.2f}/share. ",
    "数据模式 {mode}；模型 {version}，尚待金融团队核准。": "Data mode {mode}; model {version}; financial team approval pending.",
    "数据模式 {mode}；模型 {version}。正式模型按金融小组规则执行，降级和待复核项见警告。": "Data mode {mode}; model {version}. The formal finance-team rules were applied; see warnings for fallbacks and review items.",
    "仅DCF形成有效区间；相对估值独立保留为不可用。": "Only DCF produced a valid range; relative valuation remains independently marked unavailable.",
    "仅相对估值形成有效区间；DCF独立保留为不可用。": "Only relative valuation produced a valid range; DCF remains independently marked unavailable.",
    "两类方法存在重叠区间；重叠仅表示结果一致程度，不构成单独的推荐区间。": "The valuation ranges overlap. This indicates agreement between methods; the overlap is not a standalone recommended range.",
    "两类估值区间没有重叠，需要复核增长、利润率、资本成本和可比公司口径。": "The valuation ranges do not overlap. Review growth, margins, capital costs and comparable company definitions.",
    "当前仅有可用的 DCF 结果；相对估值未形成有效区间。": "Only DCF results are available; relative valuation produced no valid range.",
    "当前仅有可用的相对估值结果；DCF 未形成有效区间。": "Only relative valuation results are available; DCF produced no valid range.",
    "当前没有可用的估值区间。": "No valid valuation range is available.",
    "当前没有有效估值区间。": "No valid valuation range is available.",
}


def translator(language):
    """Keep unrecognized source/provider text verbatim, never translate evidence."""

    def translate(message):
        return ENGLISH.get(message, message) if language == "en-US" else message

    return translate
