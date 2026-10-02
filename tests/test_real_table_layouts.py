"""Minimal verbatim golden rows from official 2024 annual reports.

Midea: https://static.cninfo.com.cn/finalpage/2025-03-29/1222951181.PDF
physical pages 156-158; amounts in CNY thousands (not yuan).
CATL: https://static.cninfo.com.cn/finalpage/2025-03-15/1222806982.PDF
physical pages 114, 119-120; amounts in CNY thousands.
Rows were checked against rendered source pages, not a vendor's summary.
Tests never download PDFs, fill blank cells, or approve valuation inputs.
"""
from datetime import date

import pytest

from valuationagent.application.research import CandidateInput
from valuationagent.core.evidence import bind_evidence, evidence_context
from valuationagent.schemas.research import ResearchDraft


MIDEA_HEADER = """美的集团股份有限公司
2024  年度合并及公司利润表
(除特别注明外，金额单位为人民币千元)
              项目                   附注        2024年度         2023年度         2024年度        2023年度
                                                  合并             合并             公司            公司
                                                             (经重列)
"""
MIDEA_ROWS = [
    ("营业总收入", "409,084,266", "一、营业总收入                                    409,084,266   373,709,804        946,607      1,028,572"),
    ("营业收入", "407,149,600", "    其中：营业收入                      十八(3)    407,149,600    372,037,280        946,607      1,028,572"),
    ("营业成本", "-299,584,935", "    减：营业成本                        四(49)   (299,584,935) (276,409,404)       (40,440)       (40,060)"),
    ("利息费用", "2,453,361", "         其中：利息费用                             2,453,361     2,808,104      2,957,065      2,994,928"),
    ("销售费用", "-38,753,649", "         销售费用                     四(52)    (38,753,649)  (31,952,844)              -              -"),
]
CATL_HEADER = """宁德时代新能源科技股份有限公司                                   2024   年年度报告全文
3、合并利润表
                                                                                                                                                              单位：千元
                         项目                                                     2024   年度                                                 2023年度
"""
CATL_ROWS = [
    ("营业收入", "362,012,554", "     其中：营业收入                                                                                        362,012,554                                               400,917,045"),
    ("营业成本", "273,518,959", "     其中：营业成本                                                                                          273,518,959                                             323,982,130"),
    ("税金及附加", "2,057,466", "               税金及附加                                                                                    2,057,466                                                 1,695,508"),
    ("所得税费用", "9,175,245", "  减：所得税费用                                                                                 9,175,245                                        7,153,019"),
    ("归属于母公司股东的净利润", "50,744,682", "     1.归属于母公司股东的净利润                                                                     50,744,682                                       44,121,248"),
]


def _bind(text, metric, value, *, company="美的集团", scope="consolidated", period="2024", unit="千元"):
    block = {"block_id": "official:1", "file_id": "official", "location": {"page": 158}, "text": text}
    candidate = CandidateInput(metric=metric, raw_value=value, unit=unit, period=period,
                               scope=scope, block_id=block["block_id"], quote=text)
    draft = ResearchDraft(company=company, valuation_date=date(2025, 6, 30))
    return bind_evidence(candidate, block, evidence_context(block, [block]), draft)


@pytest.mark.parametrize("metric,value,row", MIDEA_ROWS)
def test_real_midea_four_columns_bind_year_and_scope(metric, value, row):
    warnings, checks = _bind(MIDEA_HEADER + row, metric, value)
    assert not warnings, warnings
    assert checks["column_scope"] == "consolidated"
    assert checks["year_column"] == 2024
    assert checks["unit"] == "千元"


@pytest.mark.parametrize("metric,value,row", MIDEA_ROWS)
def test_real_midea_note_column_is_a_label_boundary_not_another_account(metric, value, row):
    text = MIDEA_HEADER + "\n".join(source_row for _, _, source_row in MIDEA_ROWS)
    warnings, checks = _bind(text, metric, value)
    assert not warnings, warnings
    assert checks["source_row"] == row


@pytest.mark.parametrize("scope,period,value", [
    ("consolidated", "2023", "372,037,280"),
    ("parent", "2024", "946,607"),
    ("parent", "2023", "1,028,572"),
])
def test_real_midea_each_other_column_is_independently_addressed(scope, period, value):
    warnings, checks = _bind(MIDEA_HEADER + MIDEA_ROWS[1][2], "营业收入", value, scope=scope, period=period)
    assert not warnings, warnings
    assert checks["column_scope"] == scope
    assert checks["year_column"] == int(period)


@pytest.mark.parametrize("changes", [
    {"scope": "parent"}, {"period": "2023"}, {"unit": "元"}, {"scope": "unknown"},
])
def test_real_midea_correct_amount_cannot_cross_unit_scope_or_year(changes):
    warnings, _ = _bind(MIDEA_HEADER + MIDEA_ROWS[1][2], "营业收入", "407,149,600", **changes)
    assert warnings


def test_real_midea_dash_is_not_zero_or_an_omittable_cell():
    warnings, checks = _bind(MIDEA_HEADER + MIDEA_ROWS[4][2], "销售费用", "0", scope="parent")
    assert warnings
    assert "year_column" not in checks
    # If extraction drops placeholders, even the first value is not accepted:
    # row order alone does not prove which cells are missing.
    text = MIDEA_HEADER + MIDEA_ROWS[4][2].replace("              -", "")
    warnings, checks = _bind(text, "销售费用", "-38,753,649")
    assert warnings
    assert "year_column" not in checks


@pytest.mark.parametrize("bad_header", [
    MIDEA_HEADER.replace("公司            公司", "公司"),
    MIDEA_HEADER.replace("合并             合并", "合并             公司"),
    MIDEA_HEADER.replace("合并及公司利润表", "利润表"),
])
def test_joint_columns_require_explicit_complete_unique_perimeter_labels(bad_header):
    warnings, checks = _bind(bad_header + MIDEA_ROWS[1][2], "营业收入", "407,149,600")
    assert warnings
    assert "column_scope" not in checks


@pytest.mark.parametrize("metric,value,row", CATL_ROWS)
def test_real_catl_main_statement_integers_do_not_require_money_decimals(metric, value, row):
    warnings, checks = _bind(CATL_HEADER + row, metric, value, company="宁德时代")
    assert not warnings, warnings
    assert checks["year_column"] == 2024


def test_catl_annual_balance_sheet_relative_columns_need_explicit_december_date():
    text = """宁德时代新能源科技股份有限公司 2024年年度报告全文
1、合并资产负债表
2024    年   12  月   31  日
单位：千元
项目                         期末余额                           期初余额
货币资金                     303,511,993                        264,306,515"""
    warnings, checks = _bind(text, "货币资金", "303,511,993", company="宁德时代")
    assert not warnings, warnings
    assert checks["year_column"] == 2024
    wrong, _ = _bind(text, "货币资金", "264,306,515", company="宁德时代")
    assert any("年度列冲突" in warning for warning in wrong)
    no_date, checks = _bind(text.replace("2024    年   12  月   31  日", ""), "货币资金", "303,511,993", company="宁德时代")
    assert no_date
    assert "year_column" not in checks


def test_new_single_relative_column_does_not_inherit_previous_two_column_table():
    text = """宁德时代新能源科技股份有限公司 2024年年度报告全文
合并财务报表项目注释
单位：千元
项目 本期发生额 上期发生额
当期所得税费用 15,555,258 14,805,611
会计利润与所得税费用调整过程
单位：千元
项目 本期发生额
所得税费用 9,175,245"""
    warnings, checks = _bind(text, "所得税费用", "9,175,245", company="宁德时代")
    assert not warnings, warnings
    assert checks["year_column"] == 2024


def test_real_haitian_wrapped_capex_uses_numeric_physical_line_geometry():
    text = """佛山市海天调味食品股份有限公司2024年年度报告
合并现金流量表
2024年 1—12 月
单位：元  币种：人民币
项目                 附注          2024年度          2023年度
  收回投资收到的现金                    七、78（2）     14,004,050,000.00 14,293,300,000.00
  收到其他与投资活动有关的现金               七、78（2）        637,840,088.17    396,450,220.76
  购建固定资产、无形资产和其他               七、78（2）      1,575,700,218.89  1,924,147,446.25
长期资产支付的现金"""
    warnings, checks = _bind(
        text,
        "购建固定资产、无形资产和其他长期资产支付的现金",
        "1,575,700,218.89",
        company="佛山市海天调味食品股份有限公司",
        unit="元",
    )
    assert not warnings, warnings
    assert checks["year_column"] == 2024
    assert checks["column_alignment"] == "same_block_right_edges"
    assert checks["note_column_excluded"] is True


def test_numbered_new_statement_does_not_inherit_previous_statement_units():
    text = """美的集团股份有限公司
母公司资产负债表
单位：元
项目 2024年 2023年
其他资产 1 2
3、合并利润表
项目 2024年 2023年
营业收入 100 90"""
    warnings, _ = _bind(text, "营业收入", "100", unit="元")
    assert any("单位缺少" in warning for warning in warnings)


def test_three_year_summary_adjusted_columns_are_not_guessed():
    text = """宁德时代新能源科技股份有限公司 2024年年度报告全文
合并口径
                                                             本年比上年              2022年
              项目                     2024年        2023年        增减
                                                                          调整前         调整后
 营业收入（千元）                           362,012,554  400,917,045     -9.70% 328,593,988  328,593,988"""
    warnings, checks = _bind(text, "营业收入", "362,012,554", company="宁德时代")
    assert warnings
    assert "year_column" not in checks


MIDEA_NOTE_HEADER = """美的集团股份有限公司
2024 年度财务报表附注
(除特别注明外，金额单位为人民币千元)
四 合并财务报表项目附注(续)
"""
MIDEA_CASH_FLOW_NOTE = MIDEA_NOTE_HEADER + """
(65) 现金流量表项目附注(续)
(h) 现金流量表补充资料
将净利润调节为经营活动现金流量如下：
                                               2024 年度           2023 年度
      净利润                                     38,757,214        33,745,352
          存货的(增加)/减少                         (15,794,154)          206,064
          经营性应收项目的增加                         (14,349,722)        (9,747,941)
          经营性应付项目的增加                          50,345,636        29,692,141
"""


@pytest.mark.parametrize("metric,current,prior", [
    ("净利润", "38,757,214", "33,745,352"),
    ("存货的(增加)/减少", "-15,794,154", "206,064"),
    ("经营性应收项目的增加", "-14,349,722", "-9,747,941"),
    ("经营性应付项目的增加", "50,345,636", "29,692,141"),
])
def test_real_midea_page_256_bare_spaced_annual_header(metric, current, prior):
    for year, value in [("2024", current), ("2023", prior)]:
        warnings, checks = _bind(MIDEA_CASH_FLOW_NOTE, metric, value, period=year)
        assert not warnings, warnings
        assert checks["year_column"] == int(year)
    wrong, checks = _bind(MIDEA_CASH_FLOW_NOTE, metric, prior)
    assert any("年度列冲突" in warning for warning in wrong)
    assert "year_column" not in checks


MIDEA_DEBT_NOTE = MIDEA_NOTE_HEADER + """
(28) 短期借款
                             2024年  12月 31日    2023 年 12月 31日
     信用借款                           6,396,560         4,681,574
      保证借款                          7,708,400         1,083,216
      质押、抵押借款                      16,903,589         3,054,386
                                   31,008,549         8,819,176
"""


@pytest.mark.parametrize("metric,current,prior", [
    ("信用借款", "6,396,560", "4,681,574"),
    ("保证借款", "7,708,400", "1,083,216"),
    ("质押、抵押借款", "16,903,589", "3,054,386"),
])
def test_real_midea_page_233_bare_spaced_year_end_dates(metric, current, prior):
    warnings, checks = _bind(MIDEA_DEBT_NOTE, metric, current)
    assert not warnings, warnings
    assert checks["year_column"] == 2024
    wrong, checks = _bind(MIDEA_DEBT_NOTE, metric, prior)
    assert any("年度列冲突" in warning for warning in wrong)
    assert "year_column" not in checks


def test_real_midea_page_233_unnamed_total_cannot_be_attached_to_heading():
    warnings, checks = _bind(MIDEA_DEBT_NOTE, "短期借款", "31,008,549")
    assert warnings
    assert "year_column" not in checks


def test_real_midea_page_236_bare_dates_bind_named_row_not_unnamed_totals():
    text = MIDEA_NOTE_HEADER + """
(36) 一年内到期的非流动负债
                                     2024  年 12 月  31 日     2023  年 12 月  31 日
      一年内到期的长期借款
      ( 附注四(38))                             38,540,625             13,290,809
      一年内到期的租赁负债
      ( 附注四(40))                              1,122,108              1,166,901
                                             39,662,733             14,457,710
(37) 其他流动负债
                                     2024  年 12 月  31 日     2023  年 12 月  31 日
      预提销售返利                                 55,539,161             48,311,934
       其他                                    34,895,706             22,985,994
                                             90,434,867             71,297,928
"""
    warnings, checks = _bind(text, "预提销售返利", "55,539,161")
    assert not warnings, warnings
    assert checks["year_column"] == 2024
    wrong, checks = _bind(text, "预提销售返利", "48,311,934")
    assert any("年度列冲突" in warning for warning in wrong)
    for metric, total in [("一年内到期的非流动负债", "39,662,733"), ("其他流动负债", "90,434,867")]:
        warnings, checks = _bind(text, metric, total)
        assert warnings
        assert "year_column" not in checks


@pytest.mark.parametrize("header", [
    "2024 年 6 月 30 日 2023 年 6 月 30 日",
    "项目 2024 年 6 月 30 日 2023 年 6 月 30 日",
    "2024/06/30 2023/06/30",
    "项目 2024/06/30 2023/06/30",
    "项目 2024年上半年 2023年上半年",
    "项目 2024年第一季度 2023年第一季度",
])
def test_nonannual_headings_are_not_relabelled_as_annual_facts(header):
    text = MIDEA_NOTE_HEADER + header + "\n营业收入 100 90"
    warnings, checks = _bind(text, "营业收入", "100", period="2024")
    assert warnings
    assert "year_column" not in checks


def test_relative_balance_columns_cannot_borrow_dates_from_prior_borrowing_rate_prose():
    # CATL physical page 192: these dates describe the preceding loan-rate
    # note, not the new bonds table. In particular, the last date is 2023.
    text = """宁德时代新能源科技股份有限公司 2024 年年度报告全文
合并财务报表项目注释
截至2024年12月31日，上述借款年利率为1.74%-5.48%（2023年12月31日：1.20%-6.33%）；
36、应付债券
单位：千元
项目 期末余额 期初余额
公司债券 11,922,623 19,237,014"""
    warnings, checks = _bind(text, "公司债券", "11,922,623", period="2023", company="宁德时代")
    assert warnings
    assert "year_column" not in checks


# Verbatim physical page 136 (printed page 136) of the Midea 2024 report.
# The row itself has seven cells: before, before %, new issues, other changes,
# subtotal, after, after %. The annual cover supplies the fiscal year.
MIDEA_SHARES = """第七节 股份变动及股东情况
一、股份变动情况
1、股份变动情况
单位：股
本次变动前                   本次变动增减（＋，－）                         本次变动后
数量        比例      发行新股           其他           小计           数量        比例
三、股份总数 7,025,769,025 100.00 702,758,934 -72,572,076 630,186,858 7,655,955,883 100.00"""
MIDEA_SHARES_ROW = MIDEA_SHARES.splitlines()[-1]


def _bind_midea_shares(text=MIDEA_SHARES, value="7,655,955,883", *,
                       period="2024年度", unit="股", identity="美的集团股份有限公司 2024年度报告"):
    block = {"block_id": "midea:136", "file_id": "midea", "location": {"page": 136}, "text": text}
    row = next((line for line in text.splitlines() if line.startswith("三、股份总数")), MIDEA_SHARES_ROW)
    item = CandidateInput(metric="股份总数", raw_value=value, unit=unit,
                          period=period, scope="issuer", block_id=block["block_id"], quote=row)
    draft = ResearchDraft(company="美的集团", ticker="000333.SZ", valuation_date=date(2025, 6, 30))
    return bind_evidence(item, block, [block], draft, aliases={"common_shares"}, identity_text=identity)


def test_real_midea_share_change_table_binds_closing_issuer_quantity():
    warnings, checks = _bind_midea_shares()
    assert not warnings, warnings
    assert checks["binding"] == "issuer_share_change_table"
    assert checks["period_end"] == "2024-12-31"
    assert checks["column_alignment"] == "explicit_before_change_after_quantity"
    assert checks["reconciliation"]["before"] == "7025769025"
    assert checks["reconciliation"]["after"] == "7655955883"
    assert checks["scope"] == "issuer"


@pytest.mark.parametrize("change", [
    {"value": "7,025,769,025"},
    {"period": "2023年度"},
    {"unit": "元"},
    {"identity": "美的集团股份有限公司 2023年度报告"},
    {"text": MIDEA_SHARES.replace("单位：股", "单位：元")},
    {"text": MIDEA_SHARES.replace("本次变动后", "期末余额")},
    {"text": MIDEA_SHARES.replace("630,186,858", "630,186,857")},
    {"text": MIDEA_SHARES.replace("-72,572,076", "-")},
])
def test_midea_share_change_table_rejects_wrong_column_period_unit_or_broken_arithmetic(change):
    warnings, checks = _bind_midea_shares(**change)
    assert warnings
    assert "scope" not in checks


def test_midea_report_disclosure_share_total_uses_official_date_not_dividend_base():
    text = ("美的集团股份有限公司2024年度报告\n"
            "以截至本报告披露之日公司总股本 7,660,355,772股扣除回购专户已回购股份后，"
            "以7,631,903,546股为基数分红")
    block = {"block_id": "midea:3", "file_id": "midea", "text": text,
             "location": {"page": 3, "published_at": "2025-03-29"}}
    item = CandidateInput(metric="总股本", raw_value="7,660,355,772", unit="股", period="2025-03-29",
                          scope="issuer", block_id="midea:3",
                          quote="截至本报告披露之日公司总股本 7,660,355,772股")
    draft = ResearchDraft(company="美的集团", ticker="000333.SZ", valuation_date=date(2025, 6, 30))
    warnings, checks = bind_evidence(item, block, [block], draft, aliases={"common_shares"},
                                     identity_text="美的集团股份有限公司2024年度报告")
    assert not warnings, warnings
    assert checks["binding"] == "issuer_report_disclosure_shares"
    assert checks["period_end"] == "2025-03-29"
    assert checks["scope"] == "issuer"

    item.raw_value = "7,631,903,546"
    warnings, checks = bind_evidence(item, block, [block], draft, aliases={"common_shares"},
                                     identity_text="美的集团股份有限公司2024年度报告")
    assert warnings
    assert "scope" not in checks

    item.raw_value = "7,660,355,772"
    block["location"].pop("published_at")
    warnings, checks = bind_evidence(item, block, [block], draft, aliases={"common_shares"},
                                     identity_text="美的集团股份有限公司2024年度报告")
    assert warnings
    assert "scope" not in checks


def test_yili_explicit_dated_share_total_binds_without_inventing_a_separator():
    text = (
        "内蒙古伊利实业集团股份有限公司2024年年度报告\n"
        "公司拟向全体股东每股派发现金红利1.22元（含税），截至2025年4月8日，公司总股本\n"
        "6,365,900,705股，扣除公司回购专户股份32,859,361股，以此计算分红总额。"
    )
    block = {"block_id": "yili:2", "file_id": "yili", "text": text,
             "location": {"page": 2, "published_at": "2025-04-30"}}
    item = CandidateInput(
        metric="总股本", raw_value="6,365,900,705", unit="股",
        period="2025-04-08", scope="issuer", block_id="yili:2",
        quote="截至2025年4月8日，公司总股本6,365,900,705股",
    )
    draft = ResearchDraft(company="伊利股份", ticker="600887.SH", valuation_date=date(2025, 6, 30))
    warnings, checks = bind_evidence(
        item, block, [block], draft, aliases={"common_shares"},
        identity_text="内蒙古伊利实业集团股份有限公司2024年年度报告 600887",
    )
    assert not warnings, warnings
    assert checks["period_end"] == "2025-04-08"
    assert checks["scope"] == "issuer"
