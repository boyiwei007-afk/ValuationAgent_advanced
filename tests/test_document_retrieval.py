from valuationagent.application.document_retrieval import rank_document_blocks


def test_dated_issuer_total_is_retrieved_ahead_of_share_change_history():
    rows = [
        {"block_id": "history", "location": {"page": 71},
         "text": "股份总数 6,366,098,705 -198,000 6,365,900,705"},
        {"block_id": "dated", "location": {"page": 2},
         "text": "截至2025年4月8日，公司总股本 6,365,900,705股。"},
    ]
    assert rank_document_blocks(rows, "股份总数")[0]["block_id"] == "dated"


def block(key, text, page):
    return {"block_id": key, "text": text, "location": {"page": page}}


def test_primary_integer_statement_precedes_summary_and_contents():
    rows = [block("contents", "营业收入 所得税费用 ................... 115", 2),
            block("summary", "主要指标\n营业收入 362,012,554 400,917,045 -9.70%", 10),
            block("statement", "3、合并利润表\n单位：千元\n项目 2024年度 2023年度\n营业收入 362,012,554 400,917,045", 119)]
    assert rank_document_blocks(rows, "营业收入")[0]["block_id"] == "statement"


def test_same_page_header_helps_block_but_distant_header_does_not():
    rows = [block("header", "2024 年度合并及公司利润表", 158),
            block("values", "营业收入 407,149,600 372,037,280 946,607 1,028,572", 158),
            block("summary", "营业收入 407,149,600 372,037,280\n其他指标 营业收入 407,149,600", 180)]
    assert rank_document_blocks(rows, "营业收入")[0]["block_id"] == "values"


def test_results_are_original_blocks_not_rewritten_or_invented():
    rows = [block("one", "费用说明 未直接披露", 1)]
    assert rank_document_blocks(rows, "费用")[0] is rows[0]
    assert not rank_document_blocks(rows, "不存在的关键词")
