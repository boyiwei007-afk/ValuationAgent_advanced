"""Conservative source-to-fact binding, independent of model confidence.

Explicit contradictions and missing headers require a new, better citation.
Never infer a unit, year or company from the fact proposed by the model itself.
"""
import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

EVIDENCE_VERSION = "table-binding-v8"
SCOPE_PATTERN = r"(合并|母公司)(?:财务)?(?:报表(?:项目)?(?:附注|注释)?|资产负债表|利润表|现金流量表|口径)"
STATEMENT_TITLE = (r"(?m)^\s*(?:[一二三四五六七八九十\d]+\s*[、.．]\s*)?"
                   r"(?:(?:19|20)\d{2}\s*年度?\s*)?(?:合并及公司|合并|母公司)"
                   r"(?:资产负债表|利润表|现金流量表)(?:[（(]续[）)])?\s*$")
# The same token must locate years and remove them when checking a bare
# header. PDF layout commonly separates the year from 年度/年12月31日.
# Dates other than December 31 remain unconsumed and cannot masquerade as
# annual headers (e.g. a June 30 half-year table).
ANNUAL_HEADER_TOKEN = (r"(?<!\d)(?i:FY)?\s*((?:19|20)\d{2})"
                       r"(?:\s*年(?:\s*12\s*月\s*31\s*日|\s*度|\s*末)?|[-/]12[-/]31)?")


def compact(text):
    return re.sub(r"\s+", "", str(text)).casefold()


def numeric_tokens(text, *, exclude_percent=False):
    text = re.sub(r"\b[A-Z]{1,3}\d+\s*:", "", text)
    # PDF text extraction commonly removes the separator between adjacent
    # statement columns, for example ``29,444,936,771.4130,303,850,168.56``.
    # Splitting is safe only when both sides retain a full comma-grouped money
    # shape with exactly two decimal places; ordinary high-precision decimals
    # remain untouched.
    text = re.sub(
        r"(\d{1,3}(?:[,，]\d{3})+\.\d{2})(?=[+\-−]?(?:\d{1,3}(?:[,，]\d{3})+\.\d{2}))",
        r"\1 ",
        text,
    )
    # A PDF can omit the space between two signed amounts. Do not lose the
    # second minus sign simply because it follows the previous amount.
    pattern = r"(?:(?<![\d.])|(?=[+\-−]))\(?[+\-−]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?\)?"
    values = []
    for match in re.finditer(pattern, text):
        if exclude_percent and text[match.end():].lstrip().startswith(("%", "％")):
            continue
        raw = match[0].replace("−", "-").replace(",", "").replace("，", "")
        try:
            values.append(-Decimal(raw[1:-1]) if raw.startswith("(") and raw.endswith(")") else Decimal(raw))
        except InvalidOperation:
            pass
    return values


def evidence_context(block, file_blocks, explicit_ids=()):
    """Find local headers in document order, not download/append order.

    Only consecutive, loaded pages can supply table headers. A distant financial
    notes section may supply *scope only*, and only across a fully loaded page
    sequence. Returned excerpts remain verbatim and carry their original IDs.
    """
    location = block.get("location") or {}
    ordered = list(file_blocks)
    if location.get("page"):
        ordered = [b for _, b in sorted(enumerate(ordered),
                   key=lambda pair: (pair[1].get("location", {}).get("page", 0), pair[0]))]
    elif location.get("sheet"):
        ordered = [b for b in ordered if b.get("location", {}).get("sheet") == location["sheet"]]
    index = next(i for i, b in enumerate(ordered) if b["block_id"] == block["block_id"])
    before = ordered[:index + 1]
    page = location.get("page")
    local = before[-8:]
    if page:
        pages = {b.get("location", {}).get("page") for b in before}
        floor = page
        while floor > max(1, page - 4) and floor - 1 in pages:
            floor -= 1
        local = [b for b in local if floor <= b.get("location", {}).get("page", -1) <= page]
    elif location.get("table"):
        local = [b for b in before if b.get("location", {}).get("table") == location["table"]][:3] + local
        local = [entry for entry in local if not entry.get("location", {}).get("table")
                 or entry["location"]["table"] == location["table"]]
    if location.get("sheet") or location.get("table"):
        # Long workbooks/Word tables often keep headers dozens of rows above
        # the value. Keep the latest header of each kind in addition to nearby
        # rows; never borrow from another sheet or later in the document.
        header_patterns = [SCOPE_PATTERN, r"单位\s*[:：]?",
                           r"(?:本期|本年)(?:金额|发生额|数).*(?:上期|上年)(?:金额|发生额|数)"]
        for pattern in header_patterns:
            found = next((b for b in reversed(before[:-1]) if re.search(pattern, b["text"])
                          and (not location.get("table") or not b.get("location", {}).get("table")
                               or b["location"]["table"] == location["table"])), None)
            if found:
                local.append(found)
        year_header = next((b for b in reversed(before[:-1]) if _table_columns(b["text"] + "\n_", "")[0]
                            and (not location.get("table") or b.get("location", {}).get("table") == location["table"])), None)
        if year_header:
            local.append(year_header)
    # Explicit context cannot borrow a future header or another worksheet.
    ids = {b["block_id"] for b in local} | set(explicit_ids)
    context = [b for b in before if b["block_id"] in ids]
    if location.get("table"):
        context = [entry for entry in context if not entry.get("location", {}).get("table")
                   or entry["location"]["table"] == location["table"]]
    if page:
        context = [b for b in context if floor <= b.get("location", {}).get("page", -1) <= page]
        for candidate in reversed(before):
            headings = [line for line in candidate["text"].splitlines() if re.fullmatch(
                r"\s*(?:[一二三四五六七八九十百\d]+\s*[、.．]\s*)?"
                r"(?:合并|母公司)(?:财务)?报表(?:项目)?(?:附注|注释)\s*", line)]
            if not headings:
                continue
            start_page = candidate.get("location", {}).get("page", page)
            if all(p in pages for p in range(start_page, page + 1)):
                # Do not import this old page's currency or year headings.
                context.insert(0, {**candidate, "text": headings[-1], "context_kind": "section_scope"})
            break
    unique = {}
    for b in context:
        unique[(b["block_id"], b.get("context_kind", ""))] = b
    return list(unique.values())


def _name_in_row(name, line):
    name, line = compact(name), compact(line)
    for match in re.finditer(re.escape(name), line):
        before, after = line[:match.start()], line[match.end():]
        left = not before or not (before[-1].isalnum() or before[-1] == "_") or bool(re.search(r"\d{4}年$", before))
        right = (
            not after
            or not (after[0].isalpha() or after[0] == "_")
            or bool(re.match(
                r"^[一二三四五六七八九十百]+(?:[、.．]\d+(?:[（(]\d+[）)])?|[（(]\d+[）)])",
                after,
            ))
        )
        if left and right:
            return True
    return False


def _metric_row(item, text, aliases):
    """Prefer the requested source label; never silently select its sibling.

    A canonical name with multiple possible rows needs a narrower quotation.
    Values alone must not decide whether total revenue or operating revenue was
    intended. Short aliases also support labels wrapped over several lines.
    """
    lines = text.splitlines()
    exact = [i for i, line in enumerate(lines) if _name_in_row(item.metric, line)]
    matches = exact or [i for i, line in enumerate(lines)
                        if any(_name_in_row(alias, line) for alias in aliases)]
    quoted = [i for i in matches if compact(lines[i]) in compact(item.quote)]
    if quoted:
        matches = quoted
    amount = numeric_tokens(item.raw_value)
    blank_zero = len(amount) == 1 and amount[0] == 0
    if len(matches) == 1:
        index = matches[0]
        if amount and amount[0] not in numeric_tokens(lines[index]):
            # A zero may represent an explicitly blank annual balance-sheet
            # cell, but only the later column-alignment gate can prove that.
            # Keep the exact row available for that gate; every non-zero value
            # still has to appear in the row itself.
            if blank_zero:
                return lines[index], "\n".join(lines[:index + 1]), ""
            # A complete label on an empty row is not permission to borrow the
            # next accounting row's amount. Only a values-only continuation
            # can extend a complete label; split labels are handled below.
            following = lines[index + 1] if index + 1 < len(lines) else ""
            values_only = bool(following.strip()) and not re.search(
                r"[^\d\s.,，()（）+\-−%％|]", following
            )
            if values_only and amount[0] in numeric_tokens(following):
                return " ".join(lines[index:index + 2]), "\n".join(lines[:index + 2]), ""
            return lines[index], "\n".join(lines[:index + 1]), "科目数值缺失或冲突：不能借用相邻科目的金额"
        return lines[index], "\n".join(lines[:index + 1]), ""
    if len(matches) > 1:
        return "", "", "字段对应多行，需缩小引文明确具体科目"

    # A long Chinese statement label may be wrapped before or after its values.
    # Reconstruct only a short adjacent window that contains both the complete
    # label and the proposed amount.  Requiring the amount prevents two nearby
    # accounting rows from being silently merged into one source row.
    names = [item.metric, *aliases]
    windows = []
    for width in (2, 3):
        for start in range(len(lines) - width + 1):
            end = start + width
            window = " ".join(lines[start:end])
            joined = compact(window)
            # Some PDF extractors insert the numeric columns between the two
            # halves of a wrapped label. Compare a numbers-stripped view too.
            label_text = re.sub(
                r"[一二三四五六七八九十百]+[、.．]\s*\d+(?:[（(]\d+[）)])?",
                "",
                window,
            )
            label_joined = compact(re.sub(r"[\d０-９.,，()（）+\-−%％\s]+", "", label_text))
            if not any(
                compact(name) and (compact(name) in joined or compact(name) in label_joined)
                for name in names
            ):
                continue
            values = numeric_tokens(window)
            if amount and amount[0] not in values:
                continue
            windows.append((start, end, window))
        if windows:
            break
    quoted_windows = [entry for entry in windows if compact(entry[2]) in compact(item.quote)]
    if quoted_windows:
        windows = quoted_windows
    # Overlapping windows can describe the same wrapped row. Prefer the one
    # with the least unrelated text, but retain ambiguity between disjoint rows.
    if windows:
        starts = {entry[0] for entry in windows}
        if len(starts) == 1:
            start, end, window = min(windows, key=lambda entry: len(compact(entry[2])))
            return window, "\n".join(lines[:end]), ""
    return "", "", "字段名未与数值绑定：请引用包含该科目名称的原文行"


def _table_columns(prefix, metric_line):
    """Read explicit year/comparison/note columns without guessing blanks."""
    header, columns = "", []
    for line in prefix.splitlines()[:-1]:
        # A readable JSON table often stores each header/row as a string.
        # Remove only a valid JSON string wrapper, never arbitrary punctuation
        # or a monetary row label in an attempt to manufacture a year header.
        wrapped = line.strip().removesuffix(",")
        if wrapped.startswith('"') and wrapped.endswith('"'):
            try:
                unwrapped = json.loads(wrapped)
                if isinstance(unwrapped, str):
                    line = unwrapped
            except ValueError:
                pass
        line = re.sub(r"\b[A-Z]{1,3}\d+\s*:\s*", "", line)
        years = list(re.finditer(ANNUAL_HEADER_TOKEN, line))
        labelled = bool(re.search(r"^\s*(?:项目|科目|附注|(?i:items?|metric|notes?))(?=\s|[|:：（(]|$)", line))
        # A preceding amount row such as '营业成本 2025 2024' is not a year
        # header. Bare-year headings and explicitly labelled headers are.
        remainder = re.sub(ANNUAL_HEADER_TOKEN, "", line)
        remainder = re.sub(r"本(?:年|期)比上(?:年|期)(?:同期)?(?:增减|增长)?|同比(?:增减|增长)(?:率|变化)?", "", remainder)
        bare_years = not re.sub(r"[\s|,，()（）%％/.-]", "", remainder)
        nonannual_period = bool(re.search(r"\d\s*月|半年|季度|[-/]\d{1,2}[-/]\d{1,2}", remainder))
        explicit_single = len(years) == 1 and labelled
        if (len(years) >= 2 or explicit_single) and (labelled or bare_years) and not nonannual_period and len({m[1] for m in years}) == len(years):
            markers = [(m.start(), int(m[1])) for m in years]
            comparisons = list(re.finditer(r"本(?:年|期)比上(?:年|期)(?:同期)?(?:增减|增长)?|同比(?:增减|增长)(?:率|变化)?", line))
            markers.extend((m.start(), None) for m in comparisons)
            columns = [value for _, value in sorted(markers)]
            header = line
        elif (relative := re.findall(r"(本期|上期|本年|上年|本年度|上年度)(?:金额|发生额|数)", line)) and (
            len(relative) == 1 or len(relative) == 2 and relative[0][0] != relative[1][0]
        ) and re.fullmatch(r"\s*(?:项目|科目|补充资料)?\s*(?:(?:本期|上期|本年|上年|本年度|上年度)(?:金额|发生额|数)\s*){1,2}", line):
            reports = re.findall(r"((?:19|20)\d{2})\s*年\s*(?:年度报告|度报告)", prefix)
            if len(set(reports)) == 1:
                year = int(reports[0])
                header, columns = line, [year if label.startswith("本") else year - 1 for label in relative]
        elif re.fullmatch(r"\s*(?:项目|科目)?\s*期末余额\s+期初余额\s*", line):
            # Only an explicit annual balance-sheet date can define these
            # relative columns. Never derive dates from the proposed fact.
            dates = re.findall(r"(?m)^\s*((?:19|20)\d{2})\s*年\s*12\s*月\s*31\s*日\s*$", prefix[:prefix.index(line)])
            if dates:
                year = int(dates[-1])
                header, columns = line, [year, year - 1]
    return header, columns


def _joint_statement_columns(prefix):
    """Read explicit year × perimeter headings in a joint group/company table.

    A joint title is not itself a consolidated-scope assertion. Every year
    column must have a separate aligned-order scope label; duplicate pairs,
    missing labels, and other table shapes remain unverified.
    """
    titles = list(re.finditer(STATEMENT_TITLE, prefix))
    if not titles or "合并及公司" not in titles[-1][0]:
        return None
    lines = prefix[titles[-1].end():].splitlines()
    for index, line in enumerate(lines):
        years = re.findall(r"(?<!\d)((?:19|20)\d{2})\s*年度?", line)
        residual = re.sub(r"(?:19|20)\d{2}\s*年度?", "", line)
        residual = re.sub(r"项目|科目|资产|负债和股东权益|附注|\s+", "", residual)
        if len(years) != 4 or residual:
            continue
        for scope_line in lines[index + 1:index + 4]:
            scopes = re.findall(r"合并|母公司|公司", scope_line)
            residual_scope = re.sub(r"合并|母公司|公司|项目|科目|附注|\s+", "", scope_line)
            if len(scopes) != 4 or residual_scope:
                continue
            columns = [(int(year), "consolidated" if scope == "合并" else "parent")
                       for year, scope in zip(years, scopes)]
            if len(set(columns)) == 4 and {scope for _, scope in columns} == {"consolidated", "parent"}:
                return line + "\n" + scope_line, columns
    return None


def _joint_statement_values(row_text):
    # Explicit Chinese note references are metadata, not amounts. In contrast,
    # standalone parentheses around a number retain the accounting minus sign.
    row_text = re.sub(r"[一二三四五六七八九十百]+[（(]\d+[）)](?:[（(][a-z][）)])?[,，]?", "", row_text)
    cells = re.findall(r"(?<![\w.])(?:\([+\-−]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?\)|[+\-−]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?|(?<!\S)[-—–](?!\S))", row_text)
    if len(cells) != 4:
        return None
    return [None if cell in {"-", "—", "–"} else numeric_tokens(cell)[0] for cell in cells]


def _aligned_note_values(block_prefix, metric_line, column_count):
    """Recover omitted year cells only from stable same-block PDF columns.

    A note number plus one amount must not masquerade as two annual amounts.
    At least two preceding complete rows must independently establish every
    numeric column's right edge. Missing cells stay None; no zeros are inferred.
    Delimited/coordinate-bearing tables use their existing binding path.
    """
    if "|" in metric_line or "\t" in metric_line:
        return None
    note_ref_pattern = (
        r"[一二三四五六七八九十百]+[、.．]\s*\d+"
        r"(?:[（(]\d+[）)])?(?:[（(][a-z][）)])?"
    )
    # When PDF extraction removes the gap between two currency columns,
    # ``29,444,936,771.4130,303,850,168.56`` must be tokenised as two
    # two-decimal amounts. Keep arbitrary decimal precision everywhere else.
    grouped_decimal = (
        r"(?:\.\d{2}(?=[+\-−]?(?:\d{1,3}(?:[,，]\d{3})+\.\d{2}))|\.\d+)?"
    )
    number = re.compile(
        note_ref_pattern
        + r"|(?<![\w.])\(?[+\-−]?(?:\d{1,3}(?:[,，]\d{3})+"
        + grouped_decimal
        + r"|\d+(?:\.\d+)?)\)?"
    )

    def tokens(line):
        if "|" in line or re.search(r"\b[A-Z]{1,3}\d+\s*:", line):
            return []
        return list(number.finditer(line))

    physical_lines = block_prefix.splitlines()
    joined_metric = compact(metric_line)
    # ``_metric_row`` may reconstruct a label wrapped around its numeric row,
    # e.g. ``购建固定资产、无形资产和其他 [note/value columns]`` followed by
    # ``长期资产支付的现金``.  Column geometry belongs to the one physical
    # line carrying the values, not to the space-joined display row.
    target_indexes = [
        index for index in range(max(0, len(physical_lines) - 3), len(physical_lines))
        if compact(physical_lines[index])
        and compact(physical_lines[index]) in joined_metric
    ]
    layout_candidates = [
        (index, physical_lines[index], tokens(physical_lines[index]))
        for index in target_indexes if tokens(physical_lines[index])
    ]
    if layout_candidates:
        max_cells = max(len(entry[2]) for entry in layout_candidates)
        best = [entry for entry in layout_candidates if len(entry[2]) == max_cells]
        if len(best) != 1:
            return None
        _, layout_line, metric_tokens = best[0]
        reference_lines = physical_lines[:min(target_indexes)]
    else:
        layout_line, metric_tokens = metric_line, tokens(metric_line)
        reference_lines = physical_lines[:-1]
    # PDF text extraction commonly strips the trailing spaces from a wholly
    # blank accounting row, leaving only its label (for example, “短期借款”).
    # A populated row still needs visible column spacing; a label-only row may
    # proceed only when two complete rows below establish stable right edges.
    if metric_tokens and not re.search(r" {2,}", layout_line):
        return None

    references = []
    for line in reference_lines:
        matches = tokens(line)
        if len(matches) != column_count + 1 or not re.search(r" {2,}", line):
            continue
        first = numeric_tokens(matches[0][0])
        if not first or first[0] != first[0].to_integral_value() or not 0 < first[0] <= 999:
            continue
        references.append([match.end() for match in matches])
    if len(references) < 2:
        return None
    # Stable right-alignment is required on this exact page/chunk. Other page
    # margins, wrapped rows, and a second table with new columns must not fit.
    edges = list(zip(*references))
    if any(max(edge) - min(edge) > 2 for edge in edges):
        return None
    centers = [sum(edge) / len(edge) for edge in edges]
    values = [None] * (column_count + 1)
    for match in metric_tokens:
        positions = [i for i, center in enumerate(centers) if abs(match.end() - center) <= 2]
        if len(positions) != 1 or values[positions[0]] is not None:
            return None
        parsed = numeric_tokens(match[0])
        if re.fullmatch(note_ref_pattern, match[0]):
            # A printed reference such as 七、78（2） occupies one note cell;
            # the parenthesized sub-item is not another accounting value.
            parsed = parsed[:1]
        if len(parsed) != 1:
            return None
        values[positions[0]] = parsed[0]
    if values[0] is not None and (values[0] != values[0].to_integral_value() or not 0 < values[0] <= 999):
        return None
    return values[1:]


def _bind_issuer_shares(item, block, draft, aliases, identity_text):
    """Verify a narrow, explicit issuer share-count disclosure, not a balance.

    Common shares belong to the legal issuer, not its consolidation perimeter.
    This path therefore needs its own identity/date/unit evidence and must not
    inherit currency units or fiscal years from a nearby financial statement.
    """
    warnings = []
    checks = {"version": EVIDENCE_VERSION, "binding": "issuer_common_shares",
              "context_block_ids": [block["block_id"]], "source_row": item.quote}
    if "common_shares" not in aliases:
        return ["发行人口径仅适用于已识别的普通股股数，不能用于其他财务科目"], checks
    quote = compact(item.quote)
    if not quote or quote not in compact(block.get("text", "")):
        return ["发行人股数引文必须是同一来源片段中的连续原文"], checks

    company = compact(draft.company)
    short_company = re.sub(r"(?:股份有限公司|有限责任公司|有限公司)$", "", company)
    ticker = (draft.ticker or "").split(".")[0]
    identity = compact(identity_text + "\n" + block.get("text", ""))
    identity_matches = ((bool(company) and company in identity)
                        or (len(short_company) >= 2 and short_company in identity)
                        or (bool(ticker) and re.search(r"(?<!\d)" + re.escape(ticker) + r"(?!\d)", identity)))
    if not identity_matches:
        warnings.append("主体未匹配：发行人股数来源文件未验证为当前公司/证券代码")
    else:
        checks["issuer"] = draft.ticker or draft.company
    # These are distinct financial concepts even if the amounts happen to
    # equal the outstanding share count. Never infer total shares from them.
    if re.search(r"分红|派息|利润分配|流通股|无限售|子公司|参股公司|其他发行人|优先股|存托凭证", quote):
        return [*warnings, "发行人股数须直接披露总股份数，不能由分红基数、流通股或其他主体/证券类别推算"], checks

    # A filing may say "as of this report's disclosure" rather than print a
    # calendar day in the sentence. Only an official, dated source can turn
    # that relative date into a factual issuer share-count date. The preceding
    # total is distinct from the lower dividend-eligibility base.
    relative = re.fullmatch(
        r"(?:以)?截至本(?:年度)?报告披露之日[，,]?(?:本公司|公司)(?:的)?总股本(?:为|是|共计|[:：])?"
        r"(?P<amount>(?:\d{1,3}(?:[,，]\d{3})+|\d+))(?P<unit>股|万股|亿股)",
        quote,
    )
    if relative:
        published = (block.get("location") or {}).get("published_at")
        try:
            disclosure_date = date.fromisoformat(str(published)[:10])
        except (TypeError, ValueError):
            return [*warnings, "报告披露日股数缺少可核验的来源发布日期；请取得正式公告日期"], checks
        try:
            period = date.fromisoformat(str(item.period))
        except ValueError:
            period = None
        if period != disclosure_date:
            warnings.append("期间冲突：候选股数截止日须与正式报告披露日一致")
        if draft.valuation_date and disclosure_date > draft.valuation_date:
            warnings.append("披露时点晚于估值日，不能使用未来股数")
        if numeric_tokens(item.raw_value) != numeric_tokens(relative["amount"]):
            warnings.append("科目数值冲突：候选股数未与报告披露日总股数一致")
        if item.unit != relative["unit"]:
            warnings.append("单位冲突：报告披露日总股数须保留原文股数单位")
        checks.update(binding="issuer_report_disclosure_shares", source_row=item.quote,
                      period_end=disclosure_date.isoformat(), period=str(disclosure_date.year),
                      unit=relative["unit"])
        if not warnings:
            checks["scope"] = "issuer"
        return list(dict.fromkeys(warnings)), checks

    # HKEX final allotment results disclose the post-offering issuer total in
    # a purpose-built table.  The row is more precise than deriving shares by
    # adding offer tranches, but it often omits a trailing ``股`` because the
    # table heading already states that every entry is a share count.  Accept
    # this narrow layout only from HKEX, with the listing/trading date in the
    # same cited block and the target issuer independently matched above.
    location = block.get("location") or {}
    source_url = str(location.get("url") or location.get("source_url") or "")
    source_host = (urlsplit(source_url).hostname or "").rstrip(".").casefold()
    hkex_hosts = {"hkexnews.hk", "www.hkexnews.hk", "www1.hkexnews.hk"}
    listing_total = re.search(
        r"(?:於|于)上市(?:時|时)已(?:發行|发行)的股份(?:數目|数目)"
        r"(?:[（(](?:於|于)(?:超額配股權|超额配股权)(?:獲|获)行使前[）)])?"
        r"(?P<amount>\d{1,3}(?:[,，]\d{3})+|\d+)(?:股)?",
        quote,
    )
    listing_date = re.search(
        r"(?:開始買賣日|开始买卖日|上市日期?)(?:目前預期為|目前预期为|為|为|[:：])?"
        r"(?P<year>(?:19|20)\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日",
        quote,
    )
    if listing_total or listing_date:
        if source_host not in hkex_hosts:
            return [*warnings, "上市时已发行股数仅可由港交所正式配发结果公告核验"], checks
        if not listing_total or not listing_date:
            return [*warnings, "港交所上市股数引文须同时覆盖上市日和上市时已发行股份总数"], checks
        try:
            effective_date = date(
                int(listing_date["year"]),
                int(listing_date["month"]),
                int(listing_date["day"]),
            )
        except ValueError:
            effective_date = None
            warnings.append("上市日期无法识别")
        try:
            target_date = date.fromisoformat(str(item.period))
        except ValueError:
            target_date = None
        if effective_date is not None and target_date != effective_date:
            warnings.append("期间冲突：候选股数截止日须与港交所公告所列上市日一致")
        if draft.valuation_date and effective_date and effective_date > draft.valuation_date:
            warnings.append("上市日晚于估值日，不能提前使用发行后股数")
        published = location.get("published_at")
        try:
            publication_date = date.fromisoformat(str(published)[:10])
        except (TypeError, ValueError):
            publication_date = None
            warnings.append("港交所配发结果缺少可核验的公告发布日期")
        if draft.valuation_date and publication_date and publication_date > draft.valuation_date:
            warnings.append("配发结果公告日晚于估值日，不能使用未来信息")
        proposed = numeric_tokens(item.raw_value)
        source_amount = numeric_tokens(listing_total["amount"])
        if (
            len(proposed) != 1
            or proposed != source_amount
            or proposed[0] <= 0
            or proposed[0] != proposed[0].to_integral_value()
        ):
            warnings.append("科目数值冲突：候选股数不是上市时已发行股份总数的正整数")
        if item.unit != "股":
            warnings.append("单位冲突：港交所配发结果的已发行股份总数须以股计量")
        checks.update(
            binding="issuer_listing_issued_shares",
            source_row=item.quote,
            period_end=effective_date.isoformat() if effective_date else "",
            period=str(effective_date.year) if effective_date else "",
            unit="股",
            official_exchange="HKEX",
            publication_date=(publication_date.isoformat() if publication_date else ""),
        )
        if not warnings:
            checks["scope"] = "issuer"
        return list(dict.fromkeys(warnings)), checks

    # Some A-share annual reports use the statutory ``股本`` note for share
    # counts.  Accept it only when the row is explicitly named ``股份总数``, the
    # table exposes opening and closing balances, and the same note states the
    # issuer/date, ordinary-share class and RMB 1 par value.  This is not the
    # generic balance-sheet ``实收资本（或股本）`` shortcut prohibited below.
    text = block.get("text", "")
    capital_note_rows = [
        line for line in text.splitlines()
        if re.match(r"^\s*股份总数\s+", line)
    ]
    capital_note = (
        len(capital_note_rows) == 1
        and compact(capital_note_rows[0]) in quote
        and re.search(r"(?:^|\n)\s*\d+[、.．]\s*股本\s*(?:\n|$)", text)
        and "期初余额" in text and "期末余额" in text
        and re.search(
            r"于\s*(?P<year>(?:19|20)\d{2})\s*年\s*(?P<month>\d{1,2})\s*月\s*(?P<day>\d{1,2})\s*日[，,]?"
            r"本公司注册资本包括普通股[，,]?每股面值人民币\s*1\s*元",
            text,
        )
    )
    if capital_note:
        row = capital_note_rows[0]
        values = numeric_tokens(row)
        note_date = re.search(
            r"于\s*((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
            text,
        )
        try:
            source_date = date(*map(int, note_date.groups()))
        except (AttributeError, ValueError):
            source_date = None
        period = re.sub(r"\s+", "", item.period)
        exact_date = re.fullmatch(r"((?:19|20)\d{2})(?:年|-|/)(\d{1,2})(?:月|-|/)(\d{1,2})日?", period)
        annual = re.fullmatch(r"((?:19|20)\d{2})(?:年度?|年度?末)?", period)
        try:
            target_date = (date(*map(int, exact_date.groups())) if exact_date
                           else date(int(annual[1]), 12, 31) if annual else None)
        except ValueError:
            target_date = None
        proposed = numeric_tokens(item.raw_value)
        if len(values) != 2 or any(value <= 0 or value != value.to_integral_value() for value in values):
            warnings.append("股本附注的股份总数期初/期末列不完整或不是正整数")
        elif len(proposed) != 1 or proposed[0] != values[-1]:
            warnings.append("科目数值冲突：候选股数不是股本附注股份总数的期末余额")
        if source_date is None or target_date != source_date:
            warnings.append("期间冲突：候选股数期末日未与股本附注明确日期一致")
        else:
            checks.update(period=str(source_date.year), period_end=source_date.isoformat(), year_column=source_date.year)
        if item.unit != "股":
            warnings.append("股本附注股份总数须以股计量；不得把实收资本金额作为股数")
        else:
            checks["unit"] = "股"
        if draft.valuation_date and source_date and source_date > draft.valuation_date:
            warnings.append("股数截止日晚于估值日，不能用于该时点的历史估值")
        checks.update(
            binding="issuer_share_capital_note",
            source_row=row,
            column_alignment="explicit_opening_closing_share_total",
            par_value_evidence="ordinary shares; RMB 1 per share",
        )
        if not warnings:
            checks["scope"] = "issuer"
        return list(dict.fromkeys(warnings)), checks

    # Annual share-change tables disclose the issuer's closing shares without
    # repeating its name or fiscal date on every row. Bind the right-hand
    # 'after change / quantity' column only when the annual-report identity,
    # table unit, complete header and arithmetic all agree in the same file.
    change_rows = [line for line in block.get("text", "").splitlines()
                   if re.match(r"^\s*(?:[一二三四五六七八九十]+[、.．]\s*)?股份总数\s+", line)]
    if change_rows and any(compact(line) in quote for line in change_rows):
        return _bind_issuer_share_change(item, block, draft, identity_text, checks, warnings, change_rows)

    owners = ["本公司", "公司", *([company] if company else []), *([short_company] if len(short_company) >= 2 else [])]
    owner_pattern = "(?:" + "|".join(re.escape(owner) for owner in sorted(set(owners), key=len, reverse=True)) + ")"
    disclosure = re.compile(
        r"(?:截至|截止至?|截至至)?(?<!\d)(?P<year>(?:19|20)\d{2})年"
        r"(?:(?P<month>\d{1,2})月(?P<day>\d{1,2})日|(?:度)?末)[，,]?"
        + owner_pattern + r"(?:的)?(?:普通股股份总数|普通股股数|股份总数|总股本)"
        r"(?:为|是|共计|[:：])?(?P<amount>(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?)"
        r"(?P<unit>百万股|亿股|万股|千股|股)(?!本|息)"
    )
    matches = list(disclosure.finditer(quote))
    if len(matches) != 1:
        return [*warnings, "发行人股数缺少单一明确的截止日、当前公司总股数、数值及股数单位；请引用完整原文说明"], checks
    match = matches[0]
    try:
        source_date = date(int(match["year"]), int(match["month"] or 12), int(match["day"] or 31))
        period = re.sub(r"\s+", "", item.period)
        exact_date = re.fullmatch(r"((?:19|20)\d{2})(?:年|-|/)(\d{1,2})(?:月|-|/)(\d{1,2})日?", period)
        annual = re.fullmatch(r"((?:19|20)\d{2})(?:年度?|年度?末)?", period)
        target_date = (date(*map(int, exact_date.groups())) if exact_date
                       else date(int(annual[1]), 12, 31) if annual else None)
        if target_date != source_date:
            warnings.append("期间冲突：候选股数期末日未与原文明确截止日一致")
        else:
            checks["period"] = str(source_date.year)
            checks["period_end"] = source_date.isoformat()
        if draft.valuation_date and source_date > draft.valuation_date:
            warnings.append("股数截止日晚于估值日，不能用于该时点的历史估值")
    except ValueError:
        warnings.append("发行人股数的截止日期无法识别")
    amounts = numeric_tokens(item.raw_value)
    source_amount = numeric_tokens(match["amount"])
    if len(amounts) != 1 or amounts != source_amount or amounts[0] <= 0:
        warnings.append("科目数值冲突：候选股数未与发行人总股数原文一致")
    if item.unit != match["unit"]:
        warnings.append(f"单位冲突：原文为{match['unit']}，候选为{item.unit}；不得从金额股本推算股数")
    else:
        checks["unit"] = match["unit"]
    published = (block.get("location") or {}).get("published_at")
    if published and draft.valuation_date:
        try:
            if date.fromisoformat(str(published)[:10]) > draft.valuation_date:
                warnings.append("披露时点晚于估值日，不能用于该时点的历史估值")
        except ValueError:
            warnings.append("来源披露日期无法识别")
    if not warnings:
        checks["scope"] = "issuer"
    return list(dict.fromkeys(warnings)), checks


def _bind_issuer_share_change(item, block, draft, identity_text, checks, warnings, rows):
    """Bind the closing quantity of a complete issuer share-change table."""
    text = block.get("text", "")
    row = rows[0] if len(rows) == 1 else ""
    if len(rows) != 1 or compact(row) not in compact(item.quote):
        return [*warnings, "股份变动表总数行不唯一，需引用单一完整原文行"], checks
    prefix = text[:text.index(row)]
    heading = prefix.rfind("股份变动情况")
    local_header = prefix[heading:] if heading >= 0 else ""
    if not ("本次变动前" in local_header
            and "本次变动后" in local_header and local_header.count("数量") >= 2
            and "小计" in local_header and re.search(r"单位\s*[:：]\s*股(?!本)", local_header)):
        return [*warnings, "股份变动表缺少完整的变动前/变动后数量列与股数单位"], checks
    report_years = re.findall(r"(?<!\d)((?:19|20)\d{2})\s*年?\s*度报告", identity_text)
    if len(set(report_years)) != 1:
        return [*warnings, "股份变动表未绑定单一年度的发行人正式年报"], checks
    year = int(report_years[0])
    source_date = date(year, 12, 31)
    period = re.sub(r"\s+", "", item.period)
    exact_date = re.fullmatch(r"((?:19|20)\d{2})(?:年|-|/)(\d{1,2})(?:月|-|/)(\d{1,2})日?", period)
    annual = re.fullmatch(r"((?:19|20)\d{2})(?:年度?|年度?末)?", period)
    try:
        target_date = (date(*map(int, exact_date.groups())) if exact_date
                       else date(int(annual[1]), 12, 31) if annual else None)
    except ValueError:
        target_date = None
    if target_date != source_date:
        warnings.append("期间冲突：股份变动表期末数只对应所列年度的12月31日")
    else:
        checks.update(period=str(year), period_end=source_date.isoformat(), year_column=year)
    if draft.valuation_date and source_date > draft.valuation_date:
        warnings.append("股数截止日晚于估值日，不能用于该时点的历史估值")
    parts = re.sub(r"^\s*(?:[一二三四五六七八九十]+[、.．]\s*)?股份总数\s+", "", row).split()
    if len(parts) != 7 or not all(re.fullmatch(r"[+\-−]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?", p) for p in parts):
        return [*warnings, "股份变动表总数行列数不完整，不能猜测空白列"], checks
    values = [numeric_tokens(part)[0] for part in parts]
    before, before_pct, issued, other, change, after, after_pct = values
    if (before_pct != 100 or after_pct != 100 or before <= 0 or after <= 0
            or issued + other != change or before + change != after
            or any(value != value.to_integral_value() for value in (before, issued, other, change, after))):
        warnings.append("股份变动表期初、增减小计、期末股数或总数比例不勾稽")
    proposed = numeric_tokens(item.raw_value)
    if len(proposed) != 1 or proposed[0] != after:
        warnings.append("科目数值冲突：候选股数不是变动后数量列的股份总数")
    if item.unit != "股":
        warnings.append("单位冲突：股份变动表明确以股计量，不能使用股本金额")
    else:
        checks["unit"] = "股"
    published = (block.get("location") or {}).get("published_at")
    if published and draft.valuation_date:
        try:
            if date.fromisoformat(str(published)[:10]) > draft.valuation_date:
                warnings.append("披露时点晚于估值日，不能用于该时点的历史估值")
        except ValueError:
            warnings.append("来源披露日期无法识别")
    checks.update(binding="issuer_share_change_table", source_row=row,
                  column_alignment="explicit_before_change_after_quantity",
                  reconciliation={"before": str(before), "issued": str(issued),
                                  "other": str(other), "change": str(change), "after": str(after)})
    if not warnings:
        checks["scope"] = "issuer"
    return list(dict.fromkeys(warnings)), checks


def bind_evidence(item, block, context, draft, aliases=(), identity_text="", table_binding=None):
    """Return review reasons and machine-readable checks for one quoted value."""
    if item.role == "comparable":
        text = "\n".join(b.get("text", "") for b in context)
        identity = compact(text + "\n" + identity_text)
        warnings = []
        if not item.peer_name or compact(item.peer_name) not in identity or not item.peer_ticker or item.peer_ticker.casefold() not in identity:
            warnings.append("可比公司名称和代码未同时与原文匹配")
        if item.multiple_basis != "FY" or not re.search(r"(?i)\bFY\b|静态|年度口径", text):
            warnings.append("可比倍数分母须有明确年度FY口径，不能与TTM或预测口径混用")
        aliases = {"pe": ("pe", "p/e", "市盈率"), "ps": ("ps", "p/s", "市销率"), "ev_ebitda": ("ev/ebitda", "ev_ebitda", "企业价值倍数")}
        if item.metric not in aliases or not any(compact(a) in compact(block["text"]) for a in aliases.get(item.metric, ())):
            warnings.append("可比倍数字段未与数值所在原文匹配")
        amount = numeric_tokens(item.raw_value)
        peer_rows = [line for line in block["text"].splitlines()
                     if item.peer_ticker and item.peer_ticker.casefold() in line.casefold()
                     and any(compact(a) in compact(line) for a in aliases.get(item.metric, ()))]
        bound = False
        for line in peer_rows:
            label = next((a for a in aliases.get(item.metric, ()) if a.casefold() in line.casefold()), None)
            if label:
                values = numeric_tokens(line[line.casefold().index(label.casefold()) + len(label):])
                bound = bound or bool(amount and values and values[0] == amount[0])
        if not bound:
            warnings.append("可比数值未绑定到对应公司代码及倍数字段所在行，请缩小引文")
        try:
            as_of = date.fromisoformat(item.period)
            dates = {as_of.isoformat(), as_of.strftime("%Y/%m/%d"), f"{as_of.year}年{as_of.month}月{as_of.day}日"}
            if not any(value in text for value in dates) or (draft.valuation_date and as_of != draft.valuation_date):
                warnings.append("可比倍数的原文定价日须与估值日一致")
        except ValueError:
            warnings.append("可比倍数缺少确切定价日 YYYY-MM-DD")
        if item.unit != "ratio" or not re.search(r"倍|(?i:ratio)", text):
            warnings.append("可比倍数须由原文确认以倍计量（ratio）")
        return warnings, {"binding": "comparable", "basis": item.multiple_basis}
    if item.role != "historical":
        return [], {"binding": "user_assumption_or_policy"}
    if item.scope == "issuer":
        return _bind_issuer_shares(item, block, draft, aliases, identity_text)
    warnings, checks = [], {"version": EVIDENCE_VERSION}
    # Context after the value is not a header for that value. This matters when
    # one PDF page ends a parent statement and starts a consolidated statement.
    preceding = []
    for b in context:
        if b["block_id"] == block["block_id"] and not b.get("context_kind"):
            break
        preceding.append(b.get("text", ""))
    # Scope and unit should come from the closest table, not another table in
    # the same long page. Explicitly supplied context blocks preserve headers.
    names = sorted({item.metric, *aliases}, key=len, reverse=True)
    metric_line, block_prefix, row_error = _metric_row(item, block["text"], aliases)
    if row_error:
        warnings.append(row_error)
    prefix = "\n".join([*preceding, block_prefix or block["text"]])
    text = prefix
    prefix = re.sub(r"\b[A-Z]{1,3}\d+\s*:", "", prefix)
    scope_prefix = prefix
    # A new statement cannot inherit the previous statement's units/years.
    titles = list(re.finditer(STATEMENT_TITLE, prefix))
    if titles:
        prefix = prefix[titles[-1].start():]
    checks["context_block_ids"] = list(dict.fromkeys(b["block_id"] for b in context))
    checks["source_row"] = metric_line

    unit_pattern = r"(百万元|亿元|万元|千元|元|百万股|亿股|万股|千股|股|%)"
    units = re.findall(r"单位\s*[:：为]?\s*(?:人民币\s*)?" + unit_pattern, prefix)
    inline = re.findall(r"[\d.)）]\s*" + unit_pattern, metric_line)
    label_units = re.findall(r"[（(]\s*(?:人民币\s*)?" + unit_pattern + r"(?:\s*/\s*股)?\s*[）)]", metric_line)
    # Percentage comparison columns do not redefine monetary row units.
    if item.unit not in {"%", "ratio"}:
        inline = [unit for unit in inline if unit != "%"]
    source_unit = (label_units or inline or units or [None])[-1]
    if table_binding:
        source_unit = (label_units or inline or units or [table_binding["unit"]])[-1]
        checks["table_interpretation"] = table_binding
        anchor = table_binding["anchors"]["header"]
        if block["block_id"] == anchor["block_id"] and len(block_prefix.splitlines()) <= anchor["end_line"]:
            warnings.append("TABLE_FUTURE_HEADER: 数值行位于所选表头之前，不能借用后文年度列")
        if row_range := table_binding.get("data_ranges", {}).get(block["block_id"]):
            source_range = "\n".join(block["text"].splitlines()[row_range[0] - 1:row_range[1]])
            if not metric_line or compact(metric_line) not in compact(source_range):
                warnings.append("TABLE_ROW_OUTSIDE: 数值不在该表声明的数据行范围内")
    if item.unit in {"ratio", "%"}:
        # A ratio/percentage row can override a currency table's general unit.
        source_unit = "%" if "%" in metric_line or "％" in metric_line else "ratio" if "ratio" in metric_line.lower() or "比例" in metric_line else source_unit
    if source_unit and source_unit != item.unit:
        warnings.append(f"单位冲突：原文为{source_unit}，候选为{item.unit}")
    elif not source_unit:
        warnings.append("单位缺少原文表头或单元格依据，请补充单位所在片段")
    else:
        checks["unit"] = source_unit

    joint_columns = _joint_statement_columns(prefix)
    scopes = re.findall(SCOPE_PATTERN, scope_prefix)
    source_scope = {"合并": "consolidated", "母公司": "parent"}.get(scopes[-1] if scopes else "")
    if table_binding:
        source_scope = table_binding["scope"]
    if titles and "合并及公司" in titles[-1][0]:
        # Do not inherit another page's single-scope title for a mixed table.
        source_scope = item.scope if joint_columns and item.scope in {"consolidated", "parent"} else None
    if source_scope and source_scope != item.scope:
        warnings.append("报表口径冲突：原文为" + ("母公司" if source_scope == "parent" else "合并") + "报表")
    elif not source_scope:
        warnings.append("合并/母公司口径缺少原文标题依据，请补充报表标题")
    else:
        checks["scope"] = source_scope

    header, columns = _table_columns(prefix, metric_line)
    if table_binding:
        header, columns = table_binding["header"], table_binding["columns"]
    target = re.search(r"(?:19|20)\d{2}", item.period)
    target_year = int(target[0]) if target else None
    years = list(dict.fromkeys(int(v) for v in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", prefix)))
    years = list(dict.fromkeys([*years, *(year for year in columns if year is not None)]))
    if target_year and years and target_year not in years:
        warnings.append(f"期间冲突：候选{target_year}年未出现在对应原文表头")
    elif target_year and not years:
        warnings.append("期间缺少原文年度表头依据，请补充年份所在片段")
    elif target_year:
        checks["period"] = str(target_year)
    inline_years = list(re.finditer(r"(?<!\d)((?:19|20)\d{2})年", metric_line))
    if len(inline_years) > 1:
        # Prose can put several years on one line, without a table header.
        # Do not accept another year's amount merely because it is nearby.
        amount = numeric_tokens(item.raw_value)
        matching_years = []
        for index, year_match in enumerate(inline_years):
            end = inline_years[index + 1].start() if index + 1 < len(inline_years) else len(metric_line)
            segment = metric_line[year_match.end():end]
            if amount and amount[0] in numeric_tokens(segment, exclude_percent=item.unit not in {"%", "ratio"}):
                matching_years.append(int(year_match[1]))
        if target_year not in matching_years:
            warnings.append("年度行冲突：候选数值未绑定到原文中对应年份的说明")
            checks.pop("period", None)
        else:
            checks["inline_year"] = target_year
    # Layout/pipe tables: bind the amount's column to the year header. A
    # repeated amount is safe only when it also appears in the requested year.
    header_years = [year for year in columns if year is not None]
    if joint_columns and metric_line and len(inline_years) < 2:
        joint_header, joint = joint_columns
        row_text = metric_line
        matched = next((name for name in names if name.casefold() in row_text.casefold()), None)
        if matched:
            row_text = row_text[row_text.casefold().index(matched.casefold()) + len(matched):]
        values = _joint_statement_values(row_text)
        amount = numeric_tokens(item.raw_value)
        if values is not None and len(amount) == 1 and any(
            pair == (target_year, item.scope) and value == amount[0]
            for pair, value in zip(joint, values)
        ):
            checks.update(year_column=target_year, period=str(target_year), column_header=joint_header,
                          column_scope=item.scope, column_alignment="explicit_year_scope_pairs")
        else:
            warnings.append("年度/报表口径列冲突：合并及公司表须将数值同时绑定到明确年份及对应口径列，空白不得补零")
            checks.pop("period", None)
            checks.pop("scope", None)
    elif header_years and metric_line and len(inline_years) < 2:
        row_text = re.sub(r"\b[A-Z]{1,3}\d+\s*:", "", metric_line)
        matched = next((name for name in names if name.casefold() in row_text.casefold()), None)
        if matched:
            row_text = row_text[row_text.casefold().index(matched.casefold()) + len(matched):]
        values = numeric_tokens(row_text, exclude_percent=None not in columns and item.unit not in {"%", "ratio"})
        has_note_column = bool(re.search(r"附注|注释|(?i:\bnotes?\b)", header))
        aligned = _aligned_note_values(block_prefix, metric_line, len(columns)) if has_note_column else None
        if aligned is not None:
            values = aligned
            checks["column_alignment"] = "same_block_right_edges"
            checks["missing_year_columns"] = [year for year, value in zip(columns, values) if value is None and year is not None]
            checks["note_column_excluded"] = True
        elif (has_note_column and len(values) == len(columns) + 1
                and values[0] == values[0].to_integral_value()):
            values = values[1:]
            checks["note_column_excluded"] = True
        amount = numeric_tokens(" " + item.raw_value)
        target_is_verified_blank = (
            len(amount) == 1 and amount[0] == 0
            and len(values) == len(columns)
            and any(year == target_year and value is None
                    for year, value in zip(columns, values))
            and aligned is not None
        )
        if target_is_verified_blank:
            checks.update(
                year_column=target_year,
                period=str(target_year),
                column_header=header,
                source_blank_as_zero=True,
            )
        elif amount and len(values) == len(columns):
            positions = [columns[i] for i, value in enumerate(values) if columns[i] is not None and value == amount[0]]
            unique_positions = list(dict.fromkeys(positions))
            if target_year is None and len(unique_positions) == 1:
                # This is not an LLM guess: the year header and value column
                # are both recovered from the same source table.
                inferred_year = unique_positions[0]
                checks["year_column"] = inferred_year
                checks["period"] = str(inferred_year)
                checks["period_resolution"] = "unique_source_column"
                checks["column_header"] = header
            elif positions and target_year not in positions:
                warnings.append("年度列冲突：该数值位于" + "、".join(map(str, positions)) + "年列")
                checks.pop("period", None)
            elif target_year in positions:
                checks["year_column"] = target_year
                checks["period"] = str(target_year)
                checks["column_header"] = header
            else:
                warnings.append("候选数值未与年度表头对应的数值列绑定，请核对单元格")
        else:
            warnings.append("多年度表格列无法可靠对应，请缩小引文并提供表头和单元格位置")
    elif metric_line and len(inline_years) < 2:
        label = next((name for name in names if name.casefold() in metric_line.casefold()), None)
        value_text = metric_line[metric_line.casefold().index(label.casefold()) + len(label):] if label else metric_line
        value_text = re.sub(r"(?<!\d)(?:19|20)\d{2}年", "", value_text)
        value_text = re.split(r"[。；;]", value_text, maxsplit=1)[0]
        values = numeric_tokens(value_text, exclude_percent=item.unit not in {"%", "ratio"})
        amount = numeric_tokens(item.raw_value)
        if len(values) > 1:
            warnings.append("存在多个数值但缺少可核验的年度列，不能默认首列属于目标年度")
        elif not amount or not values or values[0] != amount[0]:
            warnings.append("科目数值冲突：候选金额不是对应科目行的数值，请缩小引文核对")

    company, ticker = compact(draft.company), (draft.ticker or "").split(".")[0]
    identity = compact(identity_text + "\n" + text)
    short_company = re.sub(r"(?:股份有限公司|有限责任公司|有限公司)$", "", company)
    if company or ticker:
        matches = (company and company in identity) or (len(short_company) >= 2 and short_company in identity) or (ticker and re.search(r"(?<!\d)" + re.escape(ticker) + r"(?!\d)", identity))
        if not matches:
            warnings.append("主体未匹配：原文或文件封面未验证为当前研究公司，请补充公司名称/证券代码依据")
        else:
            checks["issuer"] = draft.ticker or draft.company
    location = block.get("location") or {}
    published = location.get("published_at")
    if published and draft.valuation_date:
        try:
            if date.fromisoformat(str(published)[:10]) > draft.valuation_date:
                warnings.append("披露时点晚于估值日，不能用于该时点的历史估值")
        except ValueError:
            warnings.append("来源披露日期无法识别")
    return list(dict.fromkeys(warnings)), checks
