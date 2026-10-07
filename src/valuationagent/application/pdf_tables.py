import io


def read_pdf_tables(raw, page_number, strategy, table_index, check_cancel):
    import pdfplumber

    settings = {"vertical_strategy": strategy, "horizontal_strategy": strategy}
    rows, inventory = [], []
    with pdfplumber.open(io.BytesIO(raw), pages=[page_number]) as document:
        check_cancel()
        page = document.pages[0]
        if len(page.chars) > 50000 or len(page.edges) > 10000:
            raise ValueError("PDF_TABLE_COMPLEXITY: 单页字符或线段过多；请读取文本或页图，不反复检测表格。")
        tables = page.find_tables(settings)
        if len(tables) > 50:
            raise ValueError("PDF_TABLE_LIMIT: 单页检测到超过50个表格区域；请换文本视图或页图核对布局。")
        if table_index is not None and table_index > len(tables):
            raise ValueError(f"TABLE_NOT_FOUND: 本页当前策略仅检测到{len(tables)}个表；省略table_index查看目录，或换文本视图。")
        cell_count = 0
        for number, table in enumerate(tables, 1):
            check_cancel()
            table_rows = table.rows
            columns = max((len(row.cells) for row in table_rows), default=0)
            inventory.append({"table": number, "bbox": list(table.bbox), "rows": len(table_rows), "columns": columns})
            if table_index is not None and table_index != number:
                continue
            cell_count += sum(len(row.cells) for row in table_rows)
            if cell_count > 5000:
                raise ValueError("PDF_TABLE_LIMIT: 单次视图超过5000格；指定table_index缩小范围。")
            extracted = table.extract(x_tolerance=1, y_tolerance=3)
            for row_number, (row, values) in enumerate(zip(table_rows, extracted), 1):
                check_cancel()
                text = " | ".join(value if value is not None else "" for value in values)
                if len(text) > 2000:
                    raise ValueError("TABLE_ROW_TOO_LARGE: 单行表格超过2000字符；使用文本或页图核对，不截断单元格。")
                cells = [{"address": f"T{number}R{row_number}C{column}", "row": row_number, "column": column,
                    "value": value, "bbox": list(bbox) if bbox is not None else None,
                    "status": "no_separate_cell" if bbox is None else "text" if value else "empty"}
                    for column, (bbox, value) in enumerate(zip(row.cells, values), 1)]
                rows.append((text, {"page": page_number, "table": number, "row": row_number,
                    "bbox": list(row.bbox), "coordinate_system": "pdf_points_top_left",
                    "cells": values, "cell_details": cells, "decoder": "pdfplumber",
                    "decoder_version": pdfplumber.__version__, "table_strategy": strategy,
                    "x_tolerance": 1, "y_tolerance": 3}))
    return rows, {"page": page_number, "strategy": strategy, "tables": inventory,
        "selected_table": table_index, "text_layer_only": True, "semantic_verified": False,
        "instruction": "表格边界由布局检测，不是财务判断。行列是检测网格，可能受底色矩形干扰，不保证一行对应完整科目。按bbox核对表头/单位/年度；跨行文本保留换行。"
            "empty是可检测格无文本，no_separate_cell可能是合并或漏检，两者均不补0、不继承相邻值。"
            "本视图不含表外标题和附注，必要时同页pdf_geometry或页图补读；未检测到表不等于没有数据。"}
