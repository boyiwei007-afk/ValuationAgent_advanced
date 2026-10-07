import json
import re


def decode_json(raw):
    try:
        return json.loads(raw, parse_float=str)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("JSON_FORMAT: 文件不是可读取的JSON；尝试raw_text检查格式，不猜测内容。") from None


def pointer_value(payload, pointer):
    if not pointer:
        return payload
    if not pointer.startswith("/") or re.search(r"~(?![01])", pointer):
        raise ValueError("JSON_POINTER: 使用根空字符串或RFC6901路径，如/data；不执行表达式。")
    current = payload
    for token in pointer[1:].split("/"):
        key = token.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, dict):
                current = current[key]
            elif isinstance(current, list) and re.fullmatch(r"0|[1-9]\d*", key):
                current = current[int(key)]
            else:
                raise KeyError(key)
        except (KeyError, IndexError, ValueError):
            raise ValueError("JSON_POINTER: 路径不存在；先inspect_file查看数组位置。") from None
    return current


def record_inventory(raw):
    payload = decode_json(raw)
    arrays, pending = [], [("", payload, 0)]
    complete = True
    while pending and len(arrays) < 20:
        path, value, depth = pending.pop()
        if isinstance(value, list):
            arrays.append((path, value))
        elif isinstance(value, dict) and depth < 4:
            complete = complete and len(value) <= 100
            pending.extend((path + "/" + key.replace("~", "~0").replace("/", "~1"), child, depth + 1)
                for key, child in reversed(list(value.items())[:100]))
        elif isinstance(value, dict) and value:
            complete = False
    headers = [(path, rows) for path, rows in arrays if rows
        and all(isinstance(item, str) and item for item in rows) and len(rows) == len(set(rows))]
    inventory = []
    for path, rows in arrays:
        entry = {"record_path": path, "record_count": len(rows),
            "fields": sorted({key for row in rows[:100] if isinstance(row, dict) for key in row})[:60],
            "row_type": type(rows[0]).__name__ if rows else "empty",
            "sample_preview": json.dumps(rows[:1], ensure_ascii=False)[:1500], "field_sample_limit": 100}
        if rows and all(isinstance(item, str) for item in rows):
            entry.update(string_values=[item[:200] for item in rows[:60]],
                string_values_truncated=len(rows) > 60 or any(len(item) > 200 for item in rows[:60]))
        if rows and all(isinstance(row, list) for row in rows[:100]):
            entry["column_candidates"] = [{"record_columns_path": header_path, "fields": columns[:60],
                "field_count": len(columns), "requires_confirmation": True}
                for header_path, columns in headers
                if header_path.rpartition("/")[0] == path.rpartition("/")[0]
                and all(len(row) == len(columns) for row in rows[:100])]
        inventory.append(entry)
    return {"arrays": inventory, "inventory_complete": complete and not pending,
        "instruction": "用read_file(view=records,record_path=数组路径,record_filters={字段:[值]},record_fields=[列])按需选取。column_candidates仅为同级等宽的列名候选，须核对语义后显式提供record_columns_path，不自动选择或改名；无须筛列时省略record_fields，先limit=1查看真实记录及日期格式。目录最多4层/20个数组。空白保留null，不推断币种或期间。"}


def default_record_selector(document, raw):
    parts = document.provider.split(":")
    if document.provenance_type == "structured_provider" and len(parts) == 4:
        if parts[0] == "tushare":
            return {"record_path": "/data/items", "record_columns_path": "/data/fields",
                "basis": "provider_transport_contract", "policy": "tushare-columnar-json-v1"}
        if parts[0] == "infoway":
            return {"record_path": "/data", "record_columns_path": None,
                "basis": "provider_transport_contract", "policy": "infoway-object-json-v1"}
    inventory = record_inventory(raw)
    choices = [entry for entry in inventory["arrays"] if entry["row_type"] == "dict"
        or entry["record_path"] == "" and entry["row_type"] == "empty"]
    if inventory["inventory_complete"] and len(choices) == 1:
        return {"record_path": choices[0]["record_path"], "record_columns_path": None,
            "basis": "unique_object_array", "policy": "json-object-records-v1"}
    return None


def read_records(raw, args):
    payload = decode_json(raw)
    records = pointer_value(payload, args.record_path)
    if not isinstance(records, list):
        paths = [item["record_path"] for item in record_inventory(raw)["arrays"]]
        raise ValueError("JSON_RECORDS: record_path必须指向数组；本文件可用数组路径：" + json.dumps(paths, ensure_ascii=False)
            + "。先inspect_file核对行与列名数组，不猜测路径。")
    columns = None
    if args.record_columns_path is not None:
        columns = pointer_value(payload, args.record_columns_path)
        if (not isinstance(columns, list) or not columns or not all(isinstance(key, str) and key for key in columns)
                or len(columns) != len(set(columns))):
            raise ValueError("JSON_COLUMNS: 列名必须是原文非空且不重复的字符串数组。")
        if any(not isinstance(row, list) or len(row) != len(columns) for row in records):
            raise ValueError("JSON_ROW_WIDTH: 每行必须与列名数组等长；不截断、不补空值。")
        records = [dict(zip(columns, row)) for row in records]
    elif any(not isinstance(row, dict) for row in records):
        candidates = [item["record_columns_path"] for entry in record_inventory(raw)["arrays"]
            if entry["record_path"] == args.record_path for item in entry.get("column_candidates", [])]
        raise ValueError("JSON_COLUMNS: 当前不是对象记录；数组行需指定真实列名数组record_columns_path。"
            + "当前等宽候选：" + json.dumps(candidates, ensure_ascii=False) + "；核对后显式选取，不猜列名、不换来源。")
    available = {key for row in records if isinstance(row, dict) for key in row}
    unknown = (set(args.record_filters) | set(args.record_fields)) - available
    if records and unknown:
        raise ValueError("JSON_FIELDS: 指定列不存在：" + ", ".join(sorted(unknown)[:10])
            + "；当前真实列：" + ", ".join(sorted(available)[:40])
            + "。先inspect_file或不筛选读取一条records，按返回原始字段与日期格式修正；不要改用整行raw_text或其他供应商字段名。")
    selected = []
    total = 0
    for index, row in enumerate(records):
        if not isinstance(row, dict) or not all(key in row and str(row[key]) in choices for key, choices in args.record_filters.items()):
            continue
        if args.offset <= total < args.offset + args.limit:
            value = {key: row[key] for key in args.record_fields if key in row} if args.record_fields else row
            selected.append((json.dumps(value, ensure_ascii=False, indent=2), {"json_pointer": f"{args.record_path}/{index}",
                "record_index": index, "selected_fields": args.record_fields, "record_filters": args.record_filters,
                "column_pointers": {key: f"{args.record_path}/{index}/" + (str(columns.index(key)) if columns else key.replace("~", "~0").replace("/", "~1")) for key in value},
                "record_path": args.record_path, "record_columns_path": args.record_columns_path,
                "decoder": "json", "numeric_policy": "十进制数字保留字面精度；投影视图不是新增披露。"}))
        total += 1
    return selected, {"record_path": args.record_path, "record_columns_path": args.record_columns_path, "total": total, "source_record_count": len(records),
        "filters": args.record_filters, "fields": args.record_fields,
        "filter_value_samples": {key: list(dict.fromkeys(str(row[key])[:200] for row in records[:100] if key in row))[:12]
            for key in args.record_filters} if not total else {}}
