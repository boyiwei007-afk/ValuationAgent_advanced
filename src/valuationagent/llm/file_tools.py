import copy
from pathlib import PurePath


TEXT_FILES = {".txt", ".md", ".csv", ".tsv", ".html", ".htm", ".json"}


def reading_schema(schema, documents):
    groups = {}
    for document in documents:
        if document.provenance_type in {"search_snippet", "official_index"}:
            continue
        suffix = PurePath(document.name).suffix.lower()
        kind = suffix if suffix in {".pdf", ".xlsx", ".json"} else "raw" if suffix in TEXT_FILES else "text"
        groups.setdefault(kind, []).append(document.file_id)
    variants = schema.get("anyOf", [])
    if not groups or len(variants) != 2:
        return schema
    content, records = variants
    result = []
    enumerate_files = sum(map(len, groups.values())) <= 64
    for kind, file_ids in groups.items():
        branch = copy.deepcopy(content)
        properties = branch["properties"]
        views = ["text"]
        if kind == ".pdf":
            views += ["pdf_geometry", "pdf_layout", "pdf_plain", "pdf_tables"]
        else:
            for key in ("page", "table_strategy", "table_index"):
                properties.pop(key, None)
        if kind == ".xlsx":
            views.append("sheet")
        else:
            for key in ("sheet", "cell_range"):
                properties.pop(key, None)
        if kind in {".json", "raw"}:
            views.append("raw_text")
        properties["view"]["enum"] = views
        if enumerate_files:
            properties["file_id"]["enum"] = file_ids
        result.append(branch)
        if kind == ".json":
            branch = copy.deepcopy(records)
            if enumerate_files:
                branch["properties"]["file_id"]["enum"] = file_ids
            result.append(branch)
    return {**schema, "anyOf": result}
