import json

import pytest

from test_file_workspace import attach_bytes
from test_multisource_extraction import runtime_at
from valuationagent.application.file_workspace import FileRead
from valuationagent.core.tools import ToolSpec
from valuationagent.llm.qwen_tools import normalize_qwen_call


def projected(runtime):
    original = ToolSpec("read_file", "read", FileRead, lambda args: args).schema()
    _, tools = runtime.adapt_request([{"role": "system", "content": "policy"}, {"role": "user", "content": "{}"}], [original])
    return tools, original


def reply(parameters):
    return {"content": "<tool_call>\n<function=read_file>" + "".join(
        f"\n<parameter={key}>{value}</parameter>" for key, value in parameters.items()) + "\n</function>\n</tool_call>"}


def test_json_only_sources_never_advertise_pdf_or_spreadsheet_arguments(tmp_path):
    runtime = runtime_at(tmp_path)
    meta = attach_bytes(runtime, "finance.json", b'{"data":[{"amount":12}]}')
    tools, original = projected(runtime)
    serialized = json.dumps(tools)
    assert "pdf_plain" not in serialized and '"page"' not in serialized and '"sheet"' not in serialized
    assert "pdf_plain" in json.dumps(original)
    valid = normalize_qwen_call(reply({"file_id": meta["file_id"], "view": "records"}), tools)
    assert json.loads(valid["tool_calls"][0]["function"]["arguments"])["view"] == "records"
    with pytest.raises(ValueError):
        normalize_qwen_call(reply({"file_id": meta["file_id"], "view": "pdf_plain", "page": "1"}), tools)


def test_mixed_files_bind_view_to_file_identity_and_keep_original_permission_filter(tmp_path):
    runtime = runtime_at(tmp_path)
    first = attach_bytes(runtime, "financial.json", b'[{"amount":12}]')
    second = attach_bytes(runtime, "note.txt", b'Separate document')
    runtime.session.documents[-1].name = "report.pdf"
    tools, _ = projected(runtime)
    assert normalize_qwen_call(reply({"file_id": second["file_id"], "view": "pdf_geometry", "page": "2"}), tools)
    with pytest.raises(ValueError):
        normalize_qwen_call(reply({"file_id": first["file_id"], "view": "pdf_geometry", "page": "2"}), tools)
    with pytest.raises(ValueError):
        normalize_qwen_call(reply({"file_id": second["file_id"], "view": "records"}), tools)
    runtime.session.turn_control.effects = []
    assert not projected(runtime)[0]
