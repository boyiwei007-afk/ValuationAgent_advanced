"""Explicit synthetic observations and scripted reviews; not a document parser."""
import json
import re

from valuationagent.application.agent_runtime import WorkspaceAgentRuntime
from valuationagent.application.observation_extraction import STOCK_METRICS
from valuationagent.schemas.research import DocumentSummary


def fixture_observations(runtime, specifications=None):
    specifications = specifications or [{"metric": "revenue", "raw_value": "12000", "unit": "万元"}]
    session = runtime.session
    lines = [f"Issuer {session.draft.company} {session.draft.ticker}"]
    anchors, rows = {}, []
    def span(key, line, quote):
        anchors[key] = {"start_line": line, "quote": quote}
        return key
    span("issuer", 1, lines[0])
    for index, specification in enumerate(specifications):
        prefix = f"row{index}_"
        metric = specification["metric"]
        standard = specification.get("standard_metric", metric)
        unit = specification.get("unit", "元")
        period = specification.get("period", "2025")[:4]
        kind = "instant" if standard in STOCK_METRICS else "annual"
        period_end = specification.get("period") if kind == "instant" and len(specification.get("period", "")) == 10 else period + "-12-31"
        scope = "issuer" if standard in {"common_shares", "diluted_shares"} else "consolidated"
        basis = {"entity_name": session.draft.company, "entity_ticker": session.draft.ticker, "entity_refs": ["issuer"],
                 "scope": scope, "unit": unit, "currency": "CNY"}
        for key, text in [("scope", f"Reporting scope: {scope}"), ("unit", f"Units: {unit}; currency CNY"),
                          ("period", f"As of {period_end}" if kind == "instant" else f"From {period}-01-01 to {period}-12-31")]:
            lines.append(text)
            span(prefix + key, len(lines), text)
            if key != "period":
                basis[key + "_refs"] = [prefix + key]
        value = specification["raw_value"]
        lines.append(f"{metric}: {value} | comparative 900")
        span(prefix + "label", len(lines), metric)
        span(prefix + "value", len(lines), value)
        row = {"metric": metric, "standard_metric": standard, "raw_value": value,
               "value_ref": prefix + "value", "label_refs": [prefix + "label"],
               "period_kind": kind, "period_end": period_end, "period_refs": [prefix + "period"],
               "basis": basis, "semantic_role": "operating", "ebit_treatment": "exclude",
               "fcff_treatment": "include", "equity_bridge_treatment": "exclude",
               "rationale": "合成测试明确披露当前主体、报告范围、期间与计量单位；按原始科目解释，不从比较值推算当前值。"}
        if kind == "annual":
            row["period_start"] = period + "-01-01"
        for key in ("semantic_role", "ebit_treatment", "fcff_treatment", "equity_bridge_treatment", "uncertainties", "replaces"):
            if key in specification:
                row[key] = specification[key]
        rows.append(row)
    body = "\n".join(lines)
    meta = runtime.service.store.save_upload("synthetic-observations.txt", "historical_financials", "text/plain", body.encode())
    file_id = meta["file_id"]
    block_id = file_id + ":1"
    runtime.service.store.save_research_blocks(session.session_id, file_id, [
        {"block_id": block_id, "file_id": file_id, "text": body, "location": {"source_type": "uploaded_document"}}])
    session.documents.append(DocumentSummary(file_id=file_id, name=meta["original_name"], role="historical_financials",
                                             block_count=1, sha256=meta["sha256"], size_bytes=meta["size_bytes"],
                                             provenance_type="user_upload", authority_tier="B"))
    for anchor in anchors.values():
        anchor["block_id"] = block_id
    return {"file_id": file_id, "anchors": anchors, "basis": rows[0]["basis"], "rows": rows}


def session_runtime(service):
    session = service.create()
    session.draft.company, session.draft.ticker = "Synthetic issuer", "600123"
    return WorkspaceAgentRuntime(service, session)


def prepare_last(messages):
    result = json.loads(messages[-1]["content"])
    return "prepare_observation_review", {"fact_ids": [row["fact_id"] for row in result["rows"] if "fact_id" in row]}


def review_last(messages):
    result = json.loads(messages[-1]["content"])
    def period_readback(packet):
        if not packet.get("requires_period_readback"):
            return {}
        references = packet["interpretation"]["row"]["period_refs"]
        text = " ".join(packet["anchors"][key]["quote"] for key in references)
        return {"source_period_end": re.search(r"As of (\d{4}-\d{2}-\d{2})", text)[1]}
    return "review_observations", {"reviews": [{"fact_id": packet["fact_id"], "packet_id": packet["packet_id"],
        **period_readback(packet),
        "checks": {key: "supported" for key in ("entity", "amount", "period", "unit", "scope", "mapping")},
        "rationale": "合成模型已重新核对原始上下文及定位，确认主体、完整期间、单位、金额和科目映射一致；此测试不是真实LLM准确率。"}
        for packet in result["packets"]]}


def extraction_steps(args):
    return [("extract_observations", args), prepare_last, review_last]


class ObservationModel:
    def __init__(self, actions, *, turn_actions=("ingest", "report")):
        self.turn_actions = turn_actions
        self.actions = iter([*actions, ("finish_response", {"answer": "测试提取完成，未运行估值。"})])
        self.calls, self.kwargs = [], []
        self.catalog_calls = []
        self.pending = None

    def chat(self, messages, **kwargs):
        from control_fixtures import control_reply

        if reply := control_reply(kwargs, self.turn_actions):
            return reply
        action = self.pending if self.pending is not None else next(self.actions)
        name, arguments = action(messages) if callable(action) else action
        if name not in {tool["function"]["name"] for tool in kwargs["tools"]} and any(
                tool["function"]["name"] == "load_tools" for tool in kwargs["tools"]):
            self.pending = (name, arguments)
            self.catalog_calls.append(name)
            return {"tool_calls": [{"id": f"load_{len(self.catalog_calls)}", "type": "function",
                "function": {"name": "load_tools", "arguments": json.dumps({"names": [name]})}}]}
        self.pending = None
        self.calls.append(json.loads(json.dumps(messages)))
        self.kwargs.append(kwargs)
        assert name in {tool["function"]["name"] for tool in kwargs["tools"]}
        return {"tool_calls": [{"id": f"call_{len(self.calls)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}],
            "reasoning_content": "PRIVATE_TEST_REASONING"}


def run_turn(runtime, actions, content="提取资料"):
    from valuationagent.schemas.research import ResearchTurn
    model = ObservationModel(actions)
    runtime.service._clients[runtime.session.session_id] = model
    runtime.service.store.save_research(runtime.session)
    state = runtime.service.turn(runtime.session.session_id, ResearchTurn(content=content))
    return state, model
