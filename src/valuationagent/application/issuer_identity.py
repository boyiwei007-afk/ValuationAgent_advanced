from datetime import datetime, timezone
import re
import unicodedata

from valuationagent.application.json_records import decode_json, pointer_value
from valuationagent.core.tools import canonical
from valuationagent.market.tushare import FINANCIAL_KEYWORDS, normalize_a_share_ticker
from valuationagent.schemas.research import DocumentSummary


IDENTITY_CONTRACT = "tushare-stock-basic-identity-v1"


def identity_record(raw, ticker):
    payload = decode_json(raw)
    fields = pointer_value(payload, "/data/fields")
    rows = pointer_value(payload, "/data/items")
    if (not isinstance(fields, list) or not all(isinstance(field, str) for field in fields)
            or len(set(fields)) != len(fields) or not isinstance(rows, list)
            or any(not isinstance(row, list) or len(row) != len(fields) for row in rows)):
        raise ValueError("ISSUER_IDENTITY_SHAPE: 证券基础信息列或行宽无效，不猜测主体。")
    records = [dict(zip(fields, row)) for row in rows]
    if (len(records) != 1 or records[0].get("ts_code") != ticker
            or not isinstance(records[0].get("name"), str) or not records[0]["name"].strip()):
        raise ValueError("ISSUER_IDENTITY_MISSING: 未取得唯一匹配代码和名称的证券基础信息；不能把请求参数当作来源证明。")
    return records[0]


def normalized_name(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or "")).casefold()


def identity_document(session, ticker):
    selected = [document for document in session.documents
        if document.provenance_type == "structured_provider"
        and document.provider == f"tushare:{ticker}:identity:{IDENTITY_CONTRACT}"
        and document.issuer_identity.get("contract_version") == IDENTITY_CONTRACT]
    if len(selected) != 1:
        raise ValueError(f"ISSUER_IDENTITY_REQUIRED: {ticker} 缺少唯一证券名称/代码来源。先fetch_financial_history或acquire_financial_inputs获取身份快照；不能凭记忆配对名称。")
    return selected[0]


def require_identity(session, ticker, claimed_name=""):
    document = identity_document(session, ticker)
    record = document.issuer_identity.get("record", {})
    if record.get("ts_code") != ticker or not record.get("name"):
        raise ValueError("ISSUER_IDENTITY_CHANGED: 身份快照与证券代码不一致。")
    aliases = {normalized_name(str(record.get(key) or "")) for key in ("name", "fullname")}
    aliases.discard("")
    code_names = {normalized_name(ticker), normalized_name(ticker.split(".")[0])}
    if claimed_name and normalized_name(claimed_name) not in aliases | code_names:
        raise ValueError(f"ISSUER_IDENTITY_MISMATCH: {ticker} 的来源名称是 {record['name']}（{record.get('fullname') or ''}），不是 {claimed_name}。核对目标或可比的真实名称/代码后显式更正选样；不能只换标签沿用错误公司的数据。")
    if any(word in str(record.get("industry") or "") for word in FINANCIAL_KEYWORDS):
        raise ValueError(f"ISSUER_FINANCIAL_SCOPE: {ticker} 来源行业为{record['industry']}，不属于当前非金融A股研究范围。")
    cutoff = session.information_cutoff_date or session.draft.valuation_date
    listed = str(record.get("list_date") or "")
    if cutoff and re.fullmatch(r"\d{8}", listed) and listed > cutoff.strftime("%Y%m%d"):
        raise ValueError(f"ISSUER_NOT_LISTED_AT_CUTOFF: {ticker} 上市日晚于任务截止日。")
    return {"file_id": document.file_id, "sha256": document.sha256,
        "contract_version": IDENTITY_CONTRACT, "record": record,
        "limitation": "当前证券主数据用于核对身份，不证明历史名称、重组连续性或行业可比性。"}


def ensure_identity(runtime, provider, ticker):
    if getattr(provider, "provider_id", None) != "tushare":
        return None
    session, service = runtime.session, runtime.service
    ticker = normalize_a_share_ticker(ticker)
    key = f"tushare:{ticker}:identity:{IDENTITY_CONTRACT}"
    cached = next((document for document in session.documents if document.provider == key), None)
    if cached:
        from valuationagent.application.file_workspace import source_bytes

        _, raw = source_bytes(service.store, session, cached.file_id)
        if identity_record(raw, ticker) != cached.issuer_identity.get("record"):
            raise ValueError("ISSUER_IDENTITY_CHANGED: 身份解释与原始来源不一致。")
        return {"file_id": cached.file_id, "cached": True, **cached.issuer_identity}
    if key in runtime.data_signatures:
        raise ValueError("ISSUER_IDENTITY_UNAVAILABLE: 本轮已尝试证券基础信息；保留原始财务，换身份来源，不重复请求。")
    runtime.data_signatures.add(key)
    service._check_execution()
    snapshot = provider.client.query_snapshot("stock_basic", params={"ts_code": ticker},
        fields=["ts_code", "name", "fullname", "industry", "list_date", "delist_date", "list_status"])
    record = identity_record(snapshot.raw, ticker)
    meta = service.store.save_upload(f"{ticker}-identity.json", "evidence", "application/json", snapshot.raw)
    block = {"file_id": meta["file_id"], "block_id": f"{meta['file_id']}:1", "text": canonical(record),
        "location": {"source_type": "structured_provider", "source_url": "https://api.tushare.pro",
            "json_pointer": "/data/items/0", "schema_reference": "https://tushare.pro/document/2?doc_id=25"}}
    service.store.save_research_blocks(session.session_id, meta["file_id"], [block])
    identity = {"contract_version": IDENTITY_CONTRACT, "record": record,
        "retrieved_at": datetime.now(timezone.utc).isoformat()}
    document = DocumentSummary(file_id=meta["file_id"], name=meta["original_name"], role="issuer_identity",
        block_count=1, sha256=meta["sha256"], size_bytes=meta["size_bytes"], parse_status="parsed",
        provenance_type="structured_provider", authority_tier="B", source_confidence=.8,
        provider=key, source_url="https://api.tushare.pro", issuer_identity=identity,
        warnings=["当前证券主数据，不是历史时点公司名称、独立审计或业务可比性证明。"])
    session.documents.append(document)
    service.store.save_research(session)
    return {"file_id": document.file_id, "cached": False, **identity}


def validate_identity_bytes(store, session, proof):
    from valuationagent.application.file_workspace import source_bytes, source_document

    document = source_document(session, proof["file_id"])
    _, raw = source_bytes(store, session, document.file_id)
    ticker = proof["record"]["ts_code"]
    if (identity_record(raw, ticker) != proof["record"] or document.sha256 != proof["sha256"]
            or require_identity(session, ticker) != proof):
        raise ValueError("ISSUER_IDENTITY_CHANGED: 身份快照与冻结依据不一致，不能复用。")


def audit_request_identities(store, session, request):
    provider_rows = [row for row in request.input_records if row["source"].get("provider_binding")]
    limitation = "仅核对直接供应商输入的证券身份，不证明业务可比性、预测正确或所有事实经过独立审计。"
    if not provider_rows:
        return {"status": "not_applicable", "issuers": [], "limitation": limitation}
    names = {normalize_a_share_ticker(request.company.ticker): request.company.name or request.company.ticker} if request.company.ticker else {}
    names.update({item["ticker"]: item.get("name") or item["ticker"] for item in request.peer_screening if item.get("ticker")})
    names.update({peer.ticker: peer.name or peer.ticker for peer in request.peers})
    audited = {}
    for row in provider_rows:
        binding = row["source"].get("provider_binding")
        proof = binding.get("issuer_identity")
        ticker = row.get("entity_ticker")
        if not proof or ticker not in names or proof != require_identity(session, ticker, names[ticker]):
            raise ValueError("ISSUER_IDENTITY_REQUIRED: 正式输入缺少与目标或可比名称一致的身份快照。")
        if ticker not in audited:
            validate_identity_bytes(store, session, proof)
            audited[ticker] = {"ticker": ticker, "selected_name": names[ticker], "source_name": proof["record"]["name"],
                "file_id": proof["file_id"], "sha256": proof["sha256"]}
    return {"status": "verified", "issuers": list(audited.values()), "limitation": limitation}
