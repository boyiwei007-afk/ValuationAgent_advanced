from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
import time
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from valuationagent.schemas.models import (
    ChatMessage,
    RunEvent,
    RunRecord,
    RunStatus,
    ValuationOutput,
    ValuationRequest,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, default=str)


class WorkspaceStateError(ValueError):
    pass


class SQLiteRunStore:
    """Small local run store. Secrets are deliberately kept out of this class."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "workspace-agent.sqlite3"
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open one transaction-scoped connection and always release its handle."""
        connection = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS run_sources (
                    run_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_jobs (
                    session_id TEXT NOT NULL, request_id TEXT NOT NULL, body_hash TEXT NOT NULL,
                    body_json TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created REAL NOT NULL, updated REAL NOT NULL,
                    PRIMARY KEY(session_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS research_jobs_recent ON research_jobs(session_id,created DESC);
                CREATE TABLE IF NOT EXISTS event_summaries (
                    run_id TEXT NOT NULL, sequence INTEGER NOT NULL, summary_json TEXT NOT NULL,
                    PRIMARY KEY(run_id,sequence)
                );
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                    session_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_documents (
                    session_id TEXT NOT NULL, file_id TEXT NOT NULL,
                    blocks_json TEXT NOT NULL, PRIMARY KEY(session_id, file_id)
                );
                CREATE TABLE IF NOT EXISTS research_reports (
                    session_id TEXT NOT NULL, source_revision INTEGER NOT NULL,
                    report_id TEXT NOT NULL, report_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, report_id)
                );
                CREATE TABLE IF NOT EXISTS workspace_artifacts (
                    session_id TEXT NOT NULL, artifact_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, content BLOB NOT NULL,
                    PRIMARY KEY(session_id, artifact_id)
                );
                CREATE TABLE IF NOT EXISTS lineage (
                    run_id TEXT PRIMARY KEY, root_id TEXT NOT NULL, parent_id TEXT,
                    revision INTEGER NOT NULL, reason TEXT,
                    UNIQUE(root_id, revision)
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    root_id TEXT NOT NULL, cache_key TEXT NOT NULL, run_id TEXT NOT NULL,
                    tool TEXT NOT NULL, input_json TEXT NOT NULL, output_json TEXT NOT NULL,
                    PRIMARY KEY(root_id, cache_key)
                );
                CREATE TABLE IF NOT EXISTS leases (
                    run_id TEXT PRIMARY KEY, owner TEXT NOT NULL, expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    review_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    message_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_type TEXT,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    storage_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS valuation_workspaces (
                    workspace_id TEXT PRIMARY KEY,
                    research_session_id TEXT NOT NULL UNIQUE,
                    revision INTEGER NOT NULL,
                    workspace_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS valuation_workspaces_recent
                    ON valuation_workspaces(updated_at DESC);
                CREATE TABLE IF NOT EXISTS workspace_records (
                    workspace_id TEXT NOT NULL,
                    record_kind TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(workspace_id, record_kind, record_id)
                );
                CREATE INDEX IF NOT EXISTS workspace_records_by_kind
                    ON workspace_records(workspace_id, record_kind, created_at);
                """
            )

    _WORKSPACE_RECORD_IDS = {
        "requirement": "requirement_id",
        "evidence": "evidence_id",
        "evidence_usage": "usage_id",
        "fact": "fact_id",
        "assumption": "assumption_id",
        "action": "action_id",
        "checkpoint": "checkpoint_id",
        "finding": "finding_id",
        "disposition": "disposition_id",
        "decision": "decision_id",
        "model_spec": "model_spec_id",
        "calculation": "calculation_id",
        "version": "version_id",
    }

    def create_workspace(self, workspace):
        """Persist one workspace without duplicating its research-session data."""
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO valuation_workspaces VALUES(?,?,?,?,?)",
                (
                    workspace.workspace_id,
                    workspace.research_session_id,
                    workspace.revision,
                    workspace.model_dump_json(),
                    workspace.updated_at.isoformat(),
                ),
            )
        return workspace

    def freeze_run_sources(self, run_id, session):
        snapshot = {
            "research": session.model_dump(mode="json"),
            "source_blocks": {
                document.file_id: self.research_blocks(session.session_id, document.file_id)
                for document in session.documents
            },
            "research_events": [
                event.model_dump(mode="json")
                for event in self.list_events(session.session_id)
            ],
        }
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO run_sources(run_id,snapshot_json) VALUES(?,?) ON CONFLICT(run_id) DO NOTHING",
                (run_id, _json(snapshot)),
            )

    def run_sources(self, run_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_json FROM run_sources WHERE run_id=?", (run_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def get_workspace(self, workspace_id):
        from valuationagent.schemas.workspace import ValuationWorkspace

        with self._connect() as db:
            row = db.execute(
                "SELECT workspace_json FROM valuation_workspaces WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
        if row is None:
            raise KeyError(workspace_id)
        return ValuationWorkspace.model_validate_json(row[0])

    def workspace_for_research(self, research_session_id):
        with self._connect() as db:
            row = db.execute(
                "SELECT workspace_id FROM valuation_workspaces WHERE research_session_id=?",
                (research_session_id,),
            ).fetchone()
        return self.get_workspace(row[0]) if row else None

    def workspace_for_run(self, run_id):
        """Return the workspace that owns a valuation run, if any.

        The lookup deliberately uses the immutable version/model-spec records
        instead of mutable UI state.  A historical V1 therefore remains
        discoverable after V2 becomes active.
        """
        with self._connect() as db:
            rows = db.execute(
                """SELECT workspace_id,record_json FROM workspace_records
                   WHERE record_kind IN ('version','model_spec')
                   ORDER BY created_at"""
            ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row["record_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if payload.get("run_id") == run_id:
                return self.get_workspace(row["workspace_id"])
        return None

    def save_workspace(self, workspace):
        """Optimistic workspace update; stale browser tabs cannot overwrite state."""
        previous = workspace.revision
        updated = workspace.model_copy(
            update={"revision": previous + 1, "updated_at": _utc_now()}
        )
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """UPDATE valuation_workspaces
                   SET revision=?,workspace_json=?,updated_at=?
                   WHERE workspace_id=? AND revision=?""",
                (
                    updated.revision,
                    updated.model_dump_json(),
                    updated.updated_at.isoformat(),
                    workspace.workspace_id,
                    previous,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("工作区已被更新，请刷新后重试。")
        workspace.revision = updated.revision
        workspace.updated_at = updated.updated_at
        return workspace

    def list_workspaces(self, limit=30):
        from valuationagent.schemas.workspace import ValuationWorkspace
        with self._connect() as db:
            rows = db.execute(
                "SELECT workspace_json FROM valuation_workspaces ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [ValuationWorkspace.model_validate_json(row[0]) for row in rows]

    def delete_workspace(self, workspace_id, *, expected_revision):
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            workspace = db.execute("SELECT * FROM valuation_workspaces WHERE workspace_id=?", (workspace_id,)).fetchone()
            if workspace is None:
                raise KeyError(workspace_id)
            if workspace["revision"] != expected_revision:
                raise ValueError("任务已更新，请刷新历史列表后重新确认删除。")
            session_id = workspace["research_session_id"]
            if db.execute("SELECT 1 FROM research_jobs WHERE session_id=? AND status IN ('queued','running')", (session_id,)).fetchone():
                raise ValueError("任务仍在排队或执行，请先停止并等待执行结束，再删除。")
            run_ids = {row[0] for row in db.execute(
                "SELECT json_extract(record_json,'$.run_id') FROM workspace_records WHERE workspace_id=?", (workspace_id,)
            ) if row[0]}
            payload = json.loads(workspace["workspace_json"])
            if payload.get("active_run_id"):
                run_ids.add(payload["active_run_id"])
            session = db.execute("SELECT session_json FROM research_sessions WHERE session_id=?", (session_id,)).fetchone()
            if session and json.loads(session[0]).get("valuation_run_id"):
                run_ids.add(json.loads(session[0])["valuation_run_id"])
            lineage = db.execute("SELECT run_id,root_id FROM lineage").fetchall()
            roots = {row["root_id"] for row in lineage if row["run_id"] in run_ids}
            run_ids.update(row["run_id"] for row in lineage if row["root_id"] in roots)
            other_refs = db.execute("SELECT json_extract(record_json,'$.run_id') FROM workspace_records WHERE workspace_id<>?", (workspace_id,)).fetchall()
            other_refs += db.execute("SELECT json_extract(workspace_json,'$.active_run_id') FROM valuation_workspaces WHERE workspace_id<>?", (workspace_id,)).fetchall()
            other_refs += db.execute("SELECT json_extract(session_json,'$.valuation_run_id') FROM research_sessions WHERE session_id<>?", (session_id,)).fetchall()
            if run_ids.intersection(row[0] for row in other_refs):
                raise ValueError("存在其他任务引用的计算版本，不能连带删除；请先处理共享引用。")
            for run_id in {session_id, *run_ids}:
                if db.execute("SELECT 1 FROM leases WHERE run_id=? AND expires>?", (run_id, time.time())).fetchone():
                    raise ValueError("任务仍持有执行锁，请等待执行结束后再删除。")
            for run_id in run_ids:
                if db.execute("SELECT 1 FROM runs WHERE run_id=? AND status IN ('created','running')", (run_id,)).fetchone():
                    raise ValueError("计算仍在排队或执行，请先停止并等待执行结束，再删除。")
            for run_id in {session_id, *run_ids}:
                for table in ("messages", "events", "event_summaries", "leases", "run_sources", "lineage", "runs"):
                    db.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
                db.execute("DELETE FROM checkpoints WHERE run_id=? OR root_id=?", (run_id, run_id))
            for table in ("research_jobs", "research_documents", "research_reports", "workspace_artifacts", "research_sessions"):
                db.execute(f"DELETE FROM {table} WHERE session_id=?", (session_id,))
            db.execute("DELETE FROM workspace_records WHERE workspace_id=?", (workspace_id,))
            db.execute("DELETE FROM valuation_workspaces WHERE workspace_id=?", (workspace_id,))
        return {"session_id": session_id, "run_ids": sorted(run_ids)}

    def save_workspace_record(self, workspace_id, kind, record, *, immutable=False):
        id_field = self._WORKSPACE_RECORD_IDS.get(kind)
        if not id_field:
            raise ValueError(f"unsupported workspace record kind: {kind}")
        record_id = getattr(record, id_field)
        now = _utc_now().isoformat()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM valuation_workspaces WHERE workspace_id=?", (workspace_id,)).fetchone():
                raise KeyError(workspace_id)
            if immutable and db.execute(
                "SELECT 1 FROM workspace_records WHERE workspace_id=? AND record_kind=? AND record_id=?",
                (workspace_id, kind, record_id),
            ).fetchone():
                raise ValueError(f"immutable {kind} already exists: {record_id}")
            db.execute(
                """INSERT INTO workspace_records
                   (workspace_id,record_kind,record_id,record_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(workspace_id,record_kind,record_id) DO UPDATE SET
                       record_json=excluded.record_json, updated_at=excluded.updated_at""",
                (workspace_id, kind, record_id, record.model_dump_json(), now, now),
            )
        return record

    def get_workspace_record(self, workspace_id, kind, record_id, model):
        if kind not in self._WORKSPACE_RECORD_IDS:
            raise ValueError(f"unsupported workspace record kind: {kind}")
        with self._connect() as db:
            row = db.execute(
                """SELECT record_json FROM workspace_records
                   WHERE workspace_id=? AND record_kind=? AND record_id=?""",
                (workspace_id, kind, record_id),
            ).fetchone()
        if row is None:
            raise KeyError(record_id)
        return model.model_validate_json(row[0])

    def list_workspace_records(self, workspace_id, kind, model, *, limit=500):
        if kind not in self._WORKSPACE_RECORD_IDS:
            raise ValueError(f"unsupported workspace record kind: {kind}")
        with self._connect() as db:
            rows = db.execute(
                """SELECT record_json FROM workspace_records
                   WHERE workspace_id=? AND record_kind=?
                   ORDER BY created_at, rowid LIMIT ?""",
                (workspace_id, kind, max(1, min(limit, 2000))),
            ).fetchall()
        return [model.model_validate_json(row[0]) for row in rows]

    def create_run(
        self,
        run_id: str,
        request: ValuationRequest,
        *,
        parent_id: str | None = None,
        reason: str | None = None,
    ) -> RunRecord:
        now = _utc_now()
        parent = self.get_run(parent_id) if parent_id else None
        root_id = (parent.root_run_id or parent.run_id) if parent else run_id
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            revision = connection.execute(
                "SELECT COALESCE(MAX(revision),0)+1 FROM lineage WHERE root_id=?",
                (root_id,),
            ).fetchone()[0]
            if parent and revision == 1:
                connection.execute(
                    "INSERT OR IGNORE INTO lineage VALUES(?,?,?,?,?)",
                    (parent.run_id, root_id, None, 1, None),
                )
                revision = 2
            connection.execute(
                "INSERT INTO runs(run_id,status,request_json,created_at,updated_at) VALUES(?,?,?,?,?)",
                (
                    run_id,
                    RunStatus.CREATED.value,
                    request.model_dump_json(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            connection.execute(
                "INSERT INTO lineage VALUES(?,?,?,?,?)",
                (run_id, root_id, parent_id, revision, reason),
            )
        return self.get_run(run_id)

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        result: ValuationOutput | None = None,
        error: dict[str, Any] | None = None,
        review: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status=?, result_json=?,
                    error_json=?, review_json=?, updated_at=?
                WHERE run_id=?
                """,
                (
                    status.value,
                    _json(result) if result is not None else None,
                    _json(error) if error is not None else None,
                    _json(review) if review is not None else None,
                    _utc_now().isoformat(),
                    run_id,
                ),
            )
            if connection.total_changes == 0:
                raise KeyError(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            lineage = connection.execute(
                "SELECT * FROM lineage WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunRecord(
            run_id=row["run_id"],
            status=row["status"],
            request=ValuationRequest.model_validate_json(row["request_json"]),
            input_hash=hashlib.sha256(row["request_json"].encode("utf-8")).hexdigest(),
            result=ValuationOutput.model_validate_json(row["result_json"])
            if row["result_json"]
            else None,
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            review=json.loads(row["review_json"]) if row["review_json"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            root_run_id=lineage["root_id"] if lineage else run_id,
            parent_run_id=lineage["parent_id"] if lineage else None,
            revision=lineage["revision"] if lineage else 1,
            revision_reason=lineage["reason"] if lineage else None,
            workflow_version="0.5.0" if lineage else "0.1.0",
        )

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [self.get_run(row["run_id"]) for row in rows]

    def append_event(self, run_id: str, **values: Any) -> RunEvent:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence),0)+1 AS next_sequence FROM events WHERE run_id=?",
                (run_id,),
            ).fetchone()
            sequence = int(row["next_sequence"])
            event = RunEvent(run_id=run_id, sequence=sequence, **values)
            connection.execute(
                "INSERT INTO events(run_id,sequence,event_json) VALUES(?,?,?)",
                (run_id, sequence, event.model_dump_json()),
            )
            summary = event.model_dump(mode="json")
            summary["has_detail"] = bool(summary.pop("payload", None))
            connection.execute("INSERT INTO event_summaries VALUES(?,?,?)", (run_id, sequence, _json(summary)))
        return event

    def list_events(self, run_id: str, after: int = 0) -> list[RunEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_json FROM events WHERE run_id=? AND sequence>? ORDER BY sequence",
                (run_id, after),
            ).fetchall()
        return [RunEvent.model_validate_json(row["event_json"]) for row in rows]

    def add_message(
        self,
        run_id: str,
        role: str,
        content: str,
        stage: str | None = None,
        related_run_id: str | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex}",
            run_id=run_id,
            role=role,
            content=content,
            stage=stage,
            related_run_id=related_run_id,
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(message_id,run_id,created_at,message_json) VALUES(?,?,?,?)",
                (
                    message.message_id,
                    run_id,
                    message.created_at.isoformat(),
                    message.model_dump_json(),
                ),
            )
        return message

    def list_messages(self, run_id: str) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT message_json FROM messages WHERE run_id=? ORDER BY rowid",
                (run_id,),
            ).fetchall()
        return [ChatMessage.model_validate_json(row["message_json"]) for row in rows]

    def event_page(self, run_id, after=None, limit=100):
        limit = max(1, min(limit, 200))
        with self._connect() as db:
            latest = db.execute("SELECT COALESCE(MAX(sequence),0) FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
            cursor = max(0, latest - limit) if after is None else max(0, after)
            rows = db.execute("""SELECT COALESCE(s.summary_json,json_remove(e.event_json,'$.payload')) AS summary
                FROM events e LEFT JOIN event_summaries s ON e.run_id=s.run_id AND e.sequence=s.sequence
                WHERE e.run_id=? AND e.sequence>? ORDER BY e.sequence LIMIT ?""", (run_id, cursor, limit)).fetchall()
        items = [json.loads(row[0]) for row in rows]
        end = items[-1]["sequence"] if items else cursor
        return {"events": items, "cursor": end, "has_more": end < latest, "total": latest}

    def event_detail(self, run_id, sequence):
        with self._connect() as db:
            row = db.execute("SELECT event_json FROM events WHERE run_id=? AND sequence=?", (run_id, sequence)).fetchone()
        if row is None:
            raise KeyError(sequence)
        return json.loads(row[0])

    def message_page(self, run_id, before=None, limit=60):
        limit = max(1, min(100, limit))
        with self._connect() as db:
            rows = db.execute("""SELECT rowid,message_json FROM messages WHERE run_id=? AND rowid<?
                ORDER BY rowid DESC LIMIT ?""", (run_id, before or 9223372036854775807, min(100, max(1, limit)) + 1)).fetchall()
        more = len(rows) > limit
        rows = rows[:limit]
        return {"messages": [json.loads(row[1]) for row in reversed(rows)],
                "before": rows[-1][0] if rows else None, "has_more": more}

    def reserve_research_job(self, session_id, request_id, body):
        encoded = _json(body)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        now = time.time()
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM research_sessions WHERE session_id=?", (session_id,)).fetchone():
                raise KeyError(session_id)
            row = db.execute("SELECT * FROM research_jobs WHERE session_id=? AND request_id=?", (session_id, request_id)).fetchone()
            if row:
                if row["body_hash"] != digest:
                    raise ValueError("同一请求标识不能提交不同内容，请创建新请求。")
                return False
            if db.execute("SELECT 1 FROM research_jobs WHERE session_id=? AND status IN ('queued','running')", (session_id,)).fetchone():
                raise ValueError("当前研究正在执行，请等待完成或停止后继续。")
            db.execute("INSERT INTO research_jobs VALUES(?,?,?,?,?,?,?,?,?)", (session_id, request_id, digest, encoded, "queued", "queued", 0, now, now))
        return True

    def research_job(self, session_id, request_id=None):
        with self._connect() as db:
            row = db.execute("SELECT * FROM research_jobs WHERE session_id=?" + (" AND request_id=?" if request_id else " ORDER BY created DESC LIMIT 1"),
                             (session_id, request_id) if request_id else (session_id,)).fetchone()
        return dict(row) if row else None

    def update_research_job(self, session_id, request_id, *, status=None, stage=None, cancel=False):
        with self._connect() as db:
            db.execute("""UPDATE research_jobs SET status=COALESCE(?,status), stage=COALESCE(?,stage),
                cancel_requested=MAX(cancel_requested,?), updated=? WHERE session_id=? AND request_id=?""",
                       (status, stage, int(cancel), time.time(), session_id, request_id))

    def save_upload(
        self, name: str, role: str, content_type: str | None, content: bytes
    ) -> dict[str, Any]:
        if role not in {
            "historical_financials",
            "assumptions",
            "comparables",
            "evidence",
        }:
            raise ValueError("unknown file role")
        if not content:
            raise ValueError("empty files are not accepted")
        if len(content) > 50 * 1024 * 1024:
            raise ValueError("file exceeds the 50 MB limit")
        file_id = f"file_{uuid.uuid4().hex}"
        safe_suffix = Path(name).suffix.lower()
        if safe_suffix not in {
            ".pdf",
            ".docx",
            ".xlsx",
            ".csv",
            ".tsv",
            ".json",
            ".txt",
            ".md",
            ".html",
            ".htm",
        }:
            raise ValueError(
                "supported file types: PDF, DOCX, Excel, CSV/TSV, HTML, JSON, TXT, Markdown"
            )
        target = (self.upload_dir / f"{file_id}{safe_suffix}").resolve()
        if self.upload_dir not in target.parents:
            raise ValueError("invalid file name")
        target.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        created_at = _utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO files(file_id,original_name,role,content_type,sha256,size_bytes,storage_path,created_at)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    file_id,
                    Path(name).name,
                    role,
                    content_type,
                    digest,
                    len(content),
                    str(target),
                    created_at.isoformat(),
                ),
            )
        return {
            "file_id": file_id,
            "original_name": Path(name).name,
            "role": role,
            "content_type": content_type,
            "sha256": digest,
            "size_bytes": len(content),
            "created_at": created_at.isoformat(),
        }

    def get_file(self, file_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM files WHERE file_id=?", (file_id,)
            ).fetchone()
        if not row:
            raise ValueError(f"未找到文件 {file_id}")
        return dict(row)

    def revisions(self, run_id: str) -> list[RunRecord]:
        root = self.get_run(run_id).root_run_id or run_id
        with self._connect() as db:
            ids = db.execute(
                "SELECT run_id FROM lineage WHERE root_id=? ORDER BY revision", (root,)
            ).fetchall()
        return [self.get_run(row[0]) for row in ids] or [self.get_run(run_id)]

    def checkpoint(self, run_id: str, key: str) -> dict | None:
        root = self.get_run(run_id).root_run_id or run_id
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM checkpoints WHERE root_id=? AND cache_key=?", (root, key)
            ).fetchone()
        return dict(row) if row else None

    def save_checkpoint(
        self, run_id: str, key: str, tool: str, inputs: str, output: str
    ) -> None:
        root = self.get_run(run_id).root_run_id or run_id
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES(?,?,?,?,?,?)",
                (root, key, run_id, tool, inputs, output),
            )

    def artifacts(self, run_id: str) -> list[dict]:
        root = self.get_run(run_id).root_run_id or run_id
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM checkpoints WHERE root_id=?", (root,)
            ).fetchall()
        return [
            {
                "artifact_id": r["cache_key"],
                "run_id": r["run_id"],
                "tool": r["tool"],
                "inputs": json.loads(r["input_json"]),
                "output": json.loads(r["output_json"]),
            }
            for r in rows
        ]

    def acquire(self, run_id: str, owner: str) -> bool:
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM runs WHERE run_id=? UNION ALL SELECT 1 FROM research_sessions WHERE session_id=?", (run_id, run_id)).fetchone():
                raise KeyError(run_id)
            row = db.execute(
                "SELECT expires FROM leases WHERE run_id=?", (run_id,)
            ).fetchone()
            if row and row[0] > time.time():
                return False
            db.execute(
                "INSERT OR REPLACE INTO leases VALUES(?,?,?)",
                (run_id, owner, time.time() + 30),
            )
        return True

    def active(self, run_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT expires FROM leases WHERE run_id=?", (run_id,)
            ).fetchone()
        return bool(row and row[0] > time.time())

    def heartbeat(self, run_id: str, owner: str) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE leases SET expires=? WHERE run_id=? AND owner=?",
                (time.time() + 30, run_id, owner),
            )

    def release(self, run_id: str, owner: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM leases WHERE run_id=? AND owner=?", (run_id, owner))

    def create_research(self, session):
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO research_sessions VALUES(?,?,?,?)", (
                session.session_id, session.revision, session.model_dump_json(),
                session.updated_at.isoformat()))
        return session

    def get_research(self, session_id):
        from valuationagent.schemas.research import ResearchSession
        with self._connect() as db:
            row = db.execute("SELECT session_json FROM research_sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        try:
            return ResearchSession.model_validate_json(row[0])
        except ValidationError:
            raise WorkspaceStateError(
                "WORKSPACE_STATE_INCOMPATIBLE: 此工作区记录与当前数据契约不匹配（旧格式或损坏）。"
                "原始记录未修改，也不会自动套用旧流程。请重启当前服务并新建工作区；原文件保留供归档。"
            ) from None

    def save_research(self, session):
        previous = session.revision
        updated = session.model_copy(update={"revision": previous + 1, "updated_at": _utc_now()})
        with self._lock, self._connect() as db:
            cursor = db.execute(
                "UPDATE research_sessions SET revision=?,session_json=?,updated_at=? WHERE session_id=? AND revision=?",
                (updated.revision, updated.model_dump_json(), updated.updated_at.isoformat(), session.session_id, previous))
            if cursor.rowcount != 1:
                raise ValueError("会话已更新，请刷新后重试。")
        session.revision, session.updated_at = updated.revision, updated.updated_at
        return session

    def list_research(self, limit=30):
        with self._connect() as db:
            rows = db.execute("SELECT session_id FROM research_sessions ORDER BY updated_at DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()
        return [self.get_research(row[0]) for row in rows]

    def save_research_report(self, session_id, document, summary):
        with self._lock, self._connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO research_reports VALUES(?,?,?,?,?)",
                (session_id, document["source_revision"], document["report_id"],
                 _json(document), _json(summary)),
            )
            return cursor.rowcount == 1

    def research_report(self, session_id, *, summary=False, report_id=None):
        column = "summary_json" if summary else "report_json"
        with self._connect() as db:
            row = db.execute(
                f"SELECT {column} FROM research_reports WHERE session_id=?"
                + (" AND report_id=?" if report_id else "")
                + " ORDER BY source_revision DESC, rowid DESC LIMIT 1",
                (session_id, report_id) if report_id else (session_id,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_research_blocks(self, session_id, file_id, blocks):
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM research_sessions WHERE session_id=?", (session_id,)).fetchone():
                raise KeyError(session_id)
            db.execute("INSERT OR REPLACE INTO research_documents VALUES(?,?,?)", (session_id, file_id, _json(blocks)))

    def research_source_location(self, session_id, file_id):
        with self._connect() as db:
            row = db.execute("SELECT json_extract(blocks_json, '$[0].location') FROM research_documents WHERE session_id=? AND file_id=?",
                             (session_id, file_id)).fetchone()
        return json.loads(row[0]) if row and row[0] else {}

    def research_blocks(self, session_id, file_id):
        with self._connect() as db:
            row = db.execute("SELECT blocks_json FROM research_documents WHERE session_id=? AND file_id=?", (session_id, file_id)).fetchone()
        if row is None:
            raise ValueError("文件尚未加入当前会话，请先上传。")
        return json.loads(row[0])

    def save_artifact(self, session_id, metadata, content):
        if not isinstance(content, bytes) or not content or len(content) > 10 * 1024 * 1024:
            raise ValueError("ARTIFACT_SIZE_LIMIT: 输出文件必须在1字节至10 MB之间。")
        digest = hashlib.sha256(content).hexdigest()
        identity = hashlib.sha256((session_id + _json(metadata) + ("" if metadata.get("kind") == "result_report" else digest)).encode()).hexdigest()[:32]
        artifact_id = "artifact_" + identity
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM research_sessions WHERE session_id=?", (session_id,)).fetchone():
                raise ValueError("ARTIFACT_SCOPE_INVALID: 工作区不存在。")
            existing = db.execute("SELECT metadata_json, content FROM workspace_artifacts WHERE session_id=? AND artifact_id=?", (session_id, artifact_id)).fetchone()
            if existing:
                record = json.loads(existing[0])
                if record["sha256"] != hashlib.sha256(existing[1]).hexdigest() or record["size_bytes"] != len(existing[1]):
                    raise ValueError("ARTIFACT_INTEGRITY: 已有输出文件完整性检查失败。")
                return record
            count, size = db.execute("SELECT count(*), coalesce(sum(length(content)),0) FROM workspace_artifacts WHERE session_id=?", (session_id,)).fetchone()
            if count >= 100 or size + len(content) > 50 * 1024 * 1024:
                raise ValueError("ARTIFACT_QUOTA: 工作区输出达到100个文件或50 MB限额。")
            record = {**metadata, "artifact_id": artifact_id, "sha256": digest,
                      "size_bytes": len(content), "created_at": _utc_now().isoformat(), "number": count + 1}
            db.execute("INSERT INTO workspace_artifacts VALUES(?,?,?,?)", (session_id, artifact_id, _json(record), content))
            return record

    def list_artifacts(self, session_id):
        with self._connect() as db:
            rows = db.execute("SELECT metadata_json FROM workspace_artifacts WHERE session_id=? ORDER BY rowid DESC", (session_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_artifact(self, session_id, artifact_id):
        with self._connect() as db:
            row = db.execute("SELECT metadata_json, content FROM workspace_artifacts WHERE session_id=? AND artifact_id=?", (session_id, artifact_id)).fetchone()
        if row is None:
            raise ValueError("ARTIFACT_NOT_FOUND: 输出文件不属于当前工作区。")
        metadata, content = json.loads(row[0]), bytes(row[1])
        if metadata["sha256"] != hashlib.sha256(content).hexdigest() or metadata["size_bytes"] != len(content):
            raise ValueError("ARTIFACT_INTEGRITY: 输出文件完整性检查失败。")
        return metadata, content
