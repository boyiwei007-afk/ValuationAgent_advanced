"""Replay saved LLM extraction calls offline in an isolated copy, without changing source tasks."""
import argparse
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from valuationagent.application.agent_runtime import WorkspaceAgentRuntime
from valuationagent.application.observation_extraction import ExtractObservations, extract_observations
from valuationagent.application.research import ResearchService
from valuationagent.storage.sqlite import SQLiteRunStore


def replay(directory, session_id=None):
    source_path = directory.resolve() / "workspace-agent.sqlite3"
    with tempfile.TemporaryDirectory(prefix="valuation-replay-") as target:
        store = SQLiteRunStore(target)
        source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
        try:
            with store._connect() as destination:
                source.backup(destination)
        finally:
            source.close()
        store.upload_dir = directory.resolve() / "uploads"
        with store._connect() as database:
            if not session_id:
                session_id = database.execute("SELECT session_id FROM research_sessions ORDER BY updated_at DESC LIMIT 1").fetchone()[0]
            calls = [json.loads(row[0]) for row in database.execute(
                "SELECT event_json FROM events WHERE run_id=? ORDER BY sequence", (session_id,))]
        session = store.get_research(session_id)
        session.facts = []
        runtime = WorkspaceAgentRuntime(ResearchService(store), session)
        row_errors, schema_errors, saved, submitted, call_count = Counter(), 0, 0, 0, 0
        for event in calls:
            if event.get("tool") != "extract_observations" or event.get("type") != "tool.started":
                continue
            call_count += 1
            args = event["payload"]["arguments"]
            try:
                args = json.loads(args) if isinstance(args, str) else args
                parsed = ExtractObservations.model_validate(args)
            except (ValueError, ValidationError):
                schema_errors += 1
                continue
            submitted += len(parsed.rows)
            result = extract_observations(runtime, parsed)
            saved += result["saved_count"]
            row_errors.update(row["error"].split(":", 1)[0] for row in result["rows"] if row.get("error"))
        return {"source": str(directory), "session_id": session_id, "calls": call_count, "submitted_rows": submitted,
                "saved_rows_including_repeated_attempts": saved, "unique_saved_observations": len(session.facts),
                "schema_errors": schema_errors, "row_errors": dict(row_errors), "source_database_modified": False,
                "limitation": "Replays recorded model choices under current code. Does not perform new LLM review or claim a completed valuation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--session")
    args = parser.parse_args()
    print(json.dumps(replay(args.directory, args.session), ensure_ascii=False, indent=2))
