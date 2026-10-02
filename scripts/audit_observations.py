"""Read-only comparison with independently checked source-page values, never runtime inputs."""
import argparse
import json
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path


def same_number(actual, expected):
    try:
        value = Decimal(str(actual))
        return value.is_finite() and value == Decimal(str(expected))
    except InvalidOperation:
        return False


def audit(facts, case):
    results = []
    inspected = set()
    for check in case["checks"]:
        selected = []
        for fact in facts:
            proof = fact.get("verification", {}).get("reading_proof", {})
            if (fact.get("status") != "rejected" and proof.get("source_sha256") == case["source_sha256"]
                    and proof.get("resolved_value", {}).get("location", {}).get("page") in check["pages"]
                    and fact.get("standard_metric") == check["metric"]):
                selected.append(fact)
                inspected.add(fact["fact_id"])
        expected = check["expected"]
        for period, value in expected.items():
            matching = [fact for fact in selected if fact.get("period") == period]
            accurate = [fact for fact in matching if same_number(fact.get("normalized_value"), value)
                        and all(fact.get(key) == check[key] for key in ("scope", "role", "unit"))
                        and fact.get("verification", {}).get("reading_proof", {}).get("basis", {}).get("currency") == check["currency"]]
            wrong = [fact for fact in matching if fact not in accurate]
            usable = [fact for fact in accurate if fact.get("status") == "confirmed" and not fact.get("warnings")]
            results.append({"pages": check["pages"], "metric": check["metric"], "period": period,
                            "status": "wrong" if wrong else "verified" if usable else "unconfirmed" if accurate else "missing",
                            "fact_ids": [fact["fact_id"] for fact in matching],
                            "false_confirmed_ids": [fact["fact_id"] for fact in wrong if fact.get("status") == "confirmed"]})
        for fact in selected:
            if fact.get("period") not in expected:
                results.append({"pages": check["pages"], "metric": check["metric"], "period": fact.get("period"),
                                "status": "unsupported_period", "fact_ids": [fact["fact_id"]],
                                "false_confirmed_ids": [fact["fact_id"]] if fact.get("status") == "confirmed" else []})
    return {"case_id": case["case_id"], "passed": bool(results) and all(item["status"] == "verified" for item in results),
            "source_sha256": case["source_sha256"], "checks": results, "inspected_observations": len(inspected),
            "uninspected_observations": sum(fact.get("status") != "rejected" and fact["fact_id"] not in inspected for fact in facts),
            "limitation": "Only the explicitly listed pages, fields and periods are checked; this is not whole-company or valuation acceptance."}


def read_facts(database, session_id=None):
    path = Path(database).resolve(strict=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        if session_id:
            rows = connection.execute("SELECT session_json FROM research_sessions WHERE session_id=?", (session_id,)).fetchall()
        else:
            rows = connection.execute("SELECT session_json FROM research_sessions LIMIT 2").fetchall()
    if len(rows) != 1:
        raise ValueError("Choose exactly one existing research session with --session-id; the database is not modified.")
    return json.loads(rows[0][0])["facts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--session-id")
    args = parser.parse_args()
    result = audit(read_facts(args.database, args.session_id), json.loads(args.case.read_text(encoding="utf-8")))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
