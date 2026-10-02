import copy
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("observation_audit", Path(__file__).parents[1] / "scripts/audit_observations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    case = {"case_id": "synthetic", "source_sha256": "original-hash", "checks": [
        {"pages": [5], "metric": "revenue", "scope": "consolidated", "role": "historical", "unit": "元", "currency": "CNY",
         "expected": {"2025": "1200"}}]}
    fact = {"fact_id": "fact_1", "standard_metric": "revenue", "period": "2025", "normalized_value": "1200",
            "scope": "consolidated", "role": "historical", "unit": "元", "status": "confirmed", "warnings": [],
            "verification": {"reading_proof": {"source_sha256": "original-hash", "basis": {"currency": "CNY"},
                                                 "resolved_value": {"location": {"page": 5}}}}}
    return case, fact


def test_audit_deduplicates_success_and_reports_uninspected_scope_without_mutation():
    case, fact = sample()
    facts = [fact, {**fact, "fact_id": "duplicate"}, {**fact, "fact_id": "other", "standard_metric": "inventory"}]
    before = copy.deepcopy(facts)
    result = module.audit(facts, case)
    assert result["passed"] and len(result["checks"]) == 1
    assert result["uninspected_observations"] == 1
    assert facts == before


@pytest.mark.parametrize("change", [{"period": "2023"}, {"normalized_value": "12000"}, {"scope": "parent"}, {"unit": "千元"}])
def test_confirmed_flag_never_overrides_wrong_source_interpretation(change):
    case, fact = sample()
    fact.update(change)
    result = module.audit([fact], case)
    assert not result["passed"]
    assert any(item["false_confirmed_ids"] == [fact["fact_id"]] for item in result["checks"])


def test_missing_wrong_source_unconfirmed_and_warned_are_not_accuracy_passes():
    case, fact = sample()
    assert not module.audit([], case)["passed"]
    for changes in [{"status": "proposed"}, {"warnings": ["not admitted"]}, {"status": "rejected"}]:
        assert not module.audit([{**fact, **changes}], case)["passed"]
    fact["verification"]["reading_proof"]["source_sha256"] = "different"
    assert not module.audit([fact], case)["passed"]


def test_sqlite_audit_is_read_only_and_requires_unambiguous_session(tmp_path):
    case, fact = sample()
    path = tmp_path / "audit.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE research_sessions(session_id TEXT, session_json TEXT)")
        connection.execute("INSERT INTO research_sessions VALUES(?,?)", ("first", json.dumps({"facts": [fact]})))
    before = path.read_bytes()
    assert module.read_facts(path) == [fact]
    assert path.read_bytes() == before
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO research_sessions VALUES(?,?)", ("second", json.dumps({"facts": []})))
    with pytest.raises(ValueError, match="exactly one"):
        module.read_facts(path)
    assert module.read_facts(path, "first") == [fact]
    with pytest.raises(FileNotFoundError):
        module.read_facts(tmp_path / "not-created.sqlite3")
