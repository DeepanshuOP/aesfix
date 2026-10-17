"""Unit tests for ledger append-only integrity and schema validation (AGF4)."""
from __future__ import annotations

import json


from scripts.check_ledger_append_only import (
    validate_ledger_entries,
    verify_append_only,
)


def test_validate_ledger_entries_valid():
    valid_entry = json.dumps({
        "run_id": "test-uuid-1",
        "commit": "abcdef12",
        "timestamp": "2026-10-17T00:00:00Z",
        "metrics": {"gain": 0.05},
    })
    errors = validate_ledger_entries([valid_entry])
    assert len(errors) == 0


def test_validate_ledger_entries_rejects_dummy_test_entry():
    dummy_entry = json.dumps({"test": 1})
    errors = validate_ledger_entries([dummy_entry])
    assert len(errors) == 1
    assert "Missing required ledger field(s)" in errors[0]
    assert "{'test': 1}" in errors[0] or "test" in errors[0]


def test_validate_ledger_entries_rejects_invalid_json():
    errors = validate_ledger_entries(["{not valid json}"])
    assert len(errors) == 1
    assert "Invalid JSON syntax" in errors[0]


def test_verify_append_only_clean_append():
    base = [
        '{"run_id": "1", "commit": "a", "timestamp": "t1", "metrics": {}}',
        '{"run_id": "2", "commit": "b", "timestamp": "t2", "metrics": {}}',
    ]
    current = base + [
        '{"run_id": "3", "commit": "c", "timestamp": "t3", "metrics": {}}',
    ]
    errors = verify_append_only(base, current)
    assert len(errors) == 0


def test_verify_append_only_detects_modification():
    base = [
        '{"run_id": "1", "commit": "a", "timestamp": "t1", "metrics": {}}',
        '{"run_id": "2", "commit": "b", "timestamp": "t2", "metrics": {}}',
    ]
    current = [
        '{"run_id": "1", "commit": "a", "timestamp": "t1", "metrics": {}}',
        '{"run_id": "2", "commit": "modified", "timestamp": "t2", "metrics": {}}',
    ]
    errors = verify_append_only(base, current)
    assert len(errors) == 1
    assert "Ledger line 2 was modified" in errors[0]


def test_verify_append_only_detects_deletion():
    base = [
        '{"run_id": "1", "commit": "a", "timestamp": "t1", "metrics": {}}',
        '{"run_id": "2", "commit": "b", "timestamp": "t2", "metrics": {}}',
    ]
    current = [
        '{"run_id": "1", "commit": "a", "timestamp": "t1", "metrics": {}}',
    ]
    errors = verify_append_only(base, current)
    assert len(errors) == 1
    assert "Ledger truncation or deletion detected" in errors[0]
