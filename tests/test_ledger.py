"""Unit tests for evidence ledger and claim checking verification (AF-Roadmap v6 R0.4)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from aesfix.ledger import (
    append_run,
    compute_dict_hash,
    get_run_by_id,
    load_ledger,
)
from scripts.check_claims import check_directories, scan_file_for_claims
from scripts.make_tables import format_run_table, format_summary_table


def test_compute_dict_hash_deterministic():
    d1 = {"b": 2, "a": 1, "nested": {"y": "yes", "x": "no"}}
    d2 = {"nested": {"x": "no", "y": "yes"}, "a": 1, "b": 2}
    h1 = compute_dict_hash(d1)
    h2 = compute_dict_hash(d2)
    assert h1 == h2
    assert len(h1) == 64


def test_append_run_and_load(tmp_path: Path):
    ledger_file = tmp_path / "test_ledger.jsonl"
    config = {"model": "baseline", "batch_size": 1}
    manifest = {"split": "test", "num_samples": 10}
    metrics = {"hwr": 0.55, "ag_best_at_1": 0.05}

    entry = append_run(
        metrics=metrics,
        config=config,
        data_manifest=manifest,
        cost_usd=0.0025,
        ledger_path=ledger_file,
    )

    assert "run_id" in entry
    assert entry["metrics"] == metrics
    assert entry["cost_usd"] == 0.0025
    assert "hardware" in entry
    assert "commit" in entry
    assert "timestamp" in entry

    runs = load_ledger(ledger_file)
    assert len(runs) == 1
    assert runs[0]["run_id"] == entry["run_id"]

    fetched = get_run_by_id(entry["run_id"], ledger_path=ledger_file)
    assert fetched is not None
    assert fetched["run_id"] == entry["run_id"]

    assert get_run_by_id("non-existent-uuid", ledger_path=ledger_file) is None


def test_make_tables_formatting():
    run: Dict[str, Any] = {
        "run_id": "test-uuid-1234",
        "timestamp": "2026-09-30T00:00:00Z",
        "commit": "abcdef12",
        "hardware": "AMD64; CPU-only",
        "cost_usd": 0.0123,
        "config_hash": "abc1234567890def",
        "data_manifest_hash": "fedcba0987654321",
        "metrics": {"hwr": 0.62, "ag_best_at_1": 0.08},
    }
    run_table = format_run_table(run)
    assert "test-uuid-1234" in run_table
    assert "`hwr` | 0.6200" in run_table

    summary_table = format_summary_table([run])
    assert "test-uui" in summary_table
    assert "0.6200" in summary_table


def test_check_claims_untagged_violation(tmp_path: Path):
    doc = tmp_path / "claim.md"
    doc.write_text("The model achieved 0.85 SRCC on the benchmark.", encoding="utf-8")

    violations = scan_file_for_claims(doc, known_run_ids=set())
    assert len(violations) == 1
    assert "Untagged metric claim '0.85 SRCC'" in violations[0][2]


def test_check_claims_unknown_run_violation(tmp_path: Path):
    doc = tmp_path / "claim.md"
    doc.write_text("The model achieved 0.85 SRCC [run:missing-uuid].", encoding="utf-8")

    violations = scan_file_for_claims(doc, known_run_ids={"valid-uuid"})
    assert len(violations) == 1
    assert "unknown run_id 'missing-uuid'" in violations[0][2]


def test_check_claims_valid_run(tmp_path: Path):
    doc = tmp_path / "claim.md"
    doc.write_text("The model achieved 0.85 SRCC [run:valid-uuid].", encoding="utf-8")

    violations = scan_file_for_claims(doc, known_run_ids={"valid-uuid"})
    assert len(violations) == 0


def test_check_claims_directories_skips_archive(tmp_path: Path):
    docs_dir = tmp_path / "docs"
    archive_dir = docs_dir / "archive"
    archive_dir.mkdir(parents=True)
    (archive_dir / "old.md").write_text("Fabricated 0.99 SRCC here.", encoding="utf-8")

    active_doc = docs_dir / "active.md"
    active_doc.write_text("Safe explanation without metrics.", encoding="utf-8")

    total_scanned, violations = check_directories([docs_dir], tmp_path / "empty.jsonl")
    assert total_scanned == 1
    assert len(violations) == 0
