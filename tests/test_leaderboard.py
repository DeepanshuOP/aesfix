"""Unit tests for AGX Public Leaderboard & External Submission Protocol (AF-4.4).

Verifies that the leaderboard starts empty, loads verified runs from ledger,
and enforces strict participant submission schema validation and dynamic ranking.
"""
from __future__ import annotations

import json
from pathlib import Path
from aesfix.eval.leaderboard import (
    LEADERBOARD_VERSION,
    AGXLeaderboard,
    LeaderboardEntry,
)


def test_leaderboard_starts_empty_without_ledger(tmp_path: Path) -> None:
    """Verifies that the leaderboard starts empty when no ledger exists."""
    empty_ledger = tmp_path / "nonexistent_ledger.jsonl"
    board = AGXLeaderboard(ledger_path=empty_ledger)
    assert len(board.get_entries()) == 0


def test_leaderboard_loads_from_ledger(tmp_path: Path) -> None:
    """Verifies that leaderboard entries populate from verified ledger runs."""
    ledger_file = tmp_path / "test_ledger.jsonl"
    run_record = {
        "run_id": "test-run-001",
        "timestamp": "2026-09-30T00:00:00Z",
        "commit": "34cdb7f",
        "config": {
            "model_name": "TestCritic-v1",
            "organization": "Test Lab",
            "backbone": "Open-VLM-4B",
        },
        "metrics": {
            "ag_at_1": 0.0512,
            "ag_at_3": 0.0620,
            "guard_pass_rate": 0.950,
            "aspect_verification_rate": 0.650,
        },
    }
    ledger_file.write_text(json.dumps(run_record) + "\n", encoding="utf-8")

    board = AGXLeaderboard(ledger_path=ledger_file)
    entries = board.get_entries()
    assert len(entries) == 1
    assert entries[0].model_name == "TestCritic-v1"
    assert entries[0].ag_at_1 == 0.0512
    assert entries[0].rank == 1


def test_submission_payload_validation() -> None:
    """Verifies strict participant submission schema validation."""
    board = AGXLeaderboard()

    # Valid payload
    valid_payload = {
        "model_name": "TestModel-v1",
        "organization": "Independent Lab",
        "backbone": "Open-VLM-3B",
        "predictions": [
            {
                "image_id": "test_001",
                "fixes": [{"op": "adjust_exposure", "params": {"ev_delta": 0.5}}],
            }
        ],
    }
    is_valid, msg = board.validate_submission_payload(valid_payload)
    assert is_valid is True
    assert msg == "Payload valid."

    # Missing field
    invalid_payload = {
        "model_name": "IncompleteModel",
        "organization": "Lab",
    }
    is_valid_inv, msg_inv = board.validate_submission_payload(invalid_payload)
    assert is_valid_inv is False
    assert "Missing required payload field" in msg_inv

    # Empty predictions
    empty_preds_payload = {
        "model_name": "EmptyModel",
        "organization": "Lab",
        "backbone": "VLM",
        "predictions": [],
    }
    is_valid_empty, msg_empty = board.validate_submission_payload(empty_preds_payload)
    assert is_valid_empty is False
    assert "non-empty list" in msg_empty


def test_register_submission_and_dynamic_reranking(tmp_path: Path) -> None:
    """Verifies that new valid submissions are ranked correctly."""
    empty_ledger = tmp_path / "empty.jsonl"
    board = AGXLeaderboard(ledger_path=empty_ledger)
    assert len(board.get_entries()) == 0

    entry1 = LeaderboardEntry(
        model_name="Baseline-Critic",
        organization="Team A",
        backbone="Backbone A",
        ag_at_1=0.0250,
        ag_at_3=0.0300,
        guard_pass_rate=0.920,
        aspect_verification_rate=0.450,
    )
    entry2 = LeaderboardEntry(
        model_name="Novel-Critic",
        organization="Team B",
        backbone="Backbone B",
        ag_at_1=0.0450,
        ag_at_3=0.0520,
        guard_pass_rate=0.940,
        aspect_verification_rate=0.620,
    )

    board.register_submission(entry1)
    board.register_submission(entry2)

    entries = board.get_entries()
    assert len(entries) == 2
    assert entries[0].model_name == "Novel-Critic"
    assert entries[0].rank == 1
    assert entries[1].model_name == "Baseline-Critic"
    assert entries[1].rank == 2


def test_evaluate_external_submission_pipeline(tmp_path: Path) -> None:
    """Tests automated evaluation and registration of external submissions."""
    empty_ledger = tmp_path / "empty.jsonl"
    board = AGXLeaderboard(ledger_path=empty_ledger)

    payload = {
        "model_name": "External-Vision-Expert",
        "organization": "Open Vision Lab",
        "backbone": "Open-VLM-Base",
        "predictions": [
            {
                "image_id": "img_001",
                "fixes": [{"op": "adjust_contrast", "params": {"contrast_factor": 1.15}}],
            }
        ],
    }

    def custom_evaluator(preds: list) -> dict[str, float]:
        assert len(preds) == 1
        return {
            "ag_at_1": 0.0410,
            "ag_at_3": 0.0480,
            "guard_pass_rate": 0.965,
            "aspect_verification_rate": 0.610,
        }

    entry = board.evaluate_external_submission(payload, eval_fn=custom_evaluator)
    assert entry.model_name == "External-Vision-Expert"
    assert entry.ag_at_1 == 0.0410
    assert entry.rank == 1
    assert len(board.get_entries()) == 1


def test_evaluate_external_submission_invalid_raises() -> None:
    """Verifies that evaluate_external_submission rejects invalid payloads with ValueError."""
    board = AGXLeaderboard()
    invalid_payload = {"model_name": "Incomplete"}

    import pytest
    with pytest.raises(ValueError, match="Invalid submission"):
        board.evaluate_external_submission(invalid_payload)


def test_leaderboard_markdown_and_json_export(tmp_path: Path) -> None:
    """Verifies markdown rendering and JSON serialization."""
    empty_ledger = tmp_path / "empty.jsonl"
    board = AGXLeaderboard(ledger_path=empty_ledger)
    board.register_submission(
        LeaderboardEntry(
            model_name="Sample-Model",
            organization="Sample Org",
            backbone="Sample Backbone",
            ag_at_1=0.0300,
            ag_at_3=0.0350,
            guard_pass_rate=0.950,
            aspect_verification_rate=0.500,
        )
    )

    md = board.to_markdown()
    assert f"v{LEADERBOARD_VERSION}" in md
    assert "Season 1" in md
    assert "| Rank | Model Name" in md
    assert "Sample-Model" in md

    raw_json = board.to_json()
    data = json.loads(raw_json)
    assert data["version"] == LEADERBOARD_VERSION
    assert data["season"] == "Season 1"
    assert "leaderboard" in data
    assert len(data["leaderboard"]) == 1


def test_export_hf_space_app(tmp_path: Path) -> None:
    """Verifies that Gradio HF Space app.py exports cleanly with valid Python syntax."""
    empty_ledger = tmp_path / "empty.jsonl"
    board = AGXLeaderboard(ledger_path=empty_ledger)
    board.register_submission(
        LeaderboardEntry(
            model_name="Sample-Model",
            organization="Sample Org",
            backbone="Sample Backbone",
            ag_at_1=0.0300,
            ag_at_3=0.0350,
            guard_pass_rate=0.950,
            aspect_verification_rate=0.500,
        )
    )
    app_file = board.export_hf_space_app(tmp_path)

    assert app_file.exists()
    content = app_file.read_text(encoding="utf-8")
    assert "import gradio as gr" in content
    assert "LEADERBOARD_JSON" in content
    assert "Sample-Model" in content

    # Test that the exported script compiles as valid Python
    compile(content, str(app_file), "exec")
