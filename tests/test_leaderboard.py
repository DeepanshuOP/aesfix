"""Unit tests for AGX Public Leaderboard v0 and Submission Protocol (AF-2.9).

Why these tests are critical for benchmark integrity:
1. Asserts that the leaderboard initializes with at least 5 benchmarked models (AF-2.9 DoD).
2. Verifies that participant submission payloads are strictly validated against protocol requirements.
3. Tests dynamic ranking updates and HuggingFace Space markdown generation.
"""
from __future__ import annotations

import json
from aesfix.eval.leaderboard import (
    LEADERBOARD_VERSION,
    AGXLeaderboard,
    LeaderboardEntry,
)


def test_leaderboard_initial_seed_models() -> None:
    """Verifies that the leaderboard initializes with at least 5 models (DoD)."""
    board = AGXLeaderboard()
    entries = board.get_entries()

    # DoD requirement: at least 5 models on the board
    assert len(entries) >= 5

    # Check for core models
    names = [e.model_name for e in entries]
    assert "AesFix Grounded Critic (Ours)" in names
    assert "GPT-4o (Prompted Zero-Shot)" in names
    assert "Random Advice Control" in names
    assert "Oracle Inversion (Ceiling)" in names

    # Grounded critic should be rank 1 among evaluated participant models
    aesfix_entry = next(e for e in entries if e.model_name == "AesFix Grounded Critic (Ours)")
    assert aesfix_entry.rank == 1


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


def test_register_submission_and_dynamic_reranking() -> None:
    """Verifies that new valid submissions are ranked correctly."""
    board = AGXLeaderboard()
    initial_count = len(board.get_entries())

    # Add a high-performing model that takes rank 2 (below AesFix, above GPT-4o)
    new_entry = LeaderboardEntry(
        model_name="Novel-FineTuned-Critic-8B",
        organization="Research Team X",
        backbone="Llama-3-Vision-8B-LoRA",
        ag_at_1=0.0450,
        ag_at_3=0.0520,
        guard_pass_rate=0.940,
        aspect_verification_rate=0.620,
    )
    success = board.register_submission(new_entry)
    assert success is True

    entries = board.get_entries()
    assert len(entries) == initial_count + 1

    registered = next(e for e in entries if e.model_name == "Novel-FineTuned-Critic-8B")
    assert registered.rank == 2  # Directly behind AesFix (0.0824) and ahead of Qwen2-VL (0.0142)


def test_leaderboard_markdown_and_json_export() -> None:
    """Verifies HuggingFace Space markdown rendering and JSON serialization."""
    board = AGXLeaderboard()

    md = board.to_markdown()
    assert f"# AGX Public Leaderboard (v{LEADERBOARD_VERSION})" in md
    assert "| Rank | Model Name" in md
    assert "AesFix Grounded Critic (Ours)" in md
    assert "Ceiling" in md or "#1" in md

    raw_json = board.to_json()
    data = json.loads(raw_json)
    assert data["version"] == LEADERBOARD_VERSION
    assert "leaderboard" in data
    assert len(data["leaderboard"]) >= 5
