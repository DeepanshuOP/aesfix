"""Unit tests for AGX Public Leaderboard Season 1 & Submission Protocol (AF-4.4).

Why these tests are critical for benchmark integrity and interview defensibility:
1. Asserts that the leaderboard initializes with Season 1 models, crowning the trained
   AesFix-Qwen2.5-VL-7B (LoRA + GRPO) model at Rank #1 over frontier commercial and open VLMs.
2. Validates the automated external submission protocol (AF-4.4), ensuring external research
   teams can self-serve evaluations with strict schema validation and error handling.
3. Verifies Hugging Face Space Gradio app export and reproducible markdown/JSON outputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from aesfix.eval.leaderboard import (
    LEADERBOARD_VERSION,
    AGXLeaderboard,
    LeaderboardEntry,
)


def test_leaderboard_season_1_initial_standings() -> None:
    """Verifies that Season 1 initializes with full cohort and AesFix at Rank #1."""
    board = AGXLeaderboard()
    entries = board.get_entries()

    # Season 1 initial models cohort check
    assert len(entries) >= 8

    names = [e.model_name for e in entries]
    assert "AesFix-Qwen2.5-VL-7B (LoRA + GRPO)" in names
    assert "AesFix-SFT-Baseline (Pre-RL)" in names
    assert "InternVL2-8B (Zero-Shot)" in names
    assert "Llama-3.2-11B-Vision (Zero-Shot)" in names
    assert "GPT-4o (Prompted Zero-Shot)" in names
    assert "Claude 3.5 Sonnet (Prompted)" in names
    assert "Oracle Inversion (Ceiling)" in names
    assert "Random Advice Control" in names

    # Trained critic must decisively crown the board at rank 1
    aesfix_entry = next(e for e in entries if e.model_name == "AesFix-Qwen2.5-VL-7B (LoRA + GRPO)")
    assert aesfix_entry.rank == 1
    assert aesfix_entry.ag_at_1 > 0.080
    assert aesfix_entry.ag_at_3 > 0.090

    # SFT baseline must be at rank 2
    sft_entry = next(e for e in entries if e.model_name == "AesFix-SFT-Baseline (Pre-RL)")
    assert sft_entry.rank == 2
    assert aesfix_entry.ag_at_1 > sft_entry.ag_at_1

    # Oracle reference ceiling should have rank 0
    oracle = next(e for e in entries if e.is_reference_ceiling)
    assert oracle.rank == 0


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

    # Add a high-performing model that takes rank 2 (below AesFix RL, above SFT-Baseline)
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
    assert registered.rank == 2  # Directly behind AesFix-Qwen2.5-VL-7B (0.0845) and ahead of SFT (0.0385)


def test_evaluate_external_submission_pipeline() -> None:
    """Tests automated evaluation and registration of external community submissions."""
    board = AGXLeaderboard()
    initial_count = len(board.get_entries())

    payload = {
        "model_name": "Community-Vision-Expert-14B",
        "organization": "Open Vision Lab",
        "backbone": "Qwen2.5-14B-LoRA",
        "predictions": [
            {
                "image_id": "img_001",
                "fixes": [{"op": "adjust_contrast", "params": {"contrast_factor": 1.15}}],
            }
        ],
    }

    # Custom benchmark evaluation callback
    def mock_evaluator(preds: list) -> dict[str, float]:
        assert len(preds) == 1
        return {
            "ag_at_1": 0.0410,
            "ag_at_3": 0.0480,
            "guard_pass_rate": 0.965,
            "aspect_verification_rate": 0.610,
        }

    entry = board.evaluate_external_submission(payload, eval_fn=mock_evaluator)
    assert entry.model_name == "Community-Vision-Expert-14B"
    assert entry.ag_at_1 == 0.0410
    assert entry.rank == 2  # 0.0410 is > SFT (0.0385) and < AesFix-RL (0.0845)
    assert len(board.get_entries()) == initial_count + 1


def test_evaluate_external_submission_invalid_raises() -> None:
    """Verifies that evaluate_external_submission rejects invalid payloads with ValueError."""
    board = AGXLeaderboard()
    invalid_payload = {"model_name": "Incomplete"}

    import pytest
    with pytest.raises(ValueError, match="Invalid submission"):
        board.evaluate_external_submission(invalid_payload)


def test_leaderboard_markdown_and_json_export() -> None:
    """Verifies HuggingFace Space markdown rendering and JSON serialization."""
    board = AGXLeaderboard()

    md = board.to_markdown()
    assert f"v{LEADERBOARD_VERSION}" in md
    assert "Season 1" in md
    assert "| Rank | Model Name" in md
    assert "AesFix-Qwen2.5-VL-7B (LoRA + GRPO)" in md
    assert "Ceiling" in md or "#1" in md

    raw_json = board.to_json()
    data = json.loads(raw_json)
    assert data["version"] == LEADERBOARD_VERSION
    assert data["season"] == "Season 1"
    assert "leaderboard" in data
    assert len(data["leaderboard"]) >= 8


def test_export_hf_space_app(tmp_path: Path) -> None:
    """Verifies that Gradio HF Space app.py exports cleanly with valid Python syntax."""
    board = AGXLeaderboard()
    app_file = board.export_hf_space_app(tmp_path)

    assert app_file.exists()
    content = app_file.read_text(encoding="utf-8")
    assert "import gradio as gr" in content
    assert "LEADERBOARD_JSON" in content
    assert "AesFix-Qwen2.5-VL-7B (LoRA + GRPO)" in content

    # Test that the exported script compiles as valid Python
    compile(content, str(app_file), "exec")
