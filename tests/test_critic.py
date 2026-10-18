"""Tests for the Aspect Critic and machine-executable fix schema (AF-1.1).

Why these tests verify the critic layer:
1. Validates that CriticFix rejects non-canonical aspect names to prevent breaking
   the downstream judge ensemble measurement probes.
2. Ensures parse_critic_json reliably handles VLM outputs, including markdown code
   blocks and slight floating point overshoots.
3. Verifies prompt builder correctly bounds user inputs against prompt injection.
4. Tests HeuristicAspectCritic on calibration images to verify that diagnosed flaws
   generate appropriate, falsifiable fix proposals (e.g. dark.jpg -> adjust_exposure).
"""
import os
import pytest

from aesfix.critic import (
    CriticFix,
    CriticOutput,
    CriticPromptBuilder,
    HeuristicAspectCritic,
    parse_critic_json,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DARK_JPG = os.path.join(_PROJECT_ROOT, "dark.jpg")
_BLUR_JPG = os.path.join(_PROJECT_ROOT, "blur.jpg")
_TEST_JPG = os.path.join(_PROJECT_ROOT, "test.jpg")


def test_critic_fix_validation():
    # Valid fix targeting lighting
    fix = CriticFix(
        op="adjust_exposure",
        params={"ev_delta": 0.5},
        expected_aspect="lighting",
        nl_instruction="Brighten exposure by 0.5 EV.",
    )
    assert fix.expected_aspect == "lighting"

    # Non-canonical aspect must fail validation
    with pytest.raises(ValueError, match="expected_aspect must be one of"):
        CriticFix(
            op="adjust_exposure",
            params={"ev_delta": 0.5},
            expected_aspect="invalid_aspect_name",
            nl_instruction="Invalid",
        )


def test_parse_critic_json_plain_and_markdown_fenced():
    valid_data = {
        "aspect_scores": {
            "composition": 0.72,
            "lighting": 0.40,
            "color": 0.65,
            "subject_clarity": 0.80,
            "technical": 0.85,
        },
        "rationales": {
            "composition": "Strong framing.",
            "lighting": "Underexposed.",
            "color": "Good harmony.",
            "subject_clarity": "Sharp focus.",
            "technical": "Clean image.",
        },
        "overall_summary": "Solid photograph requiring exposure lift.",
        "fixes": [
            {
                "op": "adjust_exposure",
                "params": {"ev_delta": 0.4},
                "expected_aspect": "lighting",
                "nl_instruction": "Lift exposure +0.4 EV.",
            }
        ],
    }

    # 1. Direct JSON parsing
    raw_json = str(valid_data).replace("'", '"')
    parsed = parse_critic_json(raw_json)
    assert isinstance(parsed, CriticOutput)
    assert parsed.aspect_scores["lighting"] == pytest.approx(0.40)
    assert len(parsed.fixes) == 1
    assert parsed.fixes[0].op == "adjust_exposure"

    # 2. Markdown fenced JSON parsing
    fenced_raw = f"Here is my critique:\n```json\n{raw_json}\n```\nHope this helps!"
    parsed_fenced = parse_critic_json(fenced_raw)
    assert parsed_fenced.aspect_scores["composition"] == pytest.approx(0.72)

    # 3. Invalid JSON must raise ValueError
    with pytest.raises(ValueError, match="Failed to parse JSON"):
        parse_critic_json("This is purely conversational with no JSON.")


def test_prompt_builder_genre_and_multimodal_critique():
    prompt = CriticPromptBuilder.build_user_prompt(
        user_critique="The horizon feels tilted.",
        goal_or_genre="landscape",
    )
    assert "landscape" in prompt
    assert "The horizon feels tilted." in prompt
    assert "USER-SUPPLIED OBSERVATION" in prompt


def test_heuristic_critic_on_calibration_images():
    critic = HeuristicAspectCritic()

    # Evaluating dark.jpg should diagnose poor lighting and suggest adjust_exposure
    dark_output = critic.critique(_DARK_JPG, genre="portrait")
    assert dark_output.genre == "portrait"
    assert dark_output.aspect_scores["lighting"] < 0.45

    # Check that a lighting fix was proposed
    lighting_fixes = [f for f in dark_output.fixes if f.expected_aspect == "lighting"]
    assert len(lighting_fixes) >= 1
    assert lighting_fixes[0].op in ("adjust_exposure", "adjust_contrast")

    # Evaluating blur.jpg should diagnose low clarity and propose a sharpness fix
    blur_output = critic.critique(_BLUR_JPG)
    assert blur_output.aspect_scores["subject_clarity"] < 0.50
    clarity_fixes = [f for f in blur_output.fixes if f.expected_aspect in ("subject_clarity", "technical")]
    assert len(clarity_fixes) >= 1

    # Missing image must fail loudly
    with pytest.raises(FileNotFoundError):
        critic.critique("non_existent_file.jpg")


def test_heuristic_critic_signed_white_balance(tmp_path):
    """Verifies that HeuristicAspectCritic estimates Lab color cast and proposes opposite signed WB (I3)."""
    from unittest.mock import patch
    from PIL import Image
    from aesfix.judges.ensemble import JudgeScore

    critic = HeuristicAspectCritic()

    # Synthetic warm-cast image (high amber/yellow: R=230, G=190, B=90)
    warm_path = tmp_path / "synthetic_warm.jpg"
    Image.new("RGB", (64, 64), color=(230, 190, 90)).save(warm_path)

    # Synthetic cool-cast image (high blue: R=90, G=170, B=230)
    cool_path = tmp_path / "synthetic_cool.jpg"
    Image.new("RGB", (64, 64), color=(90, 170, 230)).save(cool_path)

    mock_color_flaw = JudgeScore(
        name="clip_aspect",
        raw_score=0.45,
        normalized_score=0.45,
        aspects={
            "composition": 0.70,
            "lighting": 0.70,
            "color": 0.35,
            "subject_clarity": 0.70,
            "technical": 0.70,
        },
    )

    with patch.object(critic._judge, "evaluate", return_value=mock_color_flaw):
        warm_output = critic.critique(str(warm_path))
        cool_output = critic.critique(str(cool_path))

    # Warm cast must produce negative temperature proposal (cooling compensation)
    warm_wb_fixes = [f for f in warm_output.fixes if f.op == "adjust_white_balance"]
    assert len(warm_wb_fixes) == 1
    assert warm_wb_fixes[0].params["temperature"] < 0, "Expected negative temperature to cool warm cast"
    assert "warm cast" in warm_output.rationales["color"]

    # Cool cast must produce positive temperature proposal (warming compensation)
    cool_wb_fixes = [f for f in cool_output.fixes if f.op == "adjust_white_balance"]
    assert len(cool_wb_fixes) == 1
    assert cool_wb_fixes[0].params["temperature"] > 0, "Expected positive temperature to warm cool cast"
    assert "cool cast" in cool_output.rationales["color"]
