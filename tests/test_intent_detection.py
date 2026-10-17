"""Unit tests for photographic style intent detection (F4).

Verifies that statistical intent detection cleanly separates intentional
low-key photography (e.g. dark.jpg) from genuinely underexposed photographs
(e.g. synthetic -1.5 EV test images).
"""
import os
from pathlib import Path
import pytest
from aesfix.agent.loop import AesFixAgent


@pytest.fixture
def agent() -> AesFixAgent:
    return AesFixAgent()


def test_dark_jpg_is_flagged_intentional(agent: AesFixAgent):
    # Locate dark.jpg in tests/hard_cases or root
    candidates = [
        Path("tests/hard_cases/dark.jpg"),
        Path("dark.jpg"),
        Path("web/public/samples/dark.jpg"),
    ]
    dark_path = next((p for p in candidates if p.exists()), None)
    assert dark_path is not None, "dark.jpg not found in test workspace"

    intent = agent._detect_style_intent(str(dark_path))
    assert intent["is_intentional"] is True
    assert intent["style_type"] == "low-key"
    assert intent["mean_luma"] < 0.15
    assert intent["p5_luma"] < 0.02


def test_synthetic_underexposed_images_are_not_flagged_intentional(agent: AesFixAgent):
    # Synthetic -1.5 EV images represent genuine defect, not artistic low-key
    synth_paths = [
        Path("data/synthetic_benchmark/synth_test_exp_m1_5.jpg"),
        Path("data/synthetic_benchmark/synth_temp_exp_m1_5.jpg"),
        Path("web/public/samples/underexposed_synthetic.jpg"),
    ]
    tested_count = 0
    for p in synth_paths:
        if p.exists():
            intent = agent._detect_style_intent(str(p))
            assert intent["is_intentional"] is False, f"{p} was falsely flagged intentional low-key"
            assert intent["style_type"] == "normal"
            tested_count += 1

    assert tested_count >= 2, f"Expected at least 2 synthetic images to be tested, found {tested_count}"
