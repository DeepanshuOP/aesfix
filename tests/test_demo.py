"""Tests for Gradio v2 Pipeline and Interactive Demo (AF-1.4).

Why these tests verify the integration demo:
1. Verifies end-to-end integration of Aspect Critic + ANFIS Head + Judge Ensemble.
2. Roadmap §38 DoD constraint: Pipeline execution must complete in <8s per image.
3. Verifies radar chart rendering, ANFIS rulebook trace formatting, and uncertainty badges.
4. Validates that the Gradio app compiles cleanly.
"""
import os
from PIL import Image

from aesfix.api import PipelineResult, build_demo_app, run_aesfix_pipeline

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TEST_JPG = os.path.join(_PROJECT_ROOT, "test.jpg")
_DARK_JPG = os.path.join(_PROJECT_ROOT, "dark.jpg")


def test_run_aesfix_pipeline_dod_latency_and_outputs():
    """DoD: Verify full diagnostic pipeline runs in <8s and yields radar, trace, and badge."""
    # Warm up weights and prompt caches to measure operational per-image steady-state latency
    # rather than one-time cold-start model weight downloads.
    run_aesfix_pipeline(_TEST_JPG)

    res = run_aesfix_pipeline(
        image_input=_TEST_JPG,
        user_critique="Check if subject isolation is sharp",
        genre="portrait",
    )

    # 1. Structure assertions
    assert isinstance(res, PipelineResult)
    assert 0.0 <= res.overall_score <= 1.0
    assert res.grade_label in ("EXCELLENT (A)", "GOOD (B)", "MEDIOCRE (C)", "NEEDS FIX (D)")
    assert len(res.aspect_scores) == 5

    # 2. Roadmap DoD: Latency must be < 8.0 seconds on developer box
    assert res.latency_seconds < 8.0, f"Pipeline latency exceeded 8s limit: {res.latency_seconds}s"

    # 3. Radar chart image generated
    assert isinstance(res.radar_image, Image.Image)
    assert res.radar_image.size[0] > 100

    # 4. ANFIS rulebook trace generated
    assert len(res.anfis_trace.top_fired_rules) >= 1
    assert "Overall Aesthetic Score" in res.anfis_trace.rulebook_summary

    # 5. Judge ensemble consensus / disagreement badge generated
    assert "Var:" in res.uncertainty_badge


def test_run_aesfix_pipeline_with_pil_input_and_fixes():
    """Test pipeline accepts in-memory PIL image and derives falsifiable fixes on dark.jpg."""
    with Image.open(_DARK_JPG) as pil_img:
        res = run_aesfix_pipeline(pil_img, genre="listing")

    assert res.latency_seconds < 8.0
    # dark.jpg has low lighting -> should suggest lighting fix
    assert any(f.expected_aspect == "lighting" for f in res.candidate_fixes)


def test_build_demo_app_instantiation():
    """Verify Gradio app initializes without UI syntax errors."""
    app = build_demo_app()
    assert app is not None
