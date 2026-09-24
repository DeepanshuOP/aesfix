"""Unit tests for Aesthetic Gain (AG@k) Benchmark Harness (AF-2.5 DoD).

Why these tests verify C1 Contribution:
1. Tests single-best AG@1 and compound AG@k calculation.
2. Validates that the Identity Guard vetos reward hacking by clamping gain to 0.0.
3. Tests benchmark reporting and paper markdown table generation across policies.
"""
from typing import List
import numpy as np
import pytest
from PIL import Image

from aesfix.critic.critic import CriticFix
from aesfix.eval import (
    AGBenchmarkReport,
    AGEvaluationHarness,
    AGSampleResult,
)


@pytest.fixture
def dark_synthetic_image() -> Image.Image:
    """Creates an underexposed synthetic image (mean lum ~0.20, ideal is ~0.50)."""
    arr = np.full((128, 128, 3), fill_value=50, dtype=np.uint8)
    # Add mild gradient
    arr[:, :64] = 35
    return Image.fromarray(arr, mode="RGB")


def test_ag_evaluate_single_sample_positive_gain(dark_synthetic_image: Image.Image) -> None:
    """Verifies that an appropriate fix on an underexposed photo yields verified positive gain."""
    harness = AGEvaluationHarness()

    fixes = [
        CriticFix(
            op="adjust_exposure",
            params={"ev_delta": 0.8},
            expected_aspect="lighting",
            nl_instruction="Lift exposure on underexposed photo.",
        )
    ]

    res = harness.evaluate_sample(dark_synthetic_image, fixes, k=1)

    assert isinstance(res, AGSampleResult)
    assert res.is_guard_approved is True
    assert res.verified_gain > 0.0, f"Expected positive gain, got {res.verified_gain}"
    assert res.after_score > res.before_score
    assert res.aspect_deltas.get("lighting", 0.0) > 0.0


def test_ag_guard_vetos_hallucinated_edit(dark_synthetic_image: Image.Image) -> None:
    """Verifies that an edit failing the Identity Guard receives 0.0 verified gain."""
    # Custom guard with strict threshold
    harness = AGEvaluationHarness()
    # Force strict guard threshold to simulate rejection
    harness.guard.drift_threshold = 0.01

    fixes = [
        CriticFix(
            op="adjust_exposure",
            params={"ev_delta": 1.5},
            expected_aspect="lighting",
            nl_instruction="Drastic exposure shift.",
        )
    ]

    res = harness.evaluate_sample(dark_synthetic_image, fixes, k=1)

    assert res.is_guard_approved is False
    assert res.verified_gain == 0.0  # Guard clamped gain to zero
    assert res.guard_verdict is not None
    assert res.guard_verdict.is_valid is False


def test_ag_benchmark_run_across_policies(dark_synthetic_image: Image.Image) -> None:
    """Verifies full benchmark run across candidate policy vs control baseline."""
    harness = AGEvaluationHarness()
    dataset = [dark_synthetic_image] * 4

    # Helpful policy: recommends exposure lift
    def helpful_policy(img: Image.Image) -> List[CriticFix]:
        return [
            CriticFix(op="adjust_exposure", params={"ev_delta": 0.6}, expected_aspect="lighting", nl_instruction="brighten"),
            CriticFix(op="adjust_contrast", params={"factor": 1.1}, expected_aspect="technical", nl_instruction="boost contrast"),
        ]

    # Suboptimal/harmful policy: further darkens dark image
    def harmful_policy(img: Image.Image) -> List[CriticFix]:
        return [
            CriticFix(op="adjust_exposure", params={"ev_delta": -0.6}, expected_aspect="lighting", nl_instruction="darken further")
        ]

    report_helpful = harness.run_benchmark(dataset, helpful_policy, model_name="HelpfulCritic-v1", k=2)
    report_harmful = harness.run_benchmark(dataset, harmful_policy, model_name="HarmfulCritic-v1", k=2)

    assert isinstance(report_helpful, AGBenchmarkReport)
    assert report_helpful.num_samples == 4
    assert report_helpful.mean_ag_at_1 > 0.0
    assert report_helpful.guard_pass_rate == 1.0
    assert report_helpful.aspect_verification_rate > 0.50

    # Helpful policy must outperform harmful policy
    assert report_helpful.mean_ag_at_1 > report_harmful.mean_ag_at_1

    # Verify table generation
    md_row = report_helpful.to_markdown_row()
    assert "| HelpfulCritic-v1" in md_row
    assert "%" in md_row

    json_str = report_helpful.to_json()
    assert "mean_ag_at_1" in json_str
