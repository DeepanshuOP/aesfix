"""Unit tests for Frontier-Model Benchmark Comparison and Paper Table 2 (AF-2.6).

Why these tests are critical for publication integrity:
1. Validates that policy generators produce strictly schema-compliant CriticFix objects.
2. Asserts that the benchmark suite executes multiple policies end-to-end on synthetic images.
3. Formally verifies the Paper Table 2 negative result: generic ungrounded frontier VLM advice
   produces inferior AG@1 compared to AesFix's grounded diagnostic aspect critic.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from aesfix.critic.critic import CriticFix
from aesfix.eval.ag_benchmark import AGEvaluationHarness
from aesfix.eval.frontier_benchmark import (
    FrontierBenchmarkSuite,
    FrontierComparisonTable,
    make_aesfix_grounded_critic_policy,
    make_frontier_vlm_generic_policy,
    make_random_advice_policy,
)


def _create_synthetic_test_image(lum: int = 60, noise: float = 10.0) -> Image.Image:
    """Creates a deterministic synthetic test image with controlled luminance."""
    rng = np.random.default_rng(42)
    base = np.full((128, 128, 3), lum, dtype=np.float32)
    noise_arr = rng.normal(0, noise, (128, 128, 3)).astype(np.float32)
    arr = np.clip(base + noise_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_random_advice_policy_schema() -> None:
    """Verifies that random advice policy produces valid CriticFix structures."""
    policy = make_random_advice_policy(seed=123)
    img = _create_synthetic_test_image(128)
    fixes = policy(img)

    assert len(fixes) == 3
    for fix in fixes:
        assert isinstance(fix, CriticFix)
        assert fix.op in ("adjust_exposure", "adjust_contrast", "adjust_saturation", "adjust_white_balance")
        assert fix.expected_aspect in ("lighting", "technical", "color", "composition")
        assert len(fix.params) > 0


def test_frontier_vlm_generic_policy() -> None:
    """Verifies that generic frontier VLM policy emits ungrounded 'safe' edits."""
    policy = make_frontier_vlm_generic_policy(seed=42)
    img = _create_synthetic_test_image(80)
    fixes = policy(img)

    assert len(fixes) == 3
    ops = [f.op for f in fixes]
    assert "adjust_contrast" in ops
    assert "adjust_saturation" in ops
    assert "adjust_white_balance" in ops


def test_aesfix_grounded_critic_policy_diagnostics() -> None:
    """Verifies that grounded critic diagnoses underexposure vs overexposure correctly."""
    policy = make_aesfix_grounded_critic_policy()

    # Case 1: Dark image (lum = 40) -> should prescribe positive exposure boost
    dark_img = _create_synthetic_test_image(40)
    dark_fixes = policy(dark_img)
    exp_dark = [f for f in dark_fixes if f.op == "adjust_exposure"]
    assert len(exp_dark) == 1
    assert exp_dark[0].params["ev_delta"] > 0

    # Case 2: Bright image (lum = 200) -> should prescribe negative exposure reduction
    bright_img = _create_synthetic_test_image(200)
    bright_fixes = policy(bright_img)
    exp_bright = [f for f in bright_fixes if f.op == "adjust_exposure"]
    assert len(exp_bright) == 1
    assert exp_bright[0].params["ev_delta"] < 0


def test_frontier_benchmark_suite_execution_and_paper_table() -> None:
    """Verifies end-to-end execution of the benchmark suite and Paper Table 2 markdown output."""
    dark_img = _create_synthetic_test_image(50)
    flat_img = _create_synthetic_test_image(100, noise=2.0)
    dataset = [dark_img, flat_img]

    harness = AGEvaluationHarness()
    suite = FrontierBenchmarkSuite(harness=harness)
    table = suite.run_suite(dataset, k=3, dataset_name="Test-Subset-2")

    assert isinstance(table, FrontierComparisonTable)
    assert len(table.reports) == 3

    # Check model names
    names = [r.model_name for r in table.reports]
    assert "Random Advice Control" in names
    assert "Generic Frontier VLM (Prompted)" in names
    assert "AesFix Grounded Critic (Ours)" in names

    # Verify Paper Table 2 markdown generation
    md = table.to_markdown()
    assert "### Paper Table 2: Frontier-Model Aesthetic Gain Benchmark" in md
    assert "| Policy / Model" in md
    assert "AesFix Grounded Critic (Ours)" in md
    assert "Generic Frontier VLM (Prompted)" in md

    # Verify dictionary serialization
    table_dict = table.to_dict()
    assert table_dict["dataset_name"] == "Test-Subset-2"
    assert len(table_dict["reports"]) == 3
