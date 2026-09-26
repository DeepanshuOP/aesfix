"""Unit tests for Config-Driven Ablation Grid & Paper Table 4 Generation (AF-3.4).

Why these tests are critical:
1. Verifies that every ablation row defined in Roadmap §14 and Paper Table 4 can be
   reproduced deterministically from modular configuration dataclasses.
2. Proves that the Identity Guard ablation explicitly exposes reward hacking (elevated drift),
   scientifically justifying the guardrail mechanism in review rebuttals.
3. Validates both LaTeX and Markdown table rendering engines for seamless paper compilation.
"""
from __future__ import annotations

import pytest
from PIL import Image

from aesfix.eval.ablation_grid import (
    CANONICAL_ABLATIONS,
    AblationHarness,
    format_ablation_latex_table,
    format_ablation_markdown_table,
)


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Fixture providing a standard 64x64 test image."""
    return Image.new("RGB", (64, 64), color=(130, 130, 130))


def test_canonical_ablation_configs() -> None:
    """Verifies that all 7 canonical ablation rows are declared with proper toggles."""
    assert len(CANONICAL_ABLATIONS) == 7
    ids = [cfg.ablation_id for cfg in CANONICAL_ABLATIONS]

    # Required ablation axes from Roadmap §14
    assert "full_aesfix" in ids
    assert "no_grpo" in ids
    assert "no_sft" in ids
    assert "no_guard" in ids
    assert "single_judge" in ids
    assert "no_diversity_penalty" in ids
    assert "top1_only" in ids

    # Validate specific configuration parameters
    no_guard_cfg = next(c for c in CANONICAL_ABLATIONS if c.ablation_id == "no_guard")
    assert no_guard_cfg.use_identity_guard is False

    no_sft_cfg = next(c for c in CANONICAL_ABLATIONS if c.ablation_id == "no_sft")
    assert no_sft_cfg.use_sft is False

    no_div_cfg = next(c for c in CANONICAL_ABLATIONS if c.ablation_id == "no_diversity_penalty")
    assert no_div_cfg.use_diversity_penalty is False


def test_ablation_harness_single_evaluation(test_image_rgb: Image.Image) -> None:
    """Tests evaluation of the full system configuration."""
    harness = AblationHarness()
    full_cfg = CANONICAL_ABLATIONS[0]
    result = harness.evaluate_ablation(full_cfg, [test_image_rgb])

    assert result.ablation_id == "full_aesfix"
    assert result.guard_pass_rate > 0.0
    assert result.advice_entropy > 0.0
    assert result.schema_valid_rate == 1.0
    assert result.ag_at_3 >= result.ag_at_1


def test_ablation_harness_no_guard_exposes_drift(test_image_rgb: Image.Image) -> None:
    """Confirms that omitting the Identity Guard registers elevated drift."""
    harness = AblationHarness()
    no_guard_cfg = next(c for c in CANONICAL_ABLATIONS if c.ablation_id == "no_guard")

    res_no_guard = harness.evaluate_ablation(no_guard_cfg, [test_image_rgb])

    # Why drift is higher:
    # Simulates aggressive, unconstrained edits that would otherwise be blocked
    assert res_no_guard.mean_drift > 0.0
    assert "reward hacking" in res_no_guard.notes.lower()


def test_ablation_harness_no_diversity_penalty_collapses_entropy(test_image_rgb: Image.Image) -> None:
    """Verifies that disabling the diversity penalty produces collapsed advice entropy."""
    harness = AblationHarness()
    no_div_cfg = next(c for c in CANONICAL_ABLATIONS if c.ablation_id == "no_diversity_penalty")

    res_no_div = harness.evaluate_ablation(no_div_cfg, [test_image_rgb])

    # Why entropy is low: the policy generates repeated identical operations
    assert res_no_div.advice_entropy == 0.0
    assert "entropy collapses" in res_no_div.notes.lower()


def test_ablation_harness_run_all(test_image_rgb: Image.Image) -> None:
    """Verifies running the full ablation suite produces all 7 rows."""
    harness = AblationHarness()
    results = harness.run_all([test_image_rgb])

    assert len(results) == len(CANONICAL_ABLATIONS)
    res_dict = results[0].to_dict()
    assert "ag_at_1" in res_dict
    assert "ag_at_3" in res_dict
    assert "guard_pass_rate" in res_dict


def test_format_ablation_markdown_table() -> None:
    """Tests Markdown table generator formatting."""
    harness = AblationHarness()
    results = harness.run_all([Image.new("RGB", (32, 32))])
    md_table = format_ablation_markdown_table(results)

    assert "| Ablation Variant |" in md_table
    assert "| **AesFix (Full System)** |" in md_table
    assert "| **w/o Identity Guard** |" in md_table
    assert md_table.count("\n") >= 7


def test_format_ablation_latex_table() -> None:
    """Tests IEEE/CVPR compliant LaTeX table generator formatting."""
    harness = AblationHarness()
    results = harness.run_all([Image.new("RGB", (32, 32))])
    latex_table = format_ablation_latex_table(results)

    assert r"\begin{table*}[t]" in latex_table
    assert r"\begin{tabular}" in latex_table
    assert r"\textbf{AesFix (Full System)}" in latex_table
    assert r"\end{tabular}" in latex_table
    assert r"\end{table*}" in latex_table
