"""Unit and integration tests for Aesthetic CI/CD visual linter and PR gating (AF-6.2 & Roadmap §25 F4.3).

Why these tests are critical for visual CI stability:
1. Quality Gate Enforcement: Asserts that under-threshold images reliably trigger merge-blocking
   exit codes (1) and violation reasons, while high-quality images pass cleanly (0).
2. Non-Destructive Autofix Verification: Confirms that failing assets undergo Tier-A photographic
   repair, generate valid Adobe Lightroom .xmp sidecars, and record non-negative aesthetic gains.
3. PR Review Comment Contract: Verifies that GitHub Markdown summaries render valid tables,
   diagnostic badges, and actionable guidance for developers and content pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aesfix.eval.aesthetic_ci import AestheticLinter, main

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")
_DARK_JPG = str(_PROJECT_ROOT / "dark.jpg")


def test_linter_single_image_pass() -> None:
    """Verifies that an image exceeding quality thresholds passes with no violations."""
    linter = AestheticLinter(min_overall_score=0.50, min_aspect_score=0.35)
    record = linter.lint_file(_TEST_JPG)

    assert record.path == _TEST_JPG
    assert 0.0 <= record.overall_score <= 1.0
    assert record.passed is True
    assert len(record.violation_reasons) == 0
    assert len(record.aspect_scores) == 5
    assert record.autofix_path is None


def test_linter_threshold_failure() -> None:
    """Verifies that sub-par photos fail threshold checks and flag specific violations."""
    linter = AestheticLinter(min_overall_score=0.85, min_aspect_score=0.70)
    record = linter.lint_file(_DARK_JPG)

    assert record.path == _DARK_JPG
    assert record.passed is False
    assert len(record.violation_reasons) >= 1
    assert any("below" in v for v in record.violation_reasons)
    assert record.weakest_aspect in ("lighting", "color", "technical", "composition", "subject_clarity")


def test_linter_autofix_execution(tmp_path: Path) -> None:
    """Verifies that autofix repairs failing assets and generates Lightroom XMP sidecars."""
    out_dir = str(tmp_path / "autofix_results")
    linter = AestheticLinter(
        min_overall_score=0.85,
        min_aspect_score=0.70,
        autofix=True,
        autofix_output_dir=out_dir,
    )
    record = linter.lint_file(_DARK_JPG)

    assert record.passed is False
    assert record.autofix_path is not None
    assert Path(record.autofix_path).exists()
    assert record.aesthetic_gain >= 0.0
    assert record.xmp_sidecar_preview is not None
    assert "<x:xmpmeta" in record.xmp_sidecar_preview


def test_linter_batch_paths(tmp_path: Path) -> None:
    """Verifies multi-asset directory scanning and statistical aggregation."""
    linter = AestheticLinter(min_overall_score=0.60)
    summary = linter.lint_paths([_TEST_JPG, _DARK_JPG])

    assert summary.total_scanned == 2
    assert summary.passed_count + summary.failed_count == 2
    assert 0.0 <= summary.average_score <= 1.0
    assert len(summary.records) == 2


def test_pr_comment_markdown_generation() -> None:
    """Verifies that PR comment markdown conforms to GitHub Flavored Markdown table format."""
    linter = AestheticLinter(min_overall_score=0.70)
    summary = linter.lint_paths([_TEST_JPG, _DARK_JPG])

    comment_md = linter.generate_pr_comment_markdown(summary)

    assert "##" in comment_md
    assert "AesFix Aesthetic CI/CD Quality Report" in comment_md
    assert "| Status | Asset Path | Score | Grade |" in comment_md
    assert "test.jpg" in comment_md
    assert "dark.jpg" in comment_md
    assert "Suggested Actions" in comment_md or "All visual assets meet" in comment_md


def test_cli_main_exit_codes(tmp_path: Path) -> None:
    """Verifies CLI argument handling, report output writing, and CI exit codes."""
    out_md = str(tmp_path / "pr_comment.md")

    # 1. Guaranteed failure with impossible 0.99 threshold -> exit code 1 (blocks CI merge)
    exit_fail = main([
        _DARK_JPG,
        "--threshold", "0.99",
        "--format", "markdown",
        "--output-file", out_md,
    ])
    assert exit_fail == 1
    assert Path(out_md).exists()

    # 2. Guaranteed success with 0.10 threshold -> exit code 0 (passes CI)
    out_json = str(tmp_path / "report.json")
    exit_pass = main([
        _TEST_JPG,
        "--threshold", "0.10",
        "--aspect-threshold", "0.05",
        "--format", "json",
        "--output-file", out_json,
    ])
    assert exit_pass == 0
    assert Path(out_json).exists()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_scanned"] == 1
        assert data["all_passed"] is True


def test_non_existent_file_raises() -> None:
    """Verifies that missing image paths raise FileNotFoundError defensively."""
    linter = AestheticLinter()
    with pytest.raises(FileNotFoundError, match="Image asset not found"):
        linter.lint_file("missing_non_existent_photo.jpg")
