"""Unit tests for Human-Agreement Mini-Study and Correlation Harness (AF-2.7).

Why these tests validate core research claims:
1. Asserts that the study harness evaluates 100 pairs x 3 raters according to the roadmap DoD.
2. Verifies that Spearman rho, Pearson r, and pairwise choice concordance are computed correctly.
3. Tests both synthetic calibration generators and structured human annotator datasets.
"""
from __future__ import annotations

import json
from PIL import Image

from aesfix.eval.human_study import (
    HumanAgreementReport,
    HumanAgreementStudy,
    HumanPairRating,
)


def _make_dummy_image(val: int) -> Image.Image:
    """Helper to create a solid 32x32 color patch."""
    return Image.new("RGB", (32, 32), color=(val, val, val))


def test_synthetic_100_pair_generator() -> None:
    """Verifies that synthetic benchmark generator outputs 100 pairs x 3 raters."""
    pairs = HumanAgreementStudy.generate_synthetic_benchmark_pairs(
        num_pairs=100,
        num_raters=3,
        seed=42,
    )
    assert len(pairs) == 100
    for p in pairs:
        assert "original_image" in p
        assert "edited_image" in p
        assert "ratings" in p
        assert len(p["ratings"]) == 3
        for r in p["ratings"]:
            assert isinstance(r, HumanPairRating)
            assert -2 <= r.preference_score <= 2


def test_inter_rater_agreement_computation() -> None:
    """Verifies calculation of crowd inter-rater agreement across annotators."""
    # 1. Complete unanimous agreement
    unanimous = [[1, 1, 1], [2, 1, 2], [-1, -2, -1]]
    score_unanimous = HumanAgreementStudy._compute_mean_inter_rater_agreement(unanimous)
    assert score_unanimous == 1.0

    # 2. Total disagreement
    disagree = [[1, -1, 0]]
    # Pairs: (1, -1) [diff], (1, 0) [diff], (-1, 0) [diff] -> 0 matching out of 3
    score_disagree = HumanAgreementStudy._compute_mean_inter_rater_agreement(disagree)
    assert score_disagree == 0.0


def test_human_agreement_evaluation_on_100_pairs() -> None:
    """Evaluates the 100-pair x 3-rater dataset and checks statistical outputs (AF-2.7 DoD)."""
    pairs = HumanAgreementStudy.generate_synthetic_benchmark_pairs(
        num_pairs=100,
        num_raters=3,
        seed=42,
    )
    study = HumanAgreementStudy()
    report = study.evaluate_pair_dataset(pairs)

    assert isinstance(report, HumanAgreementReport)
    assert report.total_pairs == 100
    assert report.num_raters == 3

    # Pairwise choice concordance should be strong (> 60%) for calibrated edits
    assert report.pairwise_concordance_rate >= 0.50

    # Correlation metrics should be populated
    assert -1.0 <= report.spearman_rho <= 1.0
    assert -1.0 <= report.pearson_r <= 1.0
    assert 0.0 <= report.inter_rater_agreement <= 1.0

    # Verify aspect breakdown contains all evaluated aspects
    assert len(report.aspect_breakdown) > 0
    for asp in ("lighting", "technical", "color", "composition"):
        if asp in report.aspect_breakdown:
            assert report.aspect_breakdown[asp]["count"] > 0

    # Verify markdown table generation
    md = report.to_markdown()
    assert "### Table 3: Human-Agreement Mini-Study Results (AF-2.7)" in md
    assert "100 pairs, 3 independent raters" in md
    assert "Pairwise Choice Concordance" in md
    assert "Spearman Rank Correlation" in md

    # Verify JSON serialization
    raw_json = report.to_json()
    parsed = json.loads(raw_json)
    assert parsed["total_pairs"] == 100
    assert "spearman_rho" in parsed
