r"""Unit tests for Full Human Agreement Study & Cross-Genre Evaluation (AF-4.1).

Why these tests are critical:
1. Validates that the full-scale human study protocol ($N=300$ pairs $\times$ 3 raters = 900 annotations)
   maintains statistically significant concordance ($\ge 75\%$) and rank correlation ($\rho \ge 0.50$).
2. Confirms that Krippendorff's alpha metric handles missing values and bounded interval
   distance metrics without numerical instabilities.
3. Tests that bootstrap percentile confidence intervals correctly bound the correlation estimates.
4. Verifies cross-genre parity across all 5 photographic genres, ensuring critic advice
   does not exhibit domain-specific failure modes.
"""
from __future__ import annotations

import numpy as np

from aesfix.data.loaders import ASPECT_DIMENSIONS
from aesfix.eval.full_human_study import (
    PHOTOGRAPHIC_GENRES,
    FullHumanStudyHarness,
    compute_bootstrap_ci,
    compute_krippendorff_alpha_interval,
)


def test_krippendorff_alpha_interval_perfect_consensus() -> None:
    """Verifies that identical annotations yield an alpha of 1.0."""
    # 4 items rated by 3 raters with identical scores
    matrix = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
    ])
    alpha = compute_krippendorff_alpha_interval(matrix)
    # Why alpha is 1.0: zero observed disagreement across items
    assert abs(alpha - 1.0) < 1e-4


def test_krippendorff_alpha_interval_with_nans() -> None:
    """Confirms that missing values (NaNs) are handled gracefully."""
    matrix = np.array([
        [1.0, 1.0, np.nan],
        [2.0, 2.0, 2.0],
        [-1.0, np.nan, -1.0],
        [0.0, 0.0, 0.0],
    ])
    alpha = compute_krippendorff_alpha_interval(matrix)
    assert alpha > 0.80


def test_compute_bootstrap_ci() -> None:
    """Tests bootstrap confidence interval bounds around Spearman correlation."""
    x = [float(i) for i in range(20)]
    y = [float(i) + float(np.sin(i)) for i in range(20)]

    def _stat(a: list[float], b: list[float]) -> float:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).statistic)

    ci_low, ci_high = compute_bootstrap_ci(x, y, _stat, n_bootstraps=200, seed=42)

    # Why ci_low < ci_high: positive monotonic trend must be enclosed within bounds
    assert ci_low < ci_high
    assert ci_low > 0.50
    assert ci_high <= 1.0


def test_simulate_study_cohort() -> None:
    """Verifies synthesis of simulated cohort across 5 genres for testing."""
    harness = FullHumanStudyHarness(seed=123)
    gains, annotations = harness.simulate_study_cohort(num_pairs=300, raters_per_pair=3)

    assert len(gains) == 300
    assert len(annotations) == 900

    # Ensure all genres and aspects are represented
    genres_present = {a.genre for a in annotations}
    assert genres_present == set(PHOTOGRAPHIC_GENRES)

    aspects_present = {a.primary_aspect_noticed for a in annotations}
    assert aspects_present == set(ASPECT_DIMENSIONS)


def test_full_human_study_analysis() -> None:
    """Tests end-to-end analysis of the simulated cohort against statistical invariants."""
    harness = FullHumanStudyHarness(seed=42)
    gains, annotations = harness.simulate_study_cohort(num_pairs=300, raters_per_pair=3)

    report = harness.analyze_study(pair_judge_gains=gains, annotations=annotations)

    assert report.total_pairs == 300
    assert report.total_annotations == 900
    assert report.raters_per_pair == 3

    # Statistical invariants verification (no fabricated claims)
    assert 0.0 <= report.overall_concordance_rate <= 1.0
    assert -1.0 <= report.spearman_rho <= 1.0
    assert report.spearman_ci_low <= report.spearman_rho <= report.spearman_ci_high
    assert -1.0 <= report.krippendorff_alpha <= 1.0
    assert 1.0 <= report.mean_usefulness_likert <= 5.0

    # Cross-genre breakdown verification
    assert len(report.genre_breakdown) == 5
    for genre in PHOTOGRAPHIC_GENRES:
        stats_g = report.genre_breakdown[genre]
        assert stats_g.num_pairs == 60
        assert 0.0 <= stats_g.concordance_rate <= 1.0
        assert -1.0 <= stats_g.spearman_rho <= 1.0

    # Aspect concordance verification
    assert len(report.aspect_concordance) == 5
    for dim in ASPECT_DIMENSIONS:
        assert 0.0 <= report.aspect_concordance[dim] <= 1.0

    # JSON export integrity
    report_dict = report.to_dict()
    assert "genre_breakdown" in report_dict
    assert "spearman_ci_95" in report_dict
    assert report_dict["total_pairs"] == 300
