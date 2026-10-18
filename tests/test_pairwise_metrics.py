"""Tests for pairwise preference metrics and scene-clustered bootstrap confidence intervals (Task B1 & B2).

Why this test exists:
Validates Task B1: Ensures that Spearman rank correlation between judge deltas and human preference
uses signed margins (mos - 0.5) to avoid the sign/orientation bug that yielded negative correlations
(-0.02 to -0.08) when correlating signed delta with unsigned |mos - 0.5|.
Also validates cluster bootstrap over scenes (Task B2).
"""
import pytest
from aesfix.eval.metrics import (
    pairwise_accuracy,
    pairwise_spearman,
    bootstrap_scene_ci,
    bootstrap_paired_difference_ci,
)


def test_spearman_orientation_synthetic_case() -> None:
    # 4 synthetic test pairs with unambiguous ground-truth human MOS preference
    # Pair 1: Humans strongly prefer Left (mos=0.9, signed margin = +0.4)
    # Pair 2: Humans moderately prefer Left (mos=0.7, signed margin = +0.2)
    # Pair 3: Humans moderately prefer Right (mos=0.3, signed margin = -0.2)
    # Pair 4: Humans strongly prefer Right (mos=0.1, signed margin = -0.4)
    human_mos = [0.9, 0.7, 0.3, 0.1]

    # An accurate judge predicting calibrated positive delta for left, negative delta for right:
    judge_deltas = [1.5, 0.5, -0.5, -1.5]

    # Pairwise accuracy must be 100% (4/4)
    acc = pairwise_accuracy(judge_deltas, human_mos)
    assert acc == 1.0, f"Expected 1.0 accuracy, got {acc}"

    # Signed Spearman correlation must be perfectly positive (+1.0)
    srcc = pairwise_spearman(judge_deltas, human_mos)
    assert srcc == pytest.approx(1.0, abs=1e-5), f"Expected +1.0 Spearman correlation, got {srcc}"

    # Replicate the BUGGY calculation: correlating signed delta with unsigned |mos - 0.5|
    from scipy.stats import spearmanr
    buggy_unsigned_margins = [abs(m - 0.5) for m in human_mos]  # [0.4, 0.2, 0.2, 0.4]
    buggy_srcc, _ = spearmanr(judge_deltas, buggy_unsigned_margins)
    # Unsigned margin creates a symmetric V-shape, producing negative correlation despite 100% pairwise accuracy
    assert buggy_srcc < 0.0, f"Buggy calculation should be negative, got {buggy_srcc}"
    assert buggy_srcc == pytest.approx(-0.1054, abs=1e-3)


def test_cluster_bootstrap_scene_ci() -> None:
    # 10 scenes, each with 2 pairs
    scenes = [f"scene_{i}" for i in range(10) for _ in range(2)]
    # Scene 0..6 always correct, scene 7..9 always incorrect
    is_correct = [True] * 14 + [False] * 6

    mean_acc, ci_low, ci_high = bootstrap_scene_ci(scenes, is_correct, n_resamples=500, seed=42)
    assert 0.0 <= ci_low <= mean_acc <= ci_high <= 1.0
    assert mean_acc == pytest.approx(0.70, abs=1e-5)
    # Confirm CI width is non-trivial reflecting scene clustering
    assert ci_high - ci_low > 0.10


def test_cluster_bootstrap_paired_difference_ci() -> None:
    scenes = [f"scene_{i}" for i in range(10) for _ in range(2)]
    # Judge A is 80% accurate, Judge B is 50% accurate
    correct_a = [True] * 16 + [False] * 4
    correct_b = [True] * 10 + [False] * 10

    mean_diff, ci_low, ci_high = bootstrap_paired_difference_ci(scenes, correct_a, correct_b, n_resamples=500, seed=42)
    assert ci_low <= mean_diff <= ci_high
    assert mean_diff == pytest.approx(0.30, abs=1e-5)
