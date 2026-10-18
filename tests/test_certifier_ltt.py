"""Unit tests for Learn-Then-Test (LTT) risk-controlling threshold selection (Task G).

Why this test exists:
Validates Task G: Checks that the False Improvement Rate (FIR) guarantee holds across 500
simulated runs (sim_*) on synthetic data.
Nothing from simulated data goes to the benchmark ledger.
"""
import numpy as np
import pytest
from aesfix.certify.ltt import (
    clopper_pearson_upper_p_value,
    learn_then_test_threshold,
    evaluate_decision_rule,
)


def sim_generate_pairs(n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic score deltas and human preference labels for testing.

    Data generation:
    s ~ N(0.3, 1.0)
    P(human prefers edit | s) = sigmoid(2.0 * s)
    """
    scores = rng.normal(loc=0.3, scale=1.0, size=n_pairs)
    probs = 1.0 / (1.0 + np.exp(-2.0 * scores))
    human_prefers = rng.uniform(0.0, 1.0, size=n_pairs) < probs
    return scores, human_prefers


def test_sim_clopper_pearson_exact() -> None:
    # 0 false improvements out of 10 released at alpha=0.10
    # P(X=0) = (1-0.10)^10 = 0.9^10 = 0.348678
    p0 = clopper_pearson_upper_p_value(0, 10, 0.10)
    assert p0 == pytest.approx(0.348678, abs=1e-4)

    # Empty release set is trivially safe
    assert clopper_pearson_upper_p_value(0, 0, 0.10) == 0.0

    # 10 false improvements out of 10 is maximum risk
    assert clopper_pearson_upper_p_value(10, 10, 0.10) == 1.0


def test_sim_fixed_sequence_abstain_everywhere() -> None:
    # Pathological case: scores are completely uninformative or inverted
    # Human prefers edit only 5% of the time, target alpha is 0.02
    rng = np.random.default_rng(12345)
    scores = rng.normal(size=200)
    human_prefers = rng.uniform(size=200) < 0.05  # 95% false improvements

    candidate_thresholds = np.linspace(3.0, -1.0, 41)
    thresh = learn_then_test_threshold(
        scores=scores,
        human_prefers_edit=human_prefers,
        candidate_thresholds=candidate_thresholds,
        alpha=0.02,
        delta=0.05,
    )
    # LTT should abstain or select a very conservative threshold with 0 releases
    if thresh is not None:
        fir, cov, n_rel, n_false = evaluate_decision_rule(scores, human_prefers, thresh)
        assert n_rel == 0 or fir <= 0.02


def test_sim_fir_guarantee_across_500_runs() -> None:
    """Verifies that across 500 simulated runs, the FIR violation rate does not exceed delta."""
    n_runs = 500
    alpha = 0.15
    delta = 0.10
    candidate_thresholds = np.linspace(2.5, -0.5, 31)

    rng = np.random.default_rng(42)
    violations = 0
    evaluated_runs = 0

    for _ in range(n_runs):
        # 300 calibration pairs, 300 evaluation pairs
        s_calib, y_calib = sim_generate_pairs(300, rng)
        s_eval, y_eval = sim_generate_pairs(300, rng)

        chosen_thresh = learn_then_test_threshold(
            scores=s_calib,
            human_prefers_edit=y_calib,
            candidate_thresholds=candidate_thresholds,
            alpha=alpha,
            delta=delta,
        )

        if chosen_thresh is not None:
            fir, cov, n_rel, _ = evaluate_decision_rule(s_eval, y_eval, chosen_thresh)
            if n_rel > 0:
                evaluated_runs += 1
                if fir > alpha:
                    violations += 1

    # Empirical violation rate must not exceed delta + binomial 3-sigma tolerance
    # sigma = sqrt(0.10 * 0.90 / 500) = 0.0134 -> 3-sigma ~ 0.04
    empirical_violation_rate = violations / evaluated_runs if evaluated_runs > 0 else 0.0
    print(f"\nSimulated LTT: evaluated {evaluated_runs}/{n_runs} runs, violations: {violations} ({empirical_violation_rate:.4f})")
    assert empirical_violation_rate <= delta + 0.04, (
        f"Empirical violation rate {empirical_violation_rate:.4f} exceeds target delta {delta} + tolerance"
    )
