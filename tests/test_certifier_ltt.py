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

    # Why: Empty release set has no data to reject H_0, returning 1.0
    assert clopper_pearson_upper_p_value(0, 0, 0.10) == 1.0

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
    # LTT should abstain when no threshold can reject H_0
    assert thresh is None


def test_sim_power_at_target_release() -> None:
    """Task I2 Test (a): true FIR 0.05 at lambda releasing 50%, n_cal=150, alpha=0.20.

    LTT selects a lambda with release >= 30% in >= 90% of 500 simulations.
    """
    n_sims = 500
    n_cal = 150
    alpha = 0.20
    delta = 0.10
    rng = np.random.default_rng(12345)

    passed_runs = 0
    for _ in range(n_sims):
        scores_cal = rng.uniform(0.0, 1.0, size=n_cal)
        # Why: Score >= 0.5 has true human preference rate 0.95 (true FIR = 0.05).
        # Score < 0.5 has human preference rate 0.60 (FIR = 0.40).
        probs_cal = np.where(scores_cal >= 0.5, 0.95, 0.60)
        human_cal = rng.uniform(0.0, 1.0, size=n_cal) < probs_cal

        # Candidate thresholds spanning release rates from 30% to 50% (pre-specified fixed grid)
        cand_thresholds = [0.70, 0.65, 0.60, 0.55, 0.50]

        chosen_thresh = learn_then_test_threshold(
            scores=scores_cal,
            human_prefers_edit=human_cal,
            candidate_thresholds=cand_thresholds,
            alpha=alpha,
            delta=delta,
        )
        if chosen_thresh is not None:
            rel_rate = float(np.mean(scores_cal >= chosen_thresh))
            if rel_rate >= 0.30:
                passed_runs += 1

    pass_rate = passed_runs / n_sims
    assert pass_rate >= 0.90, f"Expected pass rate >= 0.90, got {pass_rate:.4f}"


def test_sim_validity_across_500_runs() -> None:
    """Task I2 Test (b): Validity: across 500 simulations, P(FIR_eval > alpha) <= delta + MC error."""
    n_sims = 500
    n_cal = 150
    n_eval = 300
    alpha = 0.20
    delta = 0.10
    rng = np.random.default_rng(54321)

    violations = 0
    evaluated_runs = 0
    for _ in range(n_sims):
        s_cal = rng.uniform(0.0, 1.0, size=n_cal)
        p_cal = np.where(s_cal >= 0.5, 0.95, 0.60)
        y_cal = rng.uniform(0.0, 1.0, size=n_cal) < p_cal

        s_eval = rng.uniform(0.0, 1.0, size=n_eval)
        p_eval = np.where(s_eval >= 0.5, 0.95, 0.60)
        y_eval = rng.uniform(0.0, 1.0, size=n_eval) < p_eval

        # Pre-specified fixed score grid evaluated strictly from conservative to liberal
        cand_thresh = np.linspace(0.85, 0.20, 25)

        chosen_thresh = learn_then_test_threshold(
            scores=s_cal,
            human_prefers_edit=y_cal,
            candidate_thresholds=cand_thresh,
            alpha=alpha,
            delta=delta,
        )
        if chosen_thresh is not None:
            fir, cov, n_rel, _ = evaluate_decision_rule(s_eval, y_eval, chosen_thresh)
            if n_rel > 0:
                evaluated_runs += 1
                if fir > alpha:
                    violations += 1

    mc_error = 2.58 * np.sqrt(delta * (1.0 - delta) / n_sims)  # 99% CI tolerance ~ 0.035
    emp_violation_rate = violations / evaluated_runs if evaluated_runs > 0 else 0.0
    assert emp_violation_rate <= delta + mc_error, (
        f"Empirical violation rate {emp_violation_rate:.4f} exceeded delta={delta} + MC error={mc_error:.4f}"
    )
