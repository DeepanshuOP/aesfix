"""Learn-Then-Test (LTT) Risk-Controlling Threshold Selection for Image Edits (Task G).

Why this module exists:
Implements statistical risk control (Angelopoulos, Bates, Candès, Jordan, Wainwright 2021)
to bound the False Improvement Rate (FIR) on image edits.

Definition of False Improvement Rate (FIR):
Among edits released by the decision rule (score delta s >= lambda), FIR is the fraction
that humans did NOT prefer (human MOS <= 0.5, i.e. original preferred or tie).

Fixed-Sequence Testing Guarantee:
Candidate thresholds lambda are evaluated in fixed order from most conservative (highest lambda)
to most liberal (lowest lambda). Under the null hypothesis H_0: E[FIR] > alpha, the Clopper-Pearson
binomial p-value tests whether observed false improvements are consistent with FIR <= alpha.
Fixed-sequence testing stops at the first failure to reject H_0, guaranteeing family-wise error
rate control at level delta without multiple testing penalty.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binom  # type: ignore[import-untyped]


def clopper_pearson_upper_p_value(k: int, n: int, alpha: float) -> float:
    """Computes the one-sided binomial p-value for H_0: rate > alpha vs H_1: rate <= alpha.

    Args:
        k: Number of false improvements observed (loss = 1).
        n: Total number of released edits.
        alpha: Target error rate threshold (e.g. 0.05, 0.10, 0.20).

    Returns:
        p-value in [0.0, 1.0]. If n <= 0, returns 1.0 (empty release set cannot reject H_0).
    """
    if n <= 0:
        # Why: An empty release set has no statistical evidence to reject H_0.
        return 1.0
    if k < 0:
        k = 0
    if k >= n:
        return 1.0
    # Under H_0: p_true >= alpha. P(X <= k) when X ~ Binomial(n, alpha)
    return float(binom.cdf(k, n, alpha))


def learn_then_test_threshold(
    scores: Sequence[float],
    human_prefers_edit: Sequence[bool],
    candidate_thresholds: Sequence[float],
    alpha: float,
    delta: float,
) -> Optional[float]:
    """Selects the most liberal threshold lambda that controls FIR at level alpha with probability >= 1 - delta.

    Fixed-sequence testing protocol:
    1. Thresholds must be ordered from conservative (high lambda) to liberal (low lambda).
    2. At each threshold, test H_0(lambda): E[FIR(lambda)] > alpha.
    3. If p-value <= delta, reject H_0(lambda) and test the next more liberal threshold.
    4. At the first failure to reject (p-value > delta), stop and return the last rejected threshold.
    5. If even the most conservative threshold fails to reject, return None ("abstain everywhere").

    Args:
        scores: Predicted judge preference score delta s = judge(edit) - judge(original).
        human_prefers_edit: True if human majority preferred the edit (MOS > 0.5), False otherwise.
        candidate_thresholds: Monotonically decreasing sequence of candidate thresholds.
        alpha: Maximum tolerable False Improvement Rate.
        delta: Failure tolerance probability (significance level, e.g. 0.10).

    Returns:
        The chosen threshold lambda, or None if no threshold passes.
    """
    if len(scores) != len(human_prefers_edit):
        raise ValueError("Length mismatch between scores and human_prefers_edit")
    if len(scores) == 0 or len(candidate_thresholds) == 0:
        return None

    scores_arr = np.asarray(scores, dtype=np.float64)
    human_arr = np.asarray(human_prefers_edit, dtype=bool)

    # Fixed-sequence testing evaluates candidates strictly in pre-specified order.
    # The grid start must be pre-specified rather than determined by calibration data.
    last_passing_threshold: Optional[float] = None

    for thresh in candidate_thresholds:
        released_mask = scores_arr >= thresh
        n_released = int(np.sum(released_mask))
        if n_released <= 0:
            # Cannot reject H_0 with 0 released samples; fixed-sequence testing stops.
            break

        # False improvements: released edits that humans did NOT prefer
        k_false = int(np.sum(released_mask & (~human_arr)))
        p_val = clopper_pearson_upper_p_value(k_false, n_released, alpha)

        if p_val <= delta:
            # Rejection confirmed: this threshold is statistically certified safe
            last_passing_threshold = float(thresh)
        else:
            # Fixed-sequence testing stops at the first failure to reject
            break

    return last_passing_threshold


def evaluate_decision_rule(
    scores: Sequence[float],
    human_prefers_edit: Sequence[bool],
    threshold: Optional[float],
) -> Tuple[float, float, int, int]:
    """Evaluates FIR and release coverage on an evaluation split given a decision threshold.

    Args:
        scores: Score deltas s = judge(edit) - judge(original).
        human_prefers_edit: Ground truth human preference indicators.
        threshold: Decision threshold. If None, abstains on all pairs.

    Returns:
        Tuple of (realized_fir, release_rate, n_released, n_false_improvements)
    """
    total = len(scores)
    if total == 0:
        return 0.0, 0.0, 0, 0
    if threshold is None:
        return 0.0, 0.0, 0, 0

    scores_arr = np.asarray(scores, dtype=np.float64)
    human_arr = np.asarray(human_prefers_edit, dtype=bool)

    released_mask = scores_arr >= threshold
    n_released = int(np.sum(released_mask))
    if n_released == 0:
        return 0.0, 0.0, 0, 0

    n_false = int(np.sum(released_mask & (~human_arr)))
    realized_fir = float(n_false / n_released)
    release_rate = float(n_released / total)
    return realized_fir, release_rate, n_released, n_false


@dataclass
class PilotRunSummary:
    """Statistical summary of certifier pilot runs over multiple scene-disjoint splits."""
    alpha: float
    delta: float
    method: str
    mean_realized_fir: float
    max_realized_fir: float
    violation_rate: float  # Fraction of seeds where realized FIR > alpha (must be <= delta)
    mean_release_rate: float
    abstain_rate: float


def run_certifier_pilot(
    scene_ids: Sequence[str],
    scores: Sequence[float],
    human_prefers_edit: Sequence[bool],
    alphas: Sequence[float] = (0.05, 0.10, 0.20),
    delta: float = 0.10,
    n_seeds: int = 20,
    candidate_thresholds: Optional[Sequence[float]] = None,
    epsilon: float = 0.04,
) -> Dict[str, List[PilotRunSummary]]:
    """Runs the Certifier Pilot evaluation over 20 random scene-disjoint splits.

    Why scene-disjoint splitting is required:
    Pairs from the same scene share content. Splitting at the scene level preserves exchangeability.

    Label requirement:
    'pilot — expert rendering vs original (DEAR/FiveK), not agent edits'
    """
    if not (len(scene_ids) == len(scores) == len(human_prefers_edit)):
        raise ValueError("Length mismatch in input arrays")

    # Group pair indices by scene
    scene_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, s_id in enumerate(scene_ids):
        scene_to_indices[s_id].append(idx)

    unique_scenes = np.array(sorted(list(scene_to_indices.keys())))
    n_scenes = len(unique_scenes)

    results: Dict[str, List[PilotRunSummary]] = defaultdict(list)

    for alpha in alphas:
        ltt_firs: List[float] = []
        ltt_coverages: List[float] = []
        ltt_abstains = 0

        naive_zero_firs: List[float] = []
        naive_zero_coverages: List[float] = []

        naive_eps_firs: List[float] = []
        naive_eps_coverages: List[float] = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_scenes)
            half = n_scenes // 2

            calib_scenes = set(unique_scenes[perm[:half]])
            eval_scenes = set(unique_scenes[perm[half:]])

            calib_idx = [i for s in calib_scenes for i in scene_to_indices[s]]
            eval_idx = [i for s in eval_scenes for i in scene_to_indices[s]]

            s_calib = [scores[i] for i in calib_idx]
            y_calib = [human_prefers_edit[i] for i in calib_idx]

            s_eval = [scores[i] for i in eval_idx]
            y_eval = [human_prefers_edit[i] for i in eval_idx]

            if candidate_thresholds is None:
                # Pre-specified fixed score units evaluated strictly from conservative to liberal
                cand_thresh = np.linspace(1.5, -0.5, 41)
            else:
                cand_thresh = candidate_thresholds

            # 1. LTT threshold selection
            chosen_thresh = learn_then_test_threshold(
                scores=s_calib,
                human_prefers_edit=y_calib,
                candidate_thresholds=cand_thresh,
                alpha=alpha,
                delta=delta,
            )

            if chosen_thresh is None:
                ltt_abstains += 1
                fir_ltt, cov_ltt, _, _ = 0.0, 0.0, 0, 0
            else:
                fir_ltt, cov_ltt, _, _ = evaluate_decision_rule(s_eval, y_eval, chosen_thresh)

            ltt_firs.append(fir_ltt)
            ltt_coverages.append(cov_ltt)

            # 2. Naive baseline: release if s > 0
            fir_z, cov_z, _, _ = evaluate_decision_rule(s_eval, y_eval, 0.0)
            naive_zero_firs.append(fir_z)
            naive_zero_coverages.append(cov_z)

            # 3. Naive baseline: release if s > epsilon
            fir_e, cov_e, _, _ = evaluate_decision_rule(s_eval, y_eval, epsilon)
            naive_eps_firs.append(fir_e)
            naive_eps_coverages.append(cov_e)

        # Summaries for alpha
        results["ltt"].append(
            PilotRunSummary(
                alpha=alpha,
                delta=delta,
                method="LTT (Learn-Then-Test, Clopper-Pearson)",
                mean_realized_fir=float(np.mean(ltt_firs)),
                max_realized_fir=float(np.max(ltt_firs)),
                violation_rate=float(np.mean([f > alpha for f in ltt_firs])),
                mean_release_rate=float(np.mean(ltt_coverages)),
                abstain_rate=float(ltt_abstains / n_seeds),
            )
        )
        results["naive_zero"].append(
            PilotRunSummary(
                alpha=alpha,
                delta=delta,
                method="Naive (s > 0)",
                mean_realized_fir=float(np.mean(naive_zero_firs)),
                max_realized_fir=float(np.max(naive_zero_firs)),
                violation_rate=float(np.mean([f > alpha for f in naive_zero_firs])),
                mean_release_rate=float(np.mean(naive_zero_coverages)),
                abstain_rate=0.0,
            )
        )
        results["naive_eps"].append(
            PilotRunSummary(
                alpha=alpha,
                delta=delta,
                method=f"Naive (s > {epsilon})",
                mean_realized_fir=float(np.mean(naive_eps_firs)),
                max_realized_fir=float(np.max(naive_eps_firs)),
                violation_rate=float(np.mean([f > alpha for f in naive_eps_firs])),
                mean_release_rate=float(np.mean(naive_eps_coverages)),
                abstain_rate=0.0,
            )
        )

    return results
