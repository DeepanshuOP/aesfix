"""Certified Gain and Aesthetic Evaluation Metrics (Roadmap v6 Part IV.2).

Formal specification:
- AG_best@k (signed): E_x[ max_{i<=k} Delta(x, x_i) ]. Non-decreasing in k by construction.
- AG_comp@k (signed): E_x[ Delta(x, x_{1:k}) ]. May decrease with k.
- Clipped variants: max(0, Delta), reported only as secondary columns and labelled.
- Human win rate (HWR): E[ 1{humans prefer x'} + 0.5 * 1{tie} ] on the human-labelled subset.
- Harm rate (HR): P(humans prefer x | the system applied x').
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


def ag_best_at_k(deltas: Sequence[Sequence[float]], k: int) -> float:
    """Computes signed Best-of-k Aesthetic Gain (AG_best@k).

    Args:
        deltas: List of per-sample candidate edit deltas [Delta(x, x_1), ..., Delta(x, x_m)].
        k: Maximum rank cutoff (must be >= 1).

    Returns:
        Signed mean best gain across samples: E_x[ max_{i<=k} Delta(x, x_i) ].
        Non-decreasing in k by construction.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not deltas:
        return 0.0

    sample_maxes: list[float] = []
    for sample in deltas:
        if not sample:
            sample_maxes.append(0.0)
            continue
        cutoff = min(k, len(sample))
        sample_maxes.append(max(sample[:cutoff]))

    return float(sum(sample_maxes) / len(sample_maxes))


def ag_best_at_k_clipped(deltas: Sequence[Sequence[float]], k: int) -> float:
    """Computes non-negative clipped Best-of-k Aesthetic Gain (AG_best@k clipped).

    Reported only as a secondary metric; primary evaluations must use signed AG_best@k.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if not deltas:
        return 0.0

    sample_maxes: list[float] = []
    for sample in deltas:
        if not sample:
            sample_maxes.append(0.0)
            continue
        cutoff = min(k, len(sample))
        sample_maxes.append(max(0.0, max(sample[:cutoff])))

    return float(sum(sample_maxes) / len(sample_maxes))


def ag_comp_at_k(comp_deltas: Sequence[float]) -> float:
    """Computes signed Compositional Aesthetic Gain (AG_comp@k).

    Args:
        comp_deltas: Per-sample delta of the composed edit Delta(x, x_{1:k}).

    Returns:
        Signed mean compositional gain: E_x[ Delta(x, x_{1:k}) ].
    """
    if not comp_deltas:
        return 0.0
    return float(sum(comp_deltas) / len(comp_deltas))


def ag_comp_at_k_clipped(comp_deltas: Sequence[float]) -> float:
    """Computes non-negative clipped Compositional Aesthetic Gain.

    Reported only as a secondary metric; primary evaluations must use signed AG_comp@k.
    """
    if not comp_deltas:
        return 0.0
    clipped = [max(0.0, float(d)) for d in comp_deltas]
    return float(sum(clipped) / len(clipped))


def _normalize_preference(val: Any) -> float:
    """Normalizes preference to: 1.0 (win for edit x'), 0.5 (tie), 0.0 (loss/prefer original x)."""
    if isinstance(val, (int, float)):
        if val == 1.0 or val == 1:
            return 1.0
        if val == 0.5:
            return 0.5
        if val == 0.0 or val == 0:
            return 0.0
        raise ValueError(f"Numeric preference must be 1.0, 0.5, or 0.0, got {val}")

    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"win", "edit", "x_prime", "after", "1", "true"}:
            return 1.0
        if v in {"tie", "same", "equal", "draw", "0.5"}:
            return 0.5
        if v in {"loss", "orig", "original", "x", "before", "0", "false"}:
            return 0.0
        raise ValueError(f"Unrecognized preference string: {val!r}")

    if isinstance(val, bool):
        return 1.0 if val else 0.0

    raise TypeError(f"Unsupported preference value type: {type(val)}")


def human_win_rate(preferences: Sequence[Any]) -> float:
    """Computes Human Win Rate (HWR = E[ 1{humans prefer x'} + 0.5 * 1{tie} ]).

    Args:
        preferences: Sequence of pairwise preference outcomes.
            1.0 / 'win' / 'edit' / 'after': Human prefers the edited photo x'.
            0.5 / 'tie' / 'same': Human considers them equal.
            0.0 / 'loss' / 'original' / 'before': Human prefers the original photo x.

    Returns:
        Expected win rate in [0.0, 1.0] with ties counted as 0.5.
    """
    if not preferences:
        return 0.0

    scores = [_normalize_preference(p) for p in preferences]
    return float(sum(scores) / len(scores))


def harm_rate(
    preferences: Sequence[Any],
    applied_mask: Optional[Sequence[bool]] = None,
) -> float:
    """Computes Harm Rate (HR = P(humans prefer x | the system applied x')).

    Args:
        preferences: Sequence of pairwise preference outcomes.
        applied_mask: Optional boolean sequence indicating whether the system decided
            to apply the edit x' (True) or abstain/leave it alone (False). If omitted,
            all edits are considered applied.

    Returns:
        Harm rate: fraction of applied edits that worsened the photo (human preferred original x).
    """
    if not preferences:
        return 0.0

    if applied_mask is not None and len(applied_mask) != len(preferences):
        raise ValueError(
            f"Length mismatch: preferences has {len(preferences)}, applied_mask has {len(applied_mask)}"
        )

    applied_losses = 0
    total_applied = 0

    for i, p in enumerate(preferences):
        is_applied = applied_mask[i] if applied_mask is not None else True
        if not is_applied:
            continue
        total_applied += 1
        norm_val = _normalize_preference(p)
        if norm_val == 0.0:
            applied_losses += 1

    if total_applied == 0:
        return 0.0

    return float(applied_losses / total_applied)


def pairwise_accuracy(judge_deltas: Sequence[float], human_mos: Sequence[float]) -> float:
    """Computes pairwise accuracy: fraction of pairs where judge and human agree on the preferred image.

    Args:
        judge_deltas: Score difference (score_left - score_right) for each pair.
        human_mos: Fraction of human voters preferring the left image over the right image (in [0.0, 1.0]).
            mos > 0.5 means humans prefer left; mos < 0.5 means humans prefer right.

    Returns:
        Pairwise accuracy in [0.0, 1.0].
    """
    if len(judge_deltas) != len(human_mos):
        raise ValueError(f"Length mismatch: judge_deltas ({len(judge_deltas)}) vs human_mos ({len(human_mos)})")
    if not judge_deltas:
        return 0.0

    correct = 0
    total = 0
    for d, m in zip(judge_deltas, human_mos):
        if m == 0.5:
            # Neutral tie in human voting
            continue
        total += 1
        human_prefers_left = m > 0.5
        judge_prefers_left = d > 0.0
        if human_prefers_left == judge_prefers_left:
            correct += 1

    return float(correct / total) if total > 0 else 0.0


def pairwise_spearman(judge_deltas: Sequence[float], human_mos: Sequence[float]) -> float:
    """Computes Spearman rank correlation between judge preference delta and human signed preference margin.

    Orientation Convention (Task B1):
        - judge_delta: (score_left - score_right)
        - human_signed_margin: (human_mos - 0.5)
        Both quantities are positive when the left image is preferred and negative when the right
        image is preferred. Correlating signed delta with signed margin preserves directional monotonicity.
        (Correlating signed delta with unsigned |mos - 0.5| is an orientation bug that produces negative/zero correlations).

    Returns:
        Spearman rank correlation in [-1.0, 1.0].
    """
    if len(judge_deltas) != len(human_mos):
        raise ValueError(f"Length mismatch: judge_deltas ({len(judge_deltas)}) vs human_mos ({len(human_mos)})")
    if len(judge_deltas) < 2:
        return 0.0

    from scipy.stats import spearmanr  # type: ignore[import-untyped]

    signed_margins = [float(m) - 0.5 for m in human_mos]
    srcc_val, _ = spearmanr(judge_deltas, signed_margins)
    return float(srcc_val)


def bootstrap_scene_ci(
    scene_ids: Sequence[str],
    is_correct: Sequence[bool],
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Computes 95% bootstrap confidence interval over SCENES (cluster bootstrap).

    Why clustering by scene is necessary (Task B2):
    Pairs within the same scene share scene content and are not statistically independent.
    Resampling individual pairs artificially shrinks variance. Resampling scenes preserves within-scene correlation.

    Returns:
        tuple of (mean_accuracy, ci_low, ci_high)
    """
    import numpy as np
    from collections import defaultdict

    if len(scene_ids) != len(is_correct):
        raise ValueError(f"Length mismatch: {len(scene_ids)} scenes vs {len(is_correct)} outcomes")
    if not scene_ids:
        return 0.0, 0.0, 0.0

    # Group outcomes by scene
    scene_map: dict[str, list[bool]] = defaultdict(list)
    for s_id, corr in zip(scene_ids, is_correct):
        scene_map[s_id].append(corr)

    unique_scenes = np.array(list(scene_map.keys()))
    n_scenes = len(unique_scenes)

    rng = np.random.default_rng(seed)
    boot_accs = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sample_scenes = rng.choice(unique_scenes, size=n_scenes, replace=True)
        total_correct = 0
        total_pairs = 0
        for s in sample_scenes:
            vals = scene_map[s]
            total_correct += sum(vals)
            total_pairs += len(vals)
        boot_accs[b] = total_correct / total_pairs if total_pairs > 0 else 0.0

    alpha = (1.0 - ci_level) / 2.0
    ci_low = float(np.percentile(boot_accs, alpha * 100))
    ci_high = float(np.percentile(boot_accs, (1.0 - alpha) * 100))
    mean_acc = float(sum(is_correct) / len(is_correct))
    return mean_acc, ci_low, ci_high


def bootstrap_paired_difference_ci(
    scene_ids: Sequence[str],
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_resamples: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Computes 95% bootstrap CI for the paired difference (accuracy_a - accuracy_b) clustered by SCENE.

    Returns:
        tuple of (mean_diff, ci_low, ci_high)
    """
    import numpy as np
    from collections import defaultdict

    if not (len(scene_ids) == len(correct_a) == len(correct_b)):
        raise ValueError("Length mismatch between scene_ids, correct_a, and correct_b")
    if not scene_ids:
        return 0.0, 0.0, 0.0

    scene_map_a: dict[str, list[bool]] = defaultdict(list)
    scene_map_b: dict[str, list[bool]] = defaultdict(list)
    for s_id, a, b in zip(scene_ids, correct_a, correct_b):
        scene_map_a[s_id].append(a)
        scene_map_b[s_id].append(b)

    unique_scenes = np.array(list(scene_map_a.keys()))
    n_scenes = len(unique_scenes)

    rng = np.random.default_rng(seed)
    boot_diffs = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sample_scenes = rng.choice(unique_scenes, size=n_scenes, replace=True)
        corr_a = sum(sum(scene_map_a[s]) for s in sample_scenes)
        corr_b = sum(sum(scene_map_b[s]) for s in sample_scenes)
        total_p = sum(len(scene_map_a[s]) for s in sample_scenes)
        if total_p > 0:
            boot_diffs[b] = (corr_a - corr_b) / total_p

    alpha = (1.0 - ci_level) / 2.0
    ci_low = float(np.percentile(boot_diffs, alpha * 100))
    ci_high = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))
    mean_diff = float((sum(correct_a) - sum(correct_b)) / len(scene_ids))
    return mean_diff, ci_low, ci_high

