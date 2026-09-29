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
