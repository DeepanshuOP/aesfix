"""Hand-computed toy tests and Hypothesis property tests for certified metrics (Roadmap v6 Part IV.2)."""
from __future__ import annotations

import math
from typing import List

import pytest
from hypothesis import given, strategies as st

from aesfix.eval.metrics import (
    ag_best_at_k,
    ag_best_at_k_clipped,
    ag_comp_at_k,
    ag_comp_at_k_clipped,
    harm_rate,
    human_win_rate,
)


# ==============================================================================
# 1. Hand-computed toy tests
# ==============================================================================

def test_ag_best_at_k_hand_computed():
    """Verify signed AG_best@k against manual calculations."""
    deltas = [
        [-0.2, 0.5, 0.1],
        [0.3, -0.4, 0.2],
        [-0.5, -0.1, -0.3],
    ]

    # k=1: maxes are [-0.2, 0.3, -0.5] -> sum = -0.4 -> mean = -0.4/3 = -0.133333...
    res_k1 = ag_best_at_k(deltas, k=1)
    assert math.isclose(res_k1, -0.4 / 3.0, rel_tol=1e-6)

    # k=2: maxes are [0.5, 0.3, -0.1] -> sum = 0.7 -> mean = 0.7/3 = 0.233333...
    res_k2 = ag_best_at_k(deltas, k=2)
    assert math.isclose(res_k2, 0.7 / 3.0, rel_tol=1e-6)

    # k=3: maxes are [0.5, 0.3, -0.1] -> sum = 0.7 -> mean = 0.7/3 = 0.233333...
    res_k3 = ag_best_at_k(deltas, k=3)
    assert math.isclose(res_k3, 0.7 / 3.0, rel_tol=1e-6)

    # Non-decreasing property holds on toy sample
    assert res_k1 <= res_k2 <= res_k3


def test_ag_best_at_k_clipped_hand_computed():
    """Verify clipped AG_best@k against manual calculations."""
    deltas = [
        [-0.2, 0.5, 0.1],
        [0.3, -0.4, 0.2],
        [-0.5, -0.1, -0.3],
    ]

    # k=1: maxes clipped are [0.0, 0.3, 0.0] -> sum = 0.3 -> mean = 0.1
    res_k1 = ag_best_at_k_clipped(deltas, k=1)
    assert math.isclose(res_k1, 0.1, rel_tol=1e-6)

    # k=2: maxes clipped are [0.5, 0.3, 0.0] -> sum = 0.8 -> mean = 0.8/3 = 0.266666...
    res_k2 = ag_best_at_k_clipped(deltas, k=2)
    assert math.isclose(res_k2, 0.8 / 3.0, rel_tol=1e-6)


def test_ag_comp_at_k_hand_computed():
    """Verify signed AG_comp@k against manual calculations."""
    comp_deltas = [0.4, -0.2, -0.1, 0.3]
    # sum = 0.4 - 0.2 - 0.1 + 0.3 = 0.4 -> mean = 0.1
    res = ag_comp_at_k(comp_deltas)
    assert math.isclose(res, 0.1, rel_tol=1e-6)


def test_ag_comp_at_k_clipped_hand_computed():
    """Verify clipped AG_comp@k against manual calculations."""
    comp_deltas = [0.4, -0.2, -0.1, 0.3]
    # clipped = [0.4, 0.0, 0.0, 0.3] -> sum = 0.7 -> mean = 0.175
    res = ag_comp_at_k_clipped(comp_deltas)
    assert math.isclose(res, 0.175, rel_tol=1e-6)


def test_human_win_rate_hand_computed():
    """Verify HWR with ties counted as 0.5."""
    # Numeric: [win, tie, loss, win] -> [1.0, 0.5, 0.0, 1.0] -> sum = 2.5 -> mean = 2.5/4 = 0.625
    numeric_prefs = [1.0, 0.5, 0.0, 1.0]
    assert math.isclose(human_win_rate(numeric_prefs), 0.625, rel_tol=1e-6)

    # String aliases: ["win", "tie", "loss", "after", "draw"] -> [1.0, 0.5, 0.0, 1.0, 0.5] -> 3.0/5 = 0.6
    str_prefs = ["win", "tie", "loss", "after", "draw"]
    assert math.isclose(human_win_rate(str_prefs), 0.6, rel_tol=1e-6)

    # Empty preferences returns 0.0
    assert human_win_rate([]) == 0.0


def test_harm_rate_hand_computed():
    """Verify Harm Rate = P(humans prefer x | system applied x')."""
    # System applied all: [win, tie, loss, loss, win] -> 2 losses out of 5 -> 0.4
    prefs = [1.0, 0.5, 0.0, 0.0, 1.0]
    assert math.isclose(harm_rate(prefs), 0.4, rel_tol=1e-6)

    # System applied selective mask
    mask = [True, True, True, False, False]
    # Filtered to applied: [1.0, 0.5, 0.0] -> 1 loss out of 3 -> 1/3 = 0.333333...
    assert math.isclose(harm_rate(prefs, applied_mask=mask), 1.0 / 3.0, rel_tol=1e-6)

    # No applied edits returns 0.0
    assert harm_rate(prefs, applied_mask=[False] * 5) == 0.0


def test_invalid_arguments_raise():
    """Verify boundary conditions and invalid arguments."""
    with pytest.raises(ValueError, match="k must be at least 1"):
        ag_best_at_k([[0.1]], k=0)

    with pytest.raises(ValueError, match="Length mismatch"):
        harm_rate([1.0, 0.0], applied_mask=[True])


# ==============================================================================
# 2. Property testing with Hypothesis: AG_best@k is non-decreasing in k
# ==============================================================================

@given(
    deltas=st.lists(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=12,
        ),
        min_size=1,
        max_size=25,
    )
)
def test_ag_best_at_k_is_non_decreasing_property(deltas: List[List[float]]):
    """Property test (Hypothesis): AG_best@k is strictly non-decreasing in k by construction.

    For any sequence of candidate deltas for any number of images:
        AG_best@1 <= AG_best@2 <= ... <= AG_best@K
    """
    max_k = max(len(row) for row in deltas)

    gains = [ag_best_at_k(deltas, k=k) for k in range(1, max_k + 1)]

    for i in range(len(gains) - 1):
        # Allow tiny epsilon for 64-bit IEEE float representation
        assert gains[i] <= gains[i + 1] + 1e-12, (
            f"Violation at k={i+1} -> k={i+2}: gain({i+1})={gains[i]} > gain({i+2})={gains[i+1]}"
        )


@given(
    deltas=st.lists(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=1,
            max_size=12,
        ),
        min_size=1,
        max_size=25,
    )
)
def test_ag_best_at_k_clipped_is_non_decreasing_property(deltas: List[List[float]]):
    """Property test (Hypothesis): Clipped AG_best@k is also non-decreasing in k."""
    max_k = max(len(row) for row in deltas)

    clipped_gains = [ag_best_at_k_clipped(deltas, k=k) for k in range(1, max_k + 1)]

    for i in range(len(clipped_gains) - 1):
        assert clipped_gains[i] <= clipped_gains[i + 1] + 1e-12
        assert clipped_gains[i] >= 0.0
