"""Property-based and mathematical invariance tests for Tier-A executor (Task D2).

Validates:
1. Contrast mid-grey pivot: 0.5 remains invariant across all contrast factors.
2. Exposure in linear light: sRGB EOTF round-trip, +1 EV linear scaling behaviour,
   and monotonic soft highlight shoulder roll-off.
3. Unsharp mask: continuous scaling without step discontinuity, non-negative validation,
   and monotonic response with respect to amount.
"""
import math
import pytest
import torch

from aesfix.executor.tier_a import (
    TierAExecutor,
    _srgb_to_linear,
    _linear_to_srgb,
    _apply_highlight_shoulder,
)


@pytest.fixture
def executor() -> TierAExecutor:
    return TierAExecutor(device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# 1. Contrast Mid-Grey Invariance & Symmetry
# ---------------------------------------------------------------------------


def test_contrast_midgrey_pivot_invariance(executor: TierAExecutor) -> None:
    """Mid-grey (0.5) must be a mathematical fixed point for contrast adjustments."""
    # A uniform tensor of 0.5 (midtone)
    mid_grey_tensor = torch.full((1, 3, 32, 32), 0.5, dtype=torch.float32)

    for factor in [0.1, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]:
        out = executor.execute_op(mid_grey_tensor, "adjust_contrast", {"factor": factor})
        # Difference from exact 0.5 must be effectively zero
        assert torch.allclose(out, mid_grey_tensor, atol=1e-6), (
            f"Contrast factor {factor} shifted mid-grey: max|delta| = {(out - 0.5).abs().max().item()}"
        )


def test_contrast_expands_around_pivot(executor: TierAExecutor) -> None:
    """Values above 0.5 must increase and below 0.5 must decrease when factor > 1.0."""
    test_tensor = torch.tensor([[[[0.2, 0.5, 0.8]]]], dtype=torch.float32)

    # Factor > 1.0: contrast expansion
    expanded = executor.execute_op(test_tensor, "adjust_contrast", {"factor": 1.5})
    assert expanded[0, 0, 0, 0].item() < 0.2, "Values below 0.5 should decrease with factor > 1"
    assert math.isclose(expanded[0, 0, 0, 1].item(), 0.5, abs_tol=1e-6), "Midtone 0.5 must not shift"
    assert expanded[0, 0, 0, 2].item() > 0.8, "Values above 0.5 should increase with factor > 1"

    # Factor < 1.0: contrast compression
    compressed = executor.execute_op(test_tensor, "adjust_contrast", {"factor": 0.5})
    assert compressed[0, 0, 0, 0].item() > 0.2, "Values below 0.5 should rise towards 0.5"
    assert math.isclose(compressed[0, 0, 0, 1].item(), 0.5, abs_tol=1e-6), "Midtone 0.5 must not shift"
    assert compressed[0, 0, 0, 2].item() < 0.8, "Values above 0.5 should fall towards 0.5"


# ---------------------------------------------------------------------------
# 2. Exposure Linear Light & Soft Shoulder
# ---------------------------------------------------------------------------


def test_srgb_linear_roundtrip_precision() -> None:
    """sRGB -> Linear -> sRGB must be an identity mapping across the unit interval."""
    # Test 1000 evenly spaced values across [0, 1]
    srgb_values = torch.linspace(0.0, 1.0, 1000, dtype=torch.float32)
    linear = _srgb_to_linear(srgb_values)
    reconstructed = _linear_to_srgb(linear)

    assert torch.allclose(srgb_values, reconstructed, atol=1e-5), (
        f"Round-trip error exceeds 1e-5: max|delta| = {(srgb_values - reconstructed).abs().max().item()}"
    )


def test_exposure_linear_midtone_scaling(executor: TierAExecutor) -> None:
    """+1 EV in linear light maps midtone sRGB 0.5 to ~0.686 (not 1.0 as in gamma multiplication)."""
    midtone = torch.full((1, 3, 16, 16), 0.5, dtype=torch.float32)
    plus_one_ev = executor.execute_op(midtone, "adjust_exposure", {"ev_delta": 1.0})

    val = plus_one_ev[0, 0, 0, 0].item()
    # Linear calculation: ((0.5+0.055)/1.055)^2.4 = 0.2140. * 2.0 = 0.4280. Re-encode: 1.055*(0.4280^(1/2.4))-0.055 = 0.686
    assert 0.67 < val < 0.70, f"Expected +1 EV on sRGB 0.5 to yield ~0.686, got {val:.4f}"


def test_highlight_shoulder_monotonicity_and_bounds() -> None:
    """Highlight shoulder must be monotonically increasing, continuous, and strictly bounded."""
    x = torch.linspace(0.0, 5.0, 2000, dtype=torch.float32)
    y = _apply_highlight_shoulder(x, threshold=0.75)

    # 1. Identity below threshold
    below_thresh = x <= 0.75
    assert torch.allclose(y[below_thresh], x[below_thresh]), "Shoulder altered values below threshold"

    # 2. Bounded in [0, 1.0]
    assert (y >= 0.0).all() and (y <= 1.0).all(), "Shoulder output exceeded [0, 1.0]"

    # 3. Monotonically increasing: dy/dx > 0 everywhere
    diffs = y[1:] - y[:-1]
    assert (diffs >= 0.0).all(), "Shoulder violated monotonicity"


def test_exposure_identity_at_zero(executor: TierAExecutor) -> None:
    """EV delta of 0.0 must be an exact no-op."""
    rand_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    out = executor.execute_op(rand_tensor, "adjust_exposure", {"ev_delta": 0.0})
    assert torch.equal(rand_tensor, out)


# ---------------------------------------------------------------------------
# 3. Unsharp Mask Continuity & Monotonicity
# ---------------------------------------------------------------------------


def test_unsharp_mask_continuity(executor: TierAExecutor) -> None:
    """Unsharp amount must be smooth across the scale with no step discontinuity at amount=10."""
    # Synthetic checkerboard with sharp edges
    img = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    img[:, :, ::2, ::2] = 1.0
    img[:, :, 1::2, 1::2] = 1.0

    # Step through amounts from 5.0 to 15.0 in fine increments of 0.5
    amounts = [float(x) for x in range(10, 31)]  # 5.0 to 15.0 in 0.5 steps (x / 2)
    deltas = []

    for a in [x / 2.0 for x in amounts]:
        sharpened = executor.execute_op(img, "unsharp_mask", {"amount": a, "radius": 1.0})
        mean_delta = (sharpened - img).abs().mean().item()
        deltas.append(mean_delta)

    # Verify that consecutive delta differences are small and proportional (no jump at 10.0)
    diffs = [deltas[i + 1] - deltas[i] for i in range(len(deltas) - 1)]
    max_diff = max(diffs)
    min_diff = min(diffs)

    # Under a continuous linear scale, diffs between steps of 0.5 are nearly identical
    assert (max_diff - min_diff) < 0.01, f"Step discontinuity detected in unsharp amounts: diffs={diffs}"


def test_unsharp_mask_monotonic_response(executor: TierAExecutor) -> None:
    """Sharpening energy |fixed - orig| must be monotonically non-decreasing with amount."""
    img = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    img[:, :, :16, :] = 0.2
    img[:, :, 16:, :] = 0.8  # Strong step edge

    prev_delta = -1.0
    for amount in [0.0, 10.0, 25.0, 40.0, 60.0, 100.0]:
        out = executor.execute_op(img, "unsharp_mask", {"amount": amount, "radius": 1.0})
        curr_delta = (out - img).abs().mean().item()
        if amount == 0.0:
            assert curr_delta == 0.0, "amount=0 must produce zero delta"
        else:
            assert curr_delta >= prev_delta, (
                f"Monotonicity failed: amount {amount} yielded delta {curr_delta} < {prev_delta}"
            )
        prev_delta = curr_delta


def test_unsharp_mask_negative_amount_raises(executor: TierAExecutor) -> None:
    """Negative amount values must raise ValueError."""
    dummy = torch.zeros((1, 3, 16, 16), dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsharp amount must be non-negative"):
        executor.execute_op(dummy, "unsharp_mask", {"amount": -1.0})
