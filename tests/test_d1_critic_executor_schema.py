"""Tests for D1: Shared fix schema and pixel change verification.

Validates that:
1. CriticFix enforces direction in {+1, -1} with default +1.
2. HeuristicAspectCritic emits parameter names and units matching TierAExecutor.
3. Every fix emitted by HeuristicAspectCritic on the 5 legacy images changes pixels (max|Δ| > 1e-3).
"""
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from aesfix.critic.critic import CriticFix, HeuristicAspectCritic
from aesfix.executor.tier_a import TierAExecutor, _pil_to_tensor


def test_critic_fix_direction_validation() -> None:
    """CriticFix must allow direction in (+1, -1) and reject other values."""
    f1 = CriticFix(
        op="adjust_exposure",
        params={"ev_delta": 0.4},
        expected_aspect="lighting",
        direction=1,
        nl_instruction="Lift exposure",
    )
    assert f1.direction == 1

    f2 = CriticFix(
        op="adjust_exposure",
        params={"ev_delta": -0.4},
        expected_aspect="lighting",
        direction=-1,
        nl_instruction="Lower exposure",
    )
    assert f2.direction == -1

    # Default direction must be 1
    f_def = CriticFix(
        op="adjust_exposure",
        params={"ev_delta": 0.2},
        expected_aspect="lighting",
        nl_instruction="Slight lift",
    )
    assert f_def.direction == 1

    # Invalid direction values must be rejected
    for invalid in [0, 2, -2, 42]:
        with pytest.raises(ValueError, match="direction must be \\+1 or -1"):
            CriticFix(
                op="adjust_exposure",
                params={"ev_delta": 0.2},
                expected_aspect="lighting",
                direction=invalid,
                nl_instruction="Invalid test",
            )


def test_critic_fixes_change_pixels_on_legacy_images() -> None:
    """Every fix emitted by HeuristicAspectCritic must change pixels (max|Δ| > 1e-3) on legacy images."""
    critic = HeuristicAspectCritic()
    executor = TierAExecutor(device=torch.device("cpu"))

    legacy_images = ["blur.jpg", "bright.jpg", "dark.jpg", "temp.jpg", "test.jpg"]
    repo_root = Path(__file__).resolve().parent.parent

    total_fixes_tested = 0

    for img_name in legacy_images:
        img_path = repo_root / img_name
        assert img_path.exists(), f"Legacy image missing: {img_path}"

        critique = critic.critique(str(img_path))
        pil_img = Image.open(img_path).convert("RGB")
        tensor_orig = _pil_to_tensor(pil_img)

        for fix in critique.fixes:
            assert fix.direction in (1, -1), f"Invalid direction {fix.direction} in fix {fix.op}"

            # Verify parameter naming alignment
            if fix.op == "adjust_contrast":
                assert "factor" in fix.params
            elif fix.op == "adjust_white_balance":
                assert "temperature" in fix.params
            elif fix.op == "smart_crop":
                assert all(k in fix.params for k in ["top", "left", "bottom", "right"])
            elif fix.op == "denoise":
                assert "strength" in fix.params
                assert float(fix.params["strength"]) >= 1.0  # Executor 0-100 scale

            # Execute op and measure pixel delta
            tensor_fixed = executor.execute_op(tensor_orig, fix.op, fix.params)

            if tensor_fixed.shape[-2:] != tensor_orig.shape[-2:]:
                # If op is crop or horizon, interpolate to compare against original
                tensor_fixed_resampled = F.interpolate(
                    tensor_fixed,
                    size=tensor_orig.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                max_delta = (tensor_fixed_resampled - tensor_orig).abs().max().item()
            else:
                max_delta = (tensor_fixed - tensor_orig).abs().max().item()

            assert max_delta > 1e-3, (
                f"Fix {fix.op} with params {fix.params} on {img_name} produced max|Δ| = {max_delta:.6f} <= 1e-3"
            )
            total_fixes_tested += 1

    # Verify that fixes were indeed emitted and tested
    assert total_fixes_tested >= 10, f"Expected >= 10 fixes tested across legacy images, got {total_fixes_tested}"
