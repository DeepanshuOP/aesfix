"""Unit tests for IdentityGuard and Anti-Reward-Hacking Drift Gate (AF-2.2).

Why these tests verify the identity guard:
1. Validates that legitimate photographic improvements (Tier-A edits) pass the gate.
2. Validates that severe disruptions (noise, subject replacements, complete image swaps)
   are firmly rejected.
3. Tests the verify_gain protocol: confirmed that drift violations clamp gain to 0.0.
"""
from pathlib import Path
import time
import numpy as np
import pytest
import torch
from PIL import Image

from aesfix.critic.critic import CriticFix
from aesfix.executor import (
    GuardVerdict,
    IdentityGuard,
    TierAExecutor,
    compute_ssim_tensor,
)


@pytest.fixture
def sample_image() -> Image.Image:
    """Creates a standard synthetic test image."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :] = [220, 180, 140]
    arr[64:, :] = [40, 80, 160]
    return Image.fromarray(arr, mode="RGB")


def test_ssim_identical_images(sample_image: Image.Image) -> None:
    """Verifies that identical images yield SSIM = 1.0."""
    arr = np.array(sample_image, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    ssim_val = compute_ssim_tensor(t, t)
    assert pytest.approx(ssim_val, abs=1e-3) == 1.0


def test_guard_approves_identical_image(sample_image: Image.Image) -> None:
    """Verifies that identical images produce zero drift and are approved."""
    guard = IdentityGuard()
    verdict = guard.evaluate(sample_image, sample_image)

    assert isinstance(verdict, GuardVerdict)
    assert verdict.is_valid is True
    assert verdict.composite_drift == pytest.approx(0.0, abs=1e-3)
    assert verdict.rejection_reason is None


def test_guard_approves_moderate_tier_a_edit(sample_image: Image.Image) -> None:
    """Verifies that standard Tier-A exposure/contrast adjustments pass the identity guard."""
    executor = TierAExecutor()
    guard = IdentityGuard(drift_threshold=0.35)

    fixes = [
        CriticFix(
            op="adjust_exposure",
            params={"ev_delta": 0.4},
            expected_aspect="lighting",
            nl_instruction="Mildly brighten exposure.",
        ),
        CriticFix(
            op="adjust_contrast",
            params={"contrast": 15.0},
            expected_aspect="lighting",
            nl_instruction="Add mild contrast.",
        ),
    ]

    result = executor.execute_fixes(sample_image, fixes)
    verdict = guard.evaluate(sample_image, result.image)

    assert verdict.is_valid is True
    assert verdict.composite_drift < 0.35
    assert verdict.rejection_reason is None


def test_guard_rejects_catastrophic_replacement(sample_image: Image.Image) -> None:
    """Verifies that replacing the image with completely different content is rejected."""
    guard = IdentityGuard(drift_threshold=0.35)

    # Completely different image (e.g. inverted random pattern)
    random_arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    unrelated_image = Image.fromarray(random_arr, mode="RGB")

    verdict = guard.evaluate(sample_image, unrelated_image)

    assert verdict.is_valid is False
    assert verdict.composite_drift > 0.35
    assert verdict.rejection_reason is not None
    assert "Identity altered" in verdict.rejection_reason or "exceeds tolerance" in verdict.rejection_reason


def test_verify_gain_clamps_on_guard_violation(sample_image: Image.Image) -> None:
    """Verifies that an edit failing the guard receives strictly 0.0 aesthetic gain credit."""
    guard = IdentityGuard(drift_threshold=0.35)

    # Completely different image that hypothetically claims a higher aesthetic score
    random_arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    hallucinated_image = Image.fromarray(random_arr, mode="RGB")

    # Score increases from 0.40 to 0.85 (a +0.45 gain), but guard must veto it
    gain, verdict = guard.verify_gain(
        original_score=0.40,
        edited_score=0.85,
        original=sample_image,
        edited=hallucinated_image,
    )

    assert verdict.is_valid is False
    assert gain == 0.0  # Crucial: Zero gain credited to hallucination


def test_verify_gain_credits_valid_edit(sample_image: Image.Image) -> None:
    """Verifies that a valid edit receives genuine aesthetic gain credit."""
    executor = TierAExecutor()
    guard = IdentityGuard(drift_threshold=0.35)

    fix = CriticFix(
        op="adjust_exposure",
        params={"ev_delta": 0.25},
        expected_aspect="lighting",
        nl_instruction="Slight exposure lift.",
    )
    result = executor.execute_fixes(sample_image, [fix])

    gain, verdict = guard.verify_gain(
        original_score=0.50,
        edited_score=0.68,
        original=sample_image,
        edited=result.image,
    )

    assert verdict.is_valid is True
    assert gain == pytest.approx(0.18, abs=1e-2)


def test_guard_execution_latency(sample_image: Image.Image) -> None:
    """Verifies that Identity Guard evaluation completes in <30ms on CPU."""
    guard = IdentityGuard()

    start = time.perf_counter()
    _ = guard.evaluate(sample_image, sample_image)
    duration_ms = (time.perf_counter() - start) * 1000.0

    assert duration_ms < 30.0, f"Guard latency too high: {duration_ms:.2f}ms"


def test_guard_with_disk_images() -> None:
    """Verifies guard works seamlessly with file paths from disk."""
    p_test = Path("test.jpg")
    p_blur = Path("blur.jpg")

    if not p_test.exists() or not p_blur.exists():
        pytest.skip("Test images not present in root directory")

    guard = IdentityGuard(drift_threshold=0.35)
    # blur.jpg vs test.jpg represents substantial visual drift
    verdict = guard.evaluate(p_test, p_blur)
    assert isinstance(verdict, GuardVerdict)
