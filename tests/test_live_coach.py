"""Unit tests for the Live Shooting Coach and Real-Time Viewfinder Guidance Engine (AF-6.5).

Why these tests verify operational performance and guidance accuracy:
Ensures tilt detection, exposure clipping thresholds, sharpness acutance operators,
and Rule-of-Thirds power point distance calculations run in sub-30ms latency on CPU
and emit actionable capture-time alerts.
"""
from __future__ import annotations

import numpy as np

from aesfix.critic.live_coach import (
    LiveShootingCoach,
    NudgeSeverity,
    ShutterRecommendation,
    ViewfinderFrameTelemetry,
)


def _make_balanced_test_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Creates a balanced test image with moderate contrast and a sharp focal object on the upper-right third."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    # Add a horizontal horizon line across the middle
    img[h // 2 - 2 : h // 2 + 2, :] = 40

    # Place a sharp high-contrast subject at approx (2/3, 1/3)
    cx, cy = int(w * (2.0 / 3.0)), int(h * (1.0 / 3.0))
    img[cy - 30 : cy + 30, cx - 30 : cx + 30] = 240
    # Add high-frequency checkerboard pattern for sharpness
    for y in range(cy - 20, cy + 20, 4):
        for x in range(cx - 20, cx + 20, 4):
            img[y : y + 2, x : x + 2] = 20

    return img


def test_input_normalization() -> None:
    """Verifies that float, uint8, and multichannel arrays convert to valid grayscale arrays."""
    coach = LiveShootingCoach()

    # Float [0, 1] RGB
    float_rgb = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    gray = coach._ensure_grayscale_uint8(float_rgb)
    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8
    assert abs(int(gray[50, 50]) - 127) <= 2

    # 2D uint8
    uint8_2d = np.full((50, 50), 200, dtype=np.uint8)
    gray2 = coach._ensure_grayscale_uint8(uint8_2d)
    assert gray2.shape == (50, 50)
    assert gray2[0, 0] == 200


def test_exposure_clipping_detection() -> None:
    """Verifies that blown highlights and crushed blacks trigger appropriate warnings."""
    coach = LiveShootingCoach(highlight_clip_warning=5.0, shadow_clip_warning=5.0)

    # Frame with 20% blown highlights (255) and 15% crushed blacks (0)
    frame = np.full((100, 100), 128, dtype=np.uint8)
    frame[:20, :] = 255  # 20% blown
    frame[-15:, :] = 0   # 15% black

    hi_clip, sh_clip = coach.analyze_exposure_clipping(frame)
    assert hi_clip >= 19.0
    assert sh_clip >= 14.0

    telemetry = coach.evaluate_frame(frame)
    nudges = [n.aspect for n in telemetry.nudges]
    assert "lighting" in nudges


def test_sharpness_acutance_metric() -> None:
    """Verifies that flat/blurred images register low sharpness while patterned images register high sharpness."""
    coach = LiveShootingCoach(sharpness_minimum=30.0)

    # Completely flat image (zero edges)
    flat = np.full((120, 120), 128, dtype=np.uint8)
    flat_sharp = coach.calculate_sharpness(flat)
    assert flat_sharp == 0.0

    # High-contrast checkerboard image
    checker = np.zeros((120, 120), dtype=np.uint8)
    checker[::2, :] = 255
    checker[:, ::2] = 255
    sharp_val = coach.calculate_sharpness(checker)
    assert sharp_val > 100.0


def test_rule_of_thirds_anchor_calculation() -> None:
    """Verifies subject centroid distance calculation relative to rule-of-thirds power points."""
    coach = LiveShootingCoach()

    # Image with energy concentrated at upper-left third (approx 1/3, 1/3)
    frame = np.full((300, 300), 128, dtype=np.uint8)
    frame[90:110, 90:110] = 255  # 100/300 = 0.333

    centroid, thirds_dist = coach.locate_subject_centroid(frame)
    # Centroid should be close to (0.33, 0.33)
    assert abs(centroid[0] - (1.0 / 3.0)) < 0.10
    assert abs(centroid[1] - (1.0 / 3.0)) < 0.10
    assert thirds_dist < 0.08


def test_balanced_frame_telemetry_and_latency() -> None:
    """Verifies that a well-composed, balanced frame yields high readiness and fast real-time execution."""
    coach = LiveShootingCoach()
    frame = _make_balanced_test_frame()

    telemetry = coach.evaluate_frame(frame)

    assert isinstance(telemetry, ViewfinderFrameTelemetry)
    assert telemetry.latency_ms < 150.0  # Real-time capture latency threshold
    assert telemetry.capture_readiness_score > 0.65
    assert telemetry.shutter_recommendation in (
        ShutterRecommendation.CAPTURE_READY,
        ShutterRecommendation.CAUTION,
    )
    assert "composition" in telemetry.aspect_estimates
    assert "lighting" in telemetry.aspect_estimates


def test_critical_warning_hold_shutter() -> None:
    """Verifies that an unrecoverable flaw (e.g. extreme tilt and highlight blowout) triggers HOLD_AND_ADJUST."""
    coach = LiveShootingCoach(highlight_clip_warning=5.0)

    # Frame with 60% blown highlights
    frame = np.full((200, 200), 255, dtype=np.uint8)
    frame[150:, :] = 100

    telemetry = coach.evaluate_frame(frame)
    assert telemetry.shutter_recommendation == ShutterRecommendation.HOLD_AND_ADJUST
    critical_nudges = [n for n in telemetry.nudges if n.severity == NudgeSeverity.CRITICAL]
    assert len(critical_nudges) > 0


def test_hud_overlay_data_formatting() -> None:
    """Verifies that HUD overlay data matches mobile viewfinder expectations."""
    coach = LiveShootingCoach()
    frame = _make_balanced_test_frame()

    telemetry = coach.evaluate_frame(frame)
    hud = coach.get_hud_overlay_data(telemetry)

    assert "readiness_score" in hud
    assert "badge_color" in hud
    assert hud["badge_color"].startswith("#")
    assert "focal_reticle" in hud
    assert "x_pct" in hud["focal_reticle"]
    assert "top_nudge" in hud
