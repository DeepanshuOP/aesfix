"""Unit tests for shadow clipping metric (F5).

Verifies that shadow clipping is computed via perceptual luminance on a 0-1 scale:
luma = 0.299*R + 0.587*G + 0.114*B <= 0.01,
rather than checking the minimum color channel (which would falsely flag bright
saturated colors like pure yellow or pure cyan as shadow-clipped).
"""
import numpy as np
import pytest


def compute_shadow_clipping(arr: np.ndarray) -> float:
    """Computes fraction of pixels where luminance on [0, 1] scale is <= 0.01."""
    # arr has shape (H, W, 3) with float values in [0.0, 1.0]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return float(np.mean(luma <= 0.01))


def test_shadow_clipping_does_not_flag_bright_saturated_colors():
    # Pure yellow: R=1.0, G=1.0, B=0.0 -> Luma = 0.886
    # Pure cyan:   R=0.0, G=1.0, B=1.0 -> Luma = 0.701
    # Pure magenta: R=1.0, G=0.0, B=1.0 -> Luma = 0.413
    # In all these cases, min(RGB) == 0.0 <= 0.01, but the pixel is bright!
    saturated_arr = np.zeros((10, 10, 3), dtype=np.float32)
    saturated_arr[:5, :5] = [1.0, 1.0, 0.0]  # yellow
    saturated_arr[:5, 5:] = [0.0, 1.0, 1.0]  # cyan
    saturated_arr[5:, :5] = [1.0, 0.0, 1.0]  # magenta
    saturated_arr[5:, 5:] = [1.0, 1.0, 1.0]  # white

    # The buggy min-channel formula would report 75% shadow clipped
    min_channel_bug = float(np.mean(np.min(saturated_arr, axis=-1) <= 0.01))
    assert min_channel_bug == pytest.approx(0.75)

    # The correct luma-based formula reports 0% shadow clipped
    sh_clipping = compute_shadow_clipping(saturated_arr)
    assert sh_clipping == pytest.approx(0.0)


def test_shadow_clipping_flags_true_dark_pixels():
    # Synthetic array where exactly 40% of pixels have luma <= 0.01 (e.g. 0.005)
    # and 60% of pixels have luma > 0.01 (e.g. 0.50)
    arr = np.full((10, 10, 3), 0.50, dtype=np.float32)
    arr[:4, :] = 0.005  # 40 of 100 pixels are dark

    sh_clipping = compute_shadow_clipping(arr)
    assert sh_clipping == pytest.approx(0.40)


def test_shadow_clipping_unit_scale_not_byte_scale():
    # Verify that input must be normalized [0, 1], not [0, 255]
    dark_pixel = np.array([[[1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]]], dtype=np.float32)
    assert compute_shadow_clipping(dark_pixel) == 1.0

    bright_pixel = np.array([[[5.0 / 255.0, 5.0 / 255.0, 5.0 / 255.0]]], dtype=np.float32)
    # 5/255 = 0.0196 > 0.01, so it is above shadow clipping threshold
    assert compute_shadow_clipping(bright_pixel) == 0.0
