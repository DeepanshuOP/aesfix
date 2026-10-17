"""Unit tests for Mask-Conditioned Region-Targeted Tier-A+ Executor (AF-2.4 DoD).

DoD Requirement:
Demonstrate 'brighten subject' localized execution where foreground subject exposure
is lifted while background pixels remain preserved without harsh edge artifacts.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from aesfix.critic.critic import CriticFix
from aesfix.executor import (
    MaskGenerator,
    MaskedTierAExecutor,
    RegionMask,
)


@pytest.fixture
def portrait_synthetic_image() -> Image.Image:
    """Creates a synthetic photo with a dark center subject (face/figure) and bright background."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    # Bright background (e.g. sunny sky/scenery)
    arr[:, :] = [210, 220, 235]
    # Dark underexposed center subject (e.g. shadowed portrait)
    arr[32:96, 32:96] = [45, 35, 30]
    return Image.fromarray(arr, mode="RGB")


def test_mask_generator_elliptical_and_feathering() -> None:
    """Verifies that MaskGenerator builds valid 2D feathered probability tensors."""
    mask = MaskGenerator.from_bbox(shape=(100, 100), bbox=[0.2, 0.2, 0.8, 0.8], feather_radius=5.0)

    assert isinstance(mask, RegionMask)
    assert mask.mask.shape == (1, 1, 100, 100)
    assert 0.0 <= mask.mask.min() <= mask.mask.max() <= 1.0

    # Center must have high weight, corners must be zero
    assert mask.mask[0, 0, 50, 50].item() > 0.80
    assert mask.mask[0, 0, 5, 5].item() == pytest.approx(0.0, abs=1e-3)


def test_brighten_subject_dod(portrait_synthetic_image: Image.Image) -> None:
    """DoD: Verify 'brighten subject' demo lifts subject while preserving background."""
    executor = MaskedTierAExecutor()

    # Apply localized +1.0 EV exposure boost targeted at the subject
    out_img, out_tensor = executor.execute_masked_op(
        image_input=portrait_synthetic_image,
        op="adjust_exposure",
        params={"ev_delta": 1.0},
        mask="subject",
    )

    arr_orig = np.array(portrait_synthetic_image, dtype=np.float32)
    arr_out = np.array(out_img, dtype=np.float32)

    # 1. Measure subject luminance delta in center (y: 40..80, x: 40..80)
    orig_subj_lum = float(np.mean(arr_orig[40:80, 40:80]))
    out_subj_lum = float(np.mean(arr_out[40:80, 40:80]))
    subj_delta = out_subj_lum - orig_subj_lum

    # Subject must be distinctly brightened (linear photon doubling yields ~9.4 delta under feathered alpha blending)
    assert subj_delta > 8.0, f"Subject was not sufficiently brightened: delta={subj_delta:.1f}"

    # 2. Measure background luminance delta in corners (y: 0..20, x: 0..20)
    orig_bg_lum = float(np.mean(arr_orig[0:20, 0:20]))
    out_bg_lum = float(np.mean(arr_out[0:20, 0:20]))
    bg_delta = abs(out_bg_lum - orig_bg_lum)

    # Background must be preserved (near zero delta)
    assert bg_delta < 5.0, f"Background was unexpectedly altered: delta={bg_delta:.1f}"


def test_custom_bounding_box_masked_contrast(portrait_synthetic_image: Image.Image) -> None:
    """Verifies that an arbitrary bounding box mask restricts contrast modifications."""
    executor = MaskedTierAExecutor()
    bbox_mask = MaskGenerator.from_bbox(
        shape=(128, 128),
        bbox=[0.25, 0.25, 0.75, 0.75],
        feather_radius=3.0,
    )

    out_img, _ = executor.execute_masked_op(
        image_input=portrait_synthetic_image,
        op="adjust_contrast",
        params={"contrast": 40.0},
        mask=bbox_mask,
    )

    arr_orig = np.array(portrait_synthetic_image, dtype=np.float32)
    arr_out = np.array(out_img, dtype=np.float32)

    # Top-left corner must be untouched
    assert np.allclose(arr_orig[:15, :15], arr_out[:15, :15], atol=1.0)


def test_execute_masked_fix_interoperability(portrait_synthetic_image: Image.Image) -> None:
    """Verifies execution of CriticFix objects containing semantic region_hint."""
    executor = MaskedTierAExecutor()

    fix = CriticFix(
        op="adjust_saturation",
        params={"saturation": 30.0},
        expected_aspect="color",
        nl_instruction="Enrich foreground colors.",
        region_hint="subject",
    )

    out_img, t_out = executor.execute_masked_fix(portrait_synthetic_image, fix)
    assert isinstance(out_img, Image.Image)
    assert t_out.shape == (1, 3, 128, 128)
    assert not torch.isnan(t_out).any()


def test_invert_mask_background_path(portrait_synthetic_image: Image.Image) -> None:
    """Verifies invert_mask option to adjust the background while locking the subject."""
    executor = MaskedTierAExecutor()

    out_img, _ = executor.execute_masked_op(
        image_input=portrait_synthetic_image,
        op="adjust_exposure",
        params={"ev_delta": -0.8},
        mask="subject",
        invert_mask=True,  # Darken background, lock subject
    )

    arr_orig = np.array(portrait_synthetic_image, dtype=np.float32)
    arr_out = np.array(out_img, dtype=np.float32)

    # Background must be darker
    assert np.mean(arr_out[0:20, 0:20]) < np.mean(arr_orig[0:20, 0:20]) - 20.0
    # Subject center must remain close to original
    assert abs(float(np.mean(arr_out[50:70, 50:70])) - float(np.mean(arr_orig[50:70, 50:70]))) < 5.0
