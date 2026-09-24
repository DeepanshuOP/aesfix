"""Unit tests for the Synthetic Counterfactual Engine (AF-2.3 DoD).

DoD Requirement:
AG@1 with oracle advice recovers >= 90% of known degradation on the synthetic set (metric sanity).
"""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aesfix.data.synthetic import (
    SUPPORTED_DEGRADATIONS,
    SyntheticCounterfactualEngine,
    SyntheticTriplet,
)


@pytest.fixture
def clean_target_image() -> Image.Image:
    """Generates a high-quality synthetic photographic image with realistic color gradients."""
    h, w = 128, 128
    y, x = np.mgrid[:h, :w]
    # Smooth multi-tone gradient
    r = np.clip(100 + 80 * np.sin(x / 15.0), 0, 255).astype(np.uint8)
    g = np.clip(120 + 70 * np.cos(y / 15.0), 0, 255).astype(np.uint8)
    b = np.clip(140 + 60 * np.sin((x + y) / 25.0), 0, 255).astype(np.uint8)
    arr = np.stack([r, g, b], axis=-1)
    return Image.fromarray(arr, mode="RGB")


def test_supported_degradations_catalog() -> None:
    """Verifies that all canonical invertible operations are supported."""
    assert len(SUPPORTED_DEGRADATIONS) >= 5
    assert "adjust_exposure" in SUPPORTED_DEGRADATIONS
    assert "adjust_contrast" in SUPPORTED_DEGRADATIONS
    assert "adjust_saturation" in SUPPORTED_DEGRADATIONS
    assert "adjust_white_balance" in SUPPORTED_DEGRADATIONS


def test_exposure_oracle_recovery_dod(clean_target_image: Image.Image) -> None:
    """DoD: Exposure oracle inverse achieves >= 90% degradation recovery."""
    engine = SyntheticCounterfactualEngine(seed=42)
    triplet = engine.generate_triplet(clean_target_image, op="adjust_exposure", severity=1.0)

    assert isinstance(triplet, SyntheticTriplet)
    assert triplet.degradation_op == "adjust_exposure"
    assert triplet.recovery_ratio is not None
    assert triplet.recovery_ratio >= 0.90, (
        f"Exposure oracle recovery failed DoD: {triplet.recovery_ratio:.2%} (expected >= 90%)"
    )


def test_contrast_oracle_recovery_dod(clean_target_image: Image.Image) -> None:
    """DoD: Contrast oracle inverse achieves >= 90% degradation recovery."""
    engine = SyntheticCounterfactualEngine(seed=42)
    triplet = engine.generate_triplet(clean_target_image, op="adjust_contrast", severity=1.0)

    assert triplet.recovery_ratio is not None
    assert triplet.recovery_ratio >= 0.90, (
        f"Contrast oracle recovery failed DoD: {triplet.recovery_ratio:.2%} (expected >= 90%)"
    )


def test_saturation_oracle_recovery_dod(clean_target_image: Image.Image) -> None:
    """DoD: Saturation oracle inverse achieves >= 90% degradation recovery."""
    engine = SyntheticCounterfactualEngine(seed=42)
    triplet = engine.generate_triplet(clean_target_image, op="adjust_saturation", severity=1.0)

    assert triplet.recovery_ratio is not None
    assert triplet.recovery_ratio >= 0.90, (
        f"Saturation oracle recovery failed DoD: {triplet.recovery_ratio:.2%} (expected >= 90%)"
    )


def test_white_balance_oracle_recovery_dod(clean_target_image: Image.Image) -> None:
    """DoD: White balance temperature/tint oracle inverse achieves >= 90% recovery."""
    engine = SyntheticCounterfactualEngine(seed=42)
    triplet = engine.generate_triplet(clean_target_image, op="adjust_white_balance", severity=1.0)

    assert triplet.recovery_ratio is not None
    assert triplet.recovery_ratio >= 0.90, (
        f"White balance oracle recovery failed DoD: {triplet.recovery_ratio:.2%} (expected >= 90%)"
    )


def test_horizon_oracle_recovery(clean_target_image: Image.Image) -> None:
    """Verifies horizon rotation and analytical leveling recovery."""
    engine = SyntheticCounterfactualEngine(seed=42)
    triplet = engine.generate_triplet(clean_target_image, op="correct_horizon", severity=0.8)

    assert triplet.recovery_ratio is not None
    assert triplet.recovery_ratio >= 0.50


def test_generate_batch_and_export_dataset(tmp_path: Path, clean_target_image: Image.Image) -> None:
    """Verifies batch generation and JSONL dataset export for SFT warm-start."""
    engine = SyntheticCounterfactualEngine(seed=123)
    batch = engine.generate_batch([clean_target_image], count=5)

    assert len(batch) == 5
    for trip in batch:
        assert trip.recovery_ratio is not None
        assert trip.recovery_ratio > 0.70

    # Export to disk
    jsonl_path = engine.export_dataset(batch, output_dir=tmp_path / "sft_data", prefix="test_sft")
    assert jsonl_path.exists()

    # Read back and verify valid JSON lines
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    first_record = json.loads(lines[0])
    assert "oracle_fix" in first_record
    assert "degradation_op" in first_record
    assert Path(first_record["degraded_image_path"]).exists()
    assert Path(first_record["target_image_path"]).exists()


def test_oracle_recovery_with_real_image() -> None:
    """Verifies that oracle inverse recovers >= 90% degradation on a real photograph."""
    p = Path("test.jpg")
    if not p.exists():
        pytest.skip("test.jpg not present in root directory")

    real_img = Image.open(p).convert("RGB")
    engine = SyntheticCounterfactualEngine(seed=42)
    triplet = engine.generate_triplet(real_img, op="adjust_exposure", severity=1.0)

    assert triplet.recovery_ratio is not None
    assert triplet.recovery_ratio >= 0.90
