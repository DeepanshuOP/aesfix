"""Unit tests for Tier-A Deterministic Image Executor and XMP Sidecar Engine (AF-2.1 DoD).

DoD Requirements:
1. 12 canonical operations execute and round-trip successfully.
2. XMP sidecar conforms to Adobe Lightroom Camera Raw standard specifications.
3. Fast execution latency (<200ms for full chain on CPU).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aesfix.critic.critic import CriticFix
from aesfix.executor import (
    TIER_A_OPERATIONS,
    TierAExecutor,
    TierAResult,
    XMPWriter,
)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Creates a deterministic synthetic RGB image for repeatable unit testing."""
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[:64, :64] = [200, 100, 50]
    arr[64:, 64:] = [50, 150, 220]
    arr[:64, 64:] = [128, 128, 128]
    arr[64:, :64] = [20, 20, 20]
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def twelve_sample_fixes() -> list[CriticFix]:
    """Generates canonical fix configurations for each of the 12 Tier-A operations."""
    return [
        CriticFix(
            op="adjust_exposure",
            params={"ev_delta": 0.5},
            expected_aspect="lighting",
            nl_instruction="Brighten underexposed image by +0.5 EV stops.",
        ),
        CriticFix(
            op="adjust_contrast",
            params={"contrast": 20.0},
            expected_aspect="lighting",
            nl_instruction="Increase midtone contrast by 20 points.",
        ),
        CriticFix(
            op="adjust_highlights",
            params={"highlights": -25.0},
            expected_aspect="lighting",
            nl_instruction="Recover overexposed highlights by -25.",
        ),
        CriticFix(
            op="adjust_shadows",
            params={"shadows": 30.0},
            expected_aspect="lighting",
            nl_instruction="Lift deep shadows by +30 points.",
        ),
        CriticFix(
            op="adjust_white_balance",
            params={"temperature": 15.0, "tint": -5.0},
            expected_aspect="color",
            nl_instruction="Warm the color temperature and adjust green/magenta tint.",
        ),
        CriticFix(
            op="adjust_saturation",
            params={"saturation": 15.0},
            expected_aspect="color",
            nl_instruction="Enhance overall color saturation by 15%.",
        ),
        CriticFix(
            op="adjust_vibrance",
            params={"vibrance": 25.0},
            expected_aspect="color",
            nl_instruction="Boost vibrance on muted tones.",
        ),
        CriticFix(
            op="adjust_clarity",
            params={"clarity": 15.0},
            expected_aspect="technical",
            nl_instruction="Improve local midtone clarity.",
        ),
        CriticFix(
            op="unsharp_mask",
            params={"amount": 40.0, "radius": 1.2},
            expected_aspect="technical",
            nl_instruction="Sharpen fine details with an unsharp mask.",
        ),
        CriticFix(
            op="denoise",
            params={"strength": 20.0},
            expected_aspect="technical",
            nl_instruction="Smooth luminance noise across flat areas.",
        ),
        CriticFix(
            op="smart_crop",
            params={"top": 0.05, "left": 0.05, "bottom": 0.95, "right": 0.95},
            expected_aspect="composition",
            nl_instruction="Recrop image to balanced rule-of-thirds composition.",
        ),
        CriticFix(
            op="correct_horizon",
            params={"angle": 2.0},
            expected_aspect="composition",
            nl_instruction="Level tilted horizon by 2 degrees clockwise.",
        ),
    ]


def test_tier_a_operations_count() -> None:
    """Verifies that exactly 12 canonical operations are cataloged for Tier-A."""
    assert len(TIER_A_OPERATIONS) == 12
    assert "adjust_exposure" in TIER_A_OPERATIONS
    assert "adjust_contrast" in TIER_A_OPERATIONS
    assert "correct_horizon" in TIER_A_OPERATIONS
    assert "smart_crop" in TIER_A_OPERATIONS


def test_individual_ops_execution(sample_pil_image: Image.Image, twelve_sample_fixes: list[CriticFix]) -> None:
    """Verifies that every single one of the 12 operations executes without errors or NaNs."""
    executor = TierAExecutor()

    for fix in twelve_sample_fixes:
        result = executor.execute_fixes(sample_pil_image, [fix])
        assert isinstance(result, TierAResult)
        assert len(result.applied_ops) == 1
        assert result.applied_ops[0] == fix.op
        assert isinstance(result.image, Image.Image)

        # Confirm tensor properties: valid shape, range [0, 1], and finite values
        assert result.tensor.ndim == 4
        assert not torch.isnan(result.tensor).any(), f"NaN detected in op {fix.op}"
        assert not torch.isinf(result.tensor).any(), f"Inf detected in op {fix.op}"
        assert torch.min(result.tensor) >= 0.0
        assert torch.max(result.tensor) <= 1.0


def test_full_pipeline_execution_and_latency(sample_pil_image: Image.Image, twelve_sample_fixes: list[CriticFix]) -> None:
    """Verifies that chaining all 12 operations completes in <200ms on CPU."""
    executor = TierAExecutor()

    result = executor.execute_fixes(sample_pil_image, twelve_sample_fixes)

    assert len(result.applied_ops) == 12
    assert result.duration_ms < 200.0, f"Execution latency too high: {result.duration_ms:.2f}ms (expected <200ms)"
    assert result.image.size[0] > 0
    assert result.image.size[1] > 0


def test_xmp_generation_and_lightroom_compliance(twelve_sample_fixes: list[CriticFix]) -> None:
    """Verifies that generated XMP matches Adobe Camera Raw specifications and parses as valid XML."""
    xmp_str = XMPWriter.generate_xmp(twelve_sample_fixes, image_filename="sample_photo.jpg")

    # Verify standard XML parsing
    root = ET.fromstring(xmp_str)
    assert root.tag.endswith("xmpmeta")

    # Verify presence of crs namespace and key Lightroom attributes
    rdf = root.find("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}RDF")
    assert rdf is not None
    desc = rdf.find("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert desc is not None

    crs_ns = "{http://ns.adobe.com/camera-raw-settings/1.0/}"
    assert desc.attrib[f"{crs_ns}Version"] == "16.0"
    assert desc.attrib[f"{crs_ns}Exposure2012"] == "+0.50"
    assert desc.attrib[f"{crs_ns}Contrast2012"] == "+20"
    assert desc.attrib[f"{crs_ns}Highlights2012"] == "-25"
    assert desc.attrib[f"{crs_ns}Shadows2012"] == "+30"
    assert desc.attrib[f"{crs_ns}Temperature"] == "+15"
    assert desc.attrib[f"{crs_ns}Tint"] == "-5"
    assert desc.attrib[f"{crs_ns}Saturation"] == "+15"
    assert desc.attrib[f"{crs_ns}Vibrance"] == "+25"
    assert desc.attrib[f"{crs_ns}Clarity2012"] == "+15"
    assert desc.attrib[f"{crs_ns}Sharpness"] == "40"
    assert desc.attrib[f"{crs_ns}LuminanceSmoothing"] == "20"
    assert desc.attrib[f"{crs_ns}HasCrop"] == "True"
    assert float(desc.attrib[f"{crs_ns}CropAngle"]) == pytest.approx(2.0, abs=1e-3)


def test_xmp_round_trip_reconstruction(twelve_sample_fixes: list[CriticFix]) -> None:
    """Verifies DoD: 12 ops round-trip from CriticFix -> XMP -> CriticFix with identical values."""
    xmp_str = XMPWriter.generate_xmp(twelve_sample_fixes)
    parsed_fixes = XMPWriter.parse_xmp(xmp_str)

    assert len(parsed_fixes) == 12

    # Map parsed fixes by operation
    parsed_dict = {f.op: f for f in parsed_fixes}

    # Exposure check
    assert "adjust_exposure" in parsed_dict
    assert pytest.approx(parsed_dict["adjust_exposure"].params["ev_delta"], abs=1e-2) == 0.5

    # Contrast check
    assert "adjust_contrast" in parsed_dict
    assert pytest.approx(parsed_dict["adjust_contrast"].params["contrast"], abs=1e-2) == 20.0

    # Highlights & Shadows
    assert pytest.approx(parsed_dict["adjust_highlights"].params["highlights"], abs=1e-2) == -25.0
    assert pytest.approx(parsed_dict["adjust_shadows"].params["shadows"], abs=1e-2) == 30.0

    # White balance
    assert pytest.approx(parsed_dict["adjust_white_balance"].params["temperature"], abs=1e-2) == 15.0
    assert pytest.approx(parsed_dict["adjust_white_balance"].params["tint"], abs=1e-2) == -5.0

    # Saturation & Vibrance
    assert pytest.approx(parsed_dict["adjust_saturation"].params["saturation"], abs=1e-2) == 15.0
    assert pytest.approx(parsed_dict["adjust_vibrance"].params["vibrance"], abs=1e-2) == 25.0

    # Clarity & Sharpness & Denoise
    assert pytest.approx(parsed_dict["adjust_clarity"].params["clarity"], abs=1e-2) == 15.0
    assert pytest.approx(parsed_dict["unsharp_mask"].params["amount"], abs=1e-2) == 40.0
    assert pytest.approx(parsed_dict["denoise"].params["strength"], abs=1e-2) == 20.0

    # Crop & Horizon
    assert pytest.approx(parsed_dict["smart_crop"].params["top"], abs=1e-2) == 0.05
    assert pytest.approx(parsed_dict["correct_horizon"].params["angle"], abs=1e-2) == 2.0


def test_export_xmp_file(tmp_path: Path, twelve_sample_fixes: list[CriticFix]) -> None:
    """Verifies that export_xmp writes a valid physical file to disk."""
    out_file = tmp_path / "subfolder" / "photo.xmp"
    saved_path = XMPWriter.export_xmp(twelve_sample_fixes, out_file)

    assert saved_path.exists()
    content = saved_path.read_text(encoding="utf-8")
    assert "<x:xmpmeta" in content
    assert "Exposure2012" in content


def test_dict_fixes_and_real_image_execution(tmp_path: Path) -> None:
    """Verifies that executor handles plain dictionaries and real JPEG images from disk."""
    real_img_path = Path("test.jpg") if Path("test.jpg").exists() else Path("data/test.jpg")
    if not real_img_path.exists():
        pytest.skip("Neither test.jpg nor data/test.jpg found")

    executor = TierAExecutor()
    raw_dict_fixes = [
        {"op": "adjust_exposure", "params": {"ev_delta": -0.3}},
        {"op": "adjust_saturation", "params": {"saturation": 10.0}},
    ]

    result = executor.execute_fixes(real_img_path, raw_dict_fixes)
    assert len(result.applied_ops) == 2
    assert result.image.size[0] > 0
    assert "<x:xmpmeta" in result.xmp_sidecar
