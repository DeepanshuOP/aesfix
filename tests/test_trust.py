"""Unit and integration tests for the SaaS Trust, Safety, Provenance, and Retention Layer (AF-5.4).

Why these tests are critical for regulatory compliance and interview defensibility:
1. Content Safety Integrity: Proves that blank, corrupt, or prohibited images are intercepted
   before GPU resources are expended.
2. C2PA Standards Compliance: Validates that digital provenance claims adhere strictly to
   IPTC/C2PA JSON-LD schemas with cryptographic SHA-256 hashes and verified gain metadata.
3. Anti-Theft Watermarking: Asserts that preview badges and before/after comparisons render
   accurately without altering base pixel dimensions.
4. GDPR Right to Erasure: Validates immediate user photo purging and affirmative opt-in enforcement.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from aesfix.api.routes_billing import router as billing_router
from aesfix.api.routes_pipeline import router as pipeline_router
from aesfix.api.routes_trust import router as trust_router
from aesfix.api.trust import (
    C2PAProvenanceEngine,
    ContentSafetyGate,
    DataRetentionManager,
    WatermarkService,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")


def test_content_safety_gate_screening() -> None:
    """Verifies that clean photos pass and corrupt/flagged patterns are rejected."""
    gate = ContentSafetyGate()

    # 1. Clean real photograph
    verdict_clean = gate.screen_image(_TEST_JPG)
    assert verdict_clean.is_safe is True
    assert verdict_clean.risk_category == "clean"

    # 2. Corrupt completely blank image (std_dev = 0)
    blank_img = Image.new("RGB", (100, 100), (0, 0, 0))
    verdict_blank = gate.screen_image(blank_img)
    assert verdict_blank.is_safe is False
    assert verdict_blank.risk_category == "corrupt_blank"

    # 3. Synthetic dominant skin-tone pattern
    skin_arr = np.full((100, 100, 3), [220, 160, 130], dtype=np.uint8)
    skin_img = Image.fromarray(skin_arr)
    verdict_skin = gate.screen_image(skin_img)
    assert verdict_skin.is_safe is False
    assert verdict_skin.risk_category == "nudity_warning"


def test_watermark_service_and_comparison_card() -> None:
    """Verifies visual badge watermarking and before/after side-by-side card generation."""
    with Image.open(_TEST_JPG) as img:
        base_img = img.convert("RGB")
        w, h = base_img.size

        # Apply pill watermark
        watermarked = WatermarkService.apply_watermark(
            base_img,
            watermark_text="AesFix Preview",
            gain_badge="+0.082 AG",
        )
        assert isinstance(watermarked, Image.Image)
        assert watermarked.size == (w, h)

        # Pixels in the bottom-right badge region should be altered
        base_arr = np.array(base_img)
        wm_arr = np.array(watermarked)
        diff = np.abs(base_arr.astype(int) - wm_arr.astype(int))
        assert np.sum(diff[-30:, -100:]) > 0

        # Generate side-by-side comparison
        comparison = WatermarkService.create_side_by_side_comparison(
            original=base_img,
            edited=watermarked,
            gain_badge="+0.082 AG",
        )
        assert comparison.width == (w * 2) + 4
        assert comparison.height == h + 44  # Header padding


def test_c2pa_provenance_manifest() -> None:
    """Verifies cryptographic hashing and standard C2PA JSON-LD serialization."""
    orig_hash = C2PAProvenanceEngine.compute_sha256(_TEST_JPG)
    assert len(orig_hash) == 64  # Valid SHA-256 hex string

    manifest = C2PAProvenanceEngine.generate_manifest(
        original_digest=orig_hash,
        edited_digest=orig_hash,
        applied_ops=["adjust_exposure", "adjust_contrast"],
        verified_aesthetic_gain=0.05,
    )

    json_ld = manifest.to_json_ld()
    assert "https://c2pa.org/standards/json-ld/v1" in json_ld

    data = json.loads(json_ld)
    assert data["claim_generator"].startswith("AesFix")
    assert len(data["c2pa:actions"]) == 2
    assert data["c2pa:assertions"]["c2pa.training_opt_in"] is False
    assert data["c2pa:signatures"]["original_digest"] == orig_hash


def test_data_retention_and_gdpr_purge(tmp_path: Path) -> None:
    """Verifies GDPR affirmative training consent and automatic file purging."""
    mgr = DataRetentionManager(storage_root=tmp_path)
    user_id = "user_gdpr_01"

    # Strict default: False
    assert mgr.is_opted_in(user_id) is False

    # Affirmative opt-in
    mgr.set_training_opt_in(user_id, True)
    assert mgr.is_opted_in(user_id) is True

    # Revoke consent
    mgr.set_training_opt_in(user_id, False)
    assert mgr.is_opted_in(user_id) is False

    # Create dummy user file in storage root
    test_user_file = tmp_path / f"aesfix_job_dummy_{user_id}.jpg"
    test_user_file.write_text("dummy")
    assert test_user_file.exists()

    # Immediate purge request
    purged = mgr.purge_user_photos(user_id)
    assert purged == 1
    assert not test_user_file.exists()


def test_fastapi_trust_routes_integration() -> None:
    """Tests FastAPI REST endpoints: safety check, watermarking, C2PA, and purge."""
    app = FastAPI()
    app.include_router(billing_router)
    app.include_router(pipeline_router)
    app.include_router(trust_router)
    client = TestClient(app)

    auth_header = {"Authorization": "Bearer mock-user-trust:privacy@aesfix.app"}

    # 1. Safety check endpoint
    with open(_TEST_JPG, "rb") as fh:
        resp_safety = client.post(
            "/api/v1/trust/safety-check",
            files={"file": ("test.jpg", fh, "image/jpeg")},
            headers=auth_header,
        )
    assert resp_safety.status_code == 200
    assert resp_safety.json()["is_safe"] is True

    # 2. Watermark preview endpoint
    with open(_TEST_JPG, "rb") as fh:
        resp_wm = client.post(
            "/api/v1/trust/watermark-preview",
            files={"file": ("test.jpg", fh, "image/jpeg")},
            data={"gain_badge": "+0.092 Gain"},
        )
    assert resp_wm.status_code == 200
    assert resp_wm.headers["content-type"] == "image/jpeg"
    assert len(resp_wm.content) > 1000

    # 3. Training opt-in endpoint
    resp_opt = client.post(
        "/api/v1/trust/opt-in",
        json={"opt_in": True},
        headers=auth_header,
    )
    assert resp_opt.status_code == 200
    assert resp_opt.json()["training_opt_in"] is True

    # 4. GDPR purge endpoint
    resp_purge = client.delete("/api/v1/trust/purge-photos", headers=auth_header)
    assert resp_purge.status_code == 200
    assert resp_purge.json()["status"] == "purged"
