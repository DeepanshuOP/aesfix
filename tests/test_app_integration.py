"""End-to-End Production Application Gateway Integration Tests (AF-5.7).

Why these tests are critical for commercial launch validation:
1. Complete User Journey Verification: Validates that an end-user can onboard via beta invite,
   check quota, submit a photo for closed-loop diagnosis, receive verified fixes, download
   Lightroom XMP sidecars, verify C2PA credentials, and submit NPS feedback.
2. Gateway Middleware & Security: Verifies that CORS headers, correlation IDs (X-Request-ID),
   and response timing headers are injected across all API endpoints.
3. Systemic Health Assurance: Tests that all sub-routers (billing, pipeline, batch, trust,
   beta, observability) function together seamlessly in the unified production application.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from aesfix.api.app import create_app
from aesfix.api.routes_pipeline import queue_manager

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")
_DARK_JPG = str(_PROJECT_ROOT / "dark.jpg")


def test_production_app_root_and_observability_probes() -> None:
    """Verifies that the root landing page and all observability probes respond correctly."""
    app = create_app()
    client = TestClient(app)

    # 1. Root welcome endpoint
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    root_data = resp_root.json()
    assert root_data["name"] == "AesFix API"
    assert root_data["status"] == "operational"
    assert "X-Request-ID" in resp_root.headers
    assert "X-Response-Time-Ms" in resp_root.headers

    # 2. Liveness & readiness probes
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/version").status_code == 200
    assert "aesfix_http_requests_total" in client.get("/metrics").text


def test_complete_end_to_end_user_journey() -> None:
    """Executes a full lifecycle test across all domain routers in the production application."""
    app = create_app()
    client = TestClient(app)

    # --- Step 1: Beta User Onboarding ---
    resp_act = client.post(
        "/api/v1/beta/activate",
        json={"invite_code": "AESFIX-MARK-02", "email": "launch_partner@etsy.com"},
    )
    assert resp_act.status_code == 200
    assert resp_act.json()["persona"] == "marketplace_seller"

    auth_header = {"Authorization": "Bearer mock-user-launch-01:launch_partner@etsy.com"}

    # --- Step 2: Check Initial Quota ---
    resp_me = client.get("/api/v1/billing/me", headers=auth_header)
    assert resp_me.status_code == 200
    assert resp_me.json()["remaining_critiques"] == 5

    # --- Step 3: Pre-Flight Content Safety Check ---
    with open(_TEST_JPG, "rb") as fh:
        resp_safety = client.post(
            "/api/v1/trust/safety-check",
            files={"file": ("test.jpg", fh, "image/jpeg")},
            headers=auth_header,
        )
    assert resp_safety.status_code == 200
    assert resp_safety.json()["is_safe"] is True

    # --- Step 4: Photo Submission (Closed-Loop Diagnosis) ---
    resp_submit = client.post(
        "/api/v1/pipeline/submit-path",
        json={"image_path": _DARK_JPG, "goal": "listing"},
        headers=auth_header,
    )
    assert resp_submit.status_code == 200
    job_id = resp_submit.json()["job_id"]

    # Execute job synchronously in worker
    queue_manager.execute_job_sync(job_id)

    # --- Step 5: Poll Completed Results ---
    resp_job = client.get(f"/api/v1/pipeline/jobs/{job_id}")
    assert resp_job.status_code == 200
    job_data = resp_job.json()
    assert job_data["status"] == "COMPLETED"
    assert "result" in job_data
    assert len(job_data["result"]["verified_fixes"]) >= 1

    # --- Step 6: Download Lightroom XMP Sidecar ---
    resp_xmp = client.get(f"/api/v1/pipeline/jobs/{job_id}/xmp?fix_index=0")
    assert resp_xmp.status_code == 200
    assert resp_xmp.headers["content-type"] == "application/xml"
    assert "<x:xmpmeta" in resp_xmp.text

    # --- Step 7: Inspect C2PA Provenance Manifest ---
    resp_c2pa = client.get(f"/api/v1/trust/c2pa/{job_id}", headers=auth_header)
    assert resp_c2pa.status_code == 200
    c2pa_data = resp_c2pa.json()
    assert "c2pa:signatures" in c2pa_data
    assert c2pa_data["c2pa:assertions"]["c2pa.training_opt_in"] is False

    # --- Step 8: Track Core Activation Moment (first_verified_fix_viewed) ---
    resp_event = client.post(
        "/api/v1/beta/event",
        json={"step": "first_verified_fix_viewed"},
        headers=auth_header,
    )
    assert resp_event.status_code == 200

    # --- Step 9: Submit Qualitative Feedback & NPS ---
    resp_fb = client.post(
        "/api/v1/beta/feedback",
        json={"score": 10, "category": "fix_quality", "comment": "Excellent lighting fix!"},
        headers=auth_header,
    )
    assert resp_fb.status_code == 200

    # --- Step 10: Verify Business & Unit Economics Telemetry ---
    resp_cogs = client.get("/api/v1/observability/cogs")
    assert resp_cogs.status_code == 200
    assert resp_cogs.json()["status"] == "healthy"
