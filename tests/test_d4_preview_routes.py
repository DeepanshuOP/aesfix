"""Unit and integration tests for Preview Endpoints and API Hygiene (D4).

Why these tests verify operational integrity:
1. Real Preview Delivery: Ensures /jobs/{job_id}/preview/{fix_index} and /jobs/{job_id}/preview/original
   return valid PNG byte streams that can be rendered directly by browser <img> elements.
2. 404 & 400 Boundary Handling: Verifies that requesting non-existent jobs or invalid fix indices
   returns strict HTTP status codes instead of unhandled 500 crashes.
3. Metadata Hygiene: Asserts that root metadata endpoints contain no dead preprint links or
   nonexistent HuggingFace spaces.
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from aesfix.api.app import app
from aesfix.api.routes_pipeline import queue_manager

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")


def test_d4_preview_endpoints_and_metadata_hygiene() -> None:
    """Verifies preview streaming and cleaned API metadata."""
    client = TestClient(app)

    # 1. Root metadata hygiene
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    root_data = resp_root.json()
    assert root_data["status"] == "operational"
    assert "paper_preprint" not in root_data
    assert "agx_leaderboard" not in root_data
    assert root_data["docs"] == "/docs"

    # 2. Submit test job
    auth_header = {"Authorization": "Bearer mock-user-preview:photographer@aesfix.app"}
    resp_submit = client.post(
        "/api/v1/pipeline/submit-path",
        json={"image_path": _TEST_JPG, "goal": "portrait"},
        headers=auth_header,
    )
    assert resp_submit.status_code == 200
    job_id = resp_submit.json()["job_id"]

    # Execute job synchronously to generate in-memory previews
    queue_manager.execute_job_sync(job_id)

    # 3. Test original preview endpoint
    resp_orig = client.get(f"/api/v1/pipeline/jobs/{job_id}/preview/original")
    assert resp_orig.status_code == 200
    assert resp_orig.headers["content-type"] == "image/png"
    img_orig = Image.open(io.BytesIO(resp_orig.content))
    assert img_orig.format == "PNG"
    assert img_orig.size[0] > 0 and img_orig.size[1] > 0

    # 4. Test fix preview by 0-based and 1-based index
    for idx_str in ("0", "1"):
        resp_fix = client.get(f"/api/v1/pipeline/jobs/{job_id}/preview/{idx_str}")
        assert resp_fix.status_code == 200
        assert resp_fix.headers["content-type"] == "image/png"
        img_fix = Image.open(io.BytesIO(resp_fix.content))
        assert img_fix.format == "PNG"

    # 5. Test invalid index handling
    resp_404 = client.get(f"/api/v1/pipeline/jobs/{job_id}/preview/99")
    assert resp_404.status_code == 404

    resp_400 = client.get(f"/api/v1/pipeline/jobs/{job_id}/preview/invalid_idx")
    assert resp_400.status_code == 400

    resp_nonexistent_job = client.get("/api/v1/pipeline/jobs/job_nonexistent_999/preview/0")
    assert resp_nonexistent_job.status_code == 404
