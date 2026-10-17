"""Unit and integration tests for the Asynchronous Pipeline Queue and SSE Streamer (AF-5.2).

Why these tests are critical for SaaS operational integrity:
1. Closed-Loop Verification: Asserts that every queued job produces verified fixes with
   measured Aesthetic Gain (AG) badges and valid Adobe Lightroom XMP sidecars.
2. Quota Protection: Verifies that submitting a job atomically consumes user critique credits
   and rejects requests with HTTP 402 when monthly quotas are exhausted.
3. Streaming Contract: Validates that Server-Sent Events (SSE) emit valid text/event-stream
   frames across all pipeline stages (critiquing -> rulebook -> executing_fixes -> completed).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aesfix.api.billing import QuotaManager
from aesfix.api.pipeline_queue import JobStatus, PipelineQueueManager
from aesfix.api.routes_billing import router as billing_router
from aesfix.api.routes_pipeline import queue_manager, router as pipeline_router

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")
_DARK_JPG = str(_PROJECT_ROOT / "dark.jpg")


def test_queue_submission_and_quota_deduction() -> None:
    """Verifies that job submission reserves quotas and blocks when exhausted."""
    qm = QuotaManager()
    pq = PipelineQueueManager(quota_mgr=qm)
    user_id = "test_user_queue_quota"

    # User starts with 5 credits
    assert qm.get_remaining_critiques(user_id) == 5

    # Submit 1 job
    job = pq.submit_job(user_id=user_id, image_input=_TEST_JPG, goal="portrait")
    assert job.status == JobStatus.QUEUED
    assert qm.get_remaining_critiques(user_id) == 4
    assert len(job.events) == 1
    assert job.events[0].stage == "queued"

    # Consume remaining 4 credits
    for _ in range(4):
        pq.submit_job(user_id=user_id, image_input=_TEST_JPG)

    assert qm.get_remaining_critiques(user_id) == 0

    # 6th attempt should raise PermissionError
    with pytest.raises(PermissionError, match="Quota check failed"):
        pq.submit_job(user_id=user_id, image_input=_TEST_JPG)


def test_execute_job_sync_closed_loop_f1_outputs() -> None:
    """Verifies complete closed-loop execution and generation of F1.1–F1.3 artifacts."""
    qm = QuotaManager()
    pq = PipelineQueueManager(quota_mgr=qm)
    user_id = "test_user_exec"

    job = pq.submit_job(
        user_id=user_id,
        image_input=_DARK_JPG,
        user_critique="Check if photo is underexposed",
        goal="listing",
    )

    # Execute pipeline synchronously
    result = pq.execute_job_sync(job.job_id)

    # 1. Job state transitions
    assert job.status == JobStatus.COMPLETED
    assert job.error is None
    assert result.job_id == job.job_id

    # 2. F1.1 Aspect scores & Honest ANFIS/Judge state
    assert result.overall_score is None  # Untrained ANFIS grade removed from API for honesty
    assert result.grade_label is None
    assert len(result.aspect_scores) == 5
    assert result.noise_floor >= 0.0
    assert result.judge_id != ""
    assert "Single judge" in result.uncertainty_badge or "Judge Consensus" in result.uncertainty_badge
    assert result.image_verdict != ""

    # 3. F1.2 Critique card rationales
    assert len(result.critique_rationale) >= 1

    # 4. F1.3 Verified fix previews
    assert len(result.verified_fixes) >= 1
    for fix in result.verified_fixes:
        assert fix.fix_id.startswith(job.job_id)
        assert fix.op != ""
        assert fix.verdict in ("measurable improvement", "no measurable improvement", "rejected by guard")
        assert fix.judge_id == result.judge_id
        assert fix.epsilon == result.noise_floor
        assert fix.delta == round(fix.post_edit_score - fix.baseline_score, 4)
        assert "<x:xmpmeta" in fix.xmp_xml
        assert "crs:Exposure2012" in fix.xmp_xml or "crs:" in fix.xmp_xml

    # 5. Best fix ranking or honest leave-it verdict
    if result.best_fix is not None:
        assert result.best_fix.verdict == "measurable improvement"
        assert result.best_fix.aesthetic_gain >= result.verified_fixes[-1].aesthetic_gain
    else:
        assert result.image_verdict == "Leave it — no measurable improvement found"

    # 6. In-memory preview images stored
    assert job.original_image is not None
    assert len(job.preview_images) >= 1


@pytest.mark.asyncio
async def test_stream_job_events_sse() -> None:
    """Verifies that stream_job_events yields valid SSE frames with progression."""
    qm = QuotaManager()
    pq = PipelineQueueManager(quota_mgr=qm)
    user_id = "test_sse_user"

    job = pq.submit_job(user_id=user_id, image_input=_TEST_JPG)

    # Execute pipeline in worker
    pq.execute_job_sync(job.job_id)

    # Stream emitted events
    frames: list[str] = []
    async for frame in pq.stream_job_events(job.job_id):
        frames.append(frame)

    assert len(frames) >= 4  # queued, critiquing, rulebook, executing_fixes, completed
    for frame in frames:
        assert frame.startswith("data: {")
        assert frame.endswith("}\n\n")

    # Terminal event check
    last_frame = frames[-1]
    assert '"stage": "completed"' in last_frame
    assert '"progress_percent": 100' in last_frame


def test_fastapi_pipeline_routes_integration() -> None:
    """Tests FastAPI REST endpoints: submit-path, status polling, and XMP download."""
    app = FastAPI()
    app.include_router(billing_router)
    app.include_router(pipeline_router)
    client = TestClient(app)

    auth_header = {"Authorization": "Bearer mock-user-pipe-01:photographer@aesfix.app"}

    # 1. Submit image path
    resp_submit = client.post(
        "/api/v1/pipeline/submit-path",
        json={"image_path": _TEST_JPG, "goal": "portrait"},
        headers=auth_header,
    )
    assert resp_submit.status_code == 200
    data = resp_submit.json()
    job_id = data["job_id"]
    assert data["status"] == "QUEUED"
    assert f"/api/v1/pipeline/jobs/{job_id}/stream" in data["stream_url"]

    # 2. Synchronously run the submitted job so polling & xmp can be tested
    queue_manager.execute_job_sync(job_id)

    # 3. Poll job status
    resp_poll = client.get(f"/api/v1/pipeline/jobs/{job_id}")
    assert resp_poll.status_code == 200
    poll_data = resp_poll.json()
    assert poll_data["status"] == "COMPLETED"
    assert "result" in poll_data
    assert len(poll_data["result"]["verified_fixes"]) >= 1

    # 4. Download XMP sidecar
    resp_xmp = client.get(f"/api/v1/pipeline/jobs/{job_id}/xmp?fix_index=0")
    assert resp_xmp.status_code == 200
    assert resp_xmp.headers["content-type"] == "application/xml"
    assert "<x:xmpmeta" in resp_xmp.text
    assert "attachment;" in resp_xmp.headers["content-disposition"]


def test_fastapi_unauthorized_rejection() -> None:
    """Verifies that requests without valid credentials are strictly rejected with 401."""
    app = FastAPI()
    app.include_router(pipeline_router)
    client = TestClient(app)

    resp = client.post("/api/v1/pipeline/submit-path", json={"image_path": _TEST_JPG})
    assert resp.status_code == 401


def test_guard_failed_edit_with_gain_above_epsilon_shows_rejected_by_guard() -> None:
    """Verifies that an edit failing Identity Guard drift with raw Δ > ε emits 'rejected by guard'."""
    from unittest.mock import MagicMock
    from aesfix.executor.guard import GuardVerdict

    qm = QuotaManager()
    pq = PipelineQueueManager(quota_mgr=qm)

    # Mock IdentityGuard to reject with drift > threshold
    mock_guard = MagicMock()
    mock_guard.backend_name = "rgb_layout_stat"
    mock_guard.evaluate.return_value = GuardVerdict(
        is_valid=False,
        composite_drift=0.55,
        feature_drift=0.60,
        structural_drift=0.50,
        threshold=0.35,
        backend_name="rgb_layout_stat",
        rejection_reason="Perceptual drift exceeds threshold",
        guard_backend="SSIM + rgb_layout_stat",
    )
    pq.identity_guard = mock_guard

    job = pq.submit_job(user_id="test_user_guard_fail", image_input=_TEST_JPG)
    result = pq.execute_job_sync(job.job_id)

    assert len(result.verified_fixes) >= 1
    for fix in result.verified_fixes:
        assert fix.guard_passed is False
        # When guard fails, backend must explicitly return "rejected by guard" as source of truth
        assert fix.verdict == "rejected by guard"
        assert fix.aesthetic_gain == 0.0
        assert "Vetoed by Identity Guard" in fix.gain_badge

    # Best fix should be None since all fixes were vetoed by guard
    assert result.best_fix is None
    assert result.image_verdict == "Leave it — no measurable improvement found"
