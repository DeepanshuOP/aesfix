"""Unit and integration tests for Goal Conditioning and Shoot-Level Batch Diagnosis (AF-5.3).

Why these tests are critical for SaaS robustness and interview defensibility:
1. Mathematical Weight Invariants: Asserts that all Goal Mode aspect weighting profiles sum
   strictly to 1.0 and accurately encode vertical-specific priorities (e.g. listing lighting).
2. Tier Boundary Enforcement: Verifies that Free tier accounts cannot exceed a batch size of 1,
   and that batches requiring more credits than available are rejected before execution.
3. Shoot Aggregator Precision: Validates that BatchDiagnosisAggregator accurately flags substandard
   frames, detects systemic defect trends, and crowns the optimal hero shot.
4. REST Integration: Tests FastAPI endpoints for goal discovery, batch submission, and report polling.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aesfix.api.batch_goal import (
    GOAL_PROFILES,
    BatchDiagnosisAggregator,
    BatchQueueManager,
    GoalMode,
)
from aesfix.api.billing import QuotaManager, TierLevel
from aesfix.api.pipeline_queue import JobResult, PipelineQueueManager, VerifiedFixPreview
from aesfix.api.routes_batch import router as batch_router
from aesfix.api.routes_billing import router as billing_router
from aesfix.api.routes_pipeline import queue_manager, router as pipeline_router

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")
_DARK_JPG = str(_PROJECT_ROOT / "dark.jpg")


def test_goal_profiles_weights_and_vocabularies() -> None:
    """Verifies that all goal profiles have normalized weights and domain-appropriate ops."""
    assert len(GOAL_PROFILES) == 5

    for mode, profile in GOAL_PROFILES.items():
        assert profile.mode == mode
        # Weight normalization check: sum must be approximately 1.0
        total_weight = sum(profile.aspect_weights.values())
        assert abs(total_weight - 1.0) < 1e-4, f"Weights for {mode} sum to {total_weight}"

        # Domain specialization checks
        if mode == GoalMode.LISTING:
            # E-Commerce listing must heavily weight lighting
            assert profile.aspect_weights["lighting"] >= 0.30
            assert "adjust_white_balance" in profile.preferred_ops
        elif mode == GoalMode.THUMBNAIL:
            # Thumbnail must prioritize color and contrast
            assert profile.aspect_weights["color"] >= 0.25
            assert "adjust_saturation" in profile.preferred_ops


def test_batch_queue_tier_limits_and_credit_validation() -> None:
    """Verifies batch size gating by subscription tier and credit availability."""
    qm = QuotaManager()
    pq = PipelineQueueManager(quota_mgr=qm)
    bq = BatchQueueManager(pipeline_qm=pq, quota_mgr=qm)
    user_id = "test_user_batch_limits"

    # 1. FREE tier user attempts batch of 3 (Free tier limit is 1)
    user = qm.get_or_create_user(user_id)
    assert user.tier == TierLevel.FREE

    with pytest.raises(PermissionError, match="exceeds FREE tier limit of 1"):
        bq.submit_batch(user_id=user_id, image_inputs=[_TEST_JPG, _DARK_JPG, _TEST_JPG])

    # 2. Upgrade user to PRO (limit is 10), but exhaust credits down to 1
    user.tier = TierLevel.PRO
    user.critiques_used_this_month = 299  # 1 credit left
    assert qm.get_remaining_critiques(user_id) == 1

    with pytest.raises(PermissionError, match="Insufficient credits"):
        bq.submit_batch(user_id=user_id, image_inputs=[_TEST_JPG, _DARK_JPG])

    # 3. Valid batch with sufficient credits
    user.critiques_used_this_month = 0
    assert qm.get_remaining_critiques(user_id) == 300

    batch = bq.submit_batch(
        user_id=user_id,
        image_inputs=[_TEST_JPG, _DARK_JPG],
        goal=GoalMode.LISTING,
    )
    assert len(batch.job_ids) == 2
    assert qm.get_remaining_critiques(user_id) == 298


def test_batch_diagnosis_aggregator_metrics_and_hero_shot() -> None:
    """Verifies statistical aggregation, weakest aspect identification, and hero shot crown."""
    # Build 3 mock JobResults representing a real batch
    fix_exposure = VerifiedFixPreview(
        fix_id="f1", op="adjust_exposure", params={}, expected_aspect="lighting",
        reasoning="", baseline_score=0.50, post_edit_score=0.58, aesthetic_gain=0.08,
        guard_passed=True, guard_drift=0.05, gain_badge="+0.080 Verified",
        xmp_xml="<xmp/>", execution_time_ms=10.0,
    )
    fix_contrast = VerifiedFixPreview(
        fix_id="f2", op="adjust_contrast", params={}, expected_aspect="lighting",
        reasoning="", baseline_score=0.50, post_edit_score=0.54, aesthetic_gain=0.04,
        guard_passed=True, guard_drift=0.02, gain_badge="+0.040 Verified",
        xmp_xml="<xmp/>", execution_time_ms=8.0,
    )

    r1 = JobResult(
        job_id="job_hero_01",
        overall_score=0.78,
        grade_label="EXCELLENT (A)",
        aspect_scores={"composition": 0.80, "lighting": 0.75, "color": 0.80, "subject_clarity": 0.78, "technical": 0.77},
        anfis_rulebook_summary="High quality",
        top_fired_rules=["Rule 1"],
        critique_rationale={"lighting": "Well lit"},
        verified_fixes=[fix_contrast],
        best_fix=fix_contrast,
        uncertainty_badge="Consensus",
        total_latency_seconds=1.5,
    )
    r2 = JobResult(
        job_id="job_dim_02",
        overall_score=0.52,
        grade_label="MEDIOCRE (C)",
        aspect_scores={"composition": 0.70, "lighting": 0.40, "color": 0.65, "subject_clarity": 0.60, "technical": 0.55},
        anfis_rulebook_summary="Dim lighting",
        top_fired_rules=["Rule 2"],
        critique_rationale={"lighting": "Under-exposed"},
        verified_fixes=[fix_exposure],
        best_fix=fix_exposure,
        uncertainty_badge="Consensus",
        total_latency_seconds=1.6,
    )
    r3 = JobResult(
        job_id="job_dim_03",
        overall_score=0.48,
        grade_label="NEEDS FIX (D)",
        aspect_scores={"composition": 0.65, "lighting": 0.35, "color": 0.60, "subject_clarity": 0.58, "technical": 0.52},
        anfis_rulebook_summary="Harsh shadow",
        top_fired_rules=["Rule 3"],
        critique_rationale={"lighting": "Severe shadows"},
        verified_fixes=[fix_exposure],
        best_fix=fix_exposure,
        uncertainty_badge="Consensus",
        total_latency_seconds=1.4,
    )

    report = BatchDiagnosisAggregator.compile_report(
        batch_id="batch_mock_123",
        goal=GoalMode.LISTING,
        results=[r1, r2, r3],
    )

    assert report.total_photos == 3
    assert report.hero_shot_job_id == "job_hero_01"
    assert report.hero_shot_score == 0.78
    assert report.weakest_aspect == "lighting"
    # In listing mode, threshold is 0.65. r2 (0.52) and r3 (0.48) should be flagged
    assert len(report.photos_below_threshold) == 2
    assert "job_dim_02" in report.photos_below_threshold
    assert "job_dim_03" in report.photos_below_threshold

    # Defect distribution check: adjust_exposure affected 2 photos (66.7%)
    exposure_defect = next(d for d in report.defect_distribution if d.op_name == "adjust_exposure")
    assert exposure_defect.affected_photo_count == 2
    assert exposure_defect.percentage == 66.7


def test_fastapi_batch_routes_integration() -> None:
    """Tests FastAPI batch endpoints: /goals, /submit, and status polling."""
    app = FastAPI()
    app.include_router(billing_router)
    app.include_router(pipeline_router)
    app.include_router(batch_router)
    client = TestClient(app)

    auth_header = {"Authorization": "Bearer mock-user-batch-01:seller@etsy.com"}

    # 1. Discover goals
    resp_goals = client.get("/api/v1/batch/goals")
    assert resp_goals.status_code == 200
    goals = resp_goals.json()
    assert len(goals) == 5
    goal_modes = [g["mode"] for g in goals]
    assert "listing" in goal_modes
    assert "portrait" in goal_modes

    # 2. Upgrade user to PRO to allow batch submission of size > 1
    user_account = client.get("/api/v1/billing/me", headers=auth_header).json()
    uid = user_account["user_id"]
    from aesfix.api.routes_billing import quota_manager
    user = quota_manager.get_or_create_user(uid)
    user.tier = TierLevel.PRO

    # 3. Submit batch
    resp_submit = client.post(
        "/api/v1/batch/submit",
        json={"image_paths": [_TEST_JPG, _DARK_JPG], "goal": "listing"},
        headers=auth_header,
    )
    assert resp_submit.status_code == 200
    batch_data = resp_submit.json()
    batch_id = batch_data["batch_id"]
    assert batch_data["total_photos"] == 2
    assert len(batch_data["job_ids"]) == 2

    # 4. Synchronously execute constituent jobs so batch report compiles
    for jid in batch_data["job_ids"]:
        queue_manager.execute_job_sync(jid)

    # 5. Poll batch report
    resp_poll = client.get(f"/api/v1/batch/{batch_id}")
    assert resp_poll.status_code == 200
    poll_data = resp_poll.json()
    assert poll_data["status"] == "COMPLETED"
    assert poll_data["completed_photos"] == 2
    assert poll_data["report"] is not None
    assert poll_data["report"]["hero_shot_job_id"] is not None
    assert "Shoot Analysis for 2 photos" in poll_data["report"]["shoot_summary"]
