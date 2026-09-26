"""Unit and integration tests for Beta Cohort Management, Funnel Analytics, and NPS (AF-5.6).

Why these tests are critical for product analytics and interview defensibility:
1. Cohort Invariants: Validates that the 20-user beta cohort is partitioned correctly across
   Marketplace Sellers (P1), Enthusiasts (P2), and Dating Profile Optimizers (P4).
2. Activation Math Integrity: Asserts that funnel drop-offs and the core activation metric
   ('first_verified_fix_viewed') are accurately computed.
3. NPS Precision: Verifies standard Net Promoter Score calculation (promoters vs detractors).
4. REST Contract: Tests FastAPI endpoints for invite redemption, event streaming, and feedback.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aesfix.api.beta import (
    BetaCohortManager,
    BetaPersona,
    FunnelStep,
)
from aesfix.api.routes_beta import router as beta_router
from aesfix.api.routes_billing import router as billing_router
from aesfix.api.routes_pipeline import router as pipeline_router


def test_beta_cohort_initial_seed() -> None:
    """Verifies that 20 beta seats are pre-provisioned across the 3 target personas."""
    mgr = BetaCohortManager()
    summary = mgr.get_funnel_summary()

    assert summary.total_invited == 20
    assert summary.total_activated == 0
    assert summary.activation_rate_pct == 0.0

    # Persona partition count check
    sellers = [u for u in mgr._users.values() if u.persona == BetaPersona.MARKETPLACE_SELLER]
    enthusiasts = [u for u in mgr._users.values() if u.persona == BetaPersona.PHOTO_ENTHUSIAST]
    dating = [u for u in mgr._users.values() if u.persona == BetaPersona.DATING_OPTIMIZER]

    assert len(sellers) == 8
    assert len(enthusiasts) == 7
    assert len(dating) == 5


def test_beta_user_activation_and_milestones() -> None:
    """Verifies invite redemption and funnel milestone recording."""
    mgr = BetaCohortManager()
    user_id = "beta_user_01"
    code = mgr._users[user_id].invite_code

    # 1. Activate user with valid code
    user = mgr.activate_user(code, email="artisan@etsy.com")
    assert user.is_activated is True
    assert user.email == "artisan@etsy.com"
    assert FunnelStep.ONBOARDED.value in user.completed_funnel_steps

    # 2. Invalid code raises KeyError
    with pytest.raises(KeyError, match="Invalid or expired"):
        mgr.activate_user("INVALID-CODE-99")

    # 3. Advance along the activation funnel
    assert mgr.record_step(user_id, FunnelStep.PHOTO_UPLOADED) is True
    assert mgr.record_step(user_id, FunnelStep.CRITIQUE_VIEWED) is True
    assert mgr.record_step(user_id, FunnelStep.FIRST_VERIFIED_FIX_VIEWED) is True
    assert mgr.record_step(user_id, FunnelStep.XMP_DOWNLOADED) is True

    # 4. Unknown user returns False
    assert mgr.record_step("non_existent_user", FunnelStep.PHOTO_UPLOADED) is False


def test_funnel_analytics_and_nps_computation() -> None:
    """Verifies cohort conversion percentages and Net Promoter Score calculation."""
    mgr = BetaCohortManager()

    # Activate 4 users
    u1 = mgr.activate_user(mgr._users["beta_user_01"].invite_code)
    u2 = mgr.activate_user(mgr._users["beta_user_02"].invite_code)
    u3 = mgr.activate_user(mgr._users["beta_user_03"].invite_code)
    u4 = mgr.activate_user(mgr._users["beta_user_04"].invite_code)

    # 3 of 4 reach the aha moment (first_verified_fix_viewed) -> 75%
    for u in (u1, u2, u3):
        mgr.record_step(u.user_id, FunnelStep.FIRST_VERIFIED_FIX_VIEWED)

    # 2 of 4 reach power user status (xmp_downloaded) -> 50%
    for u in (u1, u2):
        mgr.record_step(u.user_id, FunnelStep.XMP_DOWNLOADED)

    # Ingest NPS ratings: 2 promoters (10, 9), 1 passive (8), 1 detractor (5)
    mgr.submit_feedback(u1.user_id, score=10, category="fix_quality", comment="Loved the lighting fix!")
    mgr.submit_feedback(u2.user_id, score=9, category="usability", comment="Very intuitive.")
    mgr.submit_feedback(u3.user_id, score=8, category="pricing", comment="Fair price.")
    mgr.submit_feedback(u4.user_id, score=5, category="critique_accuracy", comment="A bit too conservative.")

    summary = mgr.get_funnel_summary()
    assert summary.total_activated == 4
    assert summary.activation_rate_pct == 20.0  # 4 / 20 = 20%
    assert summary.aha_moment_conversion_pct == 75.0
    assert summary.power_user_conversion_pct == 50.0

    # NPS = (2 - 1) / 4 = 25.0
    assert summary.nps_score == 25.0
    assert summary.nps_breakdown["promoters"] == 2
    assert summary.nps_breakdown["detractors"] == 1


def test_fastapi_beta_routes_integration() -> None:
    """Tests FastAPI REST endpoints: activate, event tracking, feedback, and funnel."""
    app = FastAPI()
    app.include_router(billing_router)
    app.include_router(pipeline_router)
    app.include_router(beta_router)
    client = TestClient(app)

    # 1. Redeem beta invite
    resp_act = client.post(
        "/api/v1/beta/activate",
        json={"invite_code": "AESFIX-MARK-01", "email": "beta_seller@etsy.com"},
    )
    assert resp_act.status_code == 200
    assert resp_act.json()["status"] == "activated"
    assert resp_act.json()["persona"] == "marketplace_seller"

    # 2. Track milestone event with auth header
    auth_header = {"Authorization": "Bearer mock-user-beta-01:beta_seller@etsy.com"}
    resp_evt = client.post(
        "/api/v1/beta/event",
        json={"step": "first_verified_fix_viewed"},
        headers=auth_header,
    )
    assert resp_evt.status_code == 200
    assert resp_evt.json()["step"] == "first_verified_fix_viewed"

    # 3. Submit feedback
    resp_fb = client.post(
        "/api/v1/beta/feedback",
        json={
            "score": 10,
            "category": "fix_quality",
            "comment": "Saved my product shoot!",
        },
        headers=auth_header,
    )
    assert resp_fb.status_code == 200
    assert resp_fb.json()["status"] == "received"

    # 4. View funnel analytics
    resp_funnel = client.get("/api/v1/beta/funnel")
    assert resp_funnel.status_code == 200
    data = resp_funnel.json()
    assert data["total_invited"] == 20
    assert data["total_activated"] >= 1
