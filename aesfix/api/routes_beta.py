"""FastAPI Beta Cohort & Funnel Analytics Routes (AF-5.6 & Roadmap §42).

Why these endpoints are organized under /api/v1/beta:
1. Self-Serve Beta Activation: Allows invitees to redeem private codes (/activate) to unlock their accounts.
2. Lightweight Telemetry Ingestion: The frontend fires /event whenever users pass key milestones
   (upload, critique, viewing the verified fix, downloading Lightroom XMP).
3. Continuous User Voice: /feedback captures CSAT/NPS and qualitative insights from early users.
4. Funnel Health Visibility: /funnel delivers a consolidated dashboard of conversion rates and satisfaction.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from aesfix.api.beta import (
    BetaCohortManager,
    FunnelStep,
)
from aesfix.api.billing import AuthClaims
from aesfix.api.routes_billing import get_current_user

logger = logging.getLogger("aesfix.api.routes_beta")

router = APIRouter(prefix="/api/v1/beta", tags=["beta"])

# Shared beta cohort manager singleton
beta_cohort_manager = BetaCohortManager()


# --- Request & Response Models ---

class ActivateBetaRequest(BaseModel):
    """Payload to claim a private beta invitation."""
    invite_code: str = Field(description="Unique beta invitation code (e.g. AESFIX-MARK-01)")
    email: Optional[str] = Field(default=None, description="Optional user email override")


class BetaActivationResponse(BaseModel):
    """Result of beta invite redemption."""
    status: str
    user_id: str
    email: str
    persona: str
    message: str


class TrackEventRequest(BaseModel):
    """Milestone event fired by frontend."""
    step: FunnelStep = Field(description="Completed milestone in the user journey")


class SubmitFeedbackRequest(BaseModel):
    """User rating and qualitative remarks."""
    score: int = Field(ge=0, le=10, description="NPS score from 0 (detractor) to 10 (promoter)")
    category: str = Field(description="Category: critique_accuracy, fix_quality, usability, pricing")
    comment: str = Field(default="", description="Qualitative feedback or feature suggestion")


# --- Endpoints ---

@router.post("/activate", response_model=BetaActivationResponse)
def activate_beta_invite(req: ActivateBetaRequest) -> BetaActivationResponse:
    """Redeems an invitation code to join the private beta cohort."""
    try:
        user = beta_cohort_manager.activate_user(
            invite_code=req.invite_code,
            email=req.email,
        )
        return BetaActivationResponse(
            status="activated",
            user_id=user.user_id,
            email=user.email,
            persona=user.persona.value,
            message="Welcome to the AesFix Private Beta. Your account has been provisioned.",
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post("/event")
def track_funnel_event(
    req: TrackEventRequest,
    current_user: AuthClaims = Depends(get_current_user),
) -> Dict[str, Any]:
    """Records a milestone event along the activation funnel."""
    success = beta_cohort_manager.record_step(current_user.user_id, req.step)
    return {
        "status": "recorded" if success else "user_not_in_cohort",
        "user_id": current_user.user_id,
        "step": req.step.value,
    }


@router.post("/feedback")
def submit_beta_feedback(
    req: SubmitFeedbackRequest,
    current_user: AuthClaims = Depends(get_current_user),
) -> Dict[str, Any]:
    """Submits qualitative user feedback and NPS evaluation."""
    try:
        sub = beta_cohort_manager.submit_feedback(
            user_id=current_user.user_id,
            score=req.score,
            category=req.category,
            comment=req.comment,
        )
        return {
            "status": "received",
            "feedback_id": sub.feedback_id,
            "message": "Thank you for helping shape AesFix!",
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get("/funnel")
def get_funnel_analytics() -> Dict[str, Any]:
    """Returns real-time activation rates, drop-offs, and Net Promoter Score for the beta cohort."""
    summary = beta_cohort_manager.get_funnel_summary()
    return asdict(summary)
