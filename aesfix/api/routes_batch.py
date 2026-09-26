"""FastAPI Batch & Goal Conditioning Routes (AF-5.3 & Roadmap §42).

Why these endpoints are organized under /api/v1/batch:
1. Goal Mode Discovery: Allows frontends to query supported goal profiles (/goals) to render
   custom UI options (E-Commerce Listing, Portrait, YouTube Thumbnail, Dating Profile).
2. Batch Ingestion with Tier Enforcement: Accepts multi-photo collections, validates batch
   size limits against subscription tiers, and coordinates parallel job processing.
3. Shoot-Level Triage Reporting: Exposes aggregated shoot diagnostics (/batch/{batch_id})
   providing set-wide weakest aspect identification, flaw distributions, and hero-frame picks.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from aesfix.api.batch_goal import (
    GOAL_PROFILES,
    BatchQueueManager,
    GoalMode,
)
from aesfix.api.billing import AuthClaims
from aesfix.api.routes_billing import get_current_user, quota_manager
from aesfix.api.routes_pipeline import queue_manager

logger = logging.getLogger("aesfix.api.routes_batch")

router = APIRouter(prefix="/api/v1/batch", tags=["batch"])

# Shared batch queue manager wired with the pipeline queue manager
batch_queue_manager = BatchQueueManager(pipeline_qm=queue_manager, quota_mgr=quota_manager)


# --- Schemas ---

class GoalProfileResponse(BaseModel):
    """Goal mode descriptor."""
    mode: str
    display_name: str
    description: str
    aspect_weights: Dict[str, float]
    preferred_ops: List[str]
    min_quality_threshold: float
    system_prompt_hint: str


class BatchSubmitRequest(BaseModel):
    """Payload to submit multiple images for aggregate shoot diagnosis."""
    image_paths: List[str] = Field(min_length=1, description="List of image paths or cached URIs")
    goal: GoalMode = Field(default=GoalMode.GENERAL, description="Goal mode (listing, portrait, thumbnail, etc.)")
    user_critique: Optional[str] = Field(default=None, description="Optional overarching shoot context")


class BatchSubmitResponse(BaseModel):
    """Initial response upon batch ingestion."""
    batch_id: str
    goal: str
    total_photos: int
    job_ids: List[str]
    message: str


class BatchStatusResponse(BaseModel):
    """Current batch state and aggregate shoot diagnosis."""
    batch_id: str
    goal: str
    status: str
    total_photos: int
    completed_photos: int
    report: Optional[Dict[str, Any]] = None


# --- Endpoints ---

@router.get("/goals", response_model=List[GoalProfileResponse])
def list_goal_modes() -> List[GoalProfileResponse]:
    """Returns all available photographic goal modes and their tuning configurations."""
    profiles: List[GoalProfileResponse] = []
    for mode, prof in GOAL_PROFILES.items():
        profiles.append(
            GoalProfileResponse(
                mode=mode.value,
                display_name=prof.display_name,
                description=prof.description,
                aspect_weights=prof.aspect_weights,
                preferred_ops=list(prof.preferred_ops),
                min_quality_threshold=prof.min_quality_threshold,
                system_prompt_hint=prof.system_prompt_hint,
            )
        )
    return profiles


@router.post("/submit", response_model=BatchSubmitResponse)
def submit_batch(
    req: BatchSubmitRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthClaims = Depends(get_current_user),
) -> BatchSubmitResponse:
    """Submits a multi-photo batch for analysis and enqueues background processing."""
    try:
        batch = batch_queue_manager.submit_batch(
            user_id=current_user.user_id,
            image_inputs=req.image_paths,
            goal=req.goal,
            user_critique=req.user_critique,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Enqueue background execution of all constituent jobs
    for jid in batch.job_ids:
        background_tasks.add_task(asyncio.to_thread, queue_manager.execute_job_sync, jid)

    return BatchSubmitResponse(
        batch_id=batch.batch_id,
        goal=batch.goal.value,
        total_photos=len(batch.job_ids),
        job_ids=batch.job_ids,
        message="Batch enqueued successfully. Poll /api/v1/batch/{batch_id} for shoot report.",
    )


@router.get("/{batch_id}", response_model=BatchStatusResponse)
def get_batch_status(batch_id: str) -> BatchStatusResponse:
    """Retrieves batch processing progress and the aggregate shoot diagnosis report upon completion."""
    try:
        batch = batch_queue_manager.update_batch_status(batch_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch '{batch_id}' not found.",
        )

    jobs = [queue_manager.get_job(jid) for jid in batch.job_ids]
    completed_count = sum(1 for j in jobs if j is not None and j.status.value == "COMPLETED")

    report_dict = asdict(batch.report) if batch.report is not None else None

    return BatchStatusResponse(
        batch_id=batch.batch_id,
        goal=batch.goal.value,
        status=batch.status.value,
        total_photos=len(batch.job_ids),
        completed_photos=completed_count,
        report=report_dict,
    )
