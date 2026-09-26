"""FastAPI Pipeline, Job Queue, and Server-Sent Events (SSE) Routes (AF-5.2 & Roadmap §42).

Why these endpoints are organized under /api/v1/pipeline:
1. Streaming Progress: Multi-stage inference (Critic -> ANFIS -> Tier-A -> Guard -> Ensemble)
   takes 2-6 seconds. Exposing /jobs/{job_id}/stream as text/event-stream provides immediate visual
   feedback to the frontend without polling overhead.
2. Background Processing: Uses asyncio.to_thread to run CPU-bound execution off the event loop,
   preserving low-latency responsiveness for concurrent users.
3. Native XMP Delivery: Photographers can download their verified edits directly as Adobe
   Camera Raw / Lightroom .xmp sidecars via /jobs/{job_id}/xmp.
"""
from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from aesfix.api.billing import AuthClaims
from aesfix.api.pipeline_queue import PipelineQueueManager
from aesfix.api.routes_billing import get_current_user, quota_manager

logger = logging.getLogger("aesfix.api.routes_pipeline")

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])

# Shared pipeline queue manager wired with the billing quota manager
queue_manager = PipelineQueueManager(quota_mgr=quota_manager)


class SubmitJobResponse(BaseModel):
    """Initial response upon submitting a photo for analysis."""
    job_id: str
    status: str
    stream_url: str
    message: str


class SubmitJobJsonRequest(BaseModel):
    """Alternative JSON payload for automated pipelines or base64 uploads."""
    image_path: str = Field(description="Local path or cached URI to image")
    user_critique: Optional[str] = Field(default=None, description="Optional user note or question")
    goal: Optional[str] = Field(default=None, description="Genre or goal: portrait, listing, etc.")


@router.post("/submit", response_model=SubmitJobResponse)
async def submit_image_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    goal: Optional[str] = Form(None),
    user_critique: Optional[str] = Form(None),
    current_user: AuthClaims = Depends(get_current_user),
) -> SubmitJobResponse:
    """Submits an uploaded image file for closed-loop diagnosis and fix verification."""
    # Read uploaded bytes
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image format: {exc}",
        ) from exc

    try:
        job = queue_manager.submit_job(
            user_id=current_user.user_id,
            image_input=pil_img,
            user_critique=user_critique,
            goal=goal,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(exc),
        ) from exc

    # Enqueue execution in background worker thread
    background_tasks.add_task(asyncio.to_thread, queue_manager.execute_job_sync, job.job_id)

    return SubmitJobResponse(
        job_id=job.job_id,
        status=job.status.value,
        stream_url=f"/api/v1/pipeline/jobs/{job.job_id}/stream",
        message="Photo enqueued. Connect to stream_url for live SSE updates.",
    )


@router.post("/submit-path", response_model=SubmitJobResponse)
def submit_image_path(
    req: SubmitJobJsonRequest,
    background_tasks: BackgroundTasks,
    current_user: AuthClaims = Depends(get_current_user),
) -> SubmitJobResponse:
    """Submits an image via file path (useful for testing or local server integrations)."""
    try:
        job = queue_manager.submit_job(
            user_id=current_user.user_id,
            image_input=req.image_path,
            user_critique=req.user_critique,
            goal=req.goal,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to ingest image path: {exc}",
        ) from exc

    # Run background execution
    background_tasks.add_task(asyncio.to_thread, queue_manager.execute_job_sync, job.job_id)

    return SubmitJobResponse(
        job_id=job.job_id,
        status=job.status.value,
        stream_url=f"/api/v1/pipeline/jobs/{job.job_id}/stream",
        message="Photo enqueued. Connect to stream_url for live SSE updates.",
    )


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str) -> StreamingResponse:
    """Streams real-time Server-Sent Events (SSE) representing diagnosis progress."""
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    return StreamingResponse(
        queue_manager.stream_job_events(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> Dict[str, Any]:
    """Retrieves current job status, execution progress, or full diagnosis payload."""
    job = queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    resp: Dict[str, Any] = {
        "job_id": job.job_id,
        "user_id": job.user_id,
        "status": job.status.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "error": job.error,
    }

    if job.result:
        resp["result"] = asdict(job.result)

    return resp


@router.get("/jobs/{job_id}/xmp")
def download_xmp_sidecar(job_id: str, fix_index: int = 0) -> Response:
    """Downloads the Lightroom-compatible .xmp sidecar for the specified fix."""
    job = queue_manager.get_job(job_id)
    if not job or not job.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' has no completed results.",
        )

    fixes = job.result.verified_fixes
    if not fixes or fix_index >= len(fixes):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fix index {fix_index} not available.",
        )

    target_fix = fixes[fix_index]
    filename = f"aesfix_{job_id}_fix{fix_index + 1}.xmp"

    return Response(
        content=target_fix.xmp_xml,
        media_type="application/xml",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
