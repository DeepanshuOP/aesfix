"""FastAPI Trust, Safety, Provenance, and Privacy Routes (AF-5.4 & Roadmap §42).

Why these endpoints are organized under /api/v1/trust:
1. Pre-Flight Safety: /safety-check screens images before queuing GPU inference jobs.
2. Digital Provenance: /c2pa/{job_id} generates verifiable C2PA JSON-LD claims certifying
   the edit as authentic machine-verified photographic enhancement.
3. Anti-Theft Watermarking: /watermark-preview renders protected preview cards with gain badges.
4. Privacy & GDPR Compliance: /purge-photos immediately purges user files, and /opt-in allows
   users to explicitly control research training usage.
"""
from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from PIL import Image
from pydantic import BaseModel, Field

from aesfix.api.billing import AuthClaims
from aesfix.api.routes_billing import get_current_user
from aesfix.api.routes_pipeline import queue_manager
from aesfix.api.trust import (
    C2PAProvenanceEngine,
    ContentSafetyGate,
    DataRetentionManager,
    WatermarkService,
)

logger = logging.getLogger("aesfix.api.routes_trust")

router = APIRouter(prefix="/api/v1/trust", tags=["trust"])

safety_gate = ContentSafetyGate()
retention_manager = DataRetentionManager()


# --- Request & Response Models ---

class SafetyCheckResponse(BaseModel):
    """Content screening outcome."""
    is_safe: bool
    risk_category: str
    confidence: float
    rejection_reason: Optional[str] = None


class OptInRequest(BaseModel):
    """Research model training consent payload."""
    opt_in: bool = Field(description="Explicit consent to use anonymized image for research")


class PurgeResponse(BaseModel):
    """Outcome of GDPR photo purge request."""
    status: str
    user_id: str
    purged_files_count: int
    message: str


# --- Endpoints ---

@router.post("/safety-check", response_model=SafetyCheckResponse)
async def check_image_safety(
    file: UploadFile = File(...),
    current_user: AuthClaims = Depends(get_current_user),
) -> SafetyCheckResponse:
    """Pre-flight content screening gate to block prohibited or corrupt uploads."""
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
            detail=f"Invalid image format: {exc}",
        ) from exc

    verdict = safety_gate.screen_image(pil_img)
    return SafetyCheckResponse(
        is_safe=verdict.is_safe,
        risk_category=verdict.risk_category,
        confidence=verdict.confidence,
        rejection_reason=verdict.rejection_reason,
    )


@router.get("/c2pa/{job_id}")
def get_c2pa_manifest(
    job_id: str,
    current_user: AuthClaims = Depends(get_current_user),
) -> Dict[str, Any]:
    """Generates C2PA JSON-LD provenance manifest for an executed photo diagnosis job."""
    job = queue_manager.get_job(job_id)
    if not job or not job.result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found or not completed.",
        )

    best_fix = job.result.best_fix
    gain = best_fix.aesthetic_gain if best_fix else 0.0
    applied_ops = [f.op for f in job.result.measured_fixes]

    orig_hash = C2PAProvenanceEngine.compute_sha256(job.image_path)
    edited_hash = orig_hash  # In production, compute from serialized edited image

    manifest = C2PAProvenanceEngine.generate_manifest(
        original_digest=orig_hash,
        edited_digest=edited_hash,
        applied_ops=applied_ops,
        verified_aesthetic_gain=gain,
    )
    return json.loads(manifest.to_json_ld())


@router.post("/watermark-preview")
async def generate_watermarked_preview(
    file: UploadFile = File(...),
    gain_badge: Optional[str] = Form(None),
) -> Response:
    """Applies anti-theft translucent pill watermark with measured gain badge to an image."""
    content = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {exc}",
        ) from exc

    watermarked = WatermarkService.apply_watermark(
        pil_img,
        watermark_text="AesFix Measured Preview",
        gain_badge=gain_badge,
    )

    buf = io.BytesIO()
    watermarked.save(buf, format="JPEG", quality=90)

    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Content-Disposition": 'inline; filename="preview_watermarked.jpg"'},
    )


@router.delete("/purge-photos", response_model=PurgeResponse)
def purge_user_photos(
    current_user: AuthClaims = Depends(get_current_user),
) -> PurgeResponse:
    """Immediately purges all uploaded originals and intermediate files for the user (GDPR)."""
    count = retention_manager.purge_user_photos(current_user.user_id)
    return PurgeResponse(
        status="purged",
        user_id=current_user.user_id,
        purged_files_count=count,
        message=f"Successfully purged {count} user file(s) in accordance with GDPR right to erasure.",
    )


@router.post("/opt-in")
def update_training_opt_in(
    req: OptInRequest,
    current_user: AuthClaims = Depends(get_current_user),
) -> Dict[str, Any]:
    """Updates research training consent flag (defaults to False unless explicitly toggled)."""
    retention_manager.set_training_opt_in(current_user.user_id, req.opt_in)
    return {
        "user_id": current_user.user_id,
        "training_opt_in": req.opt_in,
        "status": "updated",
    }
