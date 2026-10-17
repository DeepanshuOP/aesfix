"""FastAPI Router for AesFix Closed-Loop Agent (AG4).

Provides:
- POST /api/v1/agent/run: SSE streaming per perceive-plan-act-verify step
- GET /api/v1/agent/jobs/{job_id}/trajectory: Full execution trajectory JSON
- GET /api/v1/agent/jobs/{job_id}/step/{step_idx}/preview: Preview JPEG for an individual step
- GET /api/v1/agent/jobs/{job_id}/final/preview: Final edited photograph JPEG
- GET /api/v1/agent/jobs/{job_id}/xmp: Stacked Lightroom XMP sidecar
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from aesfix.agent.llm import LLMClient
from aesfix.agent.loop import AesFixAgent, AgentStep, AgentTrajectory

logger = logging.getLogger("aesfix.api.routes_agent")

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])

# Ephemeral in-memory registry for recent agent jobs and preview images
AGENT_JOBS: Dict[str, Dict[str, Any]] = {}
MAX_STORED_JOBS = 50


class AgentRunJsonRequest(BaseModel):
    """JSON payload alternative for agent execution."""
    image_path: Optional[str] = Field(default=None, description="Local path to photo")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded photo")
    user_goal: Optional[str] = Field(default=None, description="Genre or intent goal")
    provider: Optional[str] = Field(default=None, description="gemini / openrouter / heuristic")
    model_id: Optional[str] = Field(default=None, description="Specific vision model id")
    seed: Optional[int] = Field(default=42, description="Deterministic seed")
    max_steps: Optional[int] = Field(default=6, description="Max trajectory iterations")


def _image_to_jpeg_bytes(pil_img: Image.Image, quality: int = 90, max_size: int = 1024) -> bytes:
    """Downsamples and encodes image to JPEG bytes for previews."""
    img = pil_img.copy()
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _log_agent_run_telemetry(trajectory: AgentTrajectory, job_id: str, input_hash: str) -> None:
    """Appends run telemetry (hashes + metrics only) to results/agent_runs/ (AG6)."""
    try:
        runs_dir = Path("results/agent_runs")
        runs_dir.mkdir(parents=True, exist_ok=True)
        log_file = runs_dir / "runs.jsonl"

        record = {
            "timestamp": time.time(),
            "job_id": job_id,
            "input_sha256": input_hash,
            "provider": trajectory.provider,
            "model_id": trajectory.model_id,
            "seed": trajectory.seed,
            "verdict": trajectory.verdict,
            "initial_overall_score": trajectory.initial_overall_score,
            "final_overall_score": trajectory.final_overall_score,
            "total_gain": trajectory.total_gain,
            "noise_floor": trajectory.noise_floor,
            "total_steps": trajectory.total_steps,
            "accepted_count": len(trajectory.accepted_steps),
            "rejected_count": len(trajectory.rejected_steps),
            "accepted_ops": [s.proposal.op for s in trajectory.accepted_steps],
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning("Failed appending agent telemetry to %s: %s", runs_dir, exc)


@router.post("/run")
async def run_agent_sse(
    file: Optional[UploadFile] = File(None),
    user_goal: Optional[str] = Form(None),
    provider: Optional[str] = Form(None),
    model_id: Optional[str] = Form(None),
    seed: Optional[int] = Form(42),
    max_steps: Optional[int] = Form(6),
) -> StreamingResponse:
    """Runs AesFix closed-loop agent and streams SSE updates per step."""
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required image file upload.",
        )

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

    input_hash = hashlib.sha256(content).hexdigest()
    job_id = f"job_ag_{uuid.uuid4().hex[:12]}"

    # Enforce eviction if memory cache is full
    if len(AGENT_JOBS) > MAX_STORED_JOBS:
        oldest_key = next(iter(AGENT_JOBS))
        AGENT_JOBS.pop(oldest_key, None)

    # Initialize job state
    AGENT_JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "created_at": time.time(),
        "input_hash": input_hash,
        "trajectory": None,
        "step_previews": {},
        "final_preview_bytes": None,
        "stacked_xmp": "",
    }

    # Setup agent client
    llm_client = LLMClient(provider=provider, model_id=model_id)
    agent = AesFixAgent(
        llm_client=llm_client,
        max_steps=max_steps or 6,
        seed=seed or 42,
    )

    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[Optional[Tuple[str, Dict[str, Any]]]] = asyncio.Queue()

    def on_step_executed(step: AgentStep) -> None:
        """Callback invoked synchronously within the agent loop thread."""
        step_dict = step.to_dict()
        preview_data_url = ""

        if step.preview_image is not None:
            jpeg_bytes = _image_to_jpeg_bytes(step.preview_image, quality=85, max_size=768)
            AGENT_JOBS[job_id]["step_previews"][step.step_number] = jpeg_bytes
            b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
            preview_data_url = f"data:image/jpeg;base64,{b64_str}"

        step_dict["preview_data_url"] = preview_data_url
        step_dict["preview_url"] = f"/api/v1/agent/jobs/{job_id}/step/{step.step_number}/preview"

        loop.call_soon_threadsafe(event_queue.put_nowait, ("step", step_dict))

    def run_agent_worker() -> None:
        """Runs the CPU/IO bound agent loop in a separate thread."""
        try:
            trajectory = agent.run(
                image_input=pil_img,
                user_goal=user_goal,
                step_callback=on_step_executed,
            )

            # Materialize final preview
            final_bytes = _image_to_jpeg_bytes(trajectory.final_image, quality=90, max_size=1024)
            AGENT_JOBS[job_id]["final_preview_bytes"] = final_bytes
            AGENT_JOBS[job_id]["stacked_xmp"] = trajectory.stacked_xmp
            AGENT_JOBS[job_id]["trajectory"] = trajectory.to_dict()
            AGENT_JOBS[job_id]["status"] = "completed"

            # Log telemetry
            _log_agent_run_telemetry(trajectory, job_id, input_hash)

            final_b64 = base64.b64encode(final_bytes).decode("ascii")
            complete_payload = {
                "job_id": job_id,
                "status": "completed",
                "verdict": trajectory.verdict,
                "initial_overall_score": trajectory.initial_overall_score,
                "final_overall_score": trajectory.final_overall_score,
                "total_gain": trajectory.total_gain,
                "noise_floor": trajectory.noise_floor,
                "total_steps": trajectory.total_steps,
                "accepted_steps_count": len(trajectory.accepted_steps),
                "rejected_steps_count": len(trajectory.rejected_steps),
                "stacked_xmp": trajectory.stacked_xmp,
                "provider": trajectory.provider,
                "model_id": trajectory.model_id,
                "final_image_preview": f"data:image/jpeg;base64,{final_b64}",
                "final_image_url": f"/api/v1/agent/jobs/{job_id}/final/preview",
                "trajectory_url": f"/api/v1/agent/jobs/{job_id}/trajectory",
            }
            loop.call_soon_threadsafe(event_queue.put_nowait, ("complete", complete_payload))
        except Exception as exc:
            logger.error("Agent execution failed for %s: %s", job_id, exc, exc_info=True)
            AGENT_JOBS[job_id]["status"] = "failed"
            AGENT_JOBS[job_id]["error"] = str(exc)
            err_payload = {"job_id": job_id, "status": "failed", "error": str(exc)}
            loop.call_soon_threadsafe(event_queue.put_nowait, ("error", err_payload))
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    # Launch agent thread
    asyncio.create_task(asyncio.to_thread(run_agent_worker))

    async def sse_streamer() -> AsyncGenerator[str, None]:
        # Emit initial handshake event
        init_data = {
            "job_id": job_id,
            "status": "running",
            "provider": agent.llm_client.provider,
            "model_id": agent.llm_client.model_id,
            "user_goal": user_goal or "General photographic balance",
        }
        yield f"event: init\ndata: {json.dumps(init_data)}\n\n"

        while True:
            msg = await event_queue.get()
            if msg is None:
                break
            ev_name, ev_data = msg
            yield f"event: {ev_name}\ndata: {json.dumps(ev_data)}\n\n"

    return StreamingResponse(
        sse_streamer(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/trajectory")
@router.get("/trajectory/{job_id}")
def get_agent_trajectory(job_id: str) -> Dict[str, Any]:
    """Retrieves full closed-loop agent trajectory record for job_id."""
    job = AGENT_JOBS.get(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent job '{job_id}' not found.",
        )
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "trajectory": job.get("trajectory"),
        "has_final_image": bool(job.get("final_preview_bytes")),
        "final_image_url": f"/api/v1/agent/jobs/{job_id}/final/preview" if job.get("final_preview_bytes") else None,
        "xmp_url": f"/api/v1/agent/jobs/{job_id}/xmp" if job.get("stacked_xmp") else None,
    }


@router.get("/jobs/{job_id}/step/{step_idx}/preview")
def get_step_preview(job_id: str, step_idx: int) -> Response:
    """Returns candidate preview image bytes for a specific step."""
    job = AGENT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    previews = job.get("step_previews", {})
    img_bytes = previews.get(step_idx)
    if not img_bytes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Step preview '{step_idx}' not found.")
    return Response(content=img_bytes, media_type="image/jpeg")


@router.get("/jobs/{job_id}/final/preview")
def get_final_preview(job_id: str) -> Response:
    """Returns final retouched image bytes."""
    job = AGENT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    img_bytes = job.get("final_preview_bytes")
    if not img_bytes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Final image preview not ready.")
    return Response(content=img_bytes, media_type="image/jpeg")


@router.get("/jobs/{job_id}/xmp")
def download_stacked_xmp(job_id: str) -> Response:
    """Downloads stacked Adobe Camera Raw / Lightroom XMP sidecar for accepted edits."""
    job = AGENT_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    xmp_str = job.get("stacked_xmp", "")
    if not xmp_str:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No accepted edits produced an XMP sidecar.")
    return Response(
        content=xmp_str,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="aesfix_agent_{job_id}.xmp"'},
    )
