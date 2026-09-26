"""Asynchronous Pipeline Orchestrator, Job Queue, and SSE Event Streamer (AF-5.2 & Roadmap §22, §42).

Why this orchestrator is the operational core of the AesFix SaaS:
1. The Closed Loop Realized: Coordinates the full pipeline from raw photo upload through
   Aspect Critic (F1.1, F1.2), ANFIS rule aggregation, Tier-A execution, Identity Guard
   drift verification, and Judge Ensemble gain measurement (F1.3).
2. Real-Time Observability via SSE: Multi-stage visual processing takes 2-6 seconds; synchronous
   HTTP requests risk gateway timeouts. This module provides real-time Server-Sent Events (SSE)
   so the frontend updates the progress bar and stage badge smoothly.
3. Defensible Fail-Safe Quotas: Integrates directly with QuotaManager so credits are reserved
   and deducted atomically, blocking over-consumption before expensive inference runs.
4. XMP Sidecar Generation: Packages every verified fix with an Adobe Lightroom-compatible .xmp
   preset delta so photographers can download their edits directly.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from PIL import Image

from aesfix.api.billing import QuotaManager
from aesfix.api.demo import compute_grade_label, get_demo_components
from aesfix.critic.critic import CriticFix
from aesfix.executor.guard import IdentityGuard
from aesfix.executor.tier_a import TierAExecutor

logger = logging.getLogger("aesfix.pipeline_queue")


class JobStatus(str, enum.Enum):
    """Lifecycle states of an asynchronous photo diagnosis job."""
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class JobEvent:
    """Discrete progress event emitted over Server-Sent Events (SSE)."""
    job_id: str
    stage: str
    progress_percent: int
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Formats the event for text/event-stream transmission."""
        import json
        payload = json.dumps(asdict(self))
        return f"data: {payload}\n\n"


@dataclass
class VerifiedFixPreview:
    """An executed photographic fix with verified gain badge and Lightroom XMP."""
    fix_id: str
    op: str
    params: Dict[str, Any]
    expected_aspect: str
    reasoning: str
    baseline_score: float
    post_edit_score: float
    aesthetic_gain: float  # max(0, post - baseline) if passed guard, else 0.0
    guard_passed: bool
    guard_drift: float
    gain_badge: str
    xmp_xml: str
    execution_time_ms: float


@dataclass
class JobResult:
    """Comprehensive diagnosis payload conforming to Core Features F1.1–F1.3."""
    job_id: str
    overall_score: float
    grade_label: str
    aspect_scores: Dict[str, float]
    anfis_rulebook_summary: str
    top_fired_rules: List[str]
    critique_rationale: Dict[str, str]
    verified_fixes: List[VerifiedFixPreview]
    best_fix: Optional[VerifiedFixPreview]
    uncertainty_badge: str
    total_latency_seconds: float


class PipelineJob:
    """Stateful container for an asynchronous image processing task."""

    def __init__(
        self,
        job_id: str,
        user_id: str,
        image_path: str,
        user_critique: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> None:
        self.job_id = job_id
        self.user_id = user_id
        self.image_path = image_path
        self.user_critique = user_critique
        self.goal = goal
        self.status = JobStatus.QUEUED
        self.created_at = time.time()
        self.updated_at = time.time()
        self.events: List[JobEvent] = []
        self.result: Optional[JobResult] = None
        self.error: Optional[str] = None
        self.event_subscribers: List[asyncio.Queue[JobEvent]] = []

    def record_event(self, stage: str, progress_percent: int, message: str, data: Optional[Dict[str, Any]] = None) -> JobEvent:
        """Appends a progress event and fans out to all active SSE subscribers."""
        event = JobEvent(
            job_id=self.job_id,
            stage=stage,
            progress_percent=progress_percent,
            message=message,
            data=data,
        )
        self.events.append(event)
        self.updated_at = time.time()

        for queue in list(self.event_subscribers):
            try:
                queue.put_nowait(event)
            except Exception:
                pass
        return event


class PipelineQueueManager:
    """Manages background job execution, quota deduction, and SSE streaming."""

    def __init__(
        self,
        quota_mgr: Optional[QuotaManager] = None,
        tier_a_executor: Optional[TierAExecutor] = None,
        identity_guard: Optional[IdentityGuard] = None,
    ) -> None:
        self.quota_mgr = quota_mgr or QuotaManager()
        self.tier_a_executor = tier_a_executor or TierAExecutor()
        self.identity_guard = identity_guard or IdentityGuard(drift_threshold=0.35)
        self._jobs: Dict[str, PipelineJob] = {}

    def submit_job(
        self,
        user_id: str,
        image_input: Union[str, Path, Image.Image],
        user_critique: Optional[str] = None,
        goal: Optional[str] = None,
    ) -> PipelineJob:
        """Validates quota, persists image, and enqueues job for asynchronous processing."""
        # 1. Pre-flight quota check
        can_run, reason = self.quota_mgr.can_critique(user_id)
        if not can_run:
            raise PermissionError(f"Quota check failed: {reason}")

        # 2. Materialize input image to disk
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        temp_dir = tempfile.gettempdir()
        target_path = os.path.join(temp_dir, f"aesfix_{job_id}.jpg")

        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                img.convert("RGB").save(target_path, format="JPEG", quality=95)
        elif isinstance(image_input, Image.Image):
            image_input.convert("RGB").save(target_path, format="JPEG", quality=95)
        else:
            raise ValueError(f"Unsupported image_input type: {type(image_input)}")

        # 3. Create stateful job
        job = PipelineJob(
            job_id=job_id,
            user_id=user_id,
            image_path=target_path,
            user_critique=user_critique,
            goal=goal,
        )
        self._jobs[job_id] = job

        job.record_event(
            stage="queued",
            progress_percent=5,
            message="Photo successfully uploaded and queued for diagnosis.",
        )

        # 4. Atomic quota deduction
        self.quota_mgr.consume_critique(user_id)

        return job

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        """Retrieves a job by its unique ID."""
        return self._jobs.get(job_id)

    async def stream_job_events(self, job_id: str) -> AsyncGenerator[str, None]:
        """Yields Server-Sent Events (SSE) for real-time frontend streaming."""
        job = self.get_job(job_id)
        if not job:
            error_event = JobEvent(
                job_id=job_id,
                stage="error",
                progress_percent=0,
                message=f"Job {job_id} not found.",
            )
            yield error_event.to_sse()
            return

        # Replay past events
        for past_event in list(job.events):
            yield past_event.to_sse()

        # If job is already in terminal state, finish stream
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            return

        # Subscribe to live events
        sub_queue: asyncio.Queue[JobEvent] = asyncio.Queue()
        job.event_subscribers.append(sub_queue)

        try:
            while job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
                try:
                    event = await asyncio.wait_for(sub_queue.get(), timeout=1.0)
                    yield event.to_sse()
                except asyncio.TimeoutError:
                    # Keep-alive heartbeat comment
                    yield ": ping\n\n"
        finally:
            if sub_queue in job.event_subscribers:
                job.event_subscribers.remove(sub_queue)

    def execute_job_sync(self, job_id: str) -> JobResult:
        """Executes the closed-loop pipeline synchronously (or inside a thread worker)."""
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"Job {job_id} not found.")

        job.status = JobStatus.PROCESSING
        start_time = time.perf_counter()

        try:
            critic, head, ensemble = get_demo_components()

            # --- Stage 1: Critic Analysis (F1.1, F1.2) ---
            job.record_event(
                stage="critiquing",
                progress_percent=25,
                message="Analyzing composition, lighting, color, sharpness, and technical quality...",
            )
            critic_out = critic.critique(
                job.image_path,
                user_critique=job.user_critique,
                genre=job.goal,
            )

            # --- Stage 2: ANFIS Interpretable Rulebook Trace (F1.1) ---
            job.record_event(
                stage="inferring_rulebook",
                progress_percent=45,
                message="Aggregating scores with differentiable neuro-fuzzy Takagi-Sugeno head...",
            )
            anfis_trace = head.explain(critic_out.aspect_scores, top_k=3)
            overall_score = round(anfis_trace.overall_score, 4)
            grade = compute_grade_label(overall_score)

            # Baseline Judge Ensemble Evaluation
            baseline_judge_res = ensemble.evaluate(job.image_path)
            baseline_score = round(baseline_judge_res.overall_score, 4)

            # --- Stage 3: Tier-A Execution & Verification (F1.3) ---
            job.record_event(
                stage="executing_fixes",
                progress_percent=70,
                message="Applying deterministic photographic adjustments and testing against Identity Guard...",
            )

            verified_previews: List[VerifiedFixPreview] = []
            candidate_fixes = critic_out.fixes or [
                CriticFix(
                    op="adjust_exposure",
                    params={"ev_delta": 0.25},
                    nl_instruction="Subtle exposure boost to lift midtones.",
                    expected_aspect="lighting",
                )
            ]

            temp_dir = tempfile.gettempdir()

            for idx, fix in enumerate(candidate_fixes[:3], start=1):
                # 1. Execute fix with Tier-A
                tier_a_res = self.tier_a_executor.execute_fixes(
                    image_input=job.image_path,
                    fixes=[fix],
                )

                # Save edited image temporarily for judge evaluation
                edit_path = os.path.join(temp_dir, f"edit_{job.job_id}_{idx}.jpg")
                tier_a_res.image.save(edit_path, format="JPEG", quality=95)

                # 2. Check Identity Guard drift
                guard_verdict = self.identity_guard.evaluate(
                    original=job.image_path,
                    edited=edit_path,
                )

                # 3. Score edited image with frozen judge ensemble
                post_judge_res = ensemble.evaluate(edit_path)
                post_score = round(post_judge_res.overall_score, 4)

                # Compute verified gain
                raw_delta = post_score - baseline_score
                if guard_verdict.is_valid:
                    gain = round(max(0.0, raw_delta), 4)
                    badge = f"+{gain:.3f} Verified Aesthetic Gain"
                else:
                    gain = 0.0
                    badge = "Vetoed by Identity Guard (Structural Drift)"

                reasoning_text = getattr(fix, "nl_instruction", getattr(fix, "reasoning", ""))
                verified_previews.append(
                    VerifiedFixPreview(
                        fix_id=f"{job.job_id}_fix_{idx}",
                        op=fix.op,
                        params=fix.params,
                        expected_aspect=fix.expected_aspect or "overall",
                        reasoning=reasoning_text,
                        baseline_score=baseline_score,
                        post_edit_score=post_score,
                        aesthetic_gain=gain,
                        guard_passed=guard_verdict.is_valid,
                        guard_drift=round(guard_verdict.composite_drift, 4),
                        gain_badge=badge,
                        xmp_xml=tier_a_res.xmp_sidecar,
                        execution_time_ms=round(tier_a_res.duration_ms, 2),
                    )
                )

                # Clean up temporary edited file
                try:
                    os.remove(edit_path)
                except OSError:
                    pass

            # Rank fixes by verified aesthetic gain descending
            verified_previews.sort(key=lambda f: f.aesthetic_gain, reverse=True)
            best_fix = verified_previews[0] if verified_previews else None

            # --- Stage 4: Consensus & Uncertainty Badge (F1.6) ---
            if baseline_judge_res.is_uncertain:
                uncertainty_badge = f"⚠️ High Aesthetic Disagreement (Var: {baseline_judge_res.disagreement:.3f}) — Taste-Dependent"
            else:
                uncertainty_badge = f"✅ Judge Consensus (Var: {baseline_judge_res.disagreement:.3f}) — High Confidence"

            total_elapsed = round(time.perf_counter() - start_time, 2)

            result = JobResult(
                job_id=job.job_id,
                overall_score=overall_score,
                grade_label=grade,
                aspect_scores=critic_out.aspect_scores,
                anfis_rulebook_summary=anfis_trace.rulebook_summary,
                top_fired_rules=anfis_trace.top_fired_rules,
                critique_rationale=critic_out.rationales,
                verified_fixes=verified_previews,
                best_fix=best_fix,
                uncertainty_badge=uncertainty_badge,
                total_latency_seconds=total_elapsed,
            )

            job.result = result
            job.status = JobStatus.COMPLETED

            job.record_event(
                stage="completed",
                progress_percent=100,
                message=f"Diagnosis completed in {total_elapsed}s. Best verified gain: {best_fix.gain_badge if best_fix else 'None'}",
                data={"overall_score": overall_score, "grade": grade},
            )
            return result

        except Exception as exc:
            logger.exception("Job %s execution failed: %s", job_id, exc)
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.record_event(
                stage="failed",
                progress_percent=100,
                message=f"Diagnosis failed: {exc}",
            )
            raise
