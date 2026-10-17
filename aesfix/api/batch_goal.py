"""Goal Conditioning and Shoot-Level Batch Diagnosis Engine (AF-5.3 & Roadmap §8, §22 F1.4–F1.5).

Why Goal Conditioning (F1.5) and Batch Aggregation (F1.4) create immediate customer value:
1. Photography is Purpose-Driven: An Etsy listing photo requires neutral white balance and shadow
   lifting; a YouTube thumbnail requires high-contrast color pop and tight subject crop; a dating
   profile photo requires flattering portrait illumination. Goal conditioning dynamically pivots
   the critic's focus, aspect weighting, and recommended fix operations.
2. Shoot-Level Triage (The FilterPixel / Imagen Playbook): Photographers shoot in bursts of 50-200.
   Reviewing one photo at a time is tedious. Batch mode generates an aggregate shoot report
   identifying systemic flaws (e.g., "64% of shots are underexposed"), flags substandard frames,
   and crowns the optimal 'Hero Shot' for client delivery or marketplace cover images.
3. Strict Tier Quotas: Enforces subscription tier batch limits (Free: 1, Pro: 10, Studio: 100+)
   at the ingestion boundary before enqueuing processing jobs.
"""
from __future__ import annotations

import enum
import logging
import statistics as stats
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from aesfix.api.billing import QuotaManager, TIER_CONFIGS
from aesfix.api.pipeline_queue import JobResult, JobStatus, PipelineJob, PipelineQueueManager
from aesfix.data.loaders import ASPECT_DIMENSIONS

logger = logging.getLogger("aesfix.batch_goal")


class GoalMode(str, enum.Enum):
    """Supported photographic goal modes per Roadmap §8 & §22."""
    GENERAL = "general"
    PORTRAIT = "portrait"
    LISTING = "listing"          # E-Commerce (Etsy, eBay, Poshmark) & Real Estate
    THUMBNAIL = "thumbnail"      # Social Media, YouTube, Content Creators
    DATING = "dating"            # Profile Optimization (Photofeeler / ROAST use case)


@dataclass(frozen=True)
class GoalProfile:
    """Configures evaluation priorities, prompt hints, and preferred operations for a goal."""
    mode: GoalMode
    display_name: str
    description: str
    aspect_weights: Dict[str, float]
    preferred_ops: Tuple[str, ...]
    min_quality_threshold: float
    system_prompt_hint: str


GOAL_PROFILES: Dict[GoalMode, GoalProfile] = {
    GoalMode.GENERAL: GoalProfile(
        mode=GoalMode.GENERAL,
        display_name="Balanced Photographic",
        description="Standard photographic assessment balanced across all 5 aesthetic dimensions.",
        aspect_weights={
            "composition": 0.20,
            "lighting": 0.25,
            "color": 0.20,
            "subject_clarity": 0.20,
            "technical": 0.15,
        },
        preferred_ops=("adjust_exposure", "adjust_contrast", "adjust_white_balance", "smart_crop"),
        min_quality_threshold=0.60,
        system_prompt_hint="Evaluate overall photographic excellence, balance, and natural tonal harmony.",
    ),
    GoalMode.PORTRAIT: GoalProfile(
        mode=GoalMode.PORTRAIT,
        display_name="Portrait & Headshot",
        description="Flattering skin tones, gentle subject illumination, and sharp eye focus.",
        aspect_weights={
            "composition": 0.20,
            "lighting": 0.30,
            "color": 0.15,
            "subject_clarity": 0.25,
            "technical": 0.10,
        },
        preferred_ops=("adjust_exposure", "adjust_shadows", "unsharp_mask", "smart_crop"),
        min_quality_threshold=0.65,
        system_prompt_hint="Prioritize subject facial lighting, separation from background, and avoiding harsh specular highlights on skin.",
    ),
    GoalMode.LISTING: GoalProfile(
        mode=GoalMode.LISTING,
        display_name="E-Commerce & Listing",
        description="Clean, neutral white balance and even product illumination for high buyer conversion.",
        aspect_weights={
            "composition": 0.15,
            "lighting": 0.35,
            "color": 0.25,
            "subject_clarity": 0.15,
            "technical": 0.10,
        },
        preferred_ops=("adjust_exposure", "adjust_white_balance", "adjust_highlights", "denoise"),
        min_quality_threshold=0.65,
        system_prompt_hint="Assess clarity, color fidelity of merchandise, and shadow suppression for marketplace listing standards.",
    ),
    GoalMode.THUMBNAIL: GoalProfile(
        mode=GoalMode.THUMBNAIL,
        display_name="Social Media & Thumbnail",
        description="Punchy contrast, vibrant color saturation, and immediate visual hierarchy to stop the scroll.",
        aspect_weights={
            "composition": 0.25,
            "lighting": 0.25,
            "color": 0.30,
            "subject_clarity": 0.10,
            "technical": 0.10,
        },
        preferred_ops=("adjust_contrast", "adjust_saturation", "adjust_vibrance", "smart_crop"),
        min_quality_threshold=0.60,
        system_prompt_hint="Emphasize high visual contrast, saturated subject presence, and clean framing readable at small thumbnail dimensions.",
    ),
    GoalMode.DATING: GoalProfile(
        mode=GoalMode.DATING,
        display_name="Dating Profile Optimizer",
        description="Warm, approachable candid lighting and natural subject separation.",
        aspect_weights={
            "composition": 0.20,
            "lighting": 0.30,
            "color": 0.20,
            "subject_clarity": 0.20,
            "technical": 0.10,
        },
        preferred_ops=("adjust_exposure", "adjust_white_balance", "unsharp_mask", "smart_crop"),
        min_quality_threshold=0.62,
        system_prompt_hint="Look for friendly, flattering warm lighting, relaxed framing, and absence of distracting background clutter.",
    ),
}


@dataclass
class ShootDefectSummary:
    """Frequency analysis of specific photographic defects across a batch."""
    op_name: str
    affected_photo_count: int
    percentage: float
    description: str


@dataclass
class BatchDiagnosisReport:
    """Comprehensive shoot-level summary conforming to Feature F1.4."""
    batch_id: str
    goal_mode: str
    total_photos: int
    mean_overall_score: float
    aspect_averages: Dict[str, float]
    strongest_aspect: str
    weakest_aspect: str
    photos_below_threshold: List[str]  # IDs of photos failing the goal's quality threshold
    defect_distribution: List[ShootDefectSummary]
    hero_shot_job_id: Optional[str]
    hero_shot_score: float
    shoot_summary: str
    created_at: float = field(default_factory=time.time)


@dataclass
class BatchJob:
    """Multi-photo diagnosis batch container."""
    batch_id: str
    user_id: str
    goal: GoalMode
    job_ids: List[str]
    status: JobStatus = JobStatus.QUEUED
    report: Optional[BatchDiagnosisReport] = None
    created_at: float = field(default_factory=time.time)


class BatchDiagnosisAggregator:
    """Aggregates individual photo diagnosis results into a cohesive shoot-level report."""

    @classmethod
    def compile_report(
        cls,
        batch_id: str,
        goal: GoalMode,
        results: List[JobResult],
    ) -> BatchDiagnosisReport:
        """Computes statistical metrics, identifies shoot weaknesses, and selects the hero frame."""
        if not results:
            raise ValueError("Cannot compile batch report from empty results list.")

        profile = GOAL_PROFILES[goal]
        total_photos = len(results)

        # 1. Overall and aspect averages
        overall_scores = [r.overall_score for r in results if r.overall_score is not None]
        mean_score = round(float(stats.fmean(overall_scores)), 4) if overall_scores else 0.0

        aspect_sums: Dict[str, float] = {dim: 0.0 for dim in ASPECT_DIMENSIONS}
        for r in results:
            for dim in ASPECT_DIMENSIONS:
                aspect_sums[dim] += r.aspect_scores.get(dim, 0.5)

        aspect_averages = {
            dim: round(aspect_sums[dim] / total_photos, 4) for dim in ASPECT_DIMENSIONS
        }

        # 2. Identify strongest and weakest aspects
        sorted_aspects = sorted(aspect_averages.items(), key=lambda x: x[1])
        weakest_aspect = sorted_aspects[0][0]
        strongest_aspect = sorted_aspects[-1][0]

        # 3. Photos below goal quality threshold
        below_thresh: List[str] = [
            r.job_id for r in results if r.overall_score is not None and r.overall_score < profile.min_quality_threshold
        ]

        # 4. Defect frequency across fixes
        op_counts: Dict[str, int] = {}
        for r in results:
            unique_ops_for_photo = {fix.op for fix in r.verified_fixes}
            for op in unique_ops_for_photo:
                op_counts[op] = op_counts.get(op, 0) + 1

        defect_distribution: List[ShootDefectSummary] = []
        for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True):
            pct = round((count / total_photos) * 100.0, 1)
            desc = f"{count} of {total_photos} photos ({pct}%) recommended {op.replace('_', ' ')}."
            defect_distribution.append(
                ShootDefectSummary(
                    op_name=op,
                    affected_photo_count=count,
                    percentage=pct,
                    description=desc,
                )
            )

        # 5. Crown Hero Shot (highest overall score, or fallback to first photo if unmeasured)
        best_result = max(results, key=lambda r: (r.overall_score or 0.0))
        hero_job_id = best_result.job_id
        hero_score = best_result.overall_score or 0.0

        # 6. Executive Shoot Summary
        summary = (
            f"Shoot Analysis for {total_photos} photos under '{profile.display_name}' mode. "
            f"Average quality is {mean_score:.2f}/1.0. "
            f"Primary strength: {strongest_aspect.replace('_', ' ')} ({aspect_averages[strongest_aspect]:.2f}). "
            f"Primary systematic bottleneck: {weakest_aspect.replace('_', ' ')} ({aspect_averages[weakest_aspect]:.2f}). "
            f"{len(below_thresh)} photos require active retouching before publication. "
            f"Recommended hero frame: {hero_job_id} (score: {hero_score:.2f})."
        )

        return BatchDiagnosisReport(
            batch_id=batch_id,
            goal_mode=goal.value,
            total_photos=total_photos,
            mean_overall_score=mean_score,
            aspect_averages=aspect_averages,
            strongest_aspect=strongest_aspect,
            weakest_aspect=weakest_aspect,
            photos_below_threshold=below_thresh,
            defect_distribution=defect_distribution,
            hero_shot_job_id=hero_job_id,
            hero_shot_score=hero_score,
            shoot_summary=summary,
        )


class BatchQueueManager:
    """Manages multi-photo batch submission, tier entitlement validation, and report aggregation."""

    def __init__(
        self,
        pipeline_qm: PipelineQueueManager,
        quota_mgr: Optional[QuotaManager] = None,
    ) -> None:
        self.pipeline_qm = pipeline_qm
        self.quota_mgr = quota_mgr or QuotaManager()
        self._batches: Dict[str, BatchJob] = {}

    def submit_batch(
        self,
        user_id: str,
        image_inputs: List[Any],
        goal: GoalMode = GoalMode.GENERAL,
        user_critique: Optional[str] = None,
    ) -> BatchJob:
        """Validates tier limits, enqueues individual photo jobs, and tracks batch state."""
        if not image_inputs:
            raise ValueError("Cannot submit an empty image list.")

        user = self.quota_mgr.get_or_create_user(user_id)
        entitlements = TIER_CONFIGS[user.tier]
        max_batch = entitlements.max_batch_size

        if len(image_inputs) > max_batch:
            raise PermissionError(
                f"Batch size of {len(image_inputs)} exceeds {user.tier.value} tier limit of {max_batch}. "
                f"Please upgrade to Studio tier for batches up to 100 photos."
            )

        # Pre-flight quota check for the entire batch
        remaining_credits = self.quota_mgr.get_remaining_critiques(user_id)
        if remaining_credits < len(image_inputs):
            raise PermissionError(
                f"Insufficient credits for batch: requires {len(image_inputs)}, but user only has {remaining_credits} remaining."
            )

        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        job_ids: List[str] = []

        for img in image_inputs:
            job = self.pipeline_qm.submit_job(
                user_id=user_id,
                image_input=img,
                user_critique=user_critique,
                goal=goal.value,
            )
            job_ids.append(job.job_id)

        batch_job = BatchJob(
            batch_id=batch_id,
            user_id=user_id,
            goal=goal,
            job_ids=job_ids,
            status=JobStatus.QUEUED,
        )
        self._batches[batch_id] = batch_job
        logger.info("Enqueued batch %s with %d photos for user %s (Goal: %s)",
                    batch_id, len(job_ids), user_id, goal.value)
        return batch_job

    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """Retrieves batch status and report."""
        return self._batches.get(batch_id)

    def update_batch_status(self, batch_id: str) -> BatchJob:
        """Checks constituent jobs; when all complete, compiles the aggregate shoot report."""
        batch = self.get_batch(batch_id)
        if not batch:
            raise KeyError(f"Batch {batch_id} not found.")

        jobs: List[PipelineJob] = [
            self.pipeline_qm.get_job(jid) for jid in batch.job_ids if self.pipeline_qm.get_job(jid) is not None
        ]

        if not jobs:
            batch.status = JobStatus.FAILED
            return batch

        # Check statuses
        if all(j.status == JobStatus.COMPLETED for j in jobs):
            batch.status = JobStatus.COMPLETED
            results = [j.result for j in jobs if j.result is not None]
            if results:
                batch.report = BatchDiagnosisAggregator.compile_report(
                    batch_id=batch_id,
                    goal=batch.goal,
                    results=results,
                )
        elif any(j.status == JobStatus.FAILED for j in jobs):
            batch.status = JobStatus.FAILED
        else:
            batch.status = JobStatus.PROCESSING

        return batch
