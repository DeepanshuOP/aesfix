"""AesFix Closed-Loop Agent Trajectory Loop (AG3).

Closed-loop execution cycle:
  Perceive -> Plan -> Act -> Verify -> Accept / Reject -> Stop

Verification Invariants (ALL must pass for acceptance):
1. Overall Judge Δ > ε (aesthetic gain exceeds empirical noise floor).
2. Claimed aspect moves in claimed direction by > noise floor (rsinema per-aspect heads
   where mapped: composition, light->lighting, color, quality->technical; otherwise critic scores,
   labeled as unvalidated).
3. No other aspect drops beyond its noise floor (anti-collateral damage).
4. Identity Guard passes (composite drift <= threshold).

Termination:
- max_steps = 6
- Stop after 2 consecutive rejections
- Final image, stacked XMP sidecar, and verdict ∈ {"measured improvement", "leave it"}.
- Deterministic random seed.
"""
from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from aesfix.agent.llm import LLMClient
from aesfix.agent.schema import (
    AgentProposal,
    format_validation_error_feedback,
    generate_tool_schema_prompt,
    validate_proposal,
)
from aesfix.critic.critic import CriticFix, HeuristicAspectCritic
from aesfix.executor.guard import IdentityGuard
from aesfix.executor.tier_a import TierAExecutor, XMPWriter
from aesfix.api.demo import get_demo_components
from aesfix.judges.ensemble import (
    JudgeEnsemble,
    compute_noise_floor,
)

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Record of a single perceive-plan-act-verify step in the trajectory."""
    step_number: int
    proposal: AgentProposal
    status: str  # "accepted" or "rejected"
    verdict_reason: str
    baseline_overall_score: float
    post_overall_score: float
    overall_delta: float
    noise_floor: float
    claimed_aspect: str
    claimed_aspect_delta: float
    claimed_aspect_direction: int
    aspect_scores_before: Dict[str, float]
    aspect_scores_after: Dict[str, float]
    aspect_deltas: Dict[str, float]
    guard_drift: float
    guard_passed: bool
    guard_backend: str
    execution_duration_ms: float
    xmp_xml: str
    preview_image: Optional[Image.Image] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes step record for API responses and telemetry logging."""
        return {
            "step_number": self.step_number,
            "op": self.proposal.op,
            "params": self.proposal.params,
            "aspect": self.proposal.aspect,
            "direction": self.proposal.direction,
            "rationale": self.proposal.rationale,
            "status": self.status,
            "verdict_reason": self.verdict_reason,
            "baseline_overall_score": self.baseline_overall_score,
            "post_overall_score": self.post_overall_score,
            "overall_delta": self.overall_delta,
            "noise_floor": self.noise_floor,
            "claimed_aspect": self.claimed_aspect,
            "claimed_aspect_delta": self.claimed_aspect_delta,
            "guard_drift": self.guard_drift,
            "guard_passed": self.guard_passed,
            "guard_backend": self.guard_backend,
            "aspect_scores_before": self.aspect_scores_before,
            "aspect_scores_after": self.aspect_scores_after,
            "aspect_deltas": self.aspect_deltas,
            "execution_duration_ms": self.execution_duration_ms,
        }


@dataclass
class AgentTrajectory:
    """Full execution trajectory of the AesFix agent closed loop."""
    initial_image_path: str
    final_image: Image.Image
    stacked_xmp: str
    verdict: str  # "measured improvement" or "leave it"
    initial_overall_score: float
    final_overall_score: float
    total_gain: float
    noise_floor: float
    steps: List[AgentStep] = field(default_factory=list)
    accepted_steps: List[AgentStep] = field(default_factory=list)
    rejected_steps: List[AgentStep] = field(default_factory=list)
    consecutive_rejections: int = 0
    total_steps: int = 0
    provider: str = ""
    model_id: str = ""
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Serializes trajectory metadata and steps."""
        return {
            "verdict": self.verdict,
            "initial_overall_score": self.initial_overall_score,
            "final_overall_score": self.final_overall_score,
            "total_gain": self.total_gain,
            "noise_floor": self.noise_floor,
            "total_steps": self.total_steps,
            "num_accepted": len(self.accepted_steps),
            "num_rejected": len(self.rejected_steps),
            "provider": self.provider,
            "model_id": self.model_id,
            "seed": self.seed,
            "steps": [s.to_dict() for s in self.steps],
            "accepted_ops": [s.proposal.op for s in self.accepted_steps],
        }


class AesFixAgent:
    """Closed-loop photo retouching agent operating over Tier-A visual tools."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        critic: Optional[HeuristicAspectCritic] = None,
        ensemble: Optional[JudgeEnsemble] = None,
        guard: Optional[IdentityGuard] = None,
        executor: Optional[TierAExecutor] = None,
        max_steps: int = 6,
        consecutive_rejections_stop: int = 2,
        seed: int = 42,
    ) -> None:
        self.llm_client = llm_client or LLMClient(provider="heuristic")
        self.critic = critic
        self.ensemble = ensemble
        self.guard = guard or IdentityGuard(drift_threshold=0.35)
        self.executor = executor or TierAExecutor()
        self.max_steps = max_steps
        self.consecutive_rejections_stop = consecutive_rejections_stop
        self.seed = seed

    def _set_deterministic_seed(self) -> None:
        """Sets random seeds for deterministic inference across all frameworks."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _ensure_components(self) -> Tuple[HeuristicAspectCritic, JudgeEnsemble]:
        """Loads critic and judge ensemble components if not injected."""
        critic = self.critic
        ensemble = self.ensemble
        if critic is None or ensemble is None:
            demo_critic, _, demo_ensemble = get_demo_components()
            critic = critic or demo_critic
            ensemble = ensemble or demo_ensemble
            self.critic = critic
            self.ensemble = ensemble
        return critic, ensemble

    def _extract_aspect_scores(
        self,
        image_path: str,
        ensemble: JudgeEnsemble,
        critic: HeuristicAspectCritic,
    ) -> Tuple[Dict[str, float], float]:
        """Extracts overall aesthetic score and per-aspect scores.

        Maps rsinema per-aspect heads if available:
        composition -> composition, light -> lighting, color -> color, quality -> technical.
        Otherwise falls back to critic aspect scores and labels as unvalidated.
        """
        judge_res = ensemble.evaluate(image_path)
        overall_score = round(float(judge_res.overall_score), 4)

        aspect_scores: Dict[str, float] = {}

        # 1. Attempt extraction from Rsinema heads if present in ensemble
        found_rsinema = False
        for j in getattr(ensemble, "judges", []):
            if hasattr(j, "_heads") and j._heads:
                try:
                    # Rsinema judge exposes aspect_scores
                    j_score = j.evaluate(image_path)
                    raw_aspects = getattr(j_score, "aspect_scores", {})
                    if raw_aspects:
                        found_rsinema = True
                        aspect_scores["composition (unvalidated)"] = raw_aspects.get("composition (unvalidated)", 0.5)
                        aspect_scores["lighting (unvalidated)"] = raw_aspects.get("light (unvalidated)", 0.5)
                        aspect_scores["color (unvalidated)"] = raw_aspects.get("color (unvalidated)", 0.5)
                        aspect_scores["technical (unvalidated)"] = raw_aspects.get("quality (unvalidated)", 0.5)
                        aspect_scores["sharpness (unvalidated)"] = raw_aspects.get("dof (unvalidated)", 0.5)
                        break
                except Exception as exc:
                    logger.debug("Failed extracting rsinema heads: %s", exc)

        # 2. Fall back to critic aspect scores if rsinema heads absent
        if not found_rsinema:
            critique = critic.critique(image_path)
            for k, v in critique.aspect_scores.items():
                aspect_scores[f"{k} (unvalidated)"] = round(float(v), 4)

        return aspect_scores, overall_score

    def run(
        self,
        image_input: Union[str, Path, Image.Image],
        user_goal: Optional[str] = None,
        step_callback: Optional[Callable[[AgentStep], None]] = None,
    ) -> AgentTrajectory:
        """Executes multi-step closed-loop agent trajectory on input photograph."""
        self._set_deterministic_seed()
        critic, ensemble = self._ensure_components()

        # Materialize working image to temp directory
        temp_dir = tempfile.mkdtemp(prefix="aesfix_agent_")
        orig_temp_path = os.path.join(temp_dir, "initial.jpg")

        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                current_image = img.convert("RGB")
        elif isinstance(image_input, Image.Image):
            current_image = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        current_image.save(orig_temp_path, format="JPEG", quality=95)

        # Compute noise floor on initial image
        noise_floor = round(compute_noise_floor(orig_temp_path, ensemble), 4)

        # Initial baseline perception
        current_aspects, current_overall = self._extract_aspect_scores(orig_temp_path, ensemble, critic)
        initial_overall_score = current_overall

        trajectory = AgentTrajectory(
            initial_image_path=orig_temp_path,
            final_image=current_image.copy(),
            stacked_xmp="",
            verdict="leave it",
            initial_overall_score=initial_overall_score,
            final_overall_score=initial_overall_score,
            total_gain=0.0,
            noise_floor=noise_floor,
            provider=self.llm_client.provider,
            model_id=self.llm_client.model_id,
            seed=self.seed,
        )

        accepted_fixes: List[CriticFix] = []
        consecutive_rejections = 0
        system_prompt = generate_tool_schema_prompt()
        dialogue_history: List[str] = []

        for step_idx in range(1, self.max_steps + 1):
            if consecutive_rejections >= self.consecutive_rejections_stop:
                logger.info(
                    "Stopping agent loop at step %d: reached %d consecutive rejections.",
                    step_idx,
                    self.consecutive_rejections_stop,
                )
                break

            current_temp_path = os.path.join(temp_dir, f"step_{step_idx}_in.jpg")
            current_image.save(current_temp_path, format="JPEG", quality=95)

            # Construct perception prompt with current measurements
            perception_prompt = (
                f"Step {step_idx}/{self.max_steps}.\n"
                f"Current Baseline Aesthetic Score: {current_overall:.4f} (Noise Floor ε={noise_floor:.4f}).\n"
                f"Current Aspect Scores: {json.dumps(current_aspects, indent=2)}\n"
                f"Goal: {user_goal or 'Improve overall photographic aesthetic balance without identity drift.'}\n"
            )

            if dialogue_history:
                perception_prompt += "\nRecent Attempt Feedback:\n" + "\n".join(dialogue_history[-3:])

            perception_prompt += "\nPropose your next single parametric tool call in valid JSON:"

            # Plan: Generate candidate proposal via LLMClient
            llm_resp = self.llm_client.generate(
                prompt=perception_prompt,
                image_input=current_temp_path,
                system_instruction=system_prompt,
            )

            # Validate Proposal with Pydantic
            proposal, val_err = validate_proposal(llm_resp.content)
            if val_err or proposal is None:
                # Invalid schema: feed error back and count as rejection
                err_feedback = format_validation_error_feedback(val_err or "Unknown error", llm_resp.content)
                dialogue_history.append(f"Model proposal rejected by schema validator: {val_err}")

                # Retry once immediately with explicit feedback
                retry_resp = self.llm_client.generate(
                    prompt=err_feedback,
                    image_input=current_temp_path,
                    system_instruction=system_prompt,
                )
                proposal, val_err = validate_proposal(retry_resp.content)

            if proposal is None:
                # If still invalid, record step as rejected and increment consecutive rejections
                consecutive_rejections += 1
                step_record = AgentStep(
                    step_number=step_idx,
                    proposal=AgentProposal(
                        op="adjust_exposure",
                        params={"ev_delta": 0.0},
                        aspect="lighting",
                        direction=1,
                        rationale=f"Invalid schema from model: {val_err}",
                    ),
                    status="rejected",
                    verdict_reason=f"Pydantic schema validation error: {val_err}",
                    baseline_overall_score=current_overall,
                    post_overall_score=current_overall,
                    overall_delta=0.0,
                    noise_floor=noise_floor,
                    claimed_aspect="lighting",
                    claimed_aspect_delta=0.0,
                    claimed_aspect_direction=1,
                    aspect_scores_before=current_aspects,
                    aspect_scores_after=current_aspects,
                    aspect_deltas={},
                    guard_drift=0.0,
                    guard_passed=False,
                    guard_backend="SSIM + rgb_layout_stat",
                    execution_duration_ms=0.0,
                    xmp_xml="",
                )
                trajectory.steps.append(step_record)
                trajectory.rejected_steps.append(step_record)
                if step_callback:
                    step_callback(step_record)
                continue

            # Act: Execute tool call via TierAExecutor
            t_exec_start = time.perf_counter()
            critic_fix = CriticFix(
                op=proposal.op,
                params=proposal.params,
                expected_aspect=proposal.aspect,
                nl_instruction=proposal.rationale,
            )

            exec_res = self.executor.execute_fixes(
                image_input=current_image,
                fixes=[critic_fix],
            )
            exec_time_ms = round((time.perf_counter() - t_exec_start) * 1000.0, 2)

            candidate_temp_path = os.path.join(temp_dir, f"step_{step_idx}_candidate.jpg")
            exec_res.image.save(candidate_temp_path, format="JPEG", quality=95)

            # Verify: Evaluate Identity Guard drift
            guard_verdict = self.guard.evaluate(
                original=orig_temp_path,
                edited=candidate_temp_path,
            )

            # Verify: Measure post-edit scores
            post_aspects, post_overall = self._extract_aspect_scores(
                candidate_temp_path,
                ensemble,
                critic,
            )

            overall_delta = round(post_overall - current_overall, 4)

            # Calculate per-aspect movements
            aspect_deltas: Dict[str, float] = {}
            for k in current_aspects:
                aspect_deltas[k] = round(post_aspects.get(k, 0.0) - current_aspects[k], 4)

            # Map claimed aspect to score keys
            claimed_norm = proposal.aspect.lower()
            claimed_key = f"{claimed_norm} (unvalidated)"
            claimed_delta = aspect_deltas.get(claimed_key, 0.0)

            # --- Evaluation of Acceptance Criteria ---
            # Aspect noise floor default matches overall noise floor
            aspect_noise_floor = noise_floor

            is_accepted = True
            rejection_reasons: List[str] = []

            # 1. Identity Guard passes
            if not guard_verdict.is_valid:
                is_accepted = False
                rejection_reasons.append(
                    f"Identity Guard rejected: drift {guard_verdict.composite_drift:.4f} > {guard_verdict.threshold:.4f} "
                    f"({guard_verdict.rejection_reason or 'subject altered'})"
                )

            # 2. Overall Judge Δ > ε
            if overall_delta <= noise_floor:
                is_accepted = False
                rejection_reasons.append(
                    f"Overall gain Δ={overall_delta:+.4f} does not exceed noise floor ε={noise_floor:.4f}"
                )

            # 3. Claimed aspect moves in claimed direction by > noise floor
            if proposal.direction > 0 and claimed_delta <= aspect_noise_floor:
                is_accepted = False
                rejection_reasons.append(
                    f"Claimed aspect '{proposal.aspect}' movement {claimed_delta:+.4f} did not exceed noise floor {aspect_noise_floor:.4f}"
                )
            elif proposal.direction < 0 and (-claimed_delta) <= aspect_noise_floor:
                is_accepted = False
                rejection_reasons.append(
                    f"Claimed aspect '{proposal.aspect}' inverse movement {claimed_delta:+.4f} did not exceed noise floor {aspect_noise_floor:.4f}"
                )

            # 4. No other aspect drops beyond its noise floor
            for a_key, a_delta in aspect_deltas.items():
                if a_key == claimed_key:
                    continue
                if a_delta < -aspect_noise_floor:
                    is_accepted = False
                    rejection_reasons.append(
                        f"Aspect '{a_key.replace(' (unvalidated)', '')}' regressed by {a_delta:+.4f}, exceeding noise floor {aspect_noise_floor:.4f}"
                    )
                    break

            # Construct verdict reason
            if is_accepted:
                verdict_reason = (
                    f"Accepted: Δ={overall_delta:+.4f} > ε={noise_floor:.4f}, "
                    f"claimed aspect '{proposal.aspect}' improved ({claimed_delta:+.4f}), "
                    f"other aspects preserved, guard passed (drift={guard_verdict.composite_drift:.4f})."
                )
            else:
                verdict_reason = "; ".join(rejection_reasons)

            step_record = AgentStep(
                step_number=step_idx,
                proposal=proposal,
                status="accepted" if is_accepted else "rejected",
                verdict_reason=verdict_reason,
                baseline_overall_score=current_overall,
                post_overall_score=post_overall,
                overall_delta=overall_delta,
                noise_floor=noise_floor,
                claimed_aspect=proposal.aspect,
                claimed_aspect_delta=claimed_delta,
                claimed_aspect_direction=proposal.direction,
                aspect_scores_before=current_aspects,
                aspect_scores_after=post_aspects,
                aspect_deltas=aspect_deltas,
                guard_drift=guard_verdict.composite_drift,
                guard_passed=guard_verdict.is_valid,
                guard_backend=getattr(guard_verdict, "guard_backend", f"SSIM + {guard_verdict.backend_name}"),
                execution_duration_ms=exec_time_ms,
                xmp_xml=exec_res.xmp_sidecar,
                preview_image=exec_res.image.copy(),
            )

            trajectory.steps.append(step_record)

            if is_accepted:
                # Accept step: advance state
                current_image = exec_res.image.copy()
                current_overall = post_overall
                current_aspects = post_aspects
                accepted_fixes.append(critic_fix)
                trajectory.accepted_steps.append(step_record)
                consecutive_rejections = 0
                dialogue_history.append(f"Step {step_idx}: ACCEPTED {proposal.op} ({proposal.rationale})")
            else:
                # Reject step: retain current image, increment consecutive rejections
                trajectory.rejected_steps.append(step_record)
                consecutive_rejections += 1
                dialogue_history.append(
                    f"Step {step_idx}: REJECTED {proposal.op}. Reason: {verdict_reason}"
                )

            if step_callback:
                step_callback(step_record)

        # Assemble final trajectory outputs
        trajectory.total_steps = len(trajectory.steps)
        trajectory.consecutive_rejections = consecutive_rejections
        trajectory.final_image = current_image
        trajectory.final_overall_score = current_overall
        trajectory.total_gain = round(current_overall - initial_overall_score, 4)

        if accepted_fixes:
            trajectory.verdict = "measured improvement"
            trajectory.stacked_xmp = XMPWriter.generate_xmp(accepted_fixes)
        else:
            trajectory.verdict = "leave it"
            trajectory.stacked_xmp = ""

        # Clean up temp files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        return trajectory
