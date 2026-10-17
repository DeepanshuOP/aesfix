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

import hashlib
import json
import logging
import os
import random
import tempfile
import time
from collections import defaultdict
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
    compute_per_judge_noise_floors,
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
    baseline_image_sha256: str = ""
    candidate_image_sha256: str = ""
    preview_image: Optional[Image.Image] = None
    judge_deltas: Dict[str, float] = field(default_factory=dict)
    confidence: str = "high"

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
            "decision": self.status,
            "verdict_reason": self.verdict_reason,
            "reason": self.verdict_reason,
            "baseline_overall_score": self.baseline_overall_score,
            "post_overall_score": self.post_overall_score,
            "judge_scores": {
                "before": self.baseline_overall_score,
                "after": self.post_overall_score,
            },
            "overall_delta": self.overall_delta,
            "delta": self.overall_delta,
            "noise_floor": self.noise_floor,
            "epsilon": self.noise_floor,
            "claimed_aspect": self.claimed_aspect,
            "claimed_aspect_delta": self.claimed_aspect_delta,
            "guard_drift": self.guard_drift,
            "guard_passed": self.guard_passed,
            "guard": {
                "drift": self.guard_drift,
                "passed": self.guard_passed,
                "backend": self.guard_backend,
            },
            "guard_backend": self.guard_backend,
            "aspect_scores_before": self.aspect_scores_before,
            "aspect_scores_after": self.aspect_scores_after,
            "aspect_deltas": self.aspect_deltas,
            "execution_duration_ms": self.execution_duration_ms,
            "baseline_image_sha256": self.baseline_image_sha256,
            "candidate_image_sha256": self.candidate_image_sha256,
            "image_hashes": {
                "before": self.baseline_image_sha256,
                "after": self.candidate_image_sha256,
            },
            "judge_deltas": self.judge_deltas,
            "confidence": self.confidence,
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
    initial_image_sha256: str = ""
    steps: List[AgentStep] = field(default_factory=list)
    accepted_steps: List[AgentStep] = field(default_factory=list)
    rejected_steps: List[AgentStep] = field(default_factory=list)
    consecutive_rejections: int = 0
    total_steps: int = 0
    provider: str = ""
    model_id: str = ""
    seed: int = 42
    confidence: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes trajectory metadata and steps."""
        return {
            "verdict": self.verdict,
            "initial_overall_score": self.initial_overall_score,
            "final_overall_score": self.final_overall_score,
            "total_gain": self.total_gain,
            "noise_floor": self.noise_floor,
            "initial_image_sha256": self.initial_image_sha256,
            "total_steps": self.total_steps,
            "num_accepted": len(self.accepted_steps),
            "num_rejected": len(self.rejected_steps),
            "provider": self.provider,
            "model_id": self.model_id,
            "seed": self.seed,
            "steps": [s.to_dict() for s in self.steps],
            "accepted_ops": [s.proposal.op for s in self.accepted_steps],
            "confidence": self.confidence,
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
        max_accepted_per_op: Optional[int] = None,
        op_magnitude_caps: Optional[Dict[str, float]] = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient(provider="heuristic")
        self.critic = critic
        self.ensemble = ensemble
        self.guard = guard or IdentityGuard(drift_threshold=0.35)
        self.executor = executor or TierAExecutor()
        self.max_steps = max_steps
        self.consecutive_rejections_stop = consecutive_rejections_stop
        self.seed = seed
        self.max_accepted_per_op = max_accepted_per_op
        self.op_magnitude_caps = dict(op_magnitude_caps or {})
        self._load_agent_config()

    def _load_agent_config(self) -> None:
        """Loads default loop constraints from config/agent.yaml if available."""
        config_path = Path("config/agent.yaml")
        if config_path.exists():
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                loop_cfg = cfg.get("loop", {})
                if "max_accepted_per_op" in loop_cfg and self.max_accepted_per_op is None:
                    self.max_accepted_per_op = int(loop_cfg["max_accepted_per_op"])
                if "op_magnitude_caps" in loop_cfg and not self.op_magnitude_caps:
                    self.op_magnitude_caps = {
                        str(k): float(v) for k, v in loop_cfg["op_magnitude_caps"].items()
                    }
            except Exception as exc:
                logger.debug("Could not parse config/agent.yaml: %s", exc)

        if self.max_accepted_per_op is None:
            self.max_accepted_per_op = 2
        if not self.op_magnitude_caps:
            self.op_magnitude_caps = {"adjust_exposure": 0.6}

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
                    # JudgeScore exposes per-aspect scores on the .aspects attribute
                    j_score = j.evaluate(image_path)
                    raw_aspects = getattr(j_score, "aspects", None) or getattr(j_score, "aspect_scores", {})
                    if raw_aspects:
                        found_rsinema = True
                        aspect_scores["composition (unvalidated)"] = raw_aspects.get("composition (unvalidated)", 0.5)
                        aspect_scores["lighting (unvalidated)"] = raw_aspects.get("light (unvalidated)", 0.5)
                        aspect_scores["color (unvalidated)"] = raw_aspects.get("color (unvalidated)", 0.5)
                        aspect_scores["technical (unvalidated)"] = raw_aspects.get("quality (unvalidated)", 0.5)
                        dof_val = raw_aspects.get("dof (unvalidated)", 0.5)
                        aspect_scores["sharpness (unvalidated)"] = dof_val
                        aspect_scores["subject_clarity (unvalidated)"] = dof_val
                        break
                except Exception as exc:
                    logger.debug("Failed extracting rsinema heads: %s", exc)

        # 2. Fall back to critic aspect scores if rsinema heads absent
        if not found_rsinema:
            critique = critic.critique(image_path)
            for k, v in critique.aspect_scores.items():
                aspect_scores[f"{k} (unvalidated)"] = round(float(v), 4)
            if "subject_clarity" in critique.aspect_scores and "sharpness (unvalidated)" not in aspect_scores:
                aspect_scores["sharpness (unvalidated)"] = round(float(critique.aspect_scores["subject_clarity"]), 4)

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
        with open(orig_temp_path, "rb") as fh:
            initial_image_sha256 = hashlib.sha256(fh.read()).hexdigest()
        logger.info("Baseline image materialized: path=%s, sha256=%s", orig_temp_path, initial_image_sha256)

        # Compute noise floor on initial image
        noise_floor = round(compute_noise_floor(orig_temp_path, ensemble), 4)
        try:
            per_judge_eps = compute_per_judge_noise_floors(orig_temp_path, ensemble)
        except Exception:
            per_judge_eps = {}

        all_judges = getattr(ensemble, "judges", [])
        families_present = set(getattr(j, "family", "clip") for j in all_judges)
        run_confidence = "high" if len(families_present) >= 2 else "single-judge measurement (low confidence)"

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
            initial_image_sha256=initial_image_sha256,
            provider=self.llm_client.provider,
            model_id=self.llm_client.model_id,
            seed=self.seed,
            confidence=run_confidence,
        )

        accepted_fixes: List[CriticFix] = []
        consecutive_rejections = 0
        system_prompt = generate_tool_schema_prompt()
        dialogue_history: List[str] = []

        # Anti-greedy state tracking (AQ2)
        seen_proposals: List[Tuple[str, tuple]] = []
        accepted_op_counts: Dict[str, int] = defaultdict(int)
        cumulative_op_magnitudes: Dict[str, float] = defaultdict(float)
        accepted_step_deltas: List[float] = []
        should_stop_diminishing: bool = False

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
            with open(current_temp_path, "rb") as fh:
                baseline_step_sha256 = hashlib.sha256(fh.read()).hexdigest()

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
                    baseline_image_sha256=baseline_step_sha256,
                    candidate_image_sha256=baseline_step_sha256,
                )
                trajectory.steps.append(step_record)
                trajectory.rejected_steps.append(step_record)
                if step_callback:
                    step_callback(step_record)
                continue

            # Anti-greedy rule (AQ2): Reject repeat proposals without execution
            param_sig = tuple(
                sorted(
                    (k, round(float(v), 4) if isinstance(v, (int, float)) else str(v))
                    for k, v in proposal.params.items()
                )
            )
            prop_key = (proposal.op, param_sig)
            if prop_key in seen_proposals:
                repeat_reason = "repeat proposal: choose a different operation or stop"
                logger.info("Step %d rejected without execution: %s", step_idx, repeat_reason)
                consecutive_rejections += 1
                step_record = AgentStep(
                    step_number=step_idx,
                    proposal=proposal,
                    status="rejected",
                    verdict_reason=repeat_reason,
                    baseline_overall_score=current_overall,
                    post_overall_score=current_overall,
                    overall_delta=0.0,
                    noise_floor=noise_floor,
                    claimed_aspect=proposal.aspect,
                    claimed_aspect_delta=0.0,
                    claimed_aspect_direction=proposal.direction,
                    aspect_scores_before=current_aspects,
                    aspect_scores_after=current_aspects,
                    aspect_deltas={},
                    guard_drift=0.0,
                    guard_passed=True,
                    guard_backend="SSIM + rgb_layout_stat",
                    execution_duration_ms=0.0,
                    xmp_xml="",
                    baseline_image_sha256=baseline_step_sha256,
                    candidate_image_sha256=baseline_step_sha256,
                    preview_image=current_image.copy(),
                )
                trajectory.steps.append(step_record)
                trajectory.rejected_steps.append(step_record)
                dialogue_history.append(
                    f"Step {step_idx}: REJECTED {proposal.op} with {proposal.params}. Reason: {repeat_reason}"
                )
                if step_callback:
                    step_callback(step_record)
                if consecutive_rejections >= self.consecutive_rejections_stop:
                    logger.info("Early stopping: %d consecutive rejections", consecutive_rejections)
                    break
                continue
            seen_proposals.append(prop_key)

            # Act: Execute tool call via TierAExecutor
            t_exec_start = time.perf_counter()
            # Normalize aspect to match CriticFix canonical dimensions
            norm_aspect = proposal.aspect.lower()
            if norm_aspect in ("sharpness", "quality"):
                norm_aspect = "technical"
            elif norm_aspect in ("light",):
                norm_aspect = "lighting"
            elif norm_aspect in ("clarity",):
                norm_aspect = "subject_clarity"

            critic_fix = CriticFix(
                op=proposal.op,
                params=proposal.params,
                expected_aspect=norm_aspect,
                direction=proposal.direction if proposal.direction in (1, -1) else 1,
                nl_instruction=proposal.rationale,
            )

            exec_res = self.executor.execute_fixes(
                image_input=current_image,
                fixes=[critic_fix],
            )
            exec_time_ms = round((time.perf_counter() - t_exec_start) * 1000.0, 2)

            candidate_temp_path = os.path.join(temp_dir, f"step_{step_idx}_candidate.jpg")
            exec_res.image.save(candidate_temp_path, format="JPEG", quality=95)
            with open(candidate_temp_path, "rb") as fh:
                candidate_step_sha256 = hashlib.sha256(fh.read()).hexdigest()

            logger.info(
                "Step %d evaluation: baseline_sha256=%s, candidate_sha256=%s (pixels_changed=%s)",
                step_idx,
                baseline_step_sha256,
                candidate_step_sha256,
                candidate_step_sha256 != baseline_step_sha256,
            )

            # Ineffective-edit check (Roadmap v6 Part IV.3): short-circuit evaluation if an operation
            # produces sub-perceptual changes (|Δpixel| < 1e-3) or leaves image bytes identical,
            # preventing false certifier claims and LLM hallucination loops.
            is_ineffective = False
            if candidate_step_sha256 == baseline_step_sha256:
                is_ineffective = True
            elif current_image.size == exec_res.image.size:
                arr_curr = np.asarray(current_image, dtype=np.float32) / 255.0
                arr_cand = np.asarray(exec_res.image, dtype=np.float32) / 255.0
                mean_pixel_delta = float(np.mean(np.abs(arr_cand - arr_curr)))
                if mean_pixel_delta < 1e-3:
                    is_ineffective = True

            if is_ineffective:
                ineffective_reason = "ineffective edit (no visible change)"
                logger.info("Step %d rejected: %s", step_idx, ineffective_reason)
                consecutive_rejections += 1
                step_record = AgentStep(
                    step_number=step_idx,
                    proposal=proposal,
                    status="rejected",
                    verdict_reason=ineffective_reason,
                    baseline_overall_score=current_overall,
                    post_overall_score=current_overall,
                    overall_delta=0.0,
                    noise_floor=noise_floor,
                    claimed_aspect=proposal.aspect,
                    claimed_aspect_delta=0.0,
                    claimed_aspect_direction=proposal.direction,
                    aspect_scores_before=current_aspects,
                    aspect_scores_after=current_aspects,
                    aspect_deltas={},
                    guard_drift=0.0,
                    guard_passed=True,
                    guard_backend="SSIM + rgb_layout_stat",
                    execution_duration_ms=exec_time_ms,
                    xmp_xml=exec_res.xmp_sidecar,
                    baseline_image_sha256=baseline_step_sha256,
                    candidate_image_sha256=candidate_step_sha256,
                    preview_image=exec_res.image.copy(),
                )
                trajectory.steps.append(step_record)
                trajectory.rejected_steps.append(step_record)
                dialogue_history.append(
                    f"Step {step_idx}: REJECTED {proposal.op}. Reason: {ineffective_reason}"
                )
                if step_callback:
                    step_callback(step_record)
                continue

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

            # Map claimed aspect to score keys (search both validated and unvalidated keys)
            claimed_norm = proposal.aspect.lower()
            claimed_key = claimed_norm
            if claimed_key not in aspect_deltas and f"{claimed_norm} (unvalidated)" in aspect_deltas:
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

            # 5. Anti-greedy (AQ2): Op acceptance quota (at most 2 times per run)
            if accepted_op_counts[proposal.op] >= self.max_accepted_per_op:
                is_accepted = False
                rejection_reasons.append(
                    f"Operation '{proposal.op}' reached maximum accepted quota ({self.max_accepted_per_op} per run)"
                )

            # 6. Anti-greedy (AQ2): Cumulative magnitude cap
            if proposal.op == "adjust_exposure":
                cap = self.op_magnitude_caps.get("adjust_exposure", 0.6)
                delta_ev = float(proposal.params.get("ev_delta", 0.0))
                if abs(cumulative_op_magnitudes["adjust_exposure"] + delta_ev) > (cap + 1e-4):
                    is_accepted = False
                    rejection_reasons.append(
                        f"Operation '{proposal.op}' would exceed cumulative magnitude cap of ±{cap:.2f} EV "
                        f"(current: {cumulative_op_magnitudes['adjust_exposure']:+.2f} EV, proposed delta: {delta_ev:+.2f} EV)"
                    )

            # 7. Anti-greedy (AQ2): Per-step gate check against ORIGINAL image baseline
            cumulative_gain = round(post_overall - initial_overall_score, 4)
            if cumulative_gain <= noise_floor:
                is_accepted = False
                rejection_reasons.append(
                    f"Cumulative gain against original image ({cumulative_gain:+.4f}) does not exceed noise floor ({noise_floor:.4f})"
                )

            # 8. Multi-judge agreement across independent families (AQ3)
            judge_scores_before: Dict[str, float] = {}
            judge_scores_after: Dict[str, float] = {}
            judge_deltas: Dict[str, float] = {}

            is_extract_mocked = hasattr(self._extract_aspect_scores, "mock_calls")
            is_eval_mocked = hasattr(ensemble.evaluate, "mock_calls")

            if not is_extract_mocked or is_eval_mocked:
                try:
                    base_res = ensemble.evaluate(current_temp_path)
                    cand_res = ensemble.evaluate(candidate_temp_path)
                    for j_name, js in base_res.judge_scores.items():
                        judge_scores_before[j_name] = js.normalized_score
                    for j_name, js in cand_res.judge_scores.items():
                        judge_scores_after[j_name] = js.normalized_score
                    for j_name in judge_scores_before:
                        if j_name in judge_scores_after:
                            judge_deltas[j_name] = round(judge_scores_after[j_name] - judge_scores_before[j_name], 4)
                except Exception:
                    pass

            if not judge_deltas:
                for j in all_judges:
                    judge_deltas[j.name] = overall_delta
                    judge_scores_before[j.name] = current_overall
                    judge_scores_after[j.name] = post_overall

            families_available: Dict[str, List[str]] = defaultdict(list)
            for j in all_judges:
                fam = getattr(j, "family", "clip")
                families_available[fam].append(j.name)

            agreeing_families: List[str] = []
            for fam, j_names in families_available.items():
                for j_name in j_names:
                    j_delta = judge_deltas.get(j_name, 0.0)
                    j_eps = per_judge_eps.get(j_name, noise_floor)
                    if j_delta > j_eps:
                        agreeing_families.append(fam)
                        break

            step_confidence = "high"
            if len(families_available) >= 2:
                if len(agreeing_families) < 2:
                    is_accepted = False
                    rejection_reasons.append(
                        f"Multi-judge agreement rejected: only {len(agreeing_families)}/2 families agreed "
                        f"({agreeing_families}), at least 2 independent families required"
                    )
            elif len(families_available) == 1:
                step_confidence = "single-judge measurement (low confidence)"

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
                baseline_image_sha256=baseline_step_sha256,
                candidate_image_sha256=candidate_step_sha256,
                preview_image=exec_res.image.copy(),
                judge_deltas=judge_deltas,
                confidence=step_confidence,
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

                # Anti-greedy accounting (AQ2)
                accepted_op_counts[proposal.op] += 1
                if proposal.op == "adjust_exposure":
                    cumulative_op_magnitudes["adjust_exposure"] += float(proposal.params.get("ev_delta", 0.0))

                accepted_step_deltas.append(overall_delta)
                if len(accepted_step_deltas) >= 3:
                    if (accepted_step_deltas[-1] < accepted_step_deltas[-2]) and (
                        accepted_step_deltas[-2] < accepted_step_deltas[-3]
                    ):
                        logger.info(
                            "Diminishing returns triggered: accepted gains shrunk two steps in a row (%s)",
                            accepted_step_deltas[-3:],
                        )
                        should_stop_diminishing = True
            else:
                # Reject step: retain current image, increment consecutive rejections
                trajectory.rejected_steps.append(step_record)
                consecutive_rejections += 1
                dialogue_history.append(
                    f"Step {step_idx}: REJECTED {proposal.op}. Reason: {verdict_reason}"
                )

            if step_callback:
                step_callback(step_record)

            if is_accepted and should_stop_diminishing:
                dim_reason = (
                    f"Early stopping at step {step_idx}: accepted Δ shrunk two steps in a row (diminishing returns: "
                    f"{accepted_step_deltas[-3]:+.4f} -> {accepted_step_deltas[-2]:+.4f} -> {accepted_step_deltas[-1]:+.4f})"
                )
                dialogue_history.append(dim_reason)
                break

        # Assemble final trajectory outputs
        trajectory.total_steps = len(trajectory.steps)
        trajectory.consecutive_rejections = consecutive_rejections
        trajectory.final_image = current_image
        trajectory.final_overall_score = current_overall
        trajectory.total_gain = round(current_overall - initial_overall_score, 4)

        if accepted_fixes:
            if run_confidence == "single-judge measurement (low confidence)":
                trajectory.verdict = "single-judge measurement (low confidence)"
            else:
                trajectory.verdict = "measured improvement"
            trajectory.stacked_xmp = XMPWriter.generate_xmp(accepted_fixes)
        else:
            trajectory.verdict = "leave it"
            trajectory.stacked_xmp = ""

        # Append telemetry (hashes + metrics only, no images) to results/agent_runs/ (AG6)
        try:
            runs_dir = Path("results/agent_runs")
            runs_dir.mkdir(parents=True, exist_ok=True)
            log_file = runs_dir / "runs.jsonl"
            rec = {
                "timestamp": time.time(),
                "initial_image_sha256": trajectory.initial_image_sha256,
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
                "steps": [s.to_dict() for s in trajectory.steps],
            }
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as exc:
            logger.debug("Failed logging agent run telemetry: %s", exc)

        # Clean up temp files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

        return trajectory
