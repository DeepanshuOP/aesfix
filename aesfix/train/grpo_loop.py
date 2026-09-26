"""Edit-Verified GRPO Execution Loop & Held-Out AG@k Evaluation (AF-3.3 & Paper §12-C2).

Why the Edit-Verified Execution Loop is unique:
1. Closed-Loop Visual RL: Standard vision-language RL evaluates model responses through
   static text reward models or human preference classifiers. AesFix executes candidate
   fixes on the actual input image through the deterministic Tier-A executor and measures
   empirical aesthetic delta verified by the frozen judge ensemble.
2. Anti-Hacking Guard Interlocking: Every executed candidate is intercepted by the
   Identity Guard. Any candidate that drifts beyond the structural threshold is vetoed,
   immediately zeroing aesthetic gain and recording an alert on the anti-hacking dashboard.
3. Held-Out AG@k Demonstration (Roadmap §40 DoD): Validates that reinforcement learning
   improves verified downstream execution quality (AG@3 beats SFT baseline by a significant
   margin on held-out photographs while maintaining a clean anti-hacking dashboard).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image

from aesfix.critic.critic import parse_critic_json
from aesfix.executor.guard import IdentityGuard
from aesfix.executor.tier_a import TierAExecutor
from aesfix.train.anti_hacking import AntiHackingMonitor, DashboardSnapshot
from aesfix.train.grpo_reward import (
    GRPORewardEngine,
    compute_group_advantages,
)
from aesfix.train.grpo_trainer import (
    GRPOConfig,
    GRPOLossMetrics,
    compute_grpo_loss,
)

logger = logging.getLogger("aesfix.train.grpo_loop")


@dataclass
class AGkEvaluationResult:
    """Quantitative results measuring Aesthetic Gain at k (AG@k) on held-out data."""
    num_samples: int
    ag_at_1: float
    ag_at_3: float
    guard_pass_rate: float
    schema_validity_rate: float
    mean_drift: float
    beats_baseline: bool
    baseline_ag_at_3: float
    delta_ag_at_3: float

    def to_dict(self) -> Dict[str, Any]:
        """Serializes held-out benchmark evaluation metrics."""
        return {
            "num_samples": self.num_samples,
            "ag_at_1": round(self.ag_at_1, 4),
            "ag_at_3": round(self.ag_at_3, 4),
            "guard_pass_rate": round(self.guard_pass_rate, 4),
            "schema_validity_rate": round(self.schema_validity_rate, 4),
            "mean_drift": round(self.mean_drift, 4),
            "beats_baseline": self.beats_baseline,
            "baseline_ag_at_3": round(self.baseline_ag_at_3, 4),
            "delta_ag_at_3": round(self.delta_ag_at_3, 4),
        }


@dataclass
class GRPOTrainingStepReport:
    """Summary of a single training step across rollouts, losses, and anti-hacking snapshot."""
    step: int
    loss_metrics: GRPOLossMetrics
    dashboard_snapshot: DashboardSnapshot
    mean_step_reward: float
    num_rollouts: int


def evaluate_held_out_ag(
    policy_rollout_fn: Callable[[Image.Image, int], List[str]],
    held_out_images: Sequence[Image.Image],
    executor: Optional[TierAExecutor] = None,
    judge_fn: Optional[Callable[[Union[Image.Image, str]], float]] = None,
    guard: Optional[IdentityGuard] = None,
    k: int = 3,
    baseline_ag_at_3: float = 0.045,
) -> AGkEvaluationResult:
    """Computes held-out AG@1 and AG@k to fulfill the AF-3.3 Definition of Done.

    Why AG@k is the gold-standard metric:
    In real-world photographic workflows, a critic presents up to k alternative fix plans.
    AG@k measures the maximum verifiable gain achieved among the top k proposals, provided
    the selected fix does not violate identity preservation.
    """
    exec_engine = executor or TierAExecutor()
    guard_engine = guard or IdentityGuard(drift_threshold=0.35)

    if judge_fn is None:
        judge_fn = GRPORewardEngine._default_judge_fn

    ag1_scores: List[float] = []
    agk_scores: List[float] = []
    guard_approvals: List[bool] = []
    schema_valids: List[bool] = []
    drifts: List[float] = []

    for img in held_out_images:
        base_score = judge_fn(img)
        # Sample k candidates from the policy
        candidates = policy_rollout_fn(img, k)

        sample_gains: List[float] = []

        for rank_idx, comp_text in enumerate(candidates):
            try:
                critic_out = parse_critic_json(comp_text)
                schema_valids.append(True)
            except Exception:
                schema_valids.append(False)
                sample_gains.append(0.0)
                continue

            if not critic_out.fixes:
                sample_gains.append(0.0)
                continue

            # Execute top fix of candidate
            fix = critic_out.fixes[0]
            try:
                exec_res = exec_engine.execute_fixes(image_input=img, fixes=[fix])
                edited_img = exec_res.image
                edited_score = judge_fn(edited_img)
            except Exception:
                sample_gains.append(0.0)
                continue

            # Verify through Identity Guard
            verdict = guard_engine.evaluate(original=img, edited=edited_img)
            guard_approvals.append(verdict.is_valid)
            drifts.append(verdict.composite_drift)

            if verdict.is_valid:
                gain = edited_score - base_score
                sample_gains.append(gain)
            else:
                # Identity violated: zero credit
                sample_gains.append(0.0)

            # Record AG@1 from the first candidate
            if rank_idx == 0:
                ag1_scores.append(sample_gains[-1])

        # AG@k is the maximum verified gain achieved across the k candidates
        if sample_gains:
            agk_scores.append(max(sample_gains))
        else:
            agk_scores.append(0.0)

    mean_ag1 = sum(ag1_scores) / len(ag1_scores) if ag1_scores else 0.0
    mean_agk = sum(agk_scores) / len(agk_scores) if agk_scores else 0.0
    guard_pass_rate = (
        sum(1 for a in guard_approvals if a) / len(guard_approvals)
        if guard_approvals
        else 1.0
    )
    schema_valid_rate = (
        sum(1 for s in schema_valids if s) / len(schema_valids)
        if schema_valids
        else 0.0
    )
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0

    delta_ag = mean_agk - baseline_ag_at_3
    beats_baseline = delta_ag > 0.01  # Significant margin criterion

    return AGkEvaluationResult(
        num_samples=len(held_out_images),
        ag_at_1=mean_ag1,
        ag_at_3=mean_agk,
        guard_pass_rate=guard_pass_rate,
        schema_validity_rate=schema_valid_rate,
        mean_drift=mean_drift,
        beats_baseline=beats_baseline,
        baseline_ag_at_3=baseline_ag_at_3,
        delta_ag_at_3=delta_ag,
    )


class EditVerifiedGRPOLoop:
    """Full execution engine driving GRPO training steps and anti-hacking monitoring."""

    def __init__(
        self,
        config: GRPOConfig,
        reward_engine: Optional[GRPORewardEngine] = None,
        monitor: Optional[AntiHackingMonitor] = None,
    ) -> None:
        self.config = config
        self.reward_engine = reward_engine or GRPORewardEngine()
        self.monitor = monitor or AntiHackingMonitor()
        self.current_step = 0

    def step(
        self,
        image: Image.Image,
        candidate_completions: Sequence[str],
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        judge_disagreements: Optional[Sequence[float]] = None,
        image_id: Optional[str] = None,
    ) -> GRPOTrainingStepReport:
        """Executes a single end-to-end GRPO step with verified rewards and dashboard logging."""
        self.current_step += 1

        # 1. Evaluate rollouts through multi-objective reward engine
        breakdowns = self.reward_engine.evaluate_group(
            image=image,
            completions=candidate_completions,
            image_id=image_id,
        )
        rewards = [b.total_reward for b in breakdowns]

        # 2. Compute group relative advantages
        advantages = compute_group_advantages(rewards)

        # 3. Compute clipped policy gradient loss with reference KL penalty
        loss, loss_metrics = compute_grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            clip_range=self.config.clip_range,
            kl_coeff=self.config.kl_coeff,
        )

        # 4. Log diagnostics to Anti-Hacking Monitor
        dashboard_snapshot = self.monitor.record_step(
            step=self.current_step,
            breakdowns=breakdowns,
            judge_disagreements=judge_disagreements,
        )

        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0

        return GRPOTrainingStepReport(
            step=self.current_step,
            loss_metrics=loss_metrics,
            dashboard_snapshot=dashboard_snapshot,
            mean_step_reward=round(mean_reward, 4),
            num_rollouts=len(candidate_completions),
        )
