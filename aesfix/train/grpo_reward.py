"""Group Relative Policy Optimization (GRPO) Reward Engine (AF-3.2 & Paper §12-C2).

Why Group Relative Policy Optimization (GRPO) over PPO:
1. Eliminates the Critic Value Model: In standard PPO, training a 7B parameter value model
   $V_\\phi(s)$ alongside the policy $\\pi_\\theta$ doubles GPU VRAM requirements. GRPO samples
   a group of $G$ candidate critique completions for each input photo and normalizes advantages
   directly against the group empirical distribution:
   $$A_i = \\frac{R_i - \\text{mean}(\\{R_j\\})}{\\text{std}(\\{R_j\\}) + \\epsilon}$$
2. Multi-Objective Grounding with Falsifiable Rewards: Unlike text-only RL where reward models
   are vulnerable to sycophancy or hallucination, AesFix executes proposed fixes through the
   Tier-A deterministic executor and measures true visual delta ($\\Delta \\text{AG}@1$) verified
   by the frozen judge ensemble.
3. Anti-Reward Hacking Gate: Coupled with the Identity Guard to immediately penalize and zero out
   aesthetic gain if perceptual drift exceeds the structural preservation threshold.
4. Tier-A Precompute & Cache (§12-C2): Repeated or discretized edit parameters on the same image
   hit an in-memory cache, reducing judge and tensor execution overhead by up to 10x during rollouts.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from aesfix.critic.critic import CriticOutput, parse_critic_json
from aesfix.executor.guard import IdentityGuard
from aesfix.executor.tier_a import TierAExecutor

logger = logging.getLogger("aesfix.train.grpo")


@dataclass
class RewardWeights:
    """Configurable weights for the multi-objective GRPO aesthetic reward function.

    Why these default weights:
    - schema_weight (1.0): Valid JSON adhering to CriticOutput is a hard prerequisite for execution.
    - ag_weight (5.0): Scales aesthetic gain (typically in [-0.10, +0.15]) into the primary gradient driver.
    - drift_penalty_weight (2.5): Harsh penalty when edits alter subject identity or cause pixel artifacts.
    - redundancy_penalty_weight (0.5): Discourages repeating identical operations in the fix plan.
    """
    schema_weight: float = 1.0
    ag_weight: float = 5.0
    drift_penalty_weight: float = 2.5
    redundancy_penalty_weight: float = 0.5


@dataclass
class RewardBreakdown:
    """Detailed diagnostic breakdown of a candidate critique completion's reward.

    Why this dataclass exists:
    Provides transparent auditability during training logging (W&B / TensorBoard)
    to detect reward hacking, schema degradation, or judge drift in real time.
    """
    total_reward: float
    schema_reward: float
    ag_reward: float
    drift_penalty: float
    redundancy_penalty: float
    raw_ag: float
    composite_drift: float
    is_valid_json: bool
    is_guard_approved: bool
    critic_output: Optional[CriticOutput] = None
    applied_op: Optional[str] = None


class TierARewardCache:
    """In-memory cache for executed Tier-A edits and judge evaluations (§12-C2).

    Why reward caching is a high-leverage systems optimization:
    During group rollouts, multiple completions frequently generate identical canonical
    fixes (e.g., `adjust_exposure(ev=0.4)` on the same portrait). Caching the edited
    PIL image and its judge score eliminates redundant tensor operations and neural forward passes.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Image.Image, float]] = {}

    def _hash_key(self, image_id: str, op: str, params: Dict[str, Any]) -> str:
        """Computes a deterministic hash key for image edit combinations."""
        param_str = json.dumps(params, sort_keys=True)
        raw = f"{image_id}:{op}:{param_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, image_id: str, op: str, params: Dict[str, Any]) -> Optional[Tuple[Image.Image, float]]:
        key = self._hash_key(image_id, op, params)
        return self._cache.get(key)

    def set(self, image_id: str, op: str, params: Dict[str, Any], edited_img: Image.Image, score: float) -> None:
        if len(self._cache) >= self.max_size:
            # Simple FIFO eviction to bound memory
            self._cache.pop(next(iter(self._cache)))
        key = self._hash_key(image_id, op, params)
        self._cache[key] = (edited_img, score)

    def clear(self) -> None:
        self._cache.clear()


class GRPORewardEngine:
    """Calculates multi-objective verifiable rewards for vision-language critic rollouts."""

    def __init__(
        self,
        executor: Optional[TierAExecutor] = None,
        guard: Optional[IdentityGuard] = None,
        judge_fn: Optional[Callable[[Union[Image.Image, str]], float]] = None,
        weights: Optional[RewardWeights] = None,
        cache: Optional[TierARewardCache] = None,
    ) -> None:
        self.executor = executor or TierAExecutor()
        self.guard = guard or IdentityGuard(drift_threshold=0.35)
        self.weights = weights or RewardWeights()
        self.cache = cache or TierARewardCache()

        # If no custom judge function is provided, use default mock/heuristic judge
        if judge_fn is not None:
            self.judge_fn = judge_fn
        else:
            self.judge_fn = self._default_judge_fn

    @staticmethod
    def _default_judge_fn(img: Union[Image.Image, str]) -> float:
        """Lightweight default aesthetic scorer based on contrast and brightness entropy."""
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        # Optimal mean brightness around 0.52 and contrast variance around 0.05
        mean_lum = float(np.mean(arr))
        contrast = float(np.std(arr))
        lum_score = 1.0 - abs(mean_lum - 0.52) * 2.0
        contrast_score = min(1.0, contrast / 0.20)
        return max(0.0, min(1.0, 0.5 * lum_score + 0.5 * contrast_score))

    def evaluate_candidate(
        self,
        image: Image.Image,
        completion_text: str,
        image_id: Optional[str] = None,
    ) -> RewardBreakdown:
        """Evaluates a single model completion against the multi-objective reward function.

        Why sequential gating:
        1. If the completion fails JSON parsing or schema validation, execution cannot proceed;
           we assign zero schema reward and zero AG reward immediately.
        2. If valid, we execute the top proposed fix through Tier-A.
        3. If the Identity Guard vetoes the edit (perceptual drift > threshold), AG reward is zeroed
           and a drift penalty is applied, discouraging reward hacking.
        """
        img_id = image_id or f"img_{hash(image.tobytes()[:100])}"
        schema_reward = 0.0
        ag_reward = 0.0
        drift_penalty = 0.0
        redundancy_penalty = 0.0
        raw_ag = 0.0
        composite_drift = 0.0
        is_guard_approved = False
        applied_op = None

        # 1. Parse JSON and validate CriticOutput schema
        try:
            critic_output = parse_critic_json(completion_text)
            is_valid_json = True
            schema_reward = 1.0

            # Quality bonus for non-empty rationales across all 5 dimensions
            if len(critic_output.rationales) == 5 and all(len(r.strip()) > 5 for r in critic_output.rationales.values()):
                schema_reward += 0.2
        except Exception:
            # Schema violation terminates execution branch
            return RewardBreakdown(
                total_reward=-1.0,
                schema_reward=0.0,
                ag_reward=0.0,
                drift_penalty=0.0,
                redundancy_penalty=0.0,
                raw_ag=0.0,
                composite_drift=0.0,
                is_valid_json=False,
                is_guard_approved=False,
                critic_output=None,
                applied_op=None,
            )

        # 2. Check for redundant operations within the same critique
        fix_ops = [f.op for f in critic_output.fixes]
        if len(fix_ops) != len(set(fix_ops)):
            redundancy_penalty = self.weights.redundancy_penalty_weight

        # 3. If fixes are proposed, execute the primary fix and measure aesthetic gain
        if critic_output.fixes:
            primary_fix = critic_output.fixes[0]
            applied_op = primary_fix.op

            # Check Tier-A cache first
            cached_result = self.cache.get(img_id, primary_fix.op, primary_fix.params)
            base_score = self.judge_fn(image)

            if cached_result is not None:
                edited_image, edited_score = cached_result
            else:
                try:
                    res = self.executor.execute_fixes(
                        image_input=image,
                        fixes=[primary_fix],
                    )
                    edited_image = res.image
                    edited_score = self.judge_fn(edited_image)
                    self.cache.set(img_id, primary_fix.op, primary_fix.params, edited_image, edited_score)
                except Exception as exc:
                    logger.warning("Tier-A execution failed on fix %s: %s", primary_fix, exc)
                    edited_image = image
                    edited_score = base_score

            # 4. Identity Guard verification
            guard_verdict = self.guard.evaluate(original=image, edited=edited_image)
            is_guard_approved = guard_verdict.is_valid
            composite_drift = guard_verdict.composite_drift

            if not is_guard_approved:
                # Edit vetoed: zero AG reward and penalize drift heavily
                raw_ag = 0.0
                ag_reward = 0.0
                drift_penalty = self.weights.drift_penalty_weight
            else:
                # Legitimate edit: compute aesthetic gain delta
                raw_ag = edited_score - base_score
                ag_reward = self.weights.ag_weight * raw_ag

                # Soft drift penalty for borderline edits
                if composite_drift > 0.15:
                    drift_penalty = self.weights.drift_penalty_weight * (composite_drift - 0.15)
        else:
            # Critic proposed no fixes: neutral baseline AG
            raw_ag = 0.0
            ag_reward = 0.0
            is_guard_approved = True

        total_reward = (
            self.weights.schema_weight * schema_reward
            + ag_reward
            - drift_penalty
            - redundancy_penalty
        )

        return RewardBreakdown(
            total_reward=round(total_reward, 4),
            schema_reward=round(schema_reward, 4),
            ag_reward=round(ag_reward, 4),
            drift_penalty=round(drift_penalty, 4),
            redundancy_penalty=round(redundancy_penalty, 4),
            raw_ag=round(raw_ag, 4),
            composite_drift=round(composite_drift, 4),
            is_valid_json=is_valid_json,
            is_guard_approved=is_guard_approved,
            critic_output=critic_output,
            applied_op=applied_op,
        )

    def evaluate_group(
        self,
        image: Image.Image,
        completions: Sequence[str],
        image_id: Optional[str] = None,
    ) -> List[RewardBreakdown]:
        """Evaluates a group of G candidate completions for a single prompt image."""
        return [
            self.evaluate_candidate(image=image, completion_text=comp, image_id=image_id)
            for comp in completions
        ]


def compute_group_advantages(
    rewards: Sequence[float],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Computes group-relative normalized advantages for GRPO policy updates.

    Why group normalization:
    Normalizes candidate scores relative to the current group mean and variance.
    Completions that outperform their peers receive positive advantage; inferior
    completions receive negative advantage, driving the policy towards superior critique generation.
    """
    if not rewards:
        return torch.empty(0, dtype=torch.float32)

    r_tensor = torch.tensor(rewards, dtype=torch.float32)
    if len(rewards) == 1:
        return torch.zeros_like(r_tensor)

    mean = torch.mean(r_tensor)
    std = torch.std(r_tensor, unbiased=False)

    if std < eps:
        # All candidates in group produced identical rewards
        return torch.zeros_like(r_tensor)

    advantages = (r_tensor - mean) / (std + eps)
    return advantages


def create_trl_grpo_reward_fn(
    reward_engine: GRPORewardEngine,
) -> Callable[[List[Dict[str, Any]], List[str]], List[float]]:
    """Adapts GRPORewardEngine to the standard Hugging Face TRL GRPOTrainer reward function interface.

    Why this adapter exists:
    TRL's GRPOTrainer expects reward functions of the form:
    `reward_fn(prompts: List[Dict], completions: List[str], **kwargs) -> List[float]`.
    """
    def reward_fn(prompts: List[Dict[str, Any]], completions: List[str], **kwargs: Any) -> List[float]:
        # Extract images from prompts if present, otherwise create a neutral fallback image
        rewards: List[float] = []
        for prompt, completion in zip(prompts, completions):
            image = prompt.get("image")
            if image is None:
                image = Image.new("RGB", (64, 64), color=(128, 128, 128))
            elif isinstance(image, str):
                image = Image.open(image).convert("RGB")

            breakdown = reward_engine.evaluate_candidate(
                image=image,
                completion_text=completion,
                image_id=prompt.get("image_id"),
            )
            rewards.append(breakdown.total_reward)
        return rewards

    return reward_fn
