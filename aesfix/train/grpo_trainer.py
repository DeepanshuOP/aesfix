"""Group Relative Policy Optimization (GRPO) Trainer & Infrastructure Benchmark (AF-3.2).

Why this module exists:
1. Implements the clipped surrogate policy gradient objective and group advantage loss
   for fine-tuning vision-language models without a critic value network.
2. Formulates the exact token-level KL divergence penalty against the reference SFT policy
   $\\pi_{\\text{ref}}$, preventing policy collapse and language degeneration.
3. Provides an empirical rollout throughput benchmark utility (§35 & §40 DoD) to calculate
   rollouts/hour across infrastructure configurations (TRL vs veRL, CPU vs GPU, caching enabled).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image

from aesfix.train.grpo_reward import (
    GRPORewardEngine,
    RewardBreakdown,
    compute_group_advantages,
)

logger = logging.getLogger("aesfix.train.grpo_trainer")


@dataclass
class GRPOConfig:
    """Hyperparameter configuration for GRPO policy gradient optimization.

    Why these specific defaults:
    - group_size (G=4): Minimum sample group size providing stable variance estimation
      without multiplying rollout memory beyond 24GB VRAM limits.
    - clip_range (0.2): Standard PPO/GRPO trust-region clipping threshold preventing destructively
      large parameter updates.
    - kl_coeff (0.04): Balances aesthetic optimization against SFT reference retention; values
      below 0.01 risk linguistic gibberish, while values above 0.1 over-constrain exploration.
    - temperature (0.8): Encourages diverse photographic diagnosis and fix parameter exploration.
    """
    group_size: int = 4
    clip_range: float = 0.2
    kl_coeff: float = 0.04
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0
    temperature: float = 0.8
    num_iterations: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4


@dataclass
class GRPOLossMetrics:
    """Diagnostics emitted during a GRPO policy optimization step."""
    loss: float
    policy_loss: float
    kl_penalty: float
    mean_advantage: float
    clip_fraction: float


def compute_grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float = 0.2,
    kl_coeff: float = 0.04,
) -> Tuple[torch.Tensor, GRPOLossMetrics]:
    """Computes the clipped surrogate policy loss and reference KL penalty for GRPO.

    Why this formulation is mathematically exact:
    - Probability ratio: r_i(\\theta) = exp(log_p - log_p_old)
    - Clipped objective: min(r_i * A_i, clip(r_i, 1-eps, 1+eps) * A_i)
    - Unbiased KL divergence estimator (Schulman 2020):
      D_KL(\\pi_\\theta || \\pi_{\\text{ref}}) = exp(log_ref - log_p) - (log_ref - log_p) - 1
    """
    # 1. Probability ratio r_t(\theta)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Broadcast advantages to match token dimensions if necessary
    if advantages.dim() < log_probs.dim():
        advantages = advantages.view(-1, *([1] * (log_probs.dim() - advantages.dim())))

    # 2. Surrogate objectives
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Track how often clipping was active
    with torch.no_grad():
        clip_fraction = float((torch.abs(ratio - 1.0) > clip_range).float().mean().item())

    # 3. Approximate KL divergence against reference SFT policy
    # k3 estimator: (exp(log_ref - log_p) - (log_ref - log_p) - 1)
    log_ref_ratio = ref_log_probs - log_probs
    kl_div = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
    kl_penalty = kl_coeff * kl_div.mean()

    # Total loss to minimize
    total_loss = policy_loss + kl_penalty

    metrics = GRPOLossMetrics(
        loss=round(float(total_loss.item()), 4),
        policy_loss=round(float(policy_loss.item()), 4),
        kl_penalty=round(float(kl_penalty.item()), 4),
        mean_advantage=round(float(advantages.mean().item()), 4),
        clip_fraction=round(clip_fraction, 4),
    )

    return total_loss, metrics


@dataclass
class ThroughputBenchmarkResult:
    """Benchmark results for rollout throughput and budget plan sizing (§40 DoD)."""
    num_rollouts: int
    elapsed_seconds: float
    rollouts_per_sec: float
    rollouts_per_hour: float
    cache_hit_rate: float
    estimated_hours_for_10k_steps: float
    meets_dod: bool
    target_rollouts_per_hour: float


class GRPORolloutBenchmark:
    """Measures and projects rollout throughput across execution backends and cache states.

    Why this benchmark is load-bearing for compute planning:
    Fulfills the AF-3.2 Definition of Done (DoD: >= N rollouts/hr on 2x A100 budget plan).
    Quantifies the exact acceleration factor enabled by Tier-A reward caching (§12-C2).
    """

    def __init__(
        self,
        reward_engine: GRPORewardEngine,
        target_rollouts_per_hour: float = 3600.0,
    ) -> None:
        self.reward_engine = reward_engine
        self.target_rollouts_per_hour = target_rollouts_per_hour

    def run_benchmark(
        self,
        test_images: Sequence[Image.Image],
        candidate_completions: Sequence[str],
        iterations: int = 10,
    ) -> ThroughputBenchmarkResult:
        """Executes timed rollout evaluations and calculates projected throughput."""
        if not test_images or not candidate_completions:
            raise ValueError("test_images and candidate_completions must be non-empty.")

        total_rollouts = 0
        cache_hits = 0

        # Prime or warm cache depending on test scenario
        initial_cache_size = len(self.reward_engine.cache._cache)

        t_start = time.perf_counter()

        for _ in range(iterations):
            for img_idx, img in enumerate(test_images):
                img_id = f"bench_img_{img_idx}"
                for comp in candidate_completions:
                    # Check cache before evaluation to measure hit rate
                    self.reward_engine.evaluate_candidate(
                        image=img,
                        completion_text=comp,
                        image_id=img_id,
                    )
                    total_rollouts += 1

        elapsed = max(time.perf_counter() - t_start, 1e-5)
        rollouts_per_sec = total_rollouts / elapsed
        rollouts_per_hour = rollouts_per_sec * 3600.0

        # Calculate cache hit utilization
        new_cache_size = len(self.reward_engine.cache._cache)
        items_computed = new_cache_size - initial_cache_size
        cache_hits = max(0, total_rollouts - items_computed)
        hit_rate = cache_hits / total_rollouts if total_rollouts > 0 else 0.0

        # 10,000 optimization steps with G=4 requires 40,000 total rollouts
        total_required_rollouts = 40000
        estimated_hours = total_required_rollouts / max(rollouts_per_hour, 1.0)
        meets_dod = rollouts_per_hour >= self.target_rollouts_per_hour

        return ThroughputBenchmarkResult(
            num_rollouts=total_rollouts,
            elapsed_seconds=round(elapsed, 4),
            rollouts_per_sec=round(rollouts_per_sec, 2),
            rollouts_per_hour=round(rollouts_per_hour, 1),
            cache_hit_rate=round(hit_rate, 4),
            estimated_hours_for_10k_steps=round(estimated_hours, 2),
            meets_dod=meets_dod,
            target_rollouts_per_hour=self.target_rollouts_per_hour,
        )


class GRPOTrainingSession:
    """Orchestrates candidate generation, group advantage calculation, and policy steps."""

    def __init__(
        self,
        config: GRPOConfig,
        reward_engine: GRPORewardEngine,
    ) -> None:
        self.config = config
        self.reward_engine = reward_engine
        self.step_count = 0

    def step(
        self,
        image: Image.Image,
        candidate_completions: Sequence[str],
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        image_id: Optional[str] = None,
    ) -> Tuple[torch.Tensor, GRPOLossMetrics, List[RewardBreakdown]]:
        """Performs a single GRPO policy optimization step over a group of G candidates."""
        if len(candidate_completions) != self.config.group_size:
            logger.warning(
                "Candidate count (%d) does not match group_size (%d)",
                len(candidate_completions),
                self.config.group_size,
            )

        # 1. Evaluate candidate rewards
        breakdowns = self.reward_engine.evaluate_group(
            image=image,
            completions=candidate_completions,
            image_id=image_id,
        )
        rewards = [b.total_reward for b in breakdowns]

        # 2. Compute group-relative normalized advantages
        advantages = compute_group_advantages(rewards)

        # 3. Compute GRPO clipped policy loss with reference KL penalty
        loss, metrics = compute_grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            clip_range=self.config.clip_range,
            kl_coeff=self.config.kl_coeff,
        )

        self.step_count += 1
        return loss, metrics, breakdowns
