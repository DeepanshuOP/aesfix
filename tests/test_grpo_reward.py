"""Unit tests for GRPO Multi-Objective Reward Engine & Policy Gradient Pipeline (AF-3.2).

Why these tests are critical:
1. Validates that group relative advantage calculation strictly enforces zero mean and
   unit variance across completion candidates, which is the foundational mathematical invariant of GRPO.
2. Confirms that the Identity Guard acts as an unbypassable gate, zeroing aesthetic gain
   and applying drift penalties when edits alter subject composition or cause pixel degradation.
3. Tests that Tier-A reward caching (§12-C2) successfully accelerates rollout evaluation by
   preventing redundant tensor transformations and judge evaluations on repeated edits.
4. Verifies the surrogate policy gradient loss and reference policy KL divergence penalty.
"""
from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from aesfix.executor.guard import IdentityGuard
from aesfix.train.grpo_reward import (
    GRPORewardEngine,
    RewardWeights,
    TierARewardCache,
    compute_group_advantages,
    create_trl_grpo_reward_fn,
)
from aesfix.train.grpo_trainer import (
    GRPOConfig,
    GRPORolloutBenchmark,
    GRPOTrainingSession,
    compute_grpo_loss,
)


@pytest.fixture
def sample_test_image() -> Image.Image:
    """Fixture providing a standard 64x64 test image."""
    return Image.new("RGB", (64, 64), color=(140, 140, 140))


@pytest.fixture
def valid_completion_text() -> str:
    """Fixture returning valid CriticOutput JSON proposing a subtle exposure fix."""
    return json.dumps({
        "aspect_scores": {
            "composition": 0.80,
            "lighting": 0.45,
            "color": 0.85,
            "subject_clarity": 0.80,
            "technical": 0.90,
        },
        "rationales": {
            "composition": "Strong framing along horizontal axis.",
            "lighting": "Shadows are slightly dense and require subtle lifting.",
            "color": "Pleasing palette and natural saturation.",
            "subject_clarity": "Subject plane is well focused.",
            "technical": "Low noise floor with clean details.",
        },
        "overall_summary": "Solid photograph requiring minor midtone exposure recovery.",
        "fixes": [
            {
                "op": "adjust_exposure",
                "params": {"ev": 0.35},
                "expected_aspect": "lighting",
                "nl_instruction": "Lift exposure +0.35 EV to open up midtone detail.",
            }
        ],
        "genre": "portrait",
    })


@pytest.fixture
def redundant_completion_text() -> str:
    """Fixture returning valid JSON with duplicate operations in the fix plan."""
    return json.dumps({
        "aspect_scores": {
            "composition": 0.7,
            "lighting": 0.5,
            "color": 0.7,
            "subject_clarity": 0.7,
            "technical": 0.7,
        },
        "rationales": {
            "composition": "Adequate balance.",
            "lighting": "Needs exposure.",
            "color": "Adequate.",
            "subject_clarity": "Adequate.",
            "technical": "Adequate.",
        },
        "overall_summary": "Needs work.",
        "fixes": [
            {
                "op": "adjust_exposure",
                "params": {"ev": 0.2},
                "expected_aspect": "lighting",
                "nl_instruction": "Lift exposure.",
            },
            {
                "op": "adjust_exposure",
                "params": {"ev": 0.4},
                "expected_aspect": "lighting",
                "nl_instruction": "Lift exposure again.",
            },
        ],
    })


def test_compute_group_advantages_math() -> None:
    """Verifies that group advantage computation maintains mean=0 and std=1."""
    # 1. Edge cases: empty and single candidate
    empty_adv = compute_group_advantages([])
    assert empty_adv.numel() == 0

    single_adv = compute_group_advantages([1.5])
    assert single_adv.item() == 0.0

    # 2. Identical rewards should yield zero advantages
    identical_adv = compute_group_advantages([2.0, 2.0, 2.0, 2.0])
    assert torch.allclose(identical_adv, torch.zeros(4))

    # 3. Standard diverse group: [1.0, 2.0, 3.0, 4.0]
    rewards = [1.0, 2.0, 3.0, 4.0]
    adv = compute_group_advantages(rewards)

    # Why mean must be 0 and std must be 1:
    # GRPO centers the gradient updates so half the group is pushed up and half down
    assert abs(float(adv.mean().item())) < 1e-5
    assert abs(float(adv.std(unbiased=False).item()) - 1.0) < 1e-4

    # Monotonic order: best candidate must have highest advantage
    assert adv[3] > adv[2] > adv[1] > adv[0]


def test_grpo_reward_engine_valid_candidate(
    sample_test_image: Image.Image,
    valid_completion_text: str,
) -> None:
    """Tests multi-objective evaluation of a structurally sound and verified critique."""
    engine = GRPORewardEngine()
    breakdown = engine.evaluate_candidate(
        image=sample_test_image,
        completion_text=valid_completion_text,
    )

    assert breakdown.is_valid_json is True
    assert breakdown.schema_reward >= 1.0
    assert breakdown.is_guard_approved is True
    assert breakdown.applied_op == "adjust_exposure"
    assert breakdown.drift_penalty == 0.0
    assert breakdown.redundancy_penalty == 0.0
    assert breakdown.total_reward > 0.0


def test_grpo_reward_engine_malformed_json(sample_test_image: Image.Image) -> None:
    """Confirms malformed completions are penalized immediately without execution."""
    engine = GRPORewardEngine()
    breakdown = engine.evaluate_candidate(
        image=sample_test_image,
        completion_text="Here is your advice: please brighten the image!",
    )

    assert breakdown.is_valid_json is False
    assert breakdown.total_reward == -1.0
    assert breakdown.schema_reward == 0.0
    assert breakdown.ag_reward == 0.0
    assert breakdown.applied_op is None


def test_grpo_reward_engine_redundancy_penalty(
    sample_test_image: Image.Image,
    redundant_completion_text: str,
) -> None:
    """Ensures duplicate fix operations trigger the redundancy penalty."""
    weights = RewardWeights(redundancy_penalty_weight=0.5)
    engine = GRPORewardEngine(weights=weights)

    breakdown = engine.evaluate_candidate(
        image=sample_test_image,
        completion_text=redundant_completion_text,
    )

    assert breakdown.is_valid_json is True
    assert breakdown.redundancy_penalty == 0.5


def test_grpo_reward_engine_identity_guard_veto(sample_test_image: Image.Image) -> None:
    """Verifies that an edit rejected by the Identity Guard receives zero AG and full drift penalty."""
    # Strict guard with very low threshold (0.01) guaranteed to veto any modification
    strict_guard = IdentityGuard(drift_threshold=0.01)
    engine = GRPORewardEngine(guard=strict_guard)

    completion = json.dumps({
        "aspect_scores": {
            "composition": 0.5, "lighting": 0.5, "color": 0.5,
            "subject_clarity": 0.5, "technical": 0.5,
        },
        "rationales": {
            "composition": "c", "lighting": "l", "color": "co",
            "subject_clarity": "s", "technical": "t",
        },
        "overall_summary": "summary",
        "fixes": [{"op": "adjust_exposure", "params": {"ev": 1.5}, "expected_aspect": "lighting", "nl_instruction": "brighten"}],
    })

    breakdown = engine.evaluate_candidate(image=sample_test_image, completion_text=completion)

    # Why guard veto is critical:
    # Prevents model from gaming aesthetic score through identity-destroying modifications
    assert breakdown.is_guard_approved is False
    assert breakdown.ag_reward == 0.0
    assert breakdown.drift_penalty == engine.weights.drift_penalty_weight


def test_tier_a_reward_cache_acceleration(
    sample_test_image: Image.Image,
    valid_completion_text: str,
) -> None:
    """Tests the in-memory cache hit mechanism for Tier-A rollouts (§12-C2)."""
    cache = TierARewardCache(max_size=100)
    engine = GRPORewardEngine(cache=cache)

    image_id = "test_cache_img_1"

    # First evaluation: cache miss, executed and cached
    assert cache.get(image_id, "adjust_exposure", {"ev": 0.35}) is None
    res1 = engine.evaluate_candidate(sample_test_image, valid_completion_text, image_id=image_id)

    # Cache should now hold the result
    cached = cache.get(image_id, "adjust_exposure", {"ev": 0.35})
    assert cached is not None
    edited_img, cached_score = cached
    assert isinstance(edited_img, Image.Image)

    # Second evaluation: cache hit
    res2 = engine.evaluate_candidate(sample_test_image, valid_completion_text, image_id=image_id)
    assert res1.total_reward == res2.total_reward
    assert res1.raw_ag == res2.raw_ag


def test_trl_reward_fn_adapter(
    sample_test_image: Image.Image,
    valid_completion_text: str,
) -> None:
    """Verifies adapter compatibility with Hugging Face TRL GRPOTrainer signature."""
    engine = GRPORewardEngine()
    trl_fn = create_trl_grpo_reward_fn(engine)

    prompts = [{"image": sample_test_image, "image_id": "img_1"}]
    completions = [valid_completion_text]

    rewards = trl_fn(prompts, completions)
    assert len(rewards) == 1
    assert isinstance(rewards[0], float)
    assert rewards[0] > 0.0


def test_compute_grpo_loss() -> None:
    """Verifies clipped policy gradient surrogate loss and reference KL divergence."""
    # Batch of 4 candidates with 3 tokens each
    log_probs = torch.tensor([[-1.0, -1.2, -0.8], [-2.0, -1.5, -1.8], [-0.5, -0.7, -0.6], [-1.5, -1.5, -1.5]])
    old_log_probs = log_probs.clone()
    ref_log_probs = log_probs.clone()
    advantages = torch.tensor([1.2, -0.4, 0.8, -1.6])

    loss, metrics = compute_grpo_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        advantages=advantages,
        clip_range=0.2,
        kl_coeff=0.04,
    )

    # When log_probs == ref_log_probs, KL divergence is mathematically 0.0
    assert metrics.kl_penalty == 0.0
    # When ratio is 1.0, policy loss is exactly -mean(advantages)
    expected_loss = -advantages.mean().item()
    assert abs(metrics.policy_loss - expected_loss) < 1e-4
    assert metrics.clip_fraction == 0.0


def test_grpo_rollout_benchmark(
    sample_test_image: Image.Image,
    valid_completion_text: str,
) -> None:
    """Tests rollout throughput benchmark utility and cache hit rate tracking."""
    engine = GRPORewardEngine()
    benchmark = GRPORolloutBenchmark(reward_engine=engine, target_rollouts_per_hour=100.0)

    result = benchmark.run_benchmark(
        test_images=[sample_test_image],
        candidate_completions=[valid_completion_text],
        iterations=5,
    )

    assert result.num_rollouts == 5
    assert result.rollouts_per_sec > 0.0
    assert result.rollouts_per_hour > 0.0
    # With 5 iterations on identical image/fix, cache hit rate should be > 0.5
    assert result.cache_hit_rate >= 0.50
    assert isinstance(result.meets_dod, bool)


def test_grpo_training_session(
    sample_test_image: Image.Image,
    valid_completion_text: str,
) -> None:
    """Tests end-to-end GRPO step orchestration across a candidate group."""
    config = GRPOConfig(group_size=2)
    engine = GRPORewardEngine()
    session = GRPOTrainingSession(config=config, reward_engine=engine)

    completions = [valid_completion_text, "invalid json completion"]
    log_probs = torch.tensor([[-1.0, -1.5], [-2.0, -2.5]])
    old_log_probs = log_probs.clone()
    ref_log_probs = log_probs.clone()

    loss, metrics, breakdowns = session.step(
        image=sample_test_image,
        candidate_completions=completions,
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
    )

    assert len(breakdowns) == 2
    assert breakdowns[0].is_valid_json is True
    assert breakdowns[1].is_valid_json is False
    assert session.step_count == 1
    assert isinstance(loss, torch.Tensor)
