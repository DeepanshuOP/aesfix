"""Unit tests for Edit-Verified GRPO Execution Loop & Anti-Hacking Dashboards (AF-3.3).

Why these tests are critical:
1. Validates that the Anti-Hacking Monitor reliably detects policy degradation,
   including operation mode collapse (entropy drop), guard violation spikes, and judge exploitation.
2. Confirms that the Edit-Verified GRPO training step atomically executes candidates through
   Tier-A, computes group advantages, updates policy gradients, and logs dashboard metrics.
3. Verifies that the held-out AG@k benchmark correctly evaluates execution gains, tracks
   identity guard approvals, and determines whether policy performance beats the SFT baseline.
"""
from __future__ import annotations

import json
from typing import List

import pytest
import torch
from PIL import Image

from aesfix.critic.critic import CriticFix, CriticOutput
from aesfix.train.anti_hacking import AntiHackingMonitor
from aesfix.train.grpo_loop import EditVerifiedGRPOLoop, evaluate_held_out_ag
from aesfix.train.grpo_reward import GRPORewardEngine, RewardBreakdown
from aesfix.train.grpo_trainer import GRPOConfig


@pytest.fixture
def test_image_rgb() -> Image.Image:
    """Fixture providing a standard 64x64 test image."""
    return Image.new("RGB", (64, 64), color=(120, 120, 120))


def _make_dummy_breakdown(
    op: str,
    total_reward: float = 1.0,
    raw_ag: float = 0.05,
    is_valid_json: bool = True,
    is_guard_approved: bool = True,
    composite_drift: float = 0.05,
) -> RewardBreakdown:
    """Helper creating mock reward breakdowns with specific diagnostic properties."""
    fix = CriticFix(op=op, params={}, expected_aspect="lighting", nl_instruction="test")
    critic_out = CriticOutput(
        aspect_scores={
            "composition": 0.7, "lighting": 0.7, "color": 0.7,
            "subject_clarity": 0.7, "technical": 0.7,
        },
        rationales={
            "composition": "ok", "lighting": "ok", "color": "ok",
            "subject_clarity": "ok", "technical": "ok",
        },
        overall_summary="ok",
        fixes=[fix],
    )
    return RewardBreakdown(
        total_reward=total_reward,
        schema_reward=1.0 if is_valid_json else 0.0,
        ag_reward=raw_ag * 5.0,
        drift_penalty=0.0 if is_guard_approved else 2.5,
        redundancy_penalty=0.0,
        raw_ag=raw_ag,
        composite_drift=composite_drift,
        is_valid_json=is_valid_json,
        is_guard_approved=is_guard_approved,
        critic_output=critic_out if is_valid_json else None,
        applied_op=op if is_valid_json else None,
    )


def test_anti_hacking_monitor_metrics() -> None:
    """Verifies calculation of guard violation rate, operation entropy, and rolling window."""
    monitor = AntiHackingMonitor(
        guard_violation_threshold=0.20,
        min_diversity_entropy=1.0,
        window_size=10,
    )

    # Ingest a diverse, healthy batch of rollouts across 4 operations
    healthy_batch = [
        _make_dummy_breakdown("adjust_exposure", is_guard_approved=True),
        _make_dummy_breakdown("adjust_contrast", is_guard_approved=True),
        _make_dummy_breakdown("adjust_saturation", is_guard_approved=True),
        _make_dummy_breakdown("unsharp_mask", is_guard_approved=True),
    ]

    snapshot = monitor.record_step(step=1, breakdowns=healthy_batch, judge_disagreements=[0.02, 0.03])

    # 4 distinct operations uniformly distributed -> entropy = -4 * (0.25 * log2(0.25)) = 2.0 bits
    assert abs(snapshot.advice_diversity_entropy - 2.0) < 1e-4
    assert snapshot.guard_violation_rate == 0.0
    assert snapshot.mean_judge_disagreement == 0.025
    assert len(snapshot.alerts) == 0
    assert monitor.is_healthy() is True


def test_anti_hacking_guard_violation_alert() -> None:
    """Ensures a spike in identity guard rejections triggers an anti-hacking alert."""
    monitor = AntiHackingMonitor(guard_violation_threshold=0.20)

    # Batch with 50% guard rejection rate (exceeds 20% threshold)
    compromised_batch = [
        _make_dummy_breakdown("adjust_exposure", is_guard_approved=True),
        _make_dummy_breakdown("adjust_exposure", is_guard_approved=False),
    ]

    snapshot = monitor.record_step(step=2, breakdowns=compromised_batch)

    assert snapshot.guard_violation_rate == 0.50
    assert any(a.alert_type == "guard_violation_spike" for a in snapshot.alerts)
    # Severity is critical because 50% > 40%
    critical_alerts = [a for a in snapshot.alerts if a.severity == "critical"]
    assert len(critical_alerts) >= 1
    assert monitor.is_healthy() is False


def test_anti_hacking_advice_collapse_alert() -> None:
    """Verifies that an operational mode collapse triggers an entropy alert."""
    monitor = AntiHackingMonitor(min_diversity_entropy=1.0, window_size=20)

    # 12 consecutive rollouts suggesting ONLY adjust_exposure -> entropy = 0.0 bits
    collapsed_batch = [_make_dummy_breakdown("adjust_exposure")] * 12

    snapshot = monitor.record_step(step=3, breakdowns=collapsed_batch)

    # Why entropy is 0: perfectly deterministic operation selection
    assert snapshot.advice_diversity_entropy == 0.0
    assert any(a.alert_type == "advice_collapse" for a in snapshot.alerts)


def test_anti_hacking_judge_disagreement_alert() -> None:
    """Confirms high cross-judge disagreement triggers an exploitation alert."""
    monitor = AntiHackingMonitor(max_disagreement_threshold=0.10)

    batch = [_make_dummy_breakdown("adjust_exposure")]
    snapshot = monitor.record_step(step=4, breakdowns=batch, judge_disagreements=[0.25])

    assert snapshot.mean_judge_disagreement == 0.25
    assert any(a.alert_type == "judge_disagreement_divergence" for a in snapshot.alerts)


def test_edit_verified_grpo_loop_step(test_image_rgb: Image.Image) -> None:
    """Tests an integrated training step driving candidate execution and dashboard logging."""
    config = GRPOConfig(group_size=2)
    engine = GRPORewardEngine()
    monitor = AntiHackingMonitor()
    loop = EditVerifiedGRPOLoop(config=config, reward_engine=engine, monitor=monitor)

    candidate_text_1 = json.dumps({
        "aspect_scores": {"composition": 0.8, "lighting": 0.5, "color": 0.8, "subject_clarity": 0.8, "technical": 0.8},
        "rationales": {"composition": "ok", "lighting": "low", "color": "ok", "subject_clarity": "ok", "technical": "ok"},
        "overall_summary": "ok",
        "fixes": [{"op": "adjust_exposure", "params": {"ev": 0.3}, "expected_aspect": "lighting", "nl_instruction": "brighten"}],
    })
    candidate_text_2 = json.dumps({
        "aspect_scores": {"composition": 0.8, "lighting": 0.5, "color": 0.8, "subject_clarity": 0.8, "technical": 0.8},
        "rationales": {"composition": "ok", "lighting": "low", "color": "ok", "subject_clarity": "ok", "technical": "ok"},
        "overall_summary": "ok",
        "fixes": [{"op": "adjust_contrast", "params": {"factor": 1.1}, "expected_aspect": "lighting", "nl_instruction": "contrast"}],
    })

    log_probs = torch.tensor([[-1.0, -1.2], [-1.5, -1.8]])
    old_log_probs = log_probs.clone()
    ref_log_probs = log_probs.clone()

    report = loop.step(
        image=test_image_rgb,
        candidate_completions=[candidate_text_1, candidate_text_2],
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        judge_disagreements=[0.02],
    )

    assert report.step == 1
    assert report.num_rollouts == 2
    assert report.dashboard_snapshot.num_rollouts == 2
    assert report.loss_metrics.loss is not None
    assert report.mean_step_reward > 0.0


def test_evaluate_held_out_ag(test_image_rgb: Image.Image) -> None:
    """Validates held-out AG@k evaluation and benchmark comparison against SFT baseline."""
    # Policy rollout function returning valid candidate fixes
    def mock_policy_rollout(img: Image.Image, n_candidates: int) -> List[str]:
        completions = []
        for i in range(n_candidates):
            ev = 0.2 + (i * 0.1)
            completions.append(
                json.dumps({
                    "aspect_scores": {"composition": 0.7, "lighting": 0.4, "color": 0.7, "subject_clarity": 0.7, "technical": 0.7},
                    "rationales": {"composition": "c", "lighting": "l", "color": "co", "subject_clarity": "s", "technical": "t"},
                    "overall_summary": "summary",
                    "fixes": [{"op": "adjust_exposure", "params": {"ev": ev}, "expected_aspect": "lighting", "nl_instruction": "brighten"}],
                })
            )
        return completions

    eval_result = evaluate_held_out_ag(
        policy_rollout_fn=mock_policy_rollout,
        held_out_images=[test_image_rgb],
        k=3,
        baseline_ag_at_3=0.01,
    )

    assert eval_result.num_samples == 1
    assert eval_result.guard_pass_rate == 1.0
    assert eval_result.schema_validity_rate == 1.0
    assert eval_result.ag_at_3 >= eval_result.ag_at_1
    assert isinstance(eval_result.beats_baseline, bool)

    res_dict = eval_result.to_dict()
    assert "ag_at_3" in res_dict
    assert "delta_ag_at_3" in res_dict
