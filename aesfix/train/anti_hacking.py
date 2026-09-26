"""Anti-Reward-Hacking Diagnostics & Training Observability Dashboard (AF-3.3 & Paper §12-C2).

Why Anti-Reward-Hacking Observability is essential in visual RL:
1. Mode Collapse on High-Gradient Ops: Without monitoring, reinforcement learning policies
   frequently collapse to a single high-reward operation (e.g. always predicting `adjust_exposure +0.40 EV`)
   because exposure changes produce large initial judge gain with minimal risk. We track
   operation diversity via Shannon entropy to detect and penalize policy collapse.
2. Guard Violation Tracking: If the policy learns to generate extreme parameters or subtle
   adversarial artifacts that trigger Identity Guard rejections, the guard violation rate
   spikes, signaling that the policy is attempting to "cheat" the aesthetic metric.
3. Judge Exploitation & Disagreement Drift: Models can overfit to the idiosyncratic quirks
   of a single judge (e.g. boosting high-frequency noise that fools a linear CLIP probe).
   Tracking cross-judge variance ensures gains are aesthetically authentic across perspectives.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from aesfix.critic.critic import VALID_FIX_OPERATIONS
from aesfix.train.grpo_reward import RewardBreakdown

logger = logging.getLogger("aesfix.train.anti_hacking")


@dataclass
class HackingAlert:
    """Warning emitted when training metrics indicate reward hacking or policy degeneration."""
    alert_type: str
    severity: str  # "warning" | "critical"
    message: str
    current_value: float
    threshold: float
    step: int


@dataclass
class DashboardSnapshot:
    """Point-in-time diagnostic snapshot summarizing anti-hacking metrics."""
    step: int
    num_rollouts: int
    guard_violation_rate: float
    advice_diversity_entropy: float
    mean_judge_disagreement: float
    mean_reward: float
    mean_ag: float
    mean_composite_drift: float
    op_distribution: Dict[str, float]
    alerts: List[HackingAlert] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dashboard metrics for W&B or JSON export."""
        return {
            "step": self.step,
            "num_rollouts": self.num_rollouts,
            "guard_violation_rate": round(self.guard_violation_rate, 4),
            "advice_diversity_entropy": round(self.advice_diversity_entropy, 4),
            "mean_judge_disagreement": round(self.mean_judge_disagreement, 4),
            "mean_reward": round(self.mean_reward, 4),
            "mean_ag": round(self.mean_ag, 4),
            "mean_composite_drift": round(self.mean_composite_drift, 4),
            "op_distribution": {k: round(v, 4) for k, v in self.op_distribution.items()},
            "num_alerts": len(self.alerts),
        }


class AntiHackingMonitor:
    """Monitors GRPO rollouts for reward hacking, advice collapse, and guard violations."""

    def __init__(
        self,
        guard_violation_threshold: float = 0.20,
        min_diversity_entropy: float = 1.0,
        max_disagreement_threshold: float = 0.15,
        window_size: int = 100,
    ) -> None:
        """Configures safety thresholds for training monitoring.

        Why these specific thresholds:
        - guard_violation_threshold (0.20): More than 20% guard rejections signals destructive edits.
        - min_diversity_entropy (1.0): Shannon entropy below 1.0 indicates severe operation collapse
          (the policy is heavily fixated on 1-2 operations out of 10 available).
        - max_disagreement_threshold (0.15): High variance among judges indicates the policy is
          exploiting single-model blindspots rather than producing genuine aesthetic improvements.
        """
        self.guard_violation_threshold = guard_violation_threshold
        self.min_diversity_entropy = min_diversity_entropy
        self.max_disagreement_threshold = max_disagreement_threshold
        self.window_size = window_size

        self._history: List[RewardBreakdown] = []
        self._judge_disagreements: List[float] = []
        self._step_snapshots: List[DashboardSnapshot] = []

    def record_step(
        self,
        step: int,
        breakdowns: Sequence[RewardBreakdown],
        judge_disagreements: Optional[Sequence[float]] = None,
    ) -> DashboardSnapshot:
        """Ingests step rollouts and calculates anti-hacking metrics."""
        self._history.extend(breakdowns)
        if judge_disagreements:
            self._judge_disagreements.extend(judge_disagreements)

        # Slice rolling evaluation window
        window = self._history[-self.window_size :]
        n_window = len(window)

        if n_window == 0:
            return DashboardSnapshot(
                step=step,
                num_rollouts=0,
                guard_violation_rate=0.0,
                advice_diversity_entropy=0.0,
                mean_judge_disagreement=0.0,
                mean_reward=0.0,
                mean_ag=0.0,
                mean_composite_drift=0.0,
                op_distribution={},
                alerts=[],
            )

        # 1. Guard violation rate: fraction of valid rollouts that were vetoed by Identity Guard
        vetoed_count = sum(1 for b in window if b.is_valid_json and not b.is_guard_approved)
        valid_json_count = sum(1 for b in window if b.is_valid_json)
        guard_violation_rate = (
            vetoed_count / valid_json_count if valid_json_count > 0 else 0.0
        )

        # 2. Advice diversity (Shannon entropy over proposed operations)
        op_counts: Counter[str] = Counter()
        for b in window:
            if b.applied_op:
                op_counts[b.applied_op] += 1
            elif b.critic_output and b.critic_output.fixes:
                for fix in b.critic_output.fixes:
                    op_counts[fix.op] += 1

        total_ops = sum(op_counts.values())
        op_distribution: Dict[str, float] = {}
        entropy = 0.0

        if total_ops > 0:
            for op in VALID_FIX_OPERATIONS:
                prob = op_counts[op] / total_ops
                op_distribution[op] = prob
                if prob > 0.0:
                    entropy -= prob * math.log2(prob)

        # 3. Mean metrics across window
        mean_reward = sum(b.total_reward for b in window) / n_window
        mean_ag = sum(b.raw_ag for b in window) / n_window
        mean_drift = sum(b.composite_drift for b in window) / n_window

        disagreements_window = self._judge_disagreements[-self.window_size :]
        mean_disagreement = (
            sum(disagreements_window) / len(disagreements_window)
            if disagreements_window
            else 0.0
        )

        # 4. Generate alerts based on safety thresholds
        alerts: List[HackingAlert] = []

        if guard_violation_rate > self.guard_violation_threshold:
            alerts.append(
                HackingAlert(
                    alert_type="guard_violation_spike",
                    severity="critical" if guard_violation_rate > 0.40 else "warning",
                    message=(
                        f"Identity Guard rejection rate {guard_violation_rate:.1%} exceeds "
                        f"threshold {self.guard_violation_threshold:.1%}. Policy is proposing destructive edits."
                    ),
                    current_value=guard_violation_rate,
                    threshold=self.guard_violation_threshold,
                    step=step,
                )
            )

        if total_ops >= 10 and entropy < self.min_diversity_entropy:
            alerts.append(
                HackingAlert(
                    alert_type="advice_collapse",
                    severity="warning",
                    message=(
                        f"Advice diversity entropy {entropy:.2f} bits is below threshold {self.min_diversity_entropy:.2f}. "
                        "Policy is collapsing onto a narrow subset of operations."
                    ),
                    current_value=entropy,
                    threshold=self.min_diversity_entropy,
                    step=step,
                )
            )

        if mean_disagreement > self.max_disagreement_threshold:
            alerts.append(
                HackingAlert(
                    alert_type="judge_disagreement_divergence",
                    severity="warning",
                    message=(
                        f"Cross-judge disagreement {mean_disagreement:.3f} exceeds threshold "
                        f"{self.max_disagreement_threshold:.3f}. Gains may reflect single-judge exploitation."
                    ),
                    current_value=mean_disagreement,
                    threshold=self.max_disagreement_threshold,
                    step=step,
                )
            )

        snapshot = DashboardSnapshot(
            step=step,
            num_rollouts=len(self._history),
            guard_violation_rate=guard_violation_rate,
            advice_diversity_entropy=entropy,
            mean_judge_disagreement=mean_disagreement,
            mean_reward=mean_reward,
            mean_ag=mean_ag,
            mean_composite_drift=mean_drift,
            op_distribution=op_distribution,
            alerts=alerts,
        )

        self._step_snapshots.append(snapshot)
        return snapshot

    def is_healthy(self, window_steps: int = 5) -> bool:
        """Returns True if recent steps show zero critical alerts and healthy diversity."""
        if not self._step_snapshots:
            return True
        recent = self._step_snapshots[-window_steps:]
        for snap in recent:
            if any(a.severity == "critical" for a in snap.alerts):
                return False
        return True
