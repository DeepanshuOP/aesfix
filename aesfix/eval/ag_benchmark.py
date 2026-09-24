"""Aesthetic Gain (AG@k) Benchmark & Evaluation Harness (AF-2.5 & Paper Contribution C1).

Why AG@k is the first behavioral benchmark for visual advice:
1. Beyond Passive Scoring: Traditional vision models (Q-Align, AesExpert, NIMA) merely emit
   scalar scores or freeform text critiques that are never tested. AG@k executes the model's
   own advice and measures whether the photo actually improved under an independent frozen judge.
2. Anti-Reward-Hacking Drift Gating: Evaluated edits must pass the IdentityGuard. Any edit that
   alters subject identity or introduces hallucinated artifacts has its gain strictly clamped to 0.0.
3. Aspect Falsifiability: Every fix declares an `expected_aspect`. AG@k verifies whether the specific
   aspect probe (e.g. lighting probe for an exposure fix) registered a positive delta.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from aesfix.critic.critic import CriticFix
from aesfix.executor.guard import GuardVerdict, IdentityGuard
from aesfix.executor.tier_a import TierAExecutor


@dataclass
class AGSampleResult:
    """Evaluation result for a single benchmark image under a critic's advice.

    Why this dataclass exists:
    Provides transparent before/after telemetry for auditing individual advice recommendations
    and building the AGX public leaderboard.
    """
    image_id: str
    before_score: float
    after_score: float
    raw_gain: float
    verified_gain: float
    is_guard_approved: bool
    applied_fixes: List[Dict[str, Any]]
    aspect_deltas: Dict[str, float] = field(default_factory=dict)
    guard_verdict: Optional[GuardVerdict] = None


@dataclass
class AGBenchmarkReport:
    """Consolidated benchmark report containing AG@k summary statistics and leaderboard tables.

    Why this report format:
    Directly formats Paper Table 2 and the living AGX HuggingFace Space leaderboard.
    """
    model_name: str
    k: int
    num_samples: int
    mean_ag_at_1: float
    mean_ag_at_k: float
    guard_pass_rate: float
    aspect_verification_rate: float
    sample_results: List[AGSampleResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown_row(self) -> str:
        """Formats summary metrics as a markdown table row."""
        return (
            f"| {self.model_name:<28} | {self.mean_ag_at_1:+.4f} | {self.mean_ag_at_k:+.4f} | "
            f"{self.guard_pass_rate*100:5.1f}% | {self.aspect_verification_rate*100:5.1f}% |"
        )

    def to_json(self, indent: int = 2) -> str:
        """Serializes the benchmark report to JSON format."""
        summary = {
            "model_name": self.model_name,
            "k": self.k,
            "num_samples": self.num_samples,
            "mean_ag_at_1": self.mean_ag_at_1,
            "mean_ag_at_k": self.mean_ag_at_k,
            "guard_pass_rate": self.guard_pass_rate,
            "aspect_verification_rate": self.aspect_verification_rate,
            "metadata": self.metadata,
        }
        return json.dumps(summary, indent=indent)


class AGEvaluationHarness:
    """Evaluation harness running the AG@k protocol over benchmark image sets."""

    def __init__(
        self,
        scorer_fn: Optional[Callable[[Image.Image], Tuple[float, Dict[str, float]]]] = None,
        executor: Optional[TierAExecutor] = None,
        guard: Optional[IdentityGuard] = None,
    ) -> None:
        """Initializes the AG@k harness with custom or default scoring and execution backends."""
        self.executor = executor or TierAExecutor()
        self.guard = guard or IdentityGuard(drift_threshold=0.35)
        self._scorer_fn = scorer_fn or self._default_mock_scorer

    @staticmethod
    def _default_mock_scorer(img: Image.Image) -> Tuple[float, Dict[str, float]]:
        """Fast fallback scorer for unit testing and CI pipelines without GPU dependencies."""
        arr = np.array(img, dtype=np.float32) / 255.0
        # Estimate basic photographic attributes
        mean_lum = float(np.mean(arr))
        contrast = float(np.std(arr))
        # Optimal mean luminance is ~0.50
        lighting = max(0.0, 1.0 - 2.0 * abs(mean_lum - 0.50))
        technical = min(1.0, contrast * 3.5)
        overall = 0.5 * lighting + 0.5 * technical
        aspects = {"lighting": lighting, "technical": technical, "composition": 0.6, "color": 0.6}
        return round(overall, 4), aspects

    def evaluate_sample(
        self,
        image_input: Union[Image.Image, str, Path],
        candidate_fixes: Sequence[Union[CriticFix, Dict[str, Any]]],
        image_id: str = "sample",
        k: int = 3,
    ) -> AGSampleResult:
        """Evaluates AG@k for a single image under a ranked list of candidate fixes.

        Why sequential execution:
        Applies fixes in rank order up to k, evaluating both single-best (AG@1)
        and cumulative compound improvements (AG@k).
        """
        if isinstance(image_input, (str, Path)):
            pil_orig = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_orig = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # 1. Baseline scoring before any edits
        before_score, before_aspects = self._scorer_fn(pil_orig)

        # Slice top-k fixes
        top_k_fixes = list(candidate_fixes[:k])
        if not top_k_fixes:
            return AGSampleResult(
                image_id=image_id,
                before_score=before_score,
                after_score=before_score,
                raw_gain=0.0,
                verified_gain=0.0,
                is_guard_approved=True,
                applied_fixes=[],
                aspect_deltas={},
            )

        # 2. Execute cumulative fixes sequentially
        exec_res = self.executor.execute_fixes(pil_orig, top_k_fixes)
        pil_edited = exec_res.image

        # 3. Post-execution scoring
        after_score, after_aspects = self._scorer_fn(pil_edited)
        raw_gain = max(0.0, after_score - before_score)

        # 4. Identity Guard verification
        guard_verdict = self.guard.evaluate(pil_orig, pil_edited)
        verified_gain = round(raw_gain, 4) if guard_verdict.is_valid else 0.0

        # 5. Compute per-aspect deltas
        aspect_deltas: Dict[str, float] = {}
        for asp, post_val in after_aspects.items():
            pre_val = before_aspects.get(asp, 0.5)
            aspect_deltas[asp] = round(post_val - pre_val, 4)

        serialized_fixes = [
            f.model_dump() if isinstance(f, CriticFix) else dict(f)
            for f in top_k_fixes
        ]

        return AGSampleResult(
            image_id=image_id,
            before_score=before_score,
            after_score=after_score,
            raw_gain=round(raw_gain, 4),
            verified_gain=verified_gain,
            is_guard_approved=guard_verdict.is_valid,
            applied_fixes=serialized_fixes,
            aspect_deltas=aspect_deltas,
            guard_verdict=guard_verdict,
        )

    def run_benchmark(
        self,
        images: Sequence[Union[Image.Image, str, Path]],
        critic_policy: Callable[[Image.Image], List[CriticFix]],
        model_name: str = "AspectCritic-v1",
        k: int = 3,
    ) -> AGBenchmarkReport:
        """Executes full AG@k evaluation protocol across a dataset of images.

        Why this method:
        Standardizes automated benchmarking across models, random baselines, and oracle ceilings.
        """
        sample_results: List[AGSampleResult] = []
        ag_at_1_list: List[float] = []
        ag_at_k_list: List[float] = []
        guard_passes: int = 0
        aspect_verified_count: int = 0
        aspect_target_count: int = 0

        for idx, img in enumerate(images):
            pil_img = Image.open(img).convert("RGB") if isinstance(img, (str, Path)) else img.convert("RGB")
            # Critic emits ranked fixes
            fixes = critic_policy(pil_img)

            # Evaluate AG@1 (single top fix)
            res_k1 = self.evaluate_sample(pil_img, fixes[:1], image_id=f"img_{idx:04d}_k1", k=1)
            ag_at_1_list.append(res_k1.verified_gain)

            # Evaluate AG@k (compound top-k fixes)
            res_k = self.evaluate_sample(pil_img, fixes, image_id=f"img_{idx:04d}_k{k}", k=k)
            sample_results.append(res_k)
            ag_at_k_list.append(res_k.verified_gain)

            if res_k.is_guard_approved:
                guard_passes += 1

            # Check if declared expected_aspect improved
            for fix_item in fixes[:k]:
                exp_asp = (
                    fix_item.expected_aspect
                    if isinstance(fix_item, CriticFix)
                    else fix_item.get("expected_aspect")
                )
                if exp_asp:
                    aspect_target_count += 1
                    if res_k.aspect_deltas.get(exp_asp, 0.0) > 0.0:
                        aspect_verified_count += 1

        n = len(images)
        mean_ag_1 = float(np.mean(ag_at_1_list)) if ag_at_1_list else 0.0
        mean_ag_k = float(np.mean(ag_at_k_list)) if ag_at_k_list else 0.0
        pass_rate = (guard_passes / n) if n > 0 else 0.0
        avr = (aspect_verified_count / aspect_target_count) if aspect_target_count > 0 else 0.0

        return AGBenchmarkReport(
            model_name=model_name,
            k=k,
            num_samples=n,
            mean_ag_at_1=round(mean_ag_1, 4),
            mean_ag_at_k=round(mean_ag_k, 4),
            guard_pass_rate=round(pass_rate, 4),
            aspect_verification_rate=round(avr, 4),
            sample_results=sample_results,
        )
