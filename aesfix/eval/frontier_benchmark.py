"""Visual Advice Benchmark Suite & External Advice Runner (AF-2.6 & Roadmap v6).

Evaluates photographic advice policies and external advice JSON traces against the
frozen judge ensemble and Identity Guard under the AG@k metric framework.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

from aesfix.critic.critic import CriticFix
from aesfix.eval.ag_benchmark import AGBenchmarkReport, AGEvaluationHarness

logger = logging.getLogger("aesfix.eval.frontier_benchmark")


@dataclass
class FrontierComparisonTable:
    """Consolidated multi-model comparison table representing benchmark standings."""
    reports: List[AGBenchmarkReport] = field(default_factory=list)
    dataset_name: str = "Benchmark-Subset"

    def to_markdown(self) -> str:
        """Formats the multi-model comparison into markdown table."""
        lines = [
            f"### Visual Advice Aesthetic Gain Benchmark ({self.dataset_name})",
            "",
            "| Policy / Model | Backbone | AG@1 (Best) | AG@3 (Comp) | Guard Pass % | AVR % |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
        for rep in self.reports:
            backbone = rep.metadata.get("backbone", "N/A")
            lines.append(
                f"| {rep.model_name:<28} | {backbone:<17} | {rep.mean_ag_at_1:+.4f} | {rep.mean_ag_at_k:+.4f} | "
                f"{rep.guard_pass_rate*100:5.1f}% | {rep.aspect_verification_rate*100:5.1f}% |"
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the comparison table to dictionary format."""
        return {
            "dataset_name": self.dataset_name,
            "reports": [json.loads(r.to_json()) for r in self.reports],
        }


# --- Baseline and External Advice Policies ---

def make_random_advice_policy(seed: Optional[int] = 42) -> Callable[[Image.Image], List[CriticFix]]:
    """Generates random valid photographic operations as an unguided control baseline."""
    rng = random.Random(seed)
    ops = ["adjust_exposure", "adjust_contrast", "adjust_saturation", "adjust_white_balance"]
    aspects = ["lighting", "technical", "color", "composition"]

    def policy(img: Image.Image) -> List[CriticFix]:
        fixes: List[CriticFix] = []
        for _ in range(3):
            op = rng.choice(ops)
            if op == "adjust_exposure":
                params = {"ev_delta": round(rng.uniform(-0.8, 0.8), 2)}
            elif op == "adjust_contrast":
                params = {"factor": round(rng.uniform(0.6, 1.4), 2)}
            elif op == "adjust_saturation":
                params = {"factor": round(rng.uniform(0.5, 1.5), 2)}
            else:
                params = {"temperature": round(rng.uniform(-25.0, 25.0), 1), "tint": round(rng.uniform(-15.0, 15.0), 1)}

            fixes.append(CriticFix(
                op=op,
                params=params,
                expected_aspect=rng.choice(aspects),
                nl_instruction="Random control recommendation.",
            ))
        return fixes

    return policy


def simulate_frontier_vlm_generic_policy(seed: Optional[int] = 42) -> Callable[[Image.Image], List[CriticFix]]:
    """[SIMULATED] Generates generic uncalibrated photographic edits for baseline unit tests.

    Outputs are explicitly stamped as SIMULATED and must never be attributed to real models.
    """
    rng = random.Random(seed)

    def policy(img: Image.Image) -> List[CriticFix]:
        return [
            CriticFix(
                op="adjust_contrast",
                params={"factor": round(1.05 + rng.uniform(-0.02, 0.05), 2)},
                expected_aspect="technical",
                nl_instruction="[SIMULATED] Mild contrast adjustment.",
            ),
            CriticFix(
                op="adjust_saturation",
                params={"factor": round(1.03 + rng.uniform(-0.02, 0.04), 2)},
                expected_aspect="color",
                nl_instruction="[SIMULATED] Mild saturation adjustment.",
            ),
            CriticFix(
                op="adjust_white_balance",
                params={"temperature": round(rng.uniform(2.0, 6.0), 1), "tint": 0.0},
                expected_aspect="color",
                nl_instruction="[SIMULATED] Mild white balance shift.",
            ),
        ]

    return policy


# Alias for backward compatibility in internal test suites
make_frontier_vlm_generic_policy = simulate_frontier_vlm_generic_policy


def make_advice_json_policy(
    advice_source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]
) -> Callable[[Image.Image], List[CriticFix]]:
    """Creates a policy from real external advice JSON traces.

    Expects advice_source to contain a mapping from image identifier/index to a list
    of fixes, or a list of predictions with 'fixes' dictionaries.
    """
    if isinstance(advice_source, (str, Path)):
        p = Path(advice_source)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = advice_source

    # Pre-parse advice items
    advice_lookup: Dict[str, List[CriticFix]] = {}
    if isinstance(data, dict) and "predictions" in data:
        items = data["predictions"]
    elif isinstance(data, list):
        items = data
    else:
        items = [data]

    for idx, item in enumerate(items):
        key = str(item.get("image_id", idx))
        raw_fixes = item.get("fixes", [])
        fixes: List[CriticFix] = []
        for rf in raw_fixes:
            if isinstance(rf, CriticFix):
                fixes.append(rf)
            elif isinstance(rf, dict):
                fixes.append(CriticFix(
                    op=rf["op"],
                    params=rf.get("params", {}),
                    expected_aspect=rf.get("expected_aspect", "technical"),
                    nl_instruction=rf.get("nl_instruction", ""),
                    direction=rf.get("direction", 1),
                ))
        advice_lookup[key] = fixes

    call_counter = [0]

    def policy(img: Image.Image) -> List[CriticFix]:
        idx_str = str(call_counter[0])
        call_counter[0] += 1
        return advice_lookup.get(idx_str, advice_lookup.get("0", []))

    return policy


def make_aesfix_grounded_critic_policy() -> Callable[[Image.Image], List[CriticFix]]:
    """Uses AesFix's diagnostic aspect critic to propose calibrated adjustments."""
    def policy(img: Image.Image) -> List[CriticFix]:
        arr = np.array(img, dtype=np.float32) / 255.0
        mean_lum = float(np.mean(arr))
        contrast = float(np.std(arr))

        fixes: List[CriticFix] = []

        # 1. Lighting diagnostic
        if mean_lum < 0.42:
            ev_boost = round(min(1.0, (0.50 - mean_lum) * 2.2), 2)
            fixes.append(CriticFix(
                op="adjust_exposure",
                params={"ev_delta": ev_boost},
                expected_aspect="lighting",
                nl_instruction=f"Brighten underexposed image by +{ev_boost:.2f} EV.",
            ))
        elif mean_lum > 0.58:
            ev_cut = round(max(-1.0, (0.50 - mean_lum) * 2.2), 2)
            fixes.append(CriticFix(
                op="adjust_exposure",
                params={"ev_delta": ev_cut},
                expected_aspect="lighting",
                nl_instruction=f"Tame overexposed image by {ev_cut:.2f} EV.",
            ))

        # 2. Contrast diagnostic
        if contrast < 0.18:
            fixes.append(CriticFix(
                op="adjust_contrast",
                params={"factor": 1.25},
                expected_aspect="technical",
                nl_instruction="Boost flat midtone contrast by 25%.",
            ))

        # 3. Saturation fallback
        if len(fixes) < 3:
            fixes.append(CriticFix(
                op="adjust_saturation",
                params={"factor": 1.15},
                expected_aspect="color",
                nl_instruction="Gently enrich color saturation.",
            ))

        return fixes

    return policy


class FrontierBenchmarkSuite:
    """Runner for evaluating advice policies and external advice JSON against the benchmark."""

    def __init__(self, harness: Optional[AGEvaluationHarness] = None) -> None:
        self.harness = harness or AGEvaluationHarness()

    def run_advice_benchmark(
        self,
        dataset: Sequence[Union[Image.Image, str, Path]],
        advice_source: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
        model_name: str,
        backbone: str = "External Advice JSON",
        k: int = 3,
    ) -> AGBenchmarkReport:
        """Runs evaluation against real external advice traces."""
        policy = make_advice_json_policy(advice_source)
        rep = self.harness.run_benchmark(dataset, policy, model_name=model_name, k=k)
        rep.metadata["backbone"] = backbone
        return rep

    def run_suite(
        self,
        dataset: Sequence[Union[Image.Image, str, Path]],
        policies: Optional[List[tuple[str, str, Callable[[Image.Image], List[CriticFix]]]]] = None,
        k: int = 3,
        dataset_name: str = "Benchmark-Subset",
    ) -> FrontierComparisonTable:
        """Runs benchmarking across provided policies or default baselines."""
        if policies is None:
            policies = [
                ("Random Advice Control", "Uniform Sampling", make_random_advice_policy(seed=42)),
                ("AesFix Grounded Critic", "AspectCritic-TS", make_aesfix_grounded_critic_policy()),
            ]

        reports: List[AGBenchmarkReport] = []
        for name, backbone, pol in policies:
            rep = self.harness.run_benchmark(dataset, pol, model_name=name, k=k)
            rep.metadata["backbone"] = backbone
            reports.append(rep)

        return FrontierComparisonTable(reports=reports, dataset_name=dataset_name)
