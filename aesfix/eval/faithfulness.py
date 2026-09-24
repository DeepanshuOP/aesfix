"""Faithfulness & Interpretability Evaluation Suite (AF-1.3 / Contribution C3).

Why formal faithfulness evaluation is required:
A model claiming interpretability must prove that its explanations faithfully mirror
its internal decision logic rather than merely generating plausible post-hoc rationalizations.
This module implements the two canonical faithfulness benchmarks:
1. Monotonic Intervention Compliance:
   Verifies that increasing any aesthetic attribute (e.g. +0.2 EV or sharpness)
   monotonically non-decreases the overall aesthetic rating (no paradoxical penalties).
2. Deletion / Insertion Curves:
   - Deletion: Progressively neutralizing the most important attributes should rapidly
     decay the score. Lower Deletion AUC indicates high attribution fidelity.
   - Insertion: Progressively restoring attributes from a neutral baseline should rapidly
     recover the score. Higher Insertion AUC confirms meaningful feature attribution.
3. Table 1 Benchmark:
   Directly benchmarks the learned ANFIS head against the legacy 14-rule Mamdani system
   (the CritiQ baseline), satisfying the roadmap's Table 1 Definition of Done.
"""
from __future__ import annotations

import logging
import statistics as stats
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("aesfix.eval")


@dataclass
class FaithfulnessMetrics:
    """Quantitative faithfulness metrics for an aesthetic aggregation model."""
    model_name: str
    rule_count: int
    monotonic_compliance_rate: float  # Percentage [0.0, 100.0]
    deletion_auc: float               # Lower is better
    insertion_auc: float              # Higher is better
    explainability_type: str


def evaluate_monotonic_compliance(
    predict_fn: Callable[[List[float]], float],
    samples: Sequence[Sequence[float]],
    delta: float = 0.20,
    tolerance: float = 1e-4,
) -> float:
    """Measure the percentage of single-aspect positive interventions that non-decrease score.

    Why monotonic compliance matters:
    Photographers expect that fixing a flaw (e.g. lifting underexposure) will never
    arbitrarily degrade overall image grade. Non-monotonic behavior destroys user trust.
    """
    total_interventions = 0
    passed_interventions = 0

    for sample in samples:
        base_x = [float(v) for v in sample]
        base_score = predict_fn(base_x)

        for dim in range(len(base_x)):
            # Positive perturbation check: x_dim + delta
            if base_x[dim] + delta <= 1.0 + 1e-5:
                boosted_x = list(base_x)
                boosted_x[dim] = min(1.0, boosted_x[dim] + delta)
                boosted_score = predict_fn(boosted_x)

                total_interventions += 1
                if boosted_score >= base_score - tolerance:
                    passed_interventions += 1

            # Negative perturbation check: x_dim - delta
            if base_x[dim] - delta >= -1e-5:
                reduced_x = list(base_x)
                reduced_x[dim] = max(0.0, reduced_x[dim] - delta)
                reduced_score = predict_fn(reduced_x)

                total_interventions += 1
                if reduced_score <= base_score + tolerance:
                    passed_interventions += 1

    if total_interventions == 0:
        return 100.0
    return round((passed_interventions / total_interventions) * 100.0, 2)


def evaluate_deletion_insertion(
    predict_fn: Callable[[List[float]], float],
    samples: Sequence[Sequence[float]],
    baseline_value: float = 0.50,
) -> Tuple[float, float]:
    """Compute Deletion AUC and Insertion AUC across a collection of samples.

    Features are ranked by distance from neutral baseline (most prominent first).
    Trapezoidal integration evaluates the area under the trajectory across steps.
    """
    deletion_aucs: List[float] = []
    insertion_aucs: List[float] = []

    for sample in samples:
        x = [float(v) for v in sample]
        num_features = len(x)
        if num_features == 0:
            continue

        # Sort features by saliency: largest deviation from neutral baseline
        feature_order = sorted(
            range(num_features),
            key=lambda i: abs(x[i] - baseline_value),
            reverse=True,
        )

        # 1. Deletion curve: start with full input, progressively neutralize features
        del_scores = [predict_fn(x)]
        cur_del = list(x)
        for idx in feature_order:
            cur_del[idx] = baseline_value
            del_scores.append(predict_fn(cur_del))

        # Normalized trapezoidal AUC over [0, 1] step progression
        steps = np.linspace(0.0, 1.0, len(del_scores))
        del_auc = float(np.trapezoid(del_scores, steps))
        deletion_aucs.append(del_auc)

        # 2. Insertion curve: start with all-neutral baseline, progressively restore features
        cur_ins = [baseline_value] * num_features
        ins_scores = [predict_fn(cur_ins)]
        for idx in feature_order:
            cur_ins[idx] = x[idx]
            ins_scores.append(predict_fn(cur_ins))

        ins_auc = float(np.trapezoid(ins_scores, steps))
        insertion_aucs.append(ins_auc)

    mean_del = round(float(stats.fmean(deletion_aucs)), 4) if deletion_aucs else 0.0
    mean_ins = round(float(stats.fmean(insertion_aucs)), 4) if insertion_aucs else 0.0
    return mean_del, mean_ins


def benchmark_table1_interpretability(
    anfis_model: Any,
    eval_samples: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Generate Paper Table 1 comparing ANFIS against the legacy Mamdani fuzzy baseline.

    Why this function fulfills DoD for AF-1.3:
    Roadmap §38 DoD mandates: 'Table 1 exists; legacy fuzzy included as row.'
    This function benchmarks both architectures under identical perturbation protocols.
    """
    if eval_samples is None:
        # Standardized 25-point evaluation grid covering corner, edge, and interior regions
        grid_vals = [0.15, 0.35, 0.50, 0.65, 0.85]
        eval_samples = [
            [v, v, v, v, v] for v in grid_vals
        ] + [
            [0.85, 0.15, 0.50, 0.70, 0.60],
            [0.15, 0.85, 0.70, 0.40, 0.50],
            [0.50, 0.50, 0.85, 0.20, 0.80],
            [0.90, 0.90, 0.30, 0.30, 0.50],
            [0.30, 0.70, 0.70, 0.80, 0.40],
        ]

    # Wrapper for ANFIS model
    import torch

    def _anfis_predict(x_list: List[float]) -> float:
        anfis_model.eval()
        with torch.no_grad():
            t = torch.tensor([x_list], dtype=torch.float32)
            return float(anfis_model(t).squeeze().item())

    # Wrapper for Legacy Mamdani Fuzzy System
    from baselines.legacy_fuzzy.our_system.fuzzy_system import (
        build_fuzzy_system,
        evaluate_aesthetics,
    )
    legacy_sim = build_fuzzy_system()

    def _legacy_predict(x_list: List[float]) -> float:
        _label, score = evaluate_aesthetics(
            legacy_sim,
            x_list[0], x_list[1], x_list[2], x_list[3], x_list[4],
        )
        return float(score)

    # Wrapper for Simple Linear Baseline (Weighted Mean)
    def _linear_predict(x_list: List[float]) -> float:
        weights = [0.25, 0.25, 0.15, 0.20, 0.15]
        return float(sum(w * x for w, x in zip(weights, x_list)))

    models_to_test = [
        ("ANFIS Head (C3, Ours)", _anfis_predict, anfis_model.num_rules, "Differentiable Rulebook"),
        ("Legacy Mamdani (CritiQ)", _legacy_predict, 14, "Fixed Mamdani Rules"),
        ("Linear Baseline", _linear_predict, 1, "Linear Weights"),
    ]

    table_rows: List[FaithfulnessMetrics] = []
    for name, predict_fn, rule_count, expl_type in models_to_test:
        mono_rate = evaluate_monotonic_compliance(predict_fn, eval_samples)
        del_auc, ins_auc = evaluate_deletion_insertion(predict_fn, eval_samples)

        table_rows.append(
            FaithfulnessMetrics(
                model_name=name,
                rule_count=rule_count,
                monotonic_compliance_rate=mono_rate,
                deletion_auc=del_auc,
                insertion_auc=ins_auc,
                explainability_type=expl_type,
            )
        )

    # Format Markdown Table 1 for paper and audit reports
    md_lines = [
        "| Architecture | Rules | Monotonic Compliance | Deletion AUC (↓) | Insertion AUC (↑) | Mechanism |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for row in table_rows:
        md_lines.append(
            f"| **{row.model_name}** | {row.rule_count} | {row.monotonic_compliance_rate:.1f}% | "
            f"{row.deletion_auc:.4f} | {row.insertion_auc:.4f} | {row.explainability_type} |"
        )
    markdown_table = "\n".join(md_lines)

    return {
        "metrics": table_rows,
        "markdown_table": markdown_table,
    }
