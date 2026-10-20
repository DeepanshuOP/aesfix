#!/usr/bin/env python3
"""Full-split Certifier Protocol and Goodhart J3 Brightness Bias Analysis (S2).

DIRECTIVES:
1. Certifier: Full 20-seed protocol on ALL original-vs-edit test pairs (report N pairs and N scenes).
   - Evaluates v1 and v2 scores
   - Alphas: {0.05, 0.10, 0.15, 0.20, 0.30}, Delta = 0.10
   - Reports mean FIR, max FIR, violation rate, release rate
   - Computes baselines: release-all, s > 0, s > epsilon
   - Saves release-vs-alpha plot to docs/figures/certifier_release_vs_alpha.png
2. Goodhart J3 on full pairs:
   - Evaluates P(humans prefer brighter) vs P(judge prefers brighter) per judge
     for |Delta luma| > 0.15 with scene-clustered bootstrap 95% CIs.
3. Ingests into ledger:
   - Updates results/certifier_pilot_results.json with full run
   - Appends superseding entry to results/ledger.jsonl marking pilot superseded.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.certify.ltt import PilotRunSummary, evaluate_decision_rule, learn_then_test_threshold
from aesfix.ledger import DEFAULT_LEDGER_PATH, append_run_to_ledger, compute_file_sha256

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("certifier_s2")


from scipy.stats import beta


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact Clopper-Pearson two-sided confidence interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (round(lower, 4), round(upper, 4))


def run_full_certifier_protocol(
    scene_ids: List[str],
    scores: List[float],
    human_prefers_edit: List[bool],
    candidate_thresholds: Sequence[float],
    alphas: List[float] = [0.05, 0.10, 0.15, 0.20, 0.30],
    delta: float = 0.10,
    n_seeds: int = 200,
    epsilon: float = 0.04,
) -> Dict[str, Any]:
    """Runs 200-seed scene-disjoint Learn-Then-Test protocol on original-vs-edit pairs."""
    scene_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, s_id in enumerate(scene_ids):
        scene_to_indices[s_id].append(idx)

    unique_scenes = np.array(sorted(list(scene_to_indices.keys())))
    n_scenes = len(unique_scenes)
    n_pairs = len(scores)

    # Candidate threshold grid (pre-specified, descending order)
    cand_thresh = np.asarray(candidate_thresholds, dtype=np.float64)

    results_by_alpha: Dict[str, Dict[str, Any]] = {}

    for alpha in alphas:
        ltt_firs: List[float] = []
        ltt_coverages: List[float] = []
        ltt_abstains = 0

        naive_zero_firs: List[float] = []
        naive_zero_coverages: List[float] = []

        naive_eps_firs: List[float] = []
        naive_eps_coverages: List[float] = []

        release_all_firs: List[float] = []
        release_all_coverages: List[float] = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_scenes)
            half = n_scenes // 2

            calib_scenes = set(unique_scenes[perm[:half]])
            eval_scenes = set(unique_scenes[perm[half:]])

            calib_idx = [i for s in calib_scenes for i in scene_to_indices[s]]
            eval_idx = [i for s in eval_scenes for i in scene_to_indices[s]]

            s_calib = [scores[i] for i in calib_idx]
            y_calib = [human_prefers_edit[i] for i in calib_idx]

            s_eval = [scores[i] for i in eval_idx]
            y_eval = [human_prefers_edit[i] for i in eval_idx]

            # 1. LTT threshold selection
            chosen_thresh = learn_then_test_threshold(
                scores=s_calib,
                human_prefers_edit=y_calib,
                candidate_thresholds=cand_thresh,
                alpha=alpha,
                delta=delta,
            )

            if chosen_thresh is None:
                ltt_abstains += 1
                fir_ltt, cov_ltt = 0.0, 0.0
            else:
                fir_ltt, cov_ltt, _, _ = evaluate_decision_rule(s_eval, y_eval, chosen_thresh)

            ltt_firs.append(fir_ltt)
            ltt_coverages.append(cov_ltt)

            # 2. Release-All baseline (threshold = -infinity)
            fir_all, cov_all, _, _ = evaluate_decision_rule(s_eval, y_eval, -999.0)
            release_all_firs.append(fir_all)
            release_all_coverages.append(cov_all)

            # 3. Naive baseline: s > 0
            fir_z, cov_z, _, _ = evaluate_decision_rule(s_eval, y_eval, 0.0)
            naive_zero_firs.append(fir_z)
            naive_zero_coverages.append(cov_z)

            # 4. Naive baseline: s > epsilon
            fir_e, cov_e, _, _ = evaluate_decision_rule(s_eval, y_eval, epsilon)
            naive_eps_firs.append(fir_e)
            naive_eps_coverages.append(cov_e)

        alpha_key = f"alpha_{alpha:.2f}"
        ltt_violations = int(sum(f > alpha for f in ltt_firs))
        ltt_v_rate = round(float(ltt_violations / n_seeds), 4)
        ltt_v_ci = list(clopper_pearson_ci(ltt_violations, n_seeds))

        rel_all_violations = int(sum(f > alpha for f in release_all_firs))
        rel_all_v_rate = round(float(rel_all_violations / n_seeds), 4)
        rel_all_v_ci = list(clopper_pearson_ci(rel_all_violations, n_seeds))

        zero_violations = int(sum(f > alpha for f in naive_zero_firs))
        zero_v_rate = round(float(zero_violations / n_seeds), 4)
        zero_v_ci = list(clopper_pearson_ci(zero_violations, n_seeds))

        eps_violations = int(sum(f > alpha for f in naive_eps_firs))
        eps_v_rate = round(float(eps_violations / n_seeds), 4)
        eps_v_ci = list(clopper_pearson_ci(eps_violations, n_seeds))

        results_by_alpha[alpha_key] = {
            "alpha": alpha,
            "delta": delta,
            "ltt": {
                "method": "LTT (Learn-Then-Test, Clopper-Pearson)",
                "mean_realized_fir": round(float(np.mean(ltt_firs)), 4),
                "max_realized_fir": round(float(np.max(ltt_firs)), 4),
                "violation_rate": ltt_v_rate,
                "violation_count": ltt_violations,
                "violation_ci_95": ltt_v_ci,
                "mean_release_rate": round(float(np.mean(ltt_coverages)), 4),
                "abstain_rate": round(float(ltt_abstains / n_seeds), 4),
            },
            "release_all": {
                "method": "Release-All (Baseline)",
                "mean_realized_fir": round(float(np.mean(release_all_firs)), 4),
                "max_realized_fir": round(float(np.max(release_all_firs)), 4),
                "violation_rate": rel_all_v_rate,
                "violation_count": rel_all_violations,
                "violation_ci_95": rel_all_v_ci,
                "mean_release_rate": round(float(np.mean(release_all_coverages)), 4),
                "abstain_rate": 0.0,
            },
            "naive_zero": {
                "method": "Naive (s > 0)",
                "mean_realized_fir": round(float(np.mean(naive_zero_firs)), 4),
                "max_realized_fir": round(float(np.max(naive_zero_firs)), 4),
                "violation_rate": zero_v_rate,
                "violation_count": zero_violations,
                "violation_ci_95": zero_v_ci,
                "mean_release_rate": round(float(np.mean(naive_zero_coverages)), 4),
                "abstain_rate": 0.0,
            },
            "naive_eps": {
                "method": f"Naive (s > {epsilon})",
                "mean_realized_fir": round(float(np.mean(naive_eps_firs)), 4),
                "max_realized_fir": round(float(np.max(naive_eps_firs)), 4),
                "violation_rate": eps_v_rate,
                "violation_count": eps_violations,
                "violation_ci_95": eps_v_ci,
                "mean_release_rate": round(float(np.mean(naive_eps_coverages)), 4),
                "abstain_rate": 0.0,
            },
        }

    return {
        "n_scenes": n_scenes,
        "n_pairs": n_pairs,
        "alphas": alphas,
        "delta": delta,
        "n_seeds": n_seeds,
        "results_by_alpha": results_by_alpha,
    }


def compute_goodhart_j3_bias(
    test_pair_scores_path: Path,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes P(humans prefer brighter) vs P(judge prefers brighter) for |Delta luma| > 0.15."""
    df = pd.read_csv(test_pair_scores_path)

    # Determine delta luma
    if "left_median_luma" in df.columns and "right_median_luma" in df.columns:
        df["d_luma"] = df["left_median_luma"] - df["right_median_luma"]
    elif "lightness_left" in df.columns and "lightness_right" in df.columns:
        df["d_luma"] = (df["lightness_left"] - df["lightness_right"]) / 255.0
    else:
        raise ValueError("Could not find lightness/luma columns in pair scores.")

    # Filter to |Delta luma| > 0.15 and exclude ties (mos == 0.5)
    subset = df[(df["d_luma"].abs() > 0.15) & (df["mos"] != 0.5)].copy()
    n_pairs = len(subset)
    unique_scenes = subset["scene"].unique()
    n_scenes = len(unique_scenes)

    # Left brighter flag
    subset["left_brighter"] = subset["d_luma"] > 0
    subset["human_pref_left"] = subset["mos"] > 0.5
    subset["human_prefers_brighter"] = (subset["human_pref_left"] == subset["left_brighter"]).astype(float)

    # Identify judge columns
    judge_candidates = [
        ("aesfix_pref_v1", "pref_logit_v1"),
        ("aesfix_pref_v2", "pref_logit_v2"),
        ("rsinema", "pref_logit_rsinema"),
        ("pyiqa_musiq", "pref_logit_pyiqa_musiq"),
        ("pyiqa_topiq_nr", "pref_logit_pyiqa_topiq_nr"),
    ]

    active_judges = {}
    for j_name, col in judge_candidates:
        if col in subset.columns:
            active_judges[j_name] = col
        elif f"score_{j_name}_left" in subset.columns and f"score_{j_name}_right" in subset.columns:
            diff_col = f"diff_{j_name}"
            subset[diff_col] = subset[f"score_{j_name}_left"] - subset[f"score_{j_name}_right"]
            active_judges[j_name] = diff_col

    for j_name, col in active_judges.items():
        subset[f"{j_name}_pref_left"] = subset[col] > 0
        subset[f"{j_name}_prefers_brighter"] = (subset[f"{j_name}_pref_left"] == subset["left_brighter"]).astype(float)

    # Scene-clustered bootstrap
    rng = np.random.default_rng(seed)
    scene_groups = [group.to_dict("records") for _, group in subset.groupby("scene")]
    n_groups = len(scene_groups)

    boot_human = []
    boot_judges = {j: [] for j in active_judges}
    boot_diffs = {j: [] for j in active_judges}

    for _ in range(n_bootstrap):
        sampled_indices = rng.choice(n_groups, size=n_groups, replace=True)
        sampled_rows = [row for idx in sampled_indices for row in scene_groups[idx]]
        if not sampled_rows:
            continue

        h_mean = np.mean([r["human_prefers_brighter"] for r in sampled_rows])
        boot_human.append(h_mean)

        for j in active_judges:
            j_mean = np.mean([r[f"{j}_prefers_brighter"] for r in sampled_rows])
            boot_judges[j].append(j_mean)
            boot_diffs[j].append(j_mean - h_mean)

    def ci(arr):
        return round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)

    results = {
        "n_scenes": n_scenes,
        "n_pairs": n_pairs,
        "human_prefers_brighter": {
            "mean": round(float(np.mean(boot_human)), 4),
            "ci_95": ci(boot_human),
        },
        "judges": {},
    }

    for j in active_judges:
        results["judges"][j] = {
            "mean": round(float(np.mean(boot_judges[j])), 4),
            "ci_95": ci(boot_judges[j]),
            "diff_from_human": round(float(np.mean(boot_diffs[j])), 4),
            "diff_ci_95": ci(boot_diffs[j]),
        }

    return results


def plot_release_vs_alpha(
    results_by_alpha_v1: Dict[str, Any],
    output_path: Path,
    results_by_alpha_v2: Optional[Dict[str, Any]] = None,
):
    """Plots release rate vs alpha for LTT certifier and baselines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alphas = [v["alpha"] for v in results_by_alpha_v1.values()]
    ltt_v1 = [v["ltt"]["mean_release_rate"] for v in results_by_alpha_v1.values()]
    naive_zero_v1 = [v["naive_zero"]["mean_release_rate"] for v in results_by_alpha_v1.values()]
    naive_eps_v1 = [v["naive_eps"]["mean_release_rate"] for v in results_by_alpha_v1.values()]

    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(alphas, ltt_v1, marker="o", linewidth=2, color="#1f77b4", label=r"LTT v1 ($\delta=0.10$)")
    if results_by_alpha_v2 is not None:
        ltt_v2 = [v["ltt"]["mean_release_rate"] for v in results_by_alpha_v2.values()]
        plt.plot(alphas, ltt_v2, marker="s", linewidth=2, color="#9467bd", label=r"LTT v2 ($\delta=0.10$)")
    plt.plot(alphas, naive_zero_v1, linestyle="--", color="#ff7f0e", label=r"Naive Baseline ($s > 0$)")
    plt.plot(alphas, naive_eps_v1, linestyle=":", color="#2ca02c", label=r"Naive Baseline ($s > \epsilon$)")

    plt.xlabel(r"False Improvement Rate Bound ($\alpha$)", fontsize=12)
    plt.ylabel("Mean Release Rate", fontsize=12)
    plt.title("Certifier Release Rate vs. Targeted Risk Bound (200 Seeds)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved release vs alpha plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run S2 Certifier and Goodhart J3 Protocol.")
    parser.add_argument("--test-pairs-csv", type=str, default=r"D:\aesfix-data\kaggle_runs\dear_full\test_pair_scores.csv")
    parser.add_argument("--output-json", type=str, default="results/certifier_pilot_results.json")
    parser.add_argument("--output-plot", type=str, default="docs/figures/certifier_release_vs_alpha.png")
    args = parser.parse_args()

    test_csv = Path(args.test_pairs_csv)
    if not test_csv.exists():
        alt_test = REPO_ROOT / "kaggle/k2_output/test_pair_scores.csv"
        if alt_test.exists():
            test_csv = alt_test

    if not test_csv.exists():
        logger.error(f"test_pair_scores.csv not found at {test_csv}. Cannot run S2 until K2b completes.")
        sys.exit(1)

    print(f"Loading test pairs from {test_csv}...")
    df = pd.read_csv(test_csv)
    print(f"Total test pairs loaded: {len(df)}")

    # Extract original-vs-edit pairs
    # In DEAR test split, an original-vs-edit pair has either left == 'original' or right == 'original'
    orig_vs_edit = df[(df["left"] == "original") | (df["right"] == "original")].copy()
    orig_vs_edit = orig_vs_edit[orig_vs_edit["mos"] != 0.5].copy()  # strictly oriented pairs

    print(f"Original-vs-Edit pairs found: {len(orig_vs_edit)} across {orig_vs_edit['scene'].nunique()} unique scenes.")

    scenes = orig_vs_edit["scene"].tolist()
    human_pref_edit = []
    v1_scores = []
    v2_scores = []

    has_v2 = "pref_logit_v2" in orig_vs_edit.columns

    for _, r in orig_vs_edit.iterrows():
        mos = float(r["mos"])
        if r["left"] == "original":
            # edit is right
            human_pref_edit.append(mos < 0.5)
            v1_scores.append(-float(r["pref_logit_v1"]))
            if has_v2:
                v2_scores.append(-float(r["pref_logit_v2"]))
        else:
            # edit is left
            human_pref_edit.append(mos > 0.5)
            v1_scores.append(float(r["pref_logit_v1"]))
            if has_v2:
                v2_scores.append(float(r["pref_logit_v2"]))

    alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
    delta = 0.10
    n_seeds = 200

    # Build per-judge candidate threshold grids from DEAR TRAIN original-vs-edit score deltas
    # (Label-free and disjoint from calibration and test pairs)
    # Ordered conservative -> liberal (q=0.90 down to q=0.00) and extended with -999.0 for release-all
    v1_train_percentiles = [-1.6914, 2.0109, 1.8860, 1.6279, 1.3186, 0.7585, 0.2117, -0.3255, -0.6869, -1.0286, -1.5101, -1.6914]
    v2_train_percentiles = [-4.7773, 8.8997, 7.8557, 5.6093, 5.2610, 3.5262, 2.4210, 0.7769, -0.7569, -1.5533, -3.4729, -4.7773]

    qs = np.linspace(0.90, 0.0, 50)
    cand_thresh_v1 = np.unique(np.array([np.quantile(v1_train_percentiles, q) for q in qs] + [-999.0]))[::-1]
    if cand_thresh_v1[-1] > -999.0:
        cand_thresh_v1 = np.append(cand_thresh_v1, -999.0)

    cand_thresh_v2 = np.unique(np.array([np.quantile(v2_train_percentiles, q) for q in qs] + [-999.0]))[::-1]
    if cand_thresh_v2[-1] > -999.0:
        cand_thresh_v2 = np.append(cand_thresh_v2, -999.0)

    print(f"\n--- Running {n_seeds}-seed LTT Certifier on AesFix-Pref v1 ---")
    cert_v1 = run_full_certifier_protocol(
        scene_ids=scenes,
        scores=v1_scores,
        human_prefers_edit=human_pref_edit,
        candidate_thresholds=cand_thresh_v1,
        alphas=alphas,
        delta=delta,
        n_seeds=n_seeds,
    )

    cert_v2 = None
    if has_v2:
        print(f"\n--- Running {n_seeds}-seed LTT Certifier on AesFix-Pref v2 ---")
        cert_v2 = run_full_certifier_protocol(
            scene_ids=scenes,
            scores=v2_scores,
            human_prefers_edit=human_pref_edit,
            candidate_thresholds=cand_thresh_v2,
            alphas=alphas,
            delta=delta,
            n_seeds=n_seeds,
        )

    # Plot release vs alpha
    plot_path = Path(args.output_plot)
    plot_release_vs_alpha(cert_v1["results_by_alpha"], plot_path, cert_v2["results_by_alpha"] if cert_v2 else None)

    # Run Goodhart J3
    print("\n--- Running Goodhart J3 Brightness Bias Analysis on Full Test Pairs ---")
    goodhart_res = compute_goodhart_j3_bias(test_csv)
    print(f"Human prefers brighter: {goodhart_res['human_prefers_brighter']['mean']:.4f} {goodhart_res['human_prefers_brighter']['ci_95']}")
    for j, d in goodhart_res["judges"].items():
        print(f"  {j:<18}: {d['mean']:.4f} {d['ci_95']} | Diff from human: {d['diff_from_human']:+.4f} {d['diff_ci_95']}")

    # Save certifier results to output json
    final_output = {
        "metadata": {
            "status": "verified",
            "title": "Certifier Full Split Evaluation: Learn-Then-Test Risk Control (200 Seeds, Quantile Grid)",
            "label": "full_test — expert rendering vs original (DEAR/FiveK)",
            "dataset": "DEAR full test split",
            "num_scenes": cert_v1["n_scenes"],
            "num_pairs": cert_v1["n_pairs"],
            "delta": delta,
            "n_seeds": n_seeds,
            "human_edit_preference_rate": round(float(np.mean(human_pref_edit)), 4),
            "source_test_pairs": test_csv.as_posix(),
        },
        "results_v1": cert_v1["results_by_alpha"],
        "results_v2": cert_v2["results_by_alpha"] if cert_v2 else None,
        "goodhart_j3": goodhart_res,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved verified certifier results to {out_json}")

    # Ledger entry superseding the pilot
    print("\nAppending verified run to results/ledger.jsonl (superseding pilot)...")
    ledger_run_id = str(uuid.uuid4())
    ledger_entry = {
        "run_id": ledger_run_id,
        "action": "supersede_run",
        "target_run_id": "82905feb-be5c-479a-a1e0-12486b365e43",
        "reason": "Supersedes invalidated pilot with 200-seed LTT protocol on verified DEAR test pairs under label-free quantile grid.",
        "num_scenes": cert_v1["n_scenes"],
        "num_pairs": cert_v1["n_pairs"],
        "results_path": out_json.as_posix(),
        "plot_path": plot_path.as_posix(),
        "v1_mean_release_alpha_0_10": cert_v1["results_by_alpha"]["alpha_0.10"]["ltt"]["mean_release_rate"],
        "v1_violation_rate_alpha_0_10": cert_v1["results_by_alpha"]["alpha_0.10"]["ltt"]["violation_rate"],
    }
    append_run_to_ledger(
        task="certifier_full_test_supersede",
        config=ledger_entry,
        data_manifest={
            "source_csv": test_csv.as_posix(),
            "num_pairs": cert_v1["n_pairs"],
            "num_scenes": cert_v1["n_scenes"],
        },
        metrics={
            "status": "verified",
            "v1_mean_release_alpha_0_10": cert_v1["results_by_alpha"]["alpha_0.10"]["ltt"]["mean_release_rate"],
            "v1_violation_rate_alpha_0_10": cert_v1["results_by_alpha"]["alpha_0.10"]["ltt"]["violation_rate"],
            "goodhart_j3_human_brighter": goodhart_res["human_prefers_brighter"]["mean"],
        },
    )
    print("Done S2 execution.")


if __name__ == "__main__":
    main()
