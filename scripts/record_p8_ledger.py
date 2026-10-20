#!/usr/bin/env python3
"""Records Phase 8 experimental results into results/ledger.jsonl with cryptographic provenance."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aesfix.ledger import append_run


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p8_res_p = REPO_ROOT / "results" / "p8_experiments_results.json"
    p8_res = json.load(open(p8_res_p, "r", encoding="utf-8"))

    bias_p = REPO_ROOT / "results" / "brightness_bias_ci.json"
    bias_res = json.load(open(bias_p, "r", encoding="utf-8"))

    t4_p = REPO_ROOT / "results" / "t4_clipping_summary.json"
    t4_res = json.load(open(t4_p, "r", encoding="utf-8"))

    eps_p = REPO_ROOT / "config" / "judge_epsilon.json"
    eps_res = json.load(open(eps_p, "r", encoding="utf-8"))

    config = {
        "task": "p8_comprehensive_evaluation",
        "phase": "P8",
        "author": "Deepanshu <Deepanshuop@gmail.com>",
        "n_synthetic_cases": 500,
        "n_certifier_seeds": 20,
        "n_bias_bootstrap": 5000,
        "overoptimization_k": [1, 2, 3, 5, 8, 12, 15],
        "frozen_epsilons": eps_res["frozen_epsilons"],
        "hardware_cloud": "Kaggle Tesla T4 GPU",
        "kernel_slug": "deepanshuop12/aesfix-p8-experiments",
    }

    data_manifest = {
        "p8_experiments_results_sha256": get_sha256(p8_res_p),
        "brightness_bias_ci_sha256": get_sha256(bias_p),
        "t4_clipping_summary_sha256": get_sha256(t4_p),
        "judge_epsilon_sha256": get_sha256(eps_p),
        "figure_overoptimization_k_sha256": get_sha256(REPO_ROOT / "docs" / "figures" / "overoptimization_k.png"),
        "report_sha256": get_sha256(REPO_ROOT / "docs" / "reports" / "p8_report.md"),
    }

    metrics = {
        "t0_bias_reduction_relative": bias_res["metrics"]["relative_bias_reduction"]["mean"],
        "t0_bias_reduction_relative_ci": [
            bias_res["metrics"]["relative_bias_reduction"]["ci_lower"],
            bias_res["metrics"]["relative_bias_reduction"]["ci_upper"],
        ],
        "t1_proposers": {
            cfg: {
                "recovered_rate": d.get("recovered_rate", 0.0),
                "worse_rate": d.get("worse_rate", 0.0),
                "unchanged_rate": d.get("unchanged_rate", 0.0),
                "mean_delta_e_change": d.get("mean_delta_e_change", 0.0),
            }
            for cfg, d in p8_res["t1_proposer_benchmark"].items()
        },
        "t2_overoptimization_v2_k15_gain": p8_res["t2_overoptimization_curve"]["aesfix_pref_v2"][-1]["mean_judge_gain_std"],
        "t2_overoptimization_v2_k15_de": p8_res["t2_overoptimization_curve"]["aesfix_pref_v2"][-1]["mean_delta_e_change"],
        "t2_overoptimization_v1_k15_gain": p8_res["t2_overoptimization_curve"]["aesfix_pref_v1"][-1]["mean_judge_gain_std"],
        "t2_overoptimization_v1_k15_de": p8_res["t2_overoptimization_curve"]["aesfix_pref_v1"][-1]["mean_delta_e_change"],
        "t3_policy_certifier_violations": {
            "candidate_level_p_viol_0_05": p8_res["t3_policy_certifier"]["candidate_level"][0]["p_violation"],
            "policy_level_p_viol_0_05": p8_res["t3_policy_certifier"]["policy_level"][0]["p_violation"],
            "naive_p_viol_0_05": p8_res["t3_policy_certifier"]["naive"][0]["p_violation"],
        },
        "t4_highlight_clipping_gt_1pct_rate": t4_res["highlight_clipping_stats"]["rate_gt_1pct_clipped"],
        "gpu_hours_week": 3.873,
    }

    record = append_run(
        metrics=metrics,
        config=config,
        data_manifest=data_manifest,
        cost_usd=0.0,
        run_id="p8_final_audit_run",
    )
    print(f"Successfully recorded P8 run to ledger! Run ID: {record['run_id']}")


if __name__ == "__main__":
    main()
