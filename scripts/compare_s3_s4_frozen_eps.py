#!/usr/bin/env python3
"""Re-evaluate S3 and S4 with procedure-derived frozen epsilons and produce side-by-side comparison (T0.d).

Protocol:
- Loads procedure-derived frozen epsilons from config/judge_epsilon.json
- Recomputes reward counts on S3 (8 synthetic images) and S4 (500 synthetic cases)
- Produces side-by-side table: Judge | Old ε | Old S3 Rewarded | New ε | New S3 Rewarded | Old S4 Rewarded | New S4 Rewarded
- Labels table: "rerun with procedure-derived ε (post-hoc fix, disclosed)"
- Exports results/s3_s4_epsilon_comparison.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_EPS = REPO_ROOT / "config" / "judge_epsilon.json"
CACHE_SCORES = REPO_ROOT / "results" / "synth_v2_scores_cache.json"
CASES_CSV = REPO_ROOT / "results" / "synth_v2_cases.csv"
OLD_RESULTS = REPO_ROOT / "results" / "synth_v2_oracle_results.json"
OUT_JSON = REPO_ROOT / "results" / "s3_s4_epsilon_comparison.json"


def main():
    print("=== T0.d: S3 & S4 Re-run with Procedure-Derived Frozen Epsilons ===")
    assert CONFIG_EPS.exists(), f"Missing {CONFIG_EPS}"
    assert OLD_RESULTS.exists(), f"Missing {OLD_RESULTS}"
    assert CACHE_SCORES.exists(), f"Missing {CACHE_SCORES}"
    assert CASES_CSV.exists(), f"Missing {CASES_CSV}"

    with open(CONFIG_EPS, "r", encoding="utf-8") as f:
        frozen_cfg = json.load(f)
    frozen_eps = frozen_cfg["frozen_epsilons"]
    print(f"Loaded frozen epsilons: {frozen_eps}")

    with open(OLD_RESULTS, "r", encoding="utf-8") as f:
        old_data = json.load(f)
    old_eps = old_data["epsilons"]
    old_s3 = old_data["s3_eight_images"]
    old_s4 = old_data["s4_five_hundred_cases"]

    with open(CACHE_SCORES, "r", encoding="utf-8") as f:
        score_cache = json.load(f)

    df_cases = pd.read_csv(CASES_CSV)
    n_cases = len(df_cases)
    judge_keys = ["rsinema", "pyiqa_musiq", "pyiqa_topiq_nr", "aesfix_pref_v1", "aesfix_pref_v2"]

    # 1. Recompute S3 (8 images)
    s3_table = old_s3["table"]
    new_s3_counts = {j: 0 for j in judge_keys}
    s3_new_table = []

    for rec in s3_table:
        new_rec = dict(rec)
        new_rec["new_scores"] = {}
        for j in judge_keys:
            d = rec["scores"][j]["delta"]
            new_e = frozen_eps.get(j, 0.001)
            rewards = d > new_e
            if rewards:
                new_s3_counts[j] += 1
            new_rec["new_scores"][j] = {
                "delta": d,
                "frozen_eps": new_e,
                "rewards_fix": rewards,
            }
        s3_new_table.append(new_rec)

    # 2. Recompute S4 (500 cases)
    new_s4_counts = {j: 0 for j in judge_keys}
    deg_breakdown_new = {}

    for _, row in df_cases.iterrows():
        deg_p = str(row["deg_path"])
        orc_p = str(row["orc_path"])
        deg_name = row["deg_name"]

        if deg_name not in deg_breakdown_new:
            deg_breakdown_new[deg_name] = {"total": 0, "rewards": {j: 0 for j in judge_keys}}
        deg_breakdown_new[deg_name]["total"] += 1

        s_deg = score_cache.get(deg_p, {})
        s_orc = score_cache.get(orc_p, {})

        for j in judge_keys:
            if j in s_deg and j in s_orc:
                delta = s_orc[j] - s_deg[j]
                if delta > frozen_eps.get(j, 0.001):
                    new_s4_counts[j] += 1
                    deg_breakdown_new[deg_name]["rewards"][j] += 1

    # 3. Print Side-by-Side Disclosure Table
    print("\n" + "=" * 105)
    print("SIDE-BY-SIDE DISCLOSURE: OLD (HAND-SET) VS NEW (PROCEDURE-DERIVED) EPSILONS")
    print("Label: rerun with procedure-derived ε (post-hoc fix, disclosed)")
    print("=" * 105)

    header = f"{'Judge Model':<18} | {'Old ε':<9} | {'New ε (p95)':<12} | {'Old S3 (N=8)':<14} | {'New S3 (N=8)':<14} | {'Old S4 (N=500)':<16} | {'New S4 (N=500)':<16}"
    print(header)
    print("-" * len(header))

    comparison_rows = []
    for j in judge_keys:
        oe = old_eps.get(j, 0.0)
        ne = frozen_eps.get(j, 0.0)
        os3 = f"{old_s3['reward_counts'].get(j, 0)}/8"
        ns3 = f"{new_s3_counts[j]}/8"
        os4 = f"{old_s4['judge_rewards'].get(j, 0)}/500 ({old_s4['judge_rewards'].get(j, 0)/500*100:.1f}%)"
        ns4 = f"{new_s4_counts[j]}/500 ({new_s4_counts[j]/500*100:.1f}%)"

        print(f"{j:<18} | {oe:<9.4f} | {ne:<12.5f} | {os3:<14} | {ns3:<14} | {os4:<16} | {ns4:<16}")

        comparison_rows.append({
            "judge": j,
            "old_epsilon": oe,
            "new_frozen_epsilon": ne,
            "old_s3_rewarded": old_s3["reward_counts"].get(j, 0),
            "new_s3_rewarded": new_s3_counts[j],
            "old_s4_rewarded": old_s4["judge_rewards"].get(j, 0),
            "new_s4_rewarded": new_s4_counts[j],
            "old_s4_reward_rate": round(old_s4["judge_rewards"].get(j, 0) / 500, 4),
            "new_s4_reward_rate": round(new_s4_counts[j] / 500, 4),
        })

    print("-" * len(header))

    # Overexposure specific comparison
    print("\nOverexposure (+1 EV) Reward Breakdown (N=100 Overexposure Cases):")
    print(f"{'Judge Model':<18} | {'Old Rewards':<15} | {'New Rewards':<15}")
    print("-" * 52)
    overexp_comp = {}
    for j in judge_keys:
        old_oe = old_s4.get("deg_breakdown", {}).get("overexposure", {}).get("judge_rewards", {}).get(j, 0)
        new_oe = deg_breakdown_new.get("overexposure", {}).get("rewards", {}).get(j, 0)
        print(f"{j:<18} | {old_oe}/100 ({old_oe}%)    | {new_oe}/100 ({new_oe}%)")
        overexp_comp[j] = {"old_overexposure_rewards": old_oe, "new_overexposure_rewards": new_oe}
    print("-" * 52)

    payload = {
        "metadata": {
            "title": "S3 & S4 Benchmark Re-run with Procedure-Derived Frozen Epsilons",
            "task": "T0.d",
            "disclosure_label": "rerun with procedure-derived ε (post-hoc fix, disclosed)",
            "config_path": str(CONFIG_EPS.relative_to(REPO_ROOT)),
        },
        "comparison_table": comparison_rows,
        "overexposure_comparison": overexp_comp,
        "degradation_breakdown_new": deg_breakdown_new,
        "s3_new_table": s3_new_table,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    hasher = hashlib.sha256()
    with open(OUT_JSON, "rb") as f:
        hasher.update(f.read())
    print(f"\nSaved comparison to {OUT_JSON}")
    print(f"File SHA-256: {hasher.hexdigest()}")


if __name__ == "__main__":
    main()
