#!/usr/bin/env python3
"""Computes scene-clustered bootstrap 95% CIs for brightness bias (judge - human) for v1 and v2."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = Path(r"D:\aesfix-data\kaggle_runs\dear_full\test_pair_scores.csv")
OUT_JSON = REPO_ROOT / "results" / "brightness_bias_ci.json"


def main():
    print(f"Loading test pair scores from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    if "left_median_luma" in df.columns and "right_median_luma" in df.columns:
        df["d_luma"] = df["left_median_luma"] - df["right_median_luma"]
    elif "lightness_left" in df.columns and "lightness_right" in df.columns:
        df["d_luma"] = (df["lightness_left"] - df["lightness_right"]) / 255.0

    subset = df[(df["d_luma"].abs() > 0.15) & (df["mos"] != 0.5)].copy()
    subset["left_brighter"] = subset["d_luma"] > 0
    subset["human_pref_left"] = subset["mos"] > 0.5
    subset["human_prefers_brighter"] = (subset["human_pref_left"] == subset["left_brighter"]).astype(float)

    subset["v1_pref_left"] = subset["pref_logit_v1"] > 0
    subset["v1_prefers_brighter"] = (subset["v1_pref_left"] == subset["left_brighter"]).astype(float)

    subset["v2_pref_left"] = subset["pref_logit_v2"] > 0
    subset["v2_prefers_brighter"] = (subset["v2_pref_left"] == subset["left_brighter"]).astype(float)

    scene_groups = [group.to_dict("records") for _, group in subset.groupby("scene")]
    n_groups = len(scene_groups)
    rng = np.random.default_rng(42)

    boot_h = []
    boot_v1 = []
    boot_v2 = []
    boot_diff_v1 = []
    boot_diff_v2 = []
    boot_bias_reduction_abs = []
    boot_bias_reduction_rel = []

    for _ in range(5000):
        idx = rng.choice(n_groups, size=n_groups, replace=True)
        rows = [r for i in idx for r in scene_groups[i]]
        h = np.mean([r["human_prefers_brighter"] for r in rows])
        v1 = np.mean([r["v1_prefers_brighter"] for r in rows])
        v2 = np.mean([r["v2_prefers_brighter"] for r in rows])
        diff1 = v1 - h
        diff2 = v2 - h
        boot_h.append(h)
        boot_v1.append(v1)
        boot_v2.append(v2)
        boot_diff_v1.append(diff1)
        boot_diff_v2.append(diff2)
        boot_bias_reduction_abs.append(diff1 - diff2)
        boot_bias_reduction_rel.append((diff1 - diff2) / diff1 if abs(diff1) > 1e-6 else 0.0)

    def ci(arr):
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5)),
        }

    results = {
        "metadata": {
            "title": "Scene-Clustered Bootstrap 95% CIs for Brightness Bias",
            "source_data": str(CSV_PATH),
            "n_pairs": int(len(subset)),
            "n_scenes": int(n_groups),
            "n_bootstrap": 5000,
            "seed": 42,
            "condition": "|d_luma| > 0.15 and mos != 0.5",
        },
        "metrics": {
            "human_prefers_brighter": ci(boot_h),
            "v1_prefers_brighter": ci(boot_v1),
            "v2_prefers_brighter": ci(boot_v2),
            "v1_bias_diff": ci(boot_diff_v1),
            "v2_bias_diff": ci(boot_diff_v2),
            "absolute_bias_reduction": ci(boot_bias_reduction_abs),
            "relative_bias_reduction": ci(boot_bias_reduction_rel),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved brightness bias CIs to {OUT_JSON}")
    print(f"Human: {results['metrics']['human_prefers_brighter']['mean']:.4f} [{results['metrics']['human_prefers_brighter']['ci_lower']:.4f}, {results['metrics']['human_prefers_brighter']['ci_upper']:.4f}]")
    print(f"v1 Bias: {results['metrics']['v1_bias_diff']['mean']:.4f} [{results['metrics']['v1_bias_diff']['ci_lower']:.4f}, {results['metrics']['v1_bias_diff']['ci_upper']:.4f}]")
    print(f"v2 Bias: {results['metrics']['v2_bias_diff']['mean']:.4f} [{results['metrics']['v2_bias_diff']['ci_lower']:.4f}, {results['metrics']['v2_bias_diff']['ci_upper']:.4f}]")
    print(f"Abs Reduction: {results['metrics']['absolute_bias_reduction']['mean']:.4f} [{results['metrics']['absolute_bias_reduction']['ci_lower']:.4f}, {results['metrics']['absolute_bias_reduction']['ci_upper']:.4f}]")
    print(f"Rel Reduction: {results['metrics']['relative_bias_reduction']['mean']*100:.2f}% [{results['metrics']['relative_bias_reduction']['ci_lower']*100:.2f}%, {results['metrics']['relative_bias_reduction']['ci_upper']*100:.2f}%]")


if __name__ == "__main__":
    main()
