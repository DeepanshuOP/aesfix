#!/usr/bin/env python3
"""Empirical Goodhart brightness bias analysis on DEAR test pairs and dark scenes (I5).

Computes:
(a) Human brighter preference rate among DEAR test pairs with |delta luma| > 0.15,
    with scene-clustered 95% bootstrap confidence intervals.
(b) Distribution of original scene luminance in DEAR; restriction to dark scenes (luma < 0.20),
    reporting count of dark scenes and human preference rates.
(c) Comparison against AesFix-Pref brightness preference rate (80.21%) and measurement of the
    judge's excess bias gap beyond human preference.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def compute_brightness_bias(
    dear_markup_path: Path,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    if not dear_markup_path.exists():
        raise FileNotFoundError(f"DEAR markup not found at {dear_markup_path}")

    # 1. Map scenes to original luma
    scene_orig_luma: Dict[str, float] = {}
    all_pairs: List[Dict[str, Any]] = []

    with open(dear_markup_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row["name"]
            l_left = float(row["lightness_left"]) / 255.0
            l_right = float(row["lightness_right"]) / 255.0
            mos = float(row["mos"])

            if row["left"] == "original":
                scene_orig_luma[scene] = l_left
            elif row["right"] == "original":
                scene_orig_luma[scene] = l_right

            all_pairs.append({
                "scene": scene,
                "left": row["left"],
                "right": row["right"],
                "mos": mos,
                "l_left": l_left,
                "l_right": l_right,
                "d_luma": l_left - l_right,
            })

    total_scenes = len(scene_orig_luma)
    luma_vals = np.array(list(scene_orig_luma.values()))

    # (a) Pairs with |delta luma| > 0.15 and mos != 0.5
    large_delta_pairs = [
        p for p in all_pairs
        if abs(p["d_luma"]) > 0.15 and p["mos"] != 0.5
    ]

    scene_map_a = defaultdict(list)
    for p in large_delta_pairs:
        left_is_brighter = p["d_luma"] > 0.0
        human_prefers_left = p["mos"] > 0.5
        prefers_brighter = 1.0 if (human_prefers_left == left_is_brighter) else 0.0
        scene_map_a[p["scene"]].append(prefers_brighter)

    unique_scenes_a = np.array(list(scene_map_a.keys()))
    rng = np.random.default_rng(seed)
    boot_rates_a = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled = rng.choice(unique_scenes_a, size=len(unique_scenes_a), replace=True)
        corr = sum(sum(scene_map_a[s]) for s in sampled)
        tot = sum(len(scene_map_a[s]) for s in sampled)
        boot_rates_a[b] = corr / tot if tot > 0 else 0.0

    human_brighter_rate_a = float(sum(sum(scene_map_a[s]) for s in unique_scenes_a) / len(large_delta_pairs))
    ci_low_a = float(np.percentile(boot_rates_a, 2.5))
    ci_high_a = float(np.percentile(boot_rates_a, 97.5))

    # (b) Dark scenes (original luma < 0.20)
    dark_scene_ids = {s for s, luma_val in scene_orig_luma.items() if luma_val < 0.20}
    dark_pairs = [
        p for p in all_pairs
        if p["scene"] in dark_scene_ids and p["mos"] != 0.5 and abs(p["d_luma"]) > 1e-4
    ]

    dark_scene_map = defaultdict(list)
    for p in dark_pairs:
        left_is_brighter = p["d_luma"] > 0.0
        human_prefers_left = p["mos"] > 0.5
        prefers_darker = 1.0 if (human_prefers_left != left_is_brighter) else 0.0
        dark_scene_map[p["scene"]].append(prefers_darker)

    unique_dark_scenes = np.array(list(dark_scene_map.keys()))
    boot_rates_b = np.zeros(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sampled = rng.choice(unique_dark_scenes, size=len(unique_dark_scenes), replace=True)
        corr = sum(sum(dark_scene_map[s]) for s in sampled)
        tot = sum(len(dark_scene_map[s]) for s in sampled)
        boot_rates_b[b] = corr / tot if tot > 0 else 0.0

    human_darker_rate_b = float(sum(sum(dark_scene_map[s]) for s in unique_dark_scenes) / len(dark_pairs)) if dark_pairs else 0.0
    ci_low_b = float(np.percentile(boot_rates_b, 2.5)) if dark_pairs else 0.0
    ci_high_b = float(np.percentile(boot_rates_b, 97.5)) if dark_pairs else 0.0
    human_brighter_rate_b = round(1.0 - human_darker_rate_b, 4)

    # (c) AesFix-Pref judge bias comparison
    aesfix_pref_brighter_rate = 0.8021  # Measured on DEAR brightness test pairs in K2
    excess_bias_gap = round(aesfix_pref_brighter_rate - human_brighter_rate_a, 4)

    return {
        "dataset": "DEAR test_processed_markup",
        "total_test_scenes": total_scenes,
        "total_test_pairs": len(all_pairs),
        "scene_luma_distribution": {
            "p01": round(float(np.percentile(luma_vals, 1)), 4),
            "p05": round(float(np.percentile(luma_vals, 5)), 4),
            "p50_median": round(float(np.percentile(luma_vals, 50)), 4),
            "p95": round(float(np.percentile(luma_vals, 95)), 4),
            "scenes_luma_lt_0_20_count": len(dark_scene_ids),
            "scenes_luma_lt_0_20_fraction": round(len(dark_scene_ids) / total_scenes, 4),
        },
        "part_a_large_delta_pairs": {
            "luma_delta_threshold": 0.15,
            "pairs_count": len(large_delta_pairs),
            "scenes_count": len(unique_scenes_a),
            "human_prefers_brighter_rate": round(human_brighter_rate_a, 4),
            "ci_95": [round(ci_low_a, 4), round(ci_high_a, 4)],
        },
        "part_b_dark_scenes": {
            "dark_threshold_luma": 0.20,
            "dark_scenes_count": len(dark_scene_ids),
            "dark_pairs_count": len(dark_pairs),
            "human_prefers_darker_rate": round(human_darker_rate_b, 4),
            "human_prefers_brighter_rate": human_brighter_rate_b,
            "ci_95_darker": [round(ci_low_b, 4), round(ci_high_b, 4)],
        },
        "part_c_judge_bias_gap": {
            "human_brighter_preference_rate": round(human_brighter_rate_a, 4),
            "aesfix_pref_brighter_preference_rate": aesfix_pref_brighter_rate,
            "excess_judge_bias_gap": excess_bias_gap,
            "excess_judge_bias_gap_pct": round(excess_bias_gap * 100, 2),
        },
    }


def main() -> int:
    default_csv = REPO_ROOT / "D:/aesfix-data/dear/test_processed_markup.csv"
    if not default_csv.exists():
        default_csv = Path(r"D:\aesfix-data\dear\test_processed_markup.csv")

    metrics = compute_brightness_bias(default_csv)
    out_json = REPO_ROOT / "results" / "goodhart_bias_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=== Empirical Goodhart Brightness Bias Analysis (I5) ===")
    print(f"(a) DEAR Pairs |Δ luma| > 0.15 (N={metrics['part_a_large_delta_pairs']['pairs_count']}, {metrics['part_a_large_delta_pairs']['scenes_count']} scenes):")
    print(f"    Human Brighter Preference: {metrics['part_a_large_delta_pairs']['human_prefers_brighter_rate']*100:.2f}% (95% CI: [{metrics['part_a_large_delta_pairs']['ci_95'][0]*100:.2f}%, {metrics['part_a_large_delta_pairs']['ci_95'][1]*100:.2f}%])")
    print(f"\n(b) Dark Scenes (Original Luma < 0.20): {metrics['part_b_dark_scenes']['dark_scenes_count']}/{metrics['total_test_scenes']} scenes ({metrics['scene_luma_distribution']['scenes_luma_lt_0_20_fraction']*100:.2f}%)")
    print(f"    Human Brighter Preference on Dark Scenes: {metrics['part_b_dark_scenes']['human_prefers_brighter_rate']*100:.2f}%")
    print("\n(c) Judge Bias Gap:")
    print(f"    Human baseline:     {metrics['part_c_judge_bias_gap']['human_brighter_preference_rate']*100:.2f}%")
    print(f"    AesFix-Pref:        {metrics['part_c_judge_bias_gap']['aesfix_pref_brighter_preference_rate']*100:.2f}%")
    print(f"    Excess Bias Gap:    +{metrics['part_c_judge_bias_gap']['excess_judge_bias_gap_pct']:.2f}%")
    print(f"\nSaved metrics to {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
