#!/usr/bin/env python3
"""Empirical Goodhart brightness bias analysis on DEAR test pairs and dark scenes (J3).

Corrected Methodology (J3):
1. In P5/I5(c), '0.8021' was the judge's accuracy (agreement with humans) on brightness-subset
   pairs (|delta luma| > 0.15) from the pilot run, NOT the proportion of times it preferred brighter.
2. Here we measure on the SAME pairs that have judge scores (from test_pair_scores.csv when available,
   or results/pilot_pair_scores.json):
   - P(humans prefer brighter) with scene-clustered bootstrap 95% CI
   - P(judge prefers brighter) with scene-clustered bootstrap 95% CI
   - Difference (judge - human) with scene-clustered bootstrap 95% CI
3. Dark scenes (n=6): reported as 'insufficient data (n=6 scenes)'.
   Removes any speculative claim that crowd raters treat low-key photos as defects.
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
    pilot_scores_path: Path,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    if not dear_markup_path.exists():
        raise FileNotFoundError(f"DEAR markup not found at {dear_markup_path}")

    # 1. Map scenes to original luma and all pairs from test markup
    scene_orig_luma: Dict[str, float] = {}
    all_scenes_set = set()
    all_pairs: List[Dict[str, Any]] = []

    with open(dear_markup_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row["name"]
            all_scenes_set.add(scene)
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

    total_unique_scenes = len(all_scenes_set)  # 1,283
    scenes_with_orig = len(scene_orig_luma)    # 1,233
    luma_vals = np.array(list(scene_orig_luma.values()))

    # (a) Load pairs with judge scores
    # If test_pair_scores.csv exists, use it; otherwise use pilot_pair_scores.json
    judge_pairs: List[Dict[str, Any]] = []
    source_name = "results/pilot_pair_scores.json"

    test_csv_path = REPO_ROOT / "kaggle/k2_output/test_pair_scores.csv"
    if test_csv_path.exists():
        source_name = "kaggle/k2_output/test_pair_scores.csv"
        df_k2 = pd.read_csv(test_csv_path)
        for _, r in df_k2.iterrows():
            d_luma = float(r["left_median_luma"]) - float(r["right_median_luma"])
            if abs(d_luma) > 0.15 and float(r["mos"]) != 0.5:
                left_brighter = d_luma > 0
                h_pref_brighter = float((float(r["mos"]) > 0.5) == left_brighter)
                j_pref_brighter = float((float(r["pref_logit_v1"]) > 0) == left_brighter)
                judge_pairs.append({
                    "scene": r["scene"],
                    "human_prefers_brighter": h_pref_brighter,
                    "judge_prefers_brighter": j_pref_brighter,
                    "d_luma": d_luma
                })
    else:
        with open(pilot_scores_path, "r", encoding="utf-8") as f:
            pilot_list = json.load(f)
        
        # Match pilot pairs against markup to get lightness deltas
        markup_dict = {}
        for p in all_pairs:
            markup_dict[(p["scene"], p["left"], p["right"])] = p
            markup_dict[(p["scene"], p["right"], p["left"])] = p

        for p in pilot_list:
            sc = p["scene"]
            st = p["edit_style"]
            match = markup_dict.get((sc, "original", st)) or markup_dict.get((sc, st, "original"))
            if not match:
                continue
            if match["left"] == "original":
                l_orig, l_edit = match["l_left"], match["l_right"]
            else:
                l_orig, l_edit = match["l_right"], match["l_left"]
            d_luma = l_edit - l_orig
            if abs(d_luma) > 0.15:
                edit_brighter = d_luma > 0
                h_pref_brighter = float(edit_brighter == p["human_prefers_edit"])
                j_pref_brighter = float(edit_brighter == (p["score_delta"] > 0))
                judge_pairs.append({
                    "scene": sc,
                    "human_prefers_brighter": h_pref_brighter,
                    "judge_prefers_brighter": j_pref_brighter,
                    "judge_correct": float(p["human_prefers_edit"] == (p["score_delta"] > 0)),
                    "d_luma": d_luma
                })

    # Cluster bootstrap over scenes for judge pairs with |d_luma| > 0.15
    scenes_j = sorted(list(set(p["scene"] for p in judge_pairs)))
    scene_map_j = defaultdict(list)
    for p in judge_pairs:
        scene_map_j[p["scene"]].append(p)

    rng = np.random.default_rng(seed)
    h_boots = np.zeros(n_bootstrap, dtype=np.float64)
    j_boots = np.zeros(n_bootstrap, dtype=np.float64)
    diff_boots = np.zeros(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        sampled = rng.choice(scenes_j, size=len(scenes_j), replace=True)
        samp_pairs = [p for s in sampled for p in scene_map_j[s]]
        hr = float(np.mean([p["human_prefers_brighter"] for p in samp_pairs]))
        jr = float(np.mean([p["judge_prefers_brighter"] for p in samp_pairs]))
        h_boots[b] = hr
        j_boots[b] = jr
        diff_boots[b] = jr - hr

    h_mean = float(np.mean([p["human_prefers_brighter"] for p in judge_pairs]))
    j_mean = float(np.mean([p["judge_prefers_brighter"] for p in judge_pairs]))
    diff_mean = j_mean - h_mean

    h_ci = [round(float(np.percentile(h_boots, 2.5)), 4), round(float(np.percentile(h_boots, 97.5)), 4)]
    j_ci = [round(float(np.percentile(j_boots, 2.5)), 4), round(float(np.percentile(j_boots, 97.5)), 4)]
    diff_ci = [round(float(np.percentile(diff_boots, 2.5)), 4), round(float(np.percentile(diff_boots, 97.5)), 4)]

    # (b) Dark scenes (original luma < 0.20): report as insufficient data (n=6 scenes)
    dark_scene_ids = {s for s, luma_val in scene_orig_luma.items() if luma_val < 0.20}
    dark_pairs = [
        p for p in all_pairs
        if p["scene"] in dark_scene_ids and p["mos"] != 0.5 and abs(p["d_luma"]) > 1e-4
    ]

    return {
        "dataset": "DEAR test_processed_markup",
        "total_test_scenes_unique": total_unique_scenes,
        "total_test_scenes_with_original": scenes_with_orig,
        "total_test_pairs": len(all_pairs),
        "scene_count_explanation": (
            "DEAR test split contains 1,283 unique scenes across 18,995 pairs. Exactly 1,233 of these "
            "scenes contain original-vs-edit comparisons; the remaining 50 scenes contain only edit-vs-edit pairs."
        ),
        "scene_luma_distribution": {
            "p01": round(float(np.percentile(luma_vals, 1)), 4),
            "p05": round(float(np.percentile(luma_vals, 5)), 4),
            "p50_median": round(float(np.percentile(luma_vals, 50)), 4),
            "p95": round(float(np.percentile(luma_vals, 95)), 4),
            "scenes_luma_lt_0_20_count": len(dark_scene_ids),
            "scenes_luma_lt_0_20_fraction": round(len(dark_scene_ids) / scenes_with_orig, 4),
        },
        "part_a_brightness_subset": {
            "source": source_name,
            "luma_delta_threshold": 0.15,
            "pairs_count": len(judge_pairs),
            "scenes_count": len(scenes_j),
            "p_human_prefers_brighter": round(h_mean, 4),
            "ci_95_human": h_ci,
            "p_judge_prefers_brighter": round(j_mean, 4),
            "ci_95_judge": j_ci,
            "difference_judge_minus_human": round(diff_mean, 4),
            "ci_95_difference": diff_ci,
            "clarification": (
                "The 0.8021 value previously cited in P5 was the judge's pairwise ACCURACY (agreement with humans) "
                "on brightness pairs from the initial pilot run. When measuring actual preference for the brighter image, "
                f"humans prefer brighter {h_mean*100:.2f}% and judge prefers brighter {j_mean*100:.2f}% (difference: {diff_mean*100:+.2f}%)."
            ),
        },
        "part_b_dark_scenes": {
            "status": "insufficient data (n=6 scenes)",
            "dark_threshold_luma": 0.20,
            "dark_scenes_count": len(dark_scene_ids),
            "dark_pairs_count": len(dark_pairs),
            "conclusion": (
                "Only 6 out of 1,283 scenes in DEAR have original luminance < 0.20. With n=6 scenes, "
                "there is insufficient statistical support to characterize crowd rater behavior on low-key photography."
            ),
        },
    }


def main() -> int:
    default_csv = Path(r"D:\aesfix-data\dear\test_processed_markup.csv")
    pilot_json = REPO_ROOT / "results" / "pilot_pair_scores.json"

    metrics = compute_brightness_bias(default_csv, pilot_json)
    out_json = REPO_ROOT / "results" / "goodhart_bias_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("=== Empirical Goodhart Brightness Bias Analysis (J3) ===")
    print(f"Total Unique Scenes: {metrics['total_test_scenes_unique']} (with original: {metrics['total_test_scenes_with_original']})")
    print(f"\n(a) Brightness Subset (|Δ luma| > 0.15, N={metrics['part_a_brightness_subset']['pairs_count']} pairs, {metrics['part_a_brightness_subset']['scenes_count']} scenes):")
    print(f"    P(human prefers brighter): {metrics['part_a_brightness_subset']['p_human_prefers_brighter']*100:.2f}% (95% CI: [{metrics['part_a_brightness_subset']['ci_95_human'][0]*100:.2f}%, {metrics['part_a_brightness_subset']['ci_95_human'][1]*100:.2f}%])")
    print(f"    P(judge prefers brighter): {metrics['part_a_brightness_subset']['p_judge_prefers_brighter']*100:.2f}% (95% CI: [{metrics['part_a_brightness_subset']['ci_95_judge'][0]*100:.2f}%, {metrics['part_a_brightness_subset']['ci_95_judge'][1]*100:.2f}%])")
    print(f"    Difference (Judge - Human): {metrics['part_a_brightness_subset']['difference_judge_minus_human']*100:+.2f}% (95% CI: [{metrics['part_a_brightness_subset']['ci_95_difference'][0]*100:+.2f}%, {metrics['part_a_brightness_subset']['ci_95_difference'][1]*100:+.2f}%])")
    print(f"\n(b) Dark Scenes (Original Luma < 0.20): {metrics['part_b_dark_scenes']['status']}")
    print(f"    {metrics['part_b_dark_scenes']['conclusion']}")
    print(f"\nSaved corrected metrics to {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
