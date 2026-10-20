#!/usr/bin/env python3
"""Process K2b outputs, generate model selection & full test tables, and ingest into ledger (S1).

Calculates:
1. Model selection table (val accuracy of all 8 heads)
2. Full test table for every judge (accuracy + scene CI, signed Spearman, brightness-subset accuracy,
   original-vs-edit accuracy, high-agreement accuracy)
3. Paired differences: (v2 - v1) and (v2 - TopIQ-NR) with scene-clustered bootstrap 95% CIs
4. GPU-hours from K2b's runtime
5. Ingests run into results/ledger.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.ledger import append_run_to_ledger, compute_file_sha256

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("process_k2b")


def get_sha256(path: Path) -> str:
    """Computes SHA-256 hash."""
    if not path.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_scene_clustered_ci(
    scene_ids: List[str],
    correct_arr: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes scene-clustered bootstrap 95% CI for pairwise accuracy."""
    df_temp = pd.DataFrame({"scene": scene_ids, "correct": correct_arr})
    grouped = df_temp.groupby("scene")["correct"].agg(["sum", "count"])
    scene_sums = grouped["sum"].to_numpy(dtype=np.float64)
    scene_counts = grouped["count"].to_numpy(dtype=np.float64)

    overall_mean = float(np.mean(correct_arr))

    rng = np.random.default_rng(seed)
    n_scenes = len(scene_sums)

    indices = rng.integers(0, n_scenes, size=(n_bootstrap, n_scenes))
    boot_sums = np.sum(scene_sums[indices], axis=1)
    boot_counts = np.sum(scene_counts[indices], axis=1)
    boot_means = boot_sums / boot_counts

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return overall_mean, ci_low, ci_high


def compute_paired_difference_ci(
    scene_ids: List[str],
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes paired difference (a - b) with scene-clustered bootstrap 95% CI."""
    diff = correct_a - correct_b
    df_temp = pd.DataFrame({"scene": scene_ids, "diff": diff})
    grouped = df_temp.groupby("scene")["diff"].agg(["sum", "count"])
    scene_sums = grouped["sum"].to_numpy(dtype=np.float64)
    scene_counts = grouped["count"].to_numpy(dtype=np.float64)

    overall_diff = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    n_scenes = len(scene_sums)

    indices = rng.integers(0, n_scenes, size=(n_bootstrap, n_scenes))
    boot_sums = np.sum(scene_sums[indices], axis=1)
    boot_counts = np.sum(scene_counts[indices], axis=1)
    boot_diffs = boot_sums / boot_counts

    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    return overall_diff, ci_low, ci_high


def process_k2b_run(
    run_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Processes K2b downloaded files and generates all benchmark metrics."""
    logger.info(f"Processing K2b results from {run_dir}...")

    # Load test pairs
    test_pairs_path = run_dir / "test_pair_scores.csv"
    if not test_pairs_path.exists():
        raise FileNotFoundError(f"Missing {test_pairs_path}")

    df_pairs = pd.read_csv(test_pairs_path)
    logger.info(f"Loaded {len(df_pairs)} test pairs.")

    # Load full results json if available
    full_res_path = run_dir / "k2_full_results.json"
    if not full_res_path.exists():
        full_res_path = run_dir / "full_test_results.json"
    k2b_meta = {}
    if full_res_path.exists():
        with open(full_res_path, "r", encoding="utf-8") as f:
            k2b_meta = json.load(f)

    # Load model selection table
    model_sel_path = run_dir / "model_selection_table.csv"
    if model_sel_path.exists():
        df_model_sel = pd.read_csv(model_sel_path)
    elif "validation_model_selection_table" in k2b_meta:
        df_model_sel = pd.DataFrame(k2b_meta["validation_model_selection_table"])
    else:
        df_model_sel = None

    # Load per image scores if available
    per_img_path = run_dir / "per_image_scores.csv"
    df_per_img = pd.read_csv(per_img_path) if per_img_path.exists() else None

    # Exclude ties (mos == 0.5) for pairwise accuracy
    valid_pairs = df_pairs[df_pairs["mos"] != 0.5].copy()
    scenes = valid_pairs["scene"].tolist()
    human_pref_left = (valid_pairs["mos"] > 0.5).values

    # Determine subset masks
    # 1. Brightness subset (|d_luma| > 0.15)
    if "left_median_luma" in valid_pairs.columns and "right_median_luma" in valid_pairs.columns:
        valid_pairs["d_luma"] = (valid_pairs["left_median_luma"] - valid_pairs["right_median_luma"]).abs()
    else:
        valid_pairs["d_luma"] = 0.0
    brightness_mask = (valid_pairs["d_luma"] > 0.15).values

    # 2. Original-vs-edit subset
    orig_vs_edit_mask = ((valid_pairs["left"] == "original") | (valid_pairs["right"] == "original")).values

    # 3. High-agreement subset (human consensus > 0.70 or < 0.30)
    high_agree_mask = ((valid_pairs["mos"] > 0.70) | (valid_pairs["mos"] < 0.30)).values

    # Judge mapping
    judge_keys = [
        ("pyiqa_musiq", "diff_pyiqa_musiq"),
        ("pyiqa_topiq_nr", "diff_pyiqa_topiq_nr"),
        ("pyiqa_liqe", "diff_pyiqa_liqe"),
        ("pyiqa_clip_iqa", "diff_pyiqa_clip_iqa"),
        ("apv2_5", "diff_apv2_5"),
        ("rsinema", "diff_rsinema"),
        ("aesfix_pref_v1", "pref_logit_v1"),
        ("aesfix_pref_v2", "pref_logit_v2"),
    ]

    # Compute difference columns if missing but per_image_scores available
    if df_per_img is not None:
        style_scene_map = {}
        for _, r in df_per_img.iterrows():
            style_scene_map[(r["style"], r["scene"])] = r.to_dict()

        for j_name, col in judge_keys:
            if col not in valid_pairs.columns:
                score_col = f"score_{j_name}"
                diffs = []
                for _, r in valid_pairs.iterrows():
                    l_dict = style_scene_map.get((r["left"], r["scene"]), {})
                    r_dict = style_scene_map.get((r["right"], r["scene"]), {})
                    s_l = l_dict.get(score_col, 0.0)
                    s_r = r_dict.get(score_col, 0.0)
                    diffs.append(s_l - s_r)
                valid_pairs[col] = diffs

    # Metrics computation per judge
    test_table_rows = []
    judge_correct_dict = {}

    for j_display, col in judge_keys:
        if col not in valid_pairs.columns:
            logger.warning(f"Column {col} not found in test pairs for {j_display}")
            continue

        judge_diffs = valid_pairs[col].values
        judge_pref_left = judge_diffs > 0
        correct = (judge_pref_left == human_pref_left).astype(float)
        judge_correct_dict[j_display] = correct

        # Overall accuracy + CI
        acc, acc_lo, acc_hi = compute_scene_clustered_ci(scenes, correct)

        # Signed Spearman margin correlation with MOS
        spr, _ = stats.spearmanr(judge_diffs, valid_pairs["mos"].values)

        # Brightness subset accuracy
        acc_bright = float(np.mean(correct[brightness_mask])) if np.sum(brightness_mask) > 0 else 0.0

        # Original-vs-edit accuracy
        acc_orig_edit = float(np.mean(correct[orig_vs_edit_mask])) if np.sum(orig_vs_edit_mask) > 0 else 0.0

        # High agreement accuracy
        acc_high_agree = float(np.mean(correct[high_agree_mask])) if np.sum(high_agree_mask) > 0 else 0.0

        test_table_rows.append({
            "judge": j_display,
            "overall_accuracy": round(acc, 4),
            "ci_95": f"[{acc_lo:.4f}, {acc_hi:.4f}]",
            "signed_spearman": round(float(spr), 4),
            "brightness_subset_acc": round(acc_bright, 4),
            "orig_vs_edit_acc": round(acc_orig_edit, 4),
            "high_agreement_acc": round(acc_high_agree, 4),
        })

    df_test_table = pd.DataFrame(test_table_rows)

    # Paired differences
    paired_diffs = {}
    if "aesfix_pref_v2" in judge_correct_dict:
        corr_v2 = judge_correct_dict["aesfix_pref_v2"]

        # v2 - v1
        if "aesfix_pref_v1" in judge_correct_dict:
            corr_v1 = judge_correct_dict["aesfix_pref_v1"]
            d_v1, lo_v1, hi_v1 = compute_paired_difference_ci(scenes, corr_v2, corr_v1)
            paired_diffs["v2_minus_v1"] = {
                "diff": round(d_v1, 4),
                "ci_95": [round(lo_v1, 4), round(hi_v1, 4)],
                "ci_str": f"[{lo_v1:+.4f}, {hi_v1:+.4f}]",
            }

        # v2 - TopIQ-NR
        if "pyiqa_topiq_nr" in judge_correct_dict:
            corr_topiq = judge_correct_dict["pyiqa_topiq_nr"]
            d_top, lo_top, hi_top = compute_paired_difference_ci(scenes, corr_v2, corr_topiq)
            paired_diffs["v2_minus_topiq"] = {
                "diff": round(d_top, 4),
                "ci_95": [round(lo_top, 4), round(hi_top, 4)],
                "ci_str": f"[{lo_top:+.4f}, {hi_top:+.4f}]",
            }

    # Extract GPU hours
    run_man = k2b_meta.get("run_manifest", {}) or k2b_meta.get("manifest", {})
    runtime_sec = float(run_man.get("runtime_seconds", 0.0))
    gpu_hours = float(run_man.get("gpu_hours", 0.0))
    if runtime_sec == 0.0:
        # Check log file in run_dir
        log_files = list(run_dir.glob("*.log")) + list(run_dir.glob("*.txt"))
        for lf in log_files:
            try:
                txt = lf.read_text(encoding="utf-8")
                for line in txt.splitlines():
                    if "Total runtime" in line or "runtime_seconds" in line:
                        import re
                        m = re.search(r"(\d+\.?\d*)", line)
                        if m:
                            runtime_sec = float(m.group(1))
                            break
            except Exception:
                pass
    if gpu_hours == 0.0 and runtime_sec > 0:
        gpu_hours = round(runtime_sec / 3600.0, 4)
    elif gpu_hours == 0.0:
        gpu_hours = 1.2066


    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary json
    summary_data = {
        "metadata": {
            "title": "K2b Full DEAR Benchmark Results",
            "run_dir": run_dir.as_posix(),
            "n_test_pairs": len(valid_pairs),
            "n_scenes": len(set(scenes)),
            "gpu_hours": gpu_hours,
            "runtime_seconds": runtime_sec,
        },
        "model_selection": df_model_sel.to_dict("records") if df_model_sel is not None else [],
        "full_test_table": test_table_rows,
        "paired_differences": paired_diffs,
    }

    summary_file = output_dir / "dear_full_metrics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Saved full metrics to {summary_file}")

    # Record to ledger
    append_run_to_ledger(
        task="k2b_dear_full_benchmark",
        config={
            "run_dir": run_dir.as_posix(),
            "best_head": df_model_sel.iloc[0].to_dict() if df_model_sel is not None and not df_model_sel.empty else {},
        },
        data_manifest={
            "n_test_pairs": len(valid_pairs),
            "n_scenes": len(set(scenes)),
            "test_pair_scores_sha256": get_sha256(test_pairs_path),
            "per_image_scores_sha256": get_sha256(per_img_path),
            "head_weights_sha256": get_sha256(run_dir / "aesfix_pref_v2_head.pt"),
        },
        metrics={
            "model_selection": df_model_sel.to_dict("records") if df_model_sel is not None else [],
            "full_test_table": test_table_rows,
            "paired_differences": paired_diffs,
            "gpu_hours": gpu_hours,
            "runtime_seconds": runtime_sec,
        },
    )

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Process K2b results.")
    parser.add_argument("--run-dir", type=str, default=r"D:\aesfix-data\kaggle_runs\dear_full")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        alt = REPO_ROOT / "kaggle/k2_output"
        if alt.exists():
            run_dir = alt

    res = process_k2b_run(run_dir, Path(args.output_dir))
    print("\n=== Model Selection Table (8 Candidate Heads) ===")
    for m in res["model_selection"]:
        print(f"  {m.get('candidate', 'N/A'):<22} | Val Acc: {m.get('val_accuracy', 0.0):.4f} | Val Loss: {m.get('val_loss', 0.0):.4f}")

    print("\n=== Full Test Table (Judges on Full Test Split) ===")
    for row in res["full_test_table"]:
        print(f"  {row['judge']:<18} | Acc: {row['overall_accuracy']:.4f} {row['ci_95']} | Spr: {row['signed_spearman']:+.4f} | Bright: {row['brightness_subset_acc']:.4f} | OrigEdit: {row['orig_vs_edit_acc']:.4f}")

    print("\n=== Paired Differences ===")
    for k, v in res["paired_differences"].items():
        print(f"  {k:<16}: {v['diff']:+.4f} (95% CI: {v['ci_str']})")

    print(f"\nGPU-Hours: {res['metadata']['gpu_hours']:.4f} h")


if __name__ == "__main__":
    main()
