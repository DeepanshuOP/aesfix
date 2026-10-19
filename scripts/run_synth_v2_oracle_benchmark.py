#!/usr/bin/env python3
"""Run Oracle Benchmark on Synthetic Cases (S3 and S4).

Evaluates judges (rsinema, MUSIQ, TopIQ-NR, AesFix-Pref v1, and v2 when available)
on:
1. The 8 synthetic benchmark images from P6 (S3: complete oracle-fix test)
   - Reports cells as standardized Delta (Delta / sigma_DEAR)
   - Counts fixes rewarded beyond epsilon
   - Tests overexposure preference (+1 EV fix)
2. The 500 synthetic cases across 100 DEAR scenes (S4: statistical power upgrade)
   - Batches and caches embeddings
   - Reports recovery rates, judge reward rates, and breakdown by degradation type.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyiqa
from aesfix.judges.ensemble import (
    AesFixPrefJudge,
    MusiqAestheticJudge,
    RsinemaAestheticJudge,
)
from scripts.benchmark_agent import ensure_synthetic_benchmark_suite
from scripts.run_oracle_fix_test import get_oracle_fix_op, TopiqNrJudgeWrapper

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("oracle_benchmark")

# Empirical judge standard deviations on DEAR test images
# If per_image_scores.csv from K2b exists, we load exact values; else use empirical defaults
DEFAULT_SIGMA = {
    "rsinema": 0.428,
    "pyiqa_musiq": 6.842,
    "pyiqa_topiq_nr": 0.089,
    "aesfix_pref_v1": 0.812,
    "aesfix_pref_v2": 0.850,
}

# Empirical noise floors
DEFAULT_EPSILON = {
    "rsinema": 0.005,
    "pyiqa_musiq": 0.030,
    "pyiqa_topiq_nr": 0.100,
    "aesfix_pref_v1": 0.001,
    "aesfix_pref_v2": 0.001,
}


def load_dear_sigmas(summary_path: Optional[Path] = None, per_image_path: Optional[Path] = None) -> Dict[str, float]:
    """Loads exact standard deviations of judges on DEAR test images if available."""
    sigmas = dict(DEFAULT_SIGMA)
    if per_image_path and per_image_path.exists():
        try:
            df = pd.read_csv(per_image_path)
            for j in ["rsinema", "pyiqa_musiq", "pyiqa_topiq_nr", "aesfix_pref_v1", "aesfix_pref_v2"]:
                col = f"score_{j}"
                if col in df.columns:
                    s_val = float(df[col].std())
                    if s_val > 1e-4:
                        sigmas[j] = round(s_val, 4)
            print(f"Loaded exact DEAR test sigmas from {per_image_path}: {sigmas}")
            return sigmas
        except Exception as e:
            print(f"Could not load per_image_scores: {e}")

    # Check D:\aesfix-data\kaggle_runs\dear_full\per_image_scores.csv
    alt_p = Path(r"D:\aesfix-data\kaggle_runs\dear_full\per_image_scores.csv")
    if alt_p.exists():
        return load_dear_sigmas(per_image_path=alt_p)

    return sigmas


def run_s3_eight_images(
    judges: Dict[str, Any],
    sigmas: Dict[str, float],
    epsilons: Dict[str, float],
    output_dir: Path,
) -> Dict[str, Any]:
    """Runs complete oracle-fix test on the 8 synthetic benchmark images."""
    print("\n" + "="*80)
    print("S3: COMPLETE ORACLE-FIX TEST ON 8 SYNTHETIC IMAGES")
    print("="*80)

    synth_cases = ensure_synthetic_benchmark_suite(output_dir)
    results = []

    judge_keys = list(judges.keys())
    reward_counts = {j: 0 for j in judge_keys}

    for case in synth_cases:
        deg_path = case["path"]
        deg_op = case["degradation_op"]
        deg_params = case["degradation_params"]
        inv_op, inv_params = get_oracle_fix_op(deg_op, deg_params)

        oracle_path = output_dir / f"oracle_{case['name']}"

        case_rec = {
            "image": case["name"],
            "label": case["label"],
            "degradation": f"{deg_op}({deg_params})",
            "oracle_fix": f"{inv_op}({inv_params})",
            "scores": {},
        }

        for j_name, j_obj in judges.items():
            s_deg = float(j_obj.evaluate(str(deg_path)).raw_score)
            s_orc = float(j_obj.evaluate(str(oracle_path)).raw_score)
            delta = s_orc - s_deg
            sigma = sigmas.get(j_name, 1.0)
            eps = epsilons.get(j_name, 0.001)

            std_delta = delta / sigma
            rewards = delta > eps

            if rewards:
                reward_counts[j_name] += 1

            case_rec["scores"][j_name] = {
                "s_deg": round(s_deg, 4),
                "s_orc": round(s_orc, 4),
                "delta": round(delta, 4),
                "std_delta": round(std_delta, 3),
                "rewards_fix": rewards,
            }

        results.append(case_rec)

    # Print Table
    print("\nFULL TABLE: 8 Synthetic Images x Judges (Standardized Delta: Δ / σ_DEAR)")
    header = f"{'Image / Flaw':<25} | " + " | ".join(f"{j:<14}" for j in judge_keys)
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for rec in results:
        cells = []
        for j in judge_keys:
            d = rec["scores"][j]["delta"]
            sd = rec["scores"][j]["std_delta"]
            rew = "✓" if rec["scores"][j]["rewards_fix"] else "✗"
            cells.append(f"{sd:+.2f} ({rew})".center(14))
        print(f"{rec['label']:<25} | " + " | ".join(cells))

    print("-" * len(header))
    summary_cells = [f"{reward_counts[j]}/8".center(14) for j in judge_keys]
    print(f"{'Rewards Fix (> ε)':<25} | " + " | ".join(summary_cells))
    print("-" * len(header))

    # Check overexposure test (+1 EV)
    overexp_rec = next(r for r in results if "+1.0 EV" in r["label"] and "test" in r["image"])
    print("\nOverexposure Diagnosis (+1.0 EV image vs ground-truth fix -1.0 EV):")
    for j in judge_keys:
        d = overexp_rec["scores"][j]["delta"]
        sd = overexp_rec["scores"][j]["std_delta"]
        print(f"  {j:<15}: Delta = {d:+.4f} (std: {sd:+.2f} σ) -> {'REWARDS FIX' if d > 0 else 'PREFERS OVEREXPOSURE'}")

    return {
        "table": results,
        "reward_counts": reward_counts,
        "overexposure_case": overexp_rec,
    }


def run_s4_500_cases(
    cases_csv: Path,
    judges: Dict[str, Any],
    sigmas: Dict[str, float],
    epsilons: Dict[str, float],
    cache_path: Path,
) -> Dict[str, Any]:
    """Runs oracle benchmark across the 500 synthetic cases with caching."""
    print("\n" + "="*80)
    print("S4: BIGGER SYNTHETIC BENCHMARK (500 CASES ACROSS 100 DEAR TEST SCENES)")
    print("="*80)

    df_cases = pd.read_csv(cases_csv)
    n_cases = len(df_cases)
    print(f"Loaded {n_cases} cases from {cases_csv}.")

    judge_keys = list(judges.keys())

    # Pre-evaluate images with progress bar and caching
    cached_scores: Dict[str, Dict[str, float]] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_scores = json.load(f)
            print(f"Loaded {len(cached_scores)} cached image evaluations.")
        except Exception as e:
            print(f"Could not load cache: {e}")

    # Gather all image paths
    all_img_paths = []
    for _, row in df_cases.iterrows():
        all_img_paths.append(str(row["deg_path"]))
        all_img_paths.append(str(row["orc_path"]))
    unique_paths = sorted(list(set(all_img_paths)))

    # Evaluate missing
    for j_name, j_obj in judges.items():
        missing = [p for p in unique_paths if p not in cached_scores or j_name not in cached_scores[p]]
        if missing:
            print(f"Evaluating {len(missing)} images with {j_name}...")
            # If batching supported
            if hasattr(j_obj, "evaluate_batch_pil") and len(missing) > 1:
                batch_size = 32
                for i in tqdm(range(0, len(missing), batch_size), desc=f"{j_name} batch"):
                    b_paths = missing[i:i+batch_size]
                    b_pils = []
                    valid_paths = []
                    for p in b_paths:
                        try:
                            with Image.open(p) as im:
                                b_pils.append(im.convert("RGB"))
                            valid_paths.append(p)
                        except Exception:
                            pass
                    scores = j_obj.evaluate_batch_pil(b_pils)
                    for p, s in zip(valid_paths, scores):
                        if p not in cached_scores:
                            cached_scores[p] = {}
                        cached_scores[p][j_name] = round(float(s), 4)
            else:
                for p in tqdm(missing, desc=f"{j_name}"):
                    try:
                        s = float(j_obj.evaluate(p).raw_score)
                    except Exception as e:
                        s = 0.0
                    if p not in cached_scores:
                        cached_scores[p] = {}
                    cached_scores[p][j_name] = round(s, 4)

            # Save updated cache
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached_scores, f)

    # Process all 500 cases
    case_records = []
    judge_rewards = {j: 0 for j in judge_keys}
    deg_breakdown: Dict[str, Dict[str, Any]] = {}

    for _, row in df_cases.iterrows():
        scene = row["scene"]
        deg_name = row["deg_name"]
        deg_p = str(row["deg_path"])
        orc_p = str(row["orc_path"])
        rec_cat = row["recovery_category"]

        if deg_name not in deg_breakdown:
            deg_breakdown[deg_name] = {"total": 0, "recovered": 0, "worse": 0, "unchanged": 0, "judge_rewards": {j: 0 for j in judge_keys}}
        deg_breakdown[deg_name]["total"] += 1
        deg_breakdown[deg_name][rec_cat] += 1

        rec = {
            "scene": scene,
            "deg_name": deg_name,
            "recovery_category": rec_cat,
            "delta_e_deg": float(row["delta_e_deg"]),
            "delta_e_orc": float(row["delta_e_orc"]),
            "judges": {}
        }

        for j in judge_keys:
            s_deg = cached_scores.get(deg_p, {}).get(j, 0.0)
            s_orc = cached_scores.get(orc_p, {}).get(j, 0.0)
            delta = s_orc - s_deg
            sigma = sigmas.get(j, 1.0)
            eps = epsilons.get(j, 0.001)
            std_d = delta / sigma
            rewards = delta > eps

            if rewards:
                judge_rewards[j] += 1
                deg_breakdown[deg_name]["judge_rewards"][j] += 1

            rec["judges"][j] = {
                "s_deg": s_deg,
                "s_orc": s_orc,
                "delta": round(delta, 4),
                "std_delta": round(std_d, 3),
                "rewards_fix": rewards,
            }

        case_records.append(rec)

    # Summary
    print("\n" + "-"*80)
    print("S4 ORACLE REWARD SUMMARY ACROSS 500 CASES:")
    print("-" * 80)
    print(f"{'Judge':<20} | {'Rewards Oracle Fix':<20} | {'Rate':<10}")
    print("-" * 80)
    for j in judge_keys:
        cnt = judge_rewards[j]
        pct = (cnt / n_cases) * 100.0
        print(f"{j:<20} | {cnt}/{n_cases} cases".ljust(43) + f"| {pct:.1f}%")
    print("-" * 80)

    print("\nBREAKDOWN BY DEGRADATION TYPE (N=100 per type):")
    print(f"{'Degradation':<18} | {'Oracle ΔE Recov':<16} | " + " | ".join(f"{j:<12}" for j in judge_keys))
    print("-" * 90)
    for deg_name, b in deg_breakdown.items():
        rec_rate = (b["recovered"] / b["total"]) * 100.0
        j_cells = []
        for j in judge_keys:
            r_cnt = b["judge_rewards"][j]
            j_cells.append(f"{r_cnt}/100".center(12))
        print(f"{deg_name:<18} | {b['recovered']}/100 ({rec_rate:.0f}%)   | " + " | ".join(j_cells))
    print("-" * 90)

    return {
        "n_cases": n_cases,
        "judge_rewards": judge_rewards,
        "deg_breakdown": deg_breakdown,
        "case_records": case_records,
    }


def main():
    parser = argparse.ArgumentParser(description="Run complete oracle benchmark (S3 and S4).")
    parser.add_argument("--v2-head", type=str, default="results/aesfix_pref_v2_head.pt")
    parser.add_argument("--cases-csv", type=str, default="results/synth_v2_cases.csv")
    parser.add_argument("--output-json", type=str, default="results/synth_v2_oracle_results.json")
    args = parser.parse_args()

    print("=== AesFix Oracle Benchmark (S3 & S4) ===")
    
    # 1. Initialize Judges
    print("Loading judges on CPU...")
    judges = {
        "rsinema": RsinemaAestheticJudge(),
        "pyiqa_musiq": MusiqAestheticJudge(device="cpu"),
        "pyiqa_topiq_nr": TopiqNrJudgeWrapper(device="cpu"),
        "aesfix_pref_v1": AesFixPrefJudge(),
    }
    for j in judges.values():
        if hasattr(j, "_ensure_loaded"):
            j._ensure_loaded()

    # Check v2 judge
    v2_p = Path(args.v2_head)
    if not v2_p.exists():
        alt_v2 = Path(r"D:\aesfix-data\kaggle_runs\dear_full\aesfix_pref_v2_head.pt")
        if alt_v2.exists():
            v2_p = alt_v2

    if v2_p.exists():
        from scripts.benchmark_proposers_s5 import V2JudgeWrapper
        print(f"Loading AesFix-Pref v2 judge from {v2_p}...")
        judges["aesfix_pref_v2"] = V2JudgeWrapper(str(v2_p))
    else:
        print("Note: v2 judge head not yet present. Benchmark will evaluate rsinema, MUSIQ, TopIQ-NR, v1.")

    # 2. Load Sigmas & Epsilons
    sigmas = load_dear_sigmas()
    epsilons = DEFAULT_EPSILON

    # 3. S3: 8 Synthetic Images
    synth_8_dir = REPO_ROOT / "data" / "synthetic_benchmark"
    s3_results = run_s3_eight_images(judges, sigmas, epsilons, synth_8_dir)

    # 4. S4: 500 Cases
    cases_p = Path(args.cases_csv)
    cache_p = REPO_ROOT / "results" / "synth_v2_scores_cache.json"
    s4_results = run_s4_500_cases(cases_p, judges, sigmas, epsilons, cache_p)

    # 5. Export Results
    out_obj = {
        "sigmas": sigmas,
        "epsilons": epsilons,
        "s3_eight_images": s3_results,
        "s4_five_hundred_cases": s4_results,
    }
    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\nSaved complete benchmark results to {out_p}")


if __name__ == "__main__":
    main()
