#!/usr/bin/env python3
"""T4: Caveat Check for Goodhart Entry 2 (Overexposure & Highlight Clipping).

The oracle assumes the expert render beats its +1 EV version.
On the 100 overexposure cases from synthetic benchmark v2:
- Computes highlight clipping fraction before and after the +1 EV degradation
- Computes how many +1 EV images have > 1% clipped highlights
- Deterministically samples 20 overexposure pairs (seed=42) and saves them
  to results/human_study_candidates/ for the human study.
- Exports results/t4_clipping_summary.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CASES_CSV = REPO_ROOT / "results" / "synth_v2_cases.csv"
OUT_DIR = REPO_ROOT / "results" / "human_study_candidates"
SUMMARY_JSON = REPO_ROOT / "results" / "t4_clipping_summary.json"


def compute_highlight_clipping(img: Image.Image, threshold: float = 0.99) -> float:
    """Computes the fraction of pixels where any color channel exceeds threshold."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    # A pixel is highlight clipped if max(R, G, B) >= threshold
    clipped = np.max(arr, axis=-1) >= threshold
    return float(np.mean(clipped))


def main():
    print("=== T4: Goodhart Entry 2 Caveat Check (Highlight Clipping on +1 EV) ===")
    assert CASES_CSV.exists(), f"Cases CSV not found: {CASES_CSV}"
    df = pd.read_csv(CASES_CSV)
    df_over = df[df["deg_name"] == "overexposure"].copy()
    n_total = len(df_over)
    print(f"Loaded {n_total} overexposure cases from {CASES_CSV}")

    records = []
    hl_before_list = []
    hl_after_list = []
    clipped_gt_1pct_count = 0

    for idx, row in df_over.iterrows():
        scene = row["scene"]
        deg_path = Path(row["deg_path"])
        ref_path = Path(row["ref_path"])

        with Image.open(ref_path) as im:
            ref_img = im.convert("RGB")
        with Image.open(deg_path) as im:
            deg_img = im.convert("RGB")

        hl_before = compute_highlight_clipping(ref_img)
        hl_after = compute_highlight_clipping(deg_img)
        hl_growth = hl_after - hl_before

        hl_before_list.append(hl_before)
        hl_after_list.append(hl_after)

        has_gt_1pct = hl_after > 0.01
        if has_gt_1pct:
            clipped_gt_1pct_count += 1

        records.append({
            "scene": scene,
            "ref_path": str(ref_path),
            "deg_path": str(deg_path),
            "hl_before": round(hl_before, 5),
            "hl_after": round(hl_after, 5),
            "hl_growth": round(hl_growth, 5),
            "clipped_gt_1pct": has_gt_1pct,
        })

    mean_before = float(np.mean(hl_before_list))
    median_before = float(np.median(hl_before_list))
    mean_after = float(np.mean(hl_after_list))
    median_after = float(np.median(hl_after_list))
    mean_growth = float(np.mean(np.array(hl_after_list) - np.array(hl_before_list)))

    print("\n--- Summary Statistics (N=100 Overexposure Cases) ---")
    print(f"Mean highlight clipping before (+0 EV reference):  {mean_before*100:.2f}% (median: {median_before*100:.2f}%)")
    print(f"Mean highlight clipping after  (+1 EV degraded):   {mean_after*100:.2f}% (median: {median_after*100:.2f}%)")
    print(f"Mean highlight clipping growth:                   {mean_growth*100:+.2f}%")
    print(f"Cases with > 1% clipped highlights:               {clipped_gt_1pct_count}/{n_total} ({clipped_gt_1pct_count/n_total*100:.1f}%)")

    # Sample 20 pairs deterministically with seed=42
    rng = random.Random(42)
    sample_indices = sorted(rng.sample(range(n_total), 20))
    sampled_records = [records[i] for i in sample_indices]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    study_manifest = []

    print(f"\nSaving 20 human study candidate pairs to {OUT_DIR} (seed=42)...")
    for rank, rec in enumerate(sampled_records, 1):
        scene_stem = Path(rec["scene"]).stem
        ref_dst = OUT_DIR / f"{rank:02d}_{scene_stem}_ref.jpg"
        deg_dst = OUT_DIR / f"{rank:02d}_{scene_stem}_plus1ev.jpg"

        shutil.copyfile(rec["ref_path"], ref_dst)
        shutil.copyfile(rec["deg_path"], deg_dst)

        # Hashes
        h_ref = hashlib.sha256(ref_dst.read_bytes()).hexdigest()
        h_deg = hashlib.sha256(deg_dst.read_bytes()).hexdigest()

        study_manifest.append({
            "pair_id": rank,
            "scene": rec["scene"],
            "ref_file": ref_dst.name,
            "ref_sha256": h_ref,
            "ref_hl_clipping": rec["hl_before"],
            "plus1ev_file": deg_dst.name,
            "plus1ev_sha256": h_deg,
            "plus1ev_hl_clipping": rec["hl_after"],
        })

    summary_data = {
        "metadata": {
            "task": "T4 Caveat Check for Goodhart Entry 2",
            "dataset": "DEAR synthetic benchmark v2 (100 overexposure scenes)",
            "seed_pairs": 42,
            "source_cases": str(CASES_CSV.relative_to(REPO_ROOT)),
        },
        "highlight_clipping_stats": {
            "n_scenes": n_total,
            "mean_clipping_before": round(mean_before, 5),
            "median_clipping_before": round(median_before, 5),
            "mean_clipping_after": round(mean_after, 5),
            "median_clipping_after": round(median_after, 5),
            "mean_clipping_growth": round(mean_growth, 5),
            "count_gt_1pct_clipped": clipped_gt_1pct_count,
            "rate_gt_1pct_clipped": round(clipped_gt_1pct_count / n_total, 4),
        },
        "human_study_sample_20": study_manifest,
        "all_100_cases": records,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    hasher = hashlib.sha256()
    with open(SUMMARY_JSON, "rb") as f:
        hasher.update(f.read())
    print(f"Exported T4 summary to {SUMMARY_JSON}")
    print(f"Summary SHA-256: {hasher.hexdigest()}")


if __name__ == "__main__":
    main()
