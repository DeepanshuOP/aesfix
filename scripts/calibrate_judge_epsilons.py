#!/usr/bin/env python3
"""Calibrate Judge Noise Floors (ε_j) on 50 DEAR Train Images (T0.d).

Implements the pre-registered compute_noise_floor procedure:
- JPEG q90 re-encode
- JPEG q95 re-encode
- Bilinear resize +2%
- Bilinear resize -2%

Evaluates across a fixed, seeded set of 50 DEAR train scenes strictly disjoint
from all test and synthetic evaluation sets.
Freezes ε_j = 95th percentile of |Δ_j| into config/judge_epsilon.json
along with image list, individual SHA-256 digests, and manifest hash.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calibrate_epsilons")

TRAIN_MARKUP = Path(r"D:\aesfix-data\dear\train_processed_markup.csv")
TEST_MARKUP = Path(r"D:\aesfix-data\dear\test_processed_markup.csv")
OUT_DIR = REPO_ROOT / "data" / "dear_train_50"
PERTS_DIR = REPO_ROOT / "data" / "dear_train_50_perts"
CONFIG_FILE = REPO_ROOT / "config" / "judge_epsilon.json"
HF_CACHE_DIR = REPO_ROOT / "scratch" / "hf_dataset_cache"


def sample_train_scenes(n: int = 50, seed: int = 42) -> List[str]:
    """Deterministically samples n DEAR train scenes strictly disjoint from test scenes."""
    df_tr = pd.read_csv(TRAIN_MARKUP)
    df_te = pd.read_csv(TEST_MARKUP)

    train_scenes = sorted(df_tr["name"].unique().tolist())
    test_scenes = set(df_te["name"].unique().tolist())

    disjoint_train = [s for s in train_scenes if s not in test_scenes]
    assert len(disjoint_train) == len(train_scenes), "Train and test scenes overlap!"

    rng = random.Random(seed)
    sampled = sorted(rng.sample(disjoint_train, n))
    logger.info("Sampled %d train scenes using seed=%d", len(sampled), seed)
    return sampled


def download_and_preprocess(scene: str) -> Tuple[str, Path, str]:
    """Downloads tiff16_c from HF, resizes long edge to 512px (q92), and computes sha256."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / scene

    if not out_path.exists() or out_path.stat().st_size == 0:
        raw_path = hf_hub_download(
            repo_id="vsevolodpl/DEAR",
            filename=f"images/tiff16_c/{scene}",
            repo_type="dataset",
            cache_dir=str(HF_CACHE_DIR),
        )
        with Image.open(raw_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > 512:
                scale = 512.0 / max(w, h)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                im = im.resize((nw, nh), Image.Resampling.LANCZOS)
            im.save(str(out_path), format="JPEG", quality=92)

    hasher = hashlib.sha256()
    with open(out_path, "rb") as f:
        hasher.update(f.read())
    sha = hasher.hexdigest()

    return scene, out_path, sha


def create_and_save_perturbations(scene: str, orig_path: Path) -> List[Tuple[str, str, Path]]:
    """Generates and writes the 4 canonical perturbations to disk.
    Returns list of (scene, variation_name, path).
    """
    PERTS_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(orig_path) as im:
        orig = im.convert("RGB")

    items = []
    # 0. Orig
    orig_save = PERTS_DIR / f"{scene}__orig.jpg"
    if not orig_save.exists():
        orig.save(str(orig_save), format="JPEG", quality=95)
    items.append((scene, "orig", orig_save))

    # 1. JPEG q90
    q90_p = PERTS_DIR / f"{scene}__jpeg_q90.jpg"
    if not q90_p.exists():
        orig.save(str(q90_p), format="JPEG", quality=90)
    items.append((scene, "jpeg_q90", q90_p))

    # 2. JPEG q95
    q95_p = PERTS_DIR / f"{scene}__jpeg_q95.jpg"
    if not q95_p.exists():
        orig.save(str(q95_p), format="JPEG", quality=95)
    items.append((scene, "jpeg_q95", q95_p))

    # 3. +2% resize
    up_p = PERTS_DIR / f"{scene}__resize_plus_2pct.jpg"
    if not up_p.exists():
        w, h = orig.size
        w_up, h_up = max(1, int(round(w * 1.02))), max(1, int(round(h * 1.02)))
        img_up = orig.resize((w_up, h_up), Image.Resampling.BILINEAR)
        img_up.save(str(up_p), format="JPEG", quality=95)
    items.append((scene, "resize_plus_2pct", up_p))

    # 4. -2% resize
    down_p = PERTS_DIR / f"{scene}__resize_minus_2pct.jpg"
    if not down_p.exists():
        w, h = orig.size
        w_down, h_down = max(1, int(round(w * 0.98))), max(1, int(round(h * 0.98)))
        img_down = orig.resize((w_down, h_down), Image.Resampling.BILINEAR)
        img_down.save(str(down_p), format="JPEG", quality=95)
    items.append((scene, "resize_minus_2pct", down_p))

    return items


def evaluate_rsinema_all(image_paths: List[Path], batch_size: int = 16) -> List[float]:
    """Batched evaluation for rsinema judge."""
    from aesfix.judges.ensemble import RsinemaAestheticJudge
    judge = RsinemaAestheticJudge()
    judge._ensure_loaded()
    scores: List[float] = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Eval rsinema"):
        batch_paths = image_paths[i : i + batch_size]
        pil_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = judge._processor(images=pil_imgs, return_tensors="pt")
        with torch.no_grad():
            out = judge._model(**inputs)
            cls_token = out.pooler_output
            raw_scores = (cls_token @ judge._heads["aesthetic"][0].T + judge._heads["aesthetic"][1]).squeeze(-1)
            if raw_scores.ndim == 0:
                scores.append(round(float(raw_scores.item()), 4))
            else:
                scores.extend([round(float(v), 4) for v in raw_scores.tolist()])
    return scores


def evaluate_v1_all(image_paths: List[Path], batch_size: int = 16) -> List[float]:
    """Batched evaluation for AesFix-Pref v1."""
    from aesfix.judges.ensemble import AesFixPrefJudge
    judge = AesFixPrefJudge()
    judge._ensure_loaded()
    scores: List[float] = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Eval aesfix_pref_v1"):
        batch_paths = image_paths[i : i + batch_size]
        pil_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = judge._processor(images=pil_imgs, return_tensors="pt")
        with torch.no_grad():
            vision_outputs = judge._model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_outputs[1]
            proj = judge._model.visual_projection(pooled)
            proj_norm = proj / proj.norm(p=2, dim=-1, keepdim=True)
            raw_scores = (proj_norm @ judge._weight.T + judge._bias).squeeze(-1)
            if raw_scores.ndim == 0:
                scores.append(round(float(raw_scores.item()), 4))
            else:
                scores.extend([round(float(v), 4) for v in raw_scores.tolist()])
    return scores


def evaluate_v2_all(image_paths: List[Path], batch_size: int = 8) -> List[float]:
    """Batched evaluation for AesFix-Pref v2 (SigLIP so400m)."""
    from scripts.benchmark_proposers_s5 import V2JudgeWrapper
    v2_p = REPO_ROOT / "results" / "aesfix_pref_v2_head.pt"
    if not v2_p.exists():
        v2_p = Path(r"D:\aesfix-data\kaggle_runs\dear_full\aesfix_pref_v2_head.pt")
    judge = V2JudgeWrapper(str(v2_p))
    judge._ensure_loaded()
    scores: List[float] = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Eval aesfix_pref_v2"):
        batch_paths = image_paths[i : i + batch_size]
        pil_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = judge._processor(images=pil_imgs, return_tensors="pt")
        with torch.no_grad():
            out = judge._model(pixel_values=inputs["pixel_values"])
            feat = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out[1]
            feat = feat / (torch.norm(feat, dim=-1, keepdim=True) + 1e-7)
            raw_scores = judge._head(feat.float()).squeeze(-1)
            if raw_scores.ndim == 0:
                scores.append(round(float(raw_scores.item()), 4))
            else:
                scores.extend([round(float(v), 4) for v in raw_scores.tolist()])
    return scores


def evaluate_pyiqa_all(metric_name: str, image_paths: List[Path]) -> List[float]:
    """Evaluation for pyiqa judges (topiq_nr, musiq)."""
    import pyiqa
    metric = pyiqa.create_metric(metric_name, device="cpu")
    scores = []
    for p in tqdm(image_paths, desc=f"Eval pyiqa_{metric_name}"):
        with torch.no_grad():
            val = float(metric(str(p)).squeeze().item())
        scores.append(round(val, 4))
    return scores


def main() -> None:
    t_start = time.time()
    print("=== T0.d: Calibration of Judge Noise Floors (ε_j) ===")

    # 1. Sample and verify 50 DEAR train scenes
    sampled_scenes = sample_train_scenes(50, seed=42)
    print(f"Verifying/downloading 50 train reference images in {OUT_DIR}...")
    manifest = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(download_and_preprocess, sampled_scenes))

    for scene, p, sha in sorted(results):
        manifest.append({"scene": scene, "path": str(p), "sha256": sha})

    manifest_bytes = json.dumps([m["sha256"] for m in manifest], sort_keys=True).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    print(f"Verified 50 train images. Collective Manifest SHA256: {manifest_sha256}")

    # 2. Generate and save perturbations
    print("\nGenerating 4 perturbations for each image (total 250 files)...")
    all_items: List[Tuple[str, str, Path]] = []
    for item in manifest:
        s_items = create_and_save_perturbations(item["scene"], Path(item["path"]))
        all_items.extend(s_items)
    print(f"Prepared {len(all_items)} images in {PERTS_DIR}")

    all_paths = [it[2] for it in all_items]

    # 3. Model-by-model batched evaluation
    print("\nRunning model-by-model evaluation across all 250 images...")
    scores_dict: Dict[str, List[float]] = {}

    t0 = time.time()
    scores_dict["rsinema"] = evaluate_rsinema_all(all_paths, batch_size=16)
    print(f"rsinema evaluated in {time.time()-t0:.1f}s")

    t0 = time.time()
    scores_dict["aesfix_pref_v1"] = evaluate_v1_all(all_paths, batch_size=16)
    print(f"aesfix_pref_v1 evaluated in {time.time()-t0:.1f}s")

    t0 = time.time()
    scores_dict["aesfix_pref_v2"] = evaluate_v2_all(all_paths, batch_size=8)
    print(f"aesfix_pref_v2 evaluated in {time.time()-t0:.1f}s")

    t0 = time.time()
    scores_dict["pyiqa_topiq_nr"] = evaluate_pyiqa_all("topiq_nr", all_paths)
    print(f"pyiqa_topiq_nr evaluated in {time.time()-t0:.1f}s")

    t0 = time.time()
    scores_dict["pyiqa_musiq"] = evaluate_pyiqa_all("musiq", all_paths)
    print(f"pyiqa_musiq evaluated in {time.time()-t0:.1f}s")

    # 4. Organize scores and compute |Δ_j|
    print("\nComputing perturbation absolute deltas |Δ_j|...")
    judge_keys = ["rsinema", "pyiqa_musiq", "pyiqa_topiq_nr", "aesfix_pref_v1", "aesfix_pref_v2"]
    deltas_all: Dict[str, List[float]] = {j: [] for j in judge_keys}
    max_deltas_per_img: Dict[str, List[float]] = {j: [] for j in judge_keys}
    per_image_evals = []

    # Map scores back to scenes
    score_lookup = {}
    for (sc, var_name, _), idx in zip(all_items, range(len(all_items))):
        if sc not in score_lookup:
            score_lookup[sc] = {}
        score_lookup[sc][var_name] = {j: scores_dict[j][idx] for j in judge_keys}

    for item in manifest:
        sc = item["scene"]
        base_s = score_lookup[sc]["orig"]
        img_rec = {
            "scene": sc,
            "sha256": item["sha256"],
            "baseline_scores": base_s,
            "perturbations": {},
        }
        for var_name in ["jpeg_q90", "jpeg_q95", "resize_plus_2pct", "resize_minus_2pct"]:
            pert_s = score_lookup[sc][var_name]
            deltas = {j: round(abs(pert_s[j] - base_s[j]), 5) for j in judge_keys}
            for j in judge_keys:
                deltas_all[j].append(deltas[j])
            img_rec["perturbations"][var_name] = {
                "scores": pert_s,
                "deltas": deltas,
            }

        for j in judge_keys:
            j_max = max(img_rec["perturbations"][v]["deltas"][j] for v in img_rec["perturbations"])
            max_deltas_per_img[j].append(j_max)

        per_image_evals.append(img_rec)

    # 5. Compute Percentiles and Freeze
    print("\n" + "=" * 80)
    print("CALIBRATED NOISE FLOORS (ε_j) SUMMARY (N=50 DEAR Train Images, 200 Perturbations)")
    print("=" * 80)
    header = f"{'Judge Model':<18} | {'p95(|Δ|)':<12} | {'p95(max_img |Δ|)':<18} | {'mean(|Δ|)':<12} | {'max(|Δ|)':<12}"
    print(header)
    print("-" * len(header))

    frozen_epsilons = {}
    stats_summary = {}

    for j in judge_keys:
        arr_all = np.array(deltas_all[j])
        arr_max = np.array(max_deltas_per_img[j])

        p95_val = float(np.percentile(arr_all, 95))
        p95_max_val = float(np.percentile(arr_max, 95))
        mean_val = float(np.mean(arr_all))
        max_val = float(np.max(arr_all))

        frozen_eps = round(p95_val, 5)
        # Guarantee minimum floor of 0.0001
        frozen_eps = max(0.0001, frozen_eps)
        frozen_epsilons[j] = frozen_eps

        stats_summary[j] = {
            "p95_all_perturbations": round(p95_val, 5),
            "p95_per_image_max": round(p95_max_val, 5),
            "mean_abs_delta": round(mean_val, 5),
            "max_abs_delta": round(max_val, 5),
            "frozen_epsilon": frozen_eps,
        }
        print(f"{j:<18} | {p95_val:<12.5f} | {p95_max_val:<18.5f} | {mean_val:<12.5f} | {max_val:<12.5f}")

    print("-" * len(header))
    print(f"Frozen Epsilon Values: {frozen_epsilons}")

    # 6. Save config/judge_epsilon.json
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "metadata": {
            "title": "Empirical Procedure-Derived Judge Noise Floors (ε_j)",
            "task": "T0.d",
            "protocol": "compute_noise_floor (JPEG q90, JPEG q95, +2% resize, -2% resize)",
            "dataset": "DEAR Train Split (vsevolodpl/DEAR tiff16_c, 512px q92)",
            "n_images": 50,
            "seed": 42,
            "manifest_sha256": manifest_sha256,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.time() - t_start, 2),
        },
        "frozen_epsilons": frozen_epsilons,
        "detailed_statistics": stats_summary,
        "image_manifest": manifest,
        "per_image_evaluations": per_image_evals,
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    cfg_hasher = hashlib.sha256()
    with open(CONFIG_FILE, "rb") as f:
        cfg_hasher.update(f.read())
    cfg_sha = cfg_hasher.hexdigest()

    print(f"\nSuccessfully froze judge noise floors in {CONFIG_FILE}")
    print(f"File SHA-256: {cfg_sha}")


if __name__ == "__main__":
    main()
