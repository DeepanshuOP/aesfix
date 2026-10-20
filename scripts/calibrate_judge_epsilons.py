#!/usr/bin/env python3
"""Calibrate Judge Noise Floors (ε_j) on 50 DEAR Train Images (T0.d).

Implements both calibration protocols:
1. --method null (Pre-registered Null-Transform Protocol):
   - Horizontal flip (FLIP_LEFT_RIGHT)
   - 2px translation with reflect padding
   - Same-pipeline JPEG q92 re-encode
   Evaluates invariance noise floor across 50 DEAR train scenes (150 perturbations).
2. --method legacy (Procedure-Derived Perturbation Protocol):
   - JPEG q90 re-encode
   - JPEG q95 re-encode
   - Bilinear resize +2%
   - Bilinear resize -2%

Freezes ε_j = 95th percentile of |Δ_j| into config/judge_epsilon.json
along with image list, individual SHA-256 digests, and manifest hash.
"""
from __future__ import annotations

import argparse
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

# DEAR score standard deviations for pre-registered selection rule: ε_j < 0.5 * σ_DEAR
SIGMA_DEAR = {
    "pyiqa_topiq_nr": 0.07960,
    "aesfix_pref_v1": 0.98700,
    "rsinema": 0.27580,
    "pyiqa_musiq": 6.55910,
    "aesfix_pref_v2": 6.57680,
}

HAND_ENTERED_VALUES = {
    "pyiqa_topiq_nr": 0.01707,
    "aesfix_pref_v1": 0.44138,
    "rsinema": 0.08393,
    "pyiqa_musiq": 1.78040,
    "aesfix_pref_v2": 1.85784,
}


def sample_train_scenes(n: int = 50, seed: int = 42) -> List[str]:
    """Deterministically samples n DEAR train scenes strictly disjoint from test scenes."""
    if TRAIN_MARKUP.exists() and TEST_MARKUP.exists():
        df_tr = pd.read_csv(TRAIN_MARKUP)
        df_te = pd.read_csv(TEST_MARKUP)
        train_scenes = sorted(df_tr["name"].unique().tolist())
        test_scenes = set(df_te["name"].unique().tolist())
        disjoint_train = [s for s in train_scenes if s not in test_scenes]
        rng = random.Random(seed)
        sampled = sorted(rng.sample(disjoint_train, n))
    else:
        # Fallback to local files if markup csv is unavailable
        files = sorted([p.name for p in OUT_DIR.glob("*.jpeg")])
        if len(files) >= n:
            rng = random.Random(seed)
            sampled = sorted(rng.sample(files, n))
        else:
            sampled = files
    logger.info("Sampled %d train scenes using seed=%d", len(sampled), seed)
    return sampled


def download_and_preprocess(scene: str) -> Tuple[str, Path, str]:
    """Downloads tiff16_c from HF if needed, resizes long edge to 512px (q92), and computes sha256."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / scene
    if not out_path.exists():
        out_path = OUT_DIR / f"{Path(scene).stem}.jpeg"

    if not out_path.exists() or out_path.stat().st_size == 0:
        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
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


def create_and_save_null_perturbations(scene: str, orig_path: Path) -> List[Tuple[str, str, Path]]:
    """Generates the 3 pre-registered null-transform perturbations:
    1. Horizontal flip
    2. 2px translation with reflect padding
    3. JPEG q92 re-encode
    """
    PERTS_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(orig_path) as im:
        orig = im.convert("RGB")

    items = []
    # 0. Original baseline
    orig_save = PERTS_DIR / f"{scene}__orig.jpg"
    if not orig_save.exists():
        orig.save(str(orig_save), format="JPEG", quality=95)
    items.append((scene, "orig", orig_save))

    # 1. Horizontal flip
    hflip_p = PERTS_DIR / f"{scene}__null_hflip.jpg"
    if not hflip_p.exists():
        im_hflip = orig.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im_hflip.save(str(hflip_p), format="JPEG", quality=95)
    items.append((scene, "null_hflip", hflip_p))

    # 2. 2px reflect pad translation
    trans_p = PERTS_DIR / f"{scene}__null_trans_2px.jpg"
    if not trans_p.exists():
        arr = np.array(orig)
        h, w, c = arr.shape
        padded = np.pad(arr, ((2, 0), (2, 0), (0, 0)), mode="reflect")
        im_trans = Image.fromarray(padded[:h, :w, :])
        im_trans.save(str(trans_p), format="JPEG", quality=95)
    items.append((scene, "null_trans_2px", trans_p))

    # 3. JPEG q92 re-encode
    q92_p = PERTS_DIR / f"{scene}__null_jpeg_q92.jpg"
    if not q92_p.exists():
        orig.save(str(q92_p), format="JPEG", quality=92)
    items.append((scene, "null_jpeg_q92", q92_p))

    return items


def create_and_save_legacy_perturbations(scene: str, orig_path: Path) -> List[Tuple[str, str, Path]]:
    """Generates the 4 legacy perturbations (JPEG q90, q95, +/-2% resize)."""
    PERTS_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(orig_path) as im:
        orig = im.convert("RGB")

    items = []
    orig_save = PERTS_DIR / f"{scene}__orig.jpg"
    if not orig_save.exists():
        orig.save(str(orig_save), format="JPEG", quality=95)
    items.append((scene, "orig", orig_save))

    q90_p = PERTS_DIR / f"{scene}__jpeg_q90.jpg"
    if not q90_p.exists():
        orig.save(str(q90_p), format="JPEG", quality=90)
    items.append((scene, "jpeg_q90", q90_p))

    q95_p = PERTS_DIR / f"{scene}__jpeg_q95.jpg"
    if not q95_p.exists():
        orig.save(str(q95_p), format="JPEG", quality=95)
    items.append((scene, "jpeg_q95", q95_p))

    up_p = PERTS_DIR / f"{scene}__resize_plus_2pct.jpg"
    if not up_p.exists():
        w, h = orig.size
        img_up = orig.resize((max(1, int(round(w * 1.02))), max(1, int(round(h * 1.02)))), Image.Resampling.BILINEAR)
        img_up.save(str(up_p), format="JPEG", quality=95)
    items.append((scene, "resize_plus_2pct", up_p))

    down_p = PERTS_DIR / f"{scene}__resize_minus_2pct.jpg"
    if not down_p.exists():
        w, h = orig.size
        img_down = orig.resize((max(1, int(round(w * 0.98))), max(1, int(round(h * 0.98)))), Image.Resampling.BILINEAR)
        img_down.save(str(down_p), format="JPEG", quality=95)
    items.append((scene, "resize_minus_2pct", down_p))

    return items


def evaluate_rsinema_all(image_paths: List[Path], batch_size: int = 16) -> List[float]:
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
    """Batched evaluation for AesFix-Pref v2 using float32 on CPU."""
    v2_p = REPO_ROOT / "results" / "aesfix_pref_v2_head.pt"
    if not v2_p.exists():
        v2_p = Path(r"D:\aesfix-data\kaggle_runs\dear_full\aesfix_pref_v2_head.pt")
    assert v2_p.exists(), f"v2 head checkpoint not found at {v2_p}"

    from transformers import AutoImageProcessor, SiglipVisionModel

    # Load shrunk vision model or base
    model_dir = Path("D:/aesfix-cache/models/siglip_so400m_vision_fp16")
    if model_dir.exists():
        proc = AutoImageProcessor.from_pretrained(model_dir)
        backbone = SiglipVisionModel.from_pretrained(model_dir).float().eval()
    else:
        bb_name = "google/siglip-so400m-patch14-384"
        proc = AutoImageProcessor.from_pretrained(bb_name)
        backbone = SiglipVisionModel.from_pretrained(bb_name).float().eval()

    ckpt_v2 = torch.load(v2_p, map_location="cpu", weights_only=False)
    feat_dim = int(ckpt_v2.get("feature_dim", 1152))
    state_dict = ckpt_v2.get("state_dict", ckpt_v2)
    clean_sd = {k.replace("net.", "").replace("fc.", ""): v.float() for k, v in state_dict.items()}

    head = torch.nn.Sequential(
        torch.nn.Linear(feat_dim, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 1, bias=False),
    ).eval()
    head.load_state_dict(clean_sd)

    scores: List[float] = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Eval aesfix_pref_v2"):
        batch_paths = image_paths[i : i + batch_size]
        pil_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = proc(images=pil_imgs, return_tensors="pt")["pixel_values"].float()
        with torch.no_grad():
            out = backbone(pixel_values=inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out[1]
            feat = feat / (torch.norm(feat, dim=-1, keepdim=True) + 1e-7)
            raw_scores = head(feat).squeeze(-1)
            if raw_scores.ndim == 0:
                scores.append(round(float(raw_scores.item()), 4))
            else:
                scores.extend([round(float(v), 4) for v in raw_scores.tolist()])
    return scores


def evaluate_pyiqa_all(metric_name: str, image_paths: List[Path]) -> List[float]:
    import pyiqa
    metric = pyiqa.create_metric(metric_name, device="cpu")
    scores = []
    for p in tqdm(image_paths, desc=f"Eval pyiqa_{metric_name}"):
        with torch.no_grad():
            val = float(metric(str(p)).squeeze().item())
        scores.append(round(val, 4))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Judge Noise Floors")
    parser.add_argument("--method", choices=["null", "legacy"], default="null",
                        help="Calibration method: 'null' for pre-registered null-transform, 'legacy' for procedure perturbations.")
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== Calibration of Judge Noise Floors (Method: {args.method.upper()}) ===")

    # 1. Sample and verify 50 DEAR train scenes
    sampled_scenes = sample_train_scenes(50, seed=42)
    print(f"Verifying 50 train reference images in {OUT_DIR}...")
    manifest = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(download_and_preprocess, sampled_scenes))

    for scene, p, sha in sorted(results):
        manifest.append({"scene": scene, "path": str(p), "sha256": sha})

    manifest_bytes = json.dumps([m["sha256"] for m in manifest], sort_keys=True).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    print(f"Verified 50 train images. Collective Manifest SHA256: {manifest_sha256}")

    # 2. Generate and save perturbations
    all_items: List[Tuple[str, str, Path]] = []
    if args.method == "null":
        pert_names = ["null_hflip", "null_trans_2px", "null_jpeg_q92"]
        protocol_title = "pre_registered_null_transform (hflip, 2px reflect translation, JPEG q92)"
        print(f"\nGenerating 3 null perturbations for each image (total {len(manifest)*4} files)...")
        for item in manifest:
            s_items = create_and_save_null_perturbations(item["scene"], Path(item["path"]))
            all_items.extend(s_items)
    else:
        pert_names = ["jpeg_q90", "jpeg_q95", "resize_plus_2pct", "resize_minus_2pct"]
        protocol_title = "procedure_derived_perturbation (JPEG q90, JPEG q95, +2% resize, -2% resize)"
        print(f"\nGenerating 4 legacy perturbations for each image (total {len(manifest)*5} files)...")
        for item in manifest:
            s_items = create_and_save_legacy_perturbations(item["scene"], Path(item["path"]))
            all_items.extend(s_items)

    print(f"Prepared {len(all_items)} images in {PERTS_DIR}")
    all_paths = [it[2] for it in all_items]

    # 3. Model-by-model batched evaluation
    print(f"\nRunning model-by-model evaluation across all {len(all_paths)} images...")
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

    # 4. Compute |Δ_j|
    print("\nComputing perturbation absolute deltas |Δ_j|...")
    judge_keys = ["pyiqa_topiq_nr", "aesfix_pref_v1", "rsinema", "pyiqa_musiq", "aesfix_pref_v2"]
    deltas_all: Dict[str, List[float]] = {j: [] for j in judge_keys}
    max_deltas_per_img: Dict[str, List[float]] = {j: [] for j in judge_keys}
    per_image_evals = []

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
        for var_name in pert_names:
            pert_s = score_lookup[sc][var_name]
            deltas = {j: round(abs(pert_s[j] - base_s[j]), 5) for j in judge_keys}
            for j in judge_keys:
                deltas_all[j].append(deltas[j])
            img_rec["perturbations"][var_name] = {
                "scores": pert_s,
                "deltas": deltas,
            }

        for j in judge_keys:
            j_max = max(img_rec["perturbations"][v]["deltas"][j] for v in pert_names)
            max_deltas_per_img[j].append(j_max)

        per_image_evals.append(img_rec)

    # 5. Summary & Comparison Table
    print("\n" + "=" * 95)
    print(f"CALIBRATED NOISE FLOORS (ε_j) SUMMARY (Method: {args.method.upper()}, N=50 Scenes, {len(deltas_all['rsinema'])} Perturbations)")
    print("=" * 95)
    header = f"{'Judge Model':<18} | {'Measured ε':<12} | {'Hand-Entered':<12} | {'Rel Diff %':<12} | {'0.5*σ_DEAR':<12} | {'Rule Check':<10}"
    print(header)
    print("-" * len(header))

    frozen_epsilons = {}
    stats_summary = {}
    discrepancies = []

    for j in judge_keys:
        arr_all = np.array(deltas_all[j])
        p95_val = float(np.percentile(arr_all, 95))
        mean_val = float(np.mean(arr_all))
        max_val = float(np.max(arr_all))

        frozen_eps = max(0.0001, round(p95_val, 5))
        frozen_epsilons[j] = frozen_eps

        hand_val = HAND_ENTERED_VALUES.get(j, 0.0)
        rel_diff_pct = (abs(frozen_eps - hand_val) / hand_val * 100.0) if hand_val > 0 else 0.0
        diff_flag = "DIFF > 5%" if rel_diff_pct > 5.0 else "MATCH (<5%)"

        sig_thresh = 0.5 * SIGMA_DEAR.get(j, 1.0)
        passes_rule = frozen_eps < sig_thresh
        rule_str = "PASS" if passes_rule else "FAIL"

        stats_summary[j] = {
            "p95_all_perturbations": frozen_eps,
            "mean_abs_delta": round(mean_val, 5),
            "max_abs_delta": round(max_val, 5),
            "frozen_epsilon": frozen_eps,
            "hand_entered_epsilon": hand_val,
            "rel_diff_pct": round(rel_diff_pct, 2),
            "selection_rule_0_5_sigma": round(sig_thresh, 5),
            "selection_rule_passed": passes_rule,
        }

        print(f"{j:<18} | {frozen_eps:<12.5f} | {hand_val:<12.5f} | {rel_diff_pct:<11.2f}% | {sig_thresh:<12.5f} | {rule_str:<10}")

        if rel_diff_pct > 5.0:
            discrepancies.append((j, frozen_eps, hand_val, rel_diff_pct))

    print("-" * len(header))

    if discrepancies:
        print("\nWARNING: The following values differ by more than 5% from hand-entered values:")
        for j, meas, hand, pct in discrepancies:
            print(f"  - {j}: Measured {meas:.5f} vs Hand-entered {hand:.5f} (Diff: {pct:.2f}%) -> SCRIPT VALUE WINS")
    else:
        print("\nCONFIRMATION: All measured epsilon values match hand-entered values within 5% tolerance.")

    # 6. Save config/judge_epsilon.json
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "metadata": {
            "title": f"Empirical Judge Noise Floors (ε_j) - {args.method.upper()}",
            "task": "T0.d",
            "method": args.method,
            "protocol": protocol_title,
            "dataset": "DEAR Train Split (50 scenes, 512px q92)",
            "n_scenes": 50,
            "n_perturbations_per_scene": len(pert_names),
            "n_total_perturbations": len(deltas_all['rsinema']),
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

    print(f"\nSuccessfully wrote calibrated noise floors to {CONFIG_FILE}")
    print(f"Config SHA-256: {cfg_sha}")


if __name__ == "__main__":
    main()
