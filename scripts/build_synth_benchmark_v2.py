"""Build Synthetic Benchmark v2 (S4).

Downloads 100 DEAR test scenes (tiff16_c expert C renderings),
resizes long edge to 512px (q92), and generates 500 degraded and oracle-fixed evaluation cases.
"""

import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from tqdm import tqdm

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.executor import TierAExecutor
DEAR_MARKUP_PATH = Path(r"D:\aesfix-data\dear\test_processed_markup.csv")
SYNTH_REFS_DIR = Path(r"D:\aesfix-data\synth_refs")
SYNTH_CASES_DIR = Path(r"D:\aesfix-data\synth_cases")

DEGRADATIONS = [
    ("underexposure", "adjust_exposure", {"ev_delta": -1.5}, "adjust_exposure", {"ev_delta": 1.5}),
    ("overexposure", "adjust_exposure", {"ev_delta": 1.0}, "adjust_exposure", {"ev_delta": -1.0}),
    ("low_contrast", "adjust_contrast", {"factor": 0.6}, "adjust_contrast", {"factor": 1.6667}),
    ("warm_cast", "adjust_white_balance", {"temperature": 15.0, "tint": 0.0}, "adjust_white_balance", {"temperature": -15.0, "tint": 0.0}),
    ("cool_cast", "adjust_white_balance", {"temperature": -15.0, "tint": 0.0}, "adjust_white_balance", {"temperature": 15.0, "tint": 0.0}),
]

def get_sampled_scenes(n: int = 100, seed: int = 42) -> List[str]:
    df = pd.read_csv(DEAR_MARKUP_PATH)
    scenes = sorted(df["name"].unique().tolist())
    rng = random.Random(seed)
    return sorted(rng.sample(scenes, n))

def download_and_preprocess_scene(scene: str) -> Tuple[str, bool, str]:
    out_ref_path = SYNTH_REFS_DIR / scene
    if out_ref_path.exists() and out_ref_path.stat().st_size > 0:
        return scene, True, ""
    try:
        raw_path = hf_hub_download(
            repo_id="vsevolodpl/DEAR",
            filename=f"images/tiff16_c/{scene}",
            repo_type="dataset",
        )
        with Image.open(raw_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > 512:
                scale = 512.0 / max(w, h)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                im = im.resize((nw, nh), Image.Resampling.LANCZOS)
            im.save(str(out_ref_path), format="JPEG", quality=92)
        return scene, True, ""
    except Exception as e:
        return scene, False, str(e)

def compute_lab_delta_e(img1: Image.Image, img2: Image.Image) -> float:
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.Resampling.BILINEAR)
    arr1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0
    lab1 = rgb2lab(arr1)
    lab2 = rgb2lab(arr2)
    # Mean CIEDE2000 color difference across pixels
    de = deltaE_ciede2000(lab1, lab2)
    return float(np.mean(de))

def build_benchmark() -> Dict[str, Any]:
    SYNTH_REFS_DIR.mkdir(parents=True, exist_ok=True)
    SYNTH_CASES_DIR.mkdir(parents=True, exist_ok=True)

    scenes = get_sampled_scenes(100, seed=42)
    print(f"Sampled 100 DEAR test scenes (seed 42). Downloading and preprocessing to 512px...")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(download_and_preprocess_scene, scenes), total=len(scenes), desc="Downloading Refs"))

    failed = [r for r in results if not r[1]]
    if failed:
        raise RuntimeError(f"Failed to download {len(failed)} scenes: {failed[:3]}")

    print(f"Successfully verified 100 reference images in {SYNTH_REFS_DIR}")

    executor = TierAExecutor()
    from aesfix.critic import CriticFix

    cases_meta = []
    print("Generating 500 synthetic cases (5 degradations x 100 scenes) and analytical oracles...")

    for scene in tqdm(scenes, desc="Generating Cases"):
        ref_path = SYNTH_REFS_DIR / scene
        ref_im = Image.open(ref_path).convert("RGB")

        for deg_name, deg_op, deg_params, orc_op, orc_params in DEGRADATIONS:
            deg_fix = CriticFix(
                op=deg_op,
                params=deg_params,
                expected_aspect="lighting" if "exposure" in deg_op or "contrast" in deg_op else "color",
                direction=1,
                nl_instruction=f"Apply {deg_name}",
            )
            deg_res = executor.execute_fixes(image_input=ref_im, fixes=[deg_fix])
            deg_path = SYNTH_CASES_DIR / f"{Path(scene).stem}_{deg_name}_deg.jpg"
            deg_res.image.save(str(deg_path), format="JPEG", quality=92)

            orc_fix = CriticFix(
                op=orc_op,
                params=orc_params,
                expected_aspect=deg_fix.expected_aspect,
                direction=1,
                nl_instruction=f"Apply oracle {deg_name} inverse",
            )
            orc_res = executor.execute_fixes(image_input=deg_res.image, fixes=[orc_fix])
            orc_path = SYNTH_CASES_DIR / f"{Path(scene).stem}_{deg_name}_orc.jpg"
            orc_res.image.save(str(orc_path), format="JPEG", quality=92)

            # Compute objective ground truth distance Delta E
            de_deg = compute_lab_delta_e(deg_res.image, ref_im)
            de_orc = compute_lab_delta_e(orc_res.image, ref_im)

            # Recovery categorization per pre-registered protocol
            tau = 0.5
            if de_orc < de_deg - tau:
                category = "recovered"
            elif de_orc > de_deg + tau:
                category = "worse"
            else:
                category = "unchanged"

            cases_meta.append({
                "scene": scene,
                "deg_name": deg_name,
                "deg_op": deg_op,
                "deg_params": deg_params,
                "orc_op": orc_op,
                "orc_params": orc_params,
                "ref_path": str(ref_path),
                "deg_path": str(deg_path),
                "orc_path": str(orc_path),
                "delta_e_deg": round(de_deg, 4),
                "delta_e_orc": round(de_orc, 4),
                "recovery_category": category,
            })

    out_csv = REPO_ROOT / "results" / "synth_v2_cases.csv"
    df_meta = pd.DataFrame(cases_meta)
    df_meta.to_csv(out_csv, index=False)
    print(f"Generated {len(df_meta)} cases. Saved manifest to {out_csv}")
    print("Oracle Recovery Summary:")
    print(df_meta["recovery_category"].value_counts())
    return df_meta.to_dict(orient="records")

if __name__ == "__main__":
    build_benchmark()
