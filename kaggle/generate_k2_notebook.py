import json
import base64
from pathlib import Path

def create_notebook(cells, output_path: str, accelerator: str = "GPU"):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "accelerator": accelerator
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

def md(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def code(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

# Read base64 of aesfix_pref_head.pt to embed in K2b
v1_pt_path = Path("results/aesfix_pref_head.pt")
if v1_pt_path.exists():
    with open(v1_pt_path, "rb") as f:
        v1_head_b64 = base64.b64encode(f.read()).decode("ascii")
else:
    v1_head_b64 = ""

# ==============================================================================
# K2a: aesfix-dear-512 (CPU only, accelerator NONE, internet ON)
# ==============================================================================
k2a_cells = [
    md("""# AesFix K2a: DEAR Dataset Downloader & 512px Preprocessor
**Notebook**: `aesfix-dear-512`
**Hardware**: CPU only (accelerator NONE), Internet ON.
**Role**:
1. Retrieve `HF_TOKEN` from Kaggle Secrets (or fallback to anonymous mode).
2. Download train and test markups from `vsevolodpl/DEAR`.
3. Stream all 29,886 images (4,981 scenes x 6 styles) with 6 worker threads (2 in anonymous mode).
4. Downscale long side to 512px, JPEG q92.
5. Exponential backoff with jitter on HTTP 429, honoring Retry-After (up to 8 retries).
6. Delete downloaded raw file (resolving symlinks) and check disk space every 500 files.
7. Resumable across executions; writes `manifest.csv` and verifies <1% missing threshold."""),

    code("""# Cell 1: Environment & Secrets Setup
import os
import sys
import time
import shutil
import random
import hashlib
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

start_time = time.time()

# Verify HF_TOKEN from Kaggle Secrets only
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    val = user_secrets.get_secret("HF_TOKEN")
    if val and str(val).strip():
        hf_token = str(val).strip()
        print("[Secrets] HF_TOKEN retrieved and verified successfully from Kaggle Secrets.")
except Exception as e:
    pass

if hf_token:
    os.environ["HF_TOKEN"] = str(hf_token).strip()
    num_workers = 6
    print("[Auth] Authenticated mode enabled with 6 workers.")
else:
    num_workers = 2
    print("[Auth] anonymous mode")

print(f"Python: {sys.version}")
"""),

    code("""# Cell 2: Dependencies Installation
!pip install --no-cache-dir huggingface_hub requests pillow pandas tqdm
"""),

    code("""# Cell 3: Fetch Markups & Define Dataset Partitions
from huggingface_hub import hf_hub_download

OUT_BASE = Path("/kaggle/working")
IMG_DIR = OUT_BASE / "images_512"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Resume source if previously mounted
INPUT_PREV = Path("/kaggle/input/aesfix-dear-512")

print("Downloading official train and test markups from vsevolodpl/DEAR...")
hf_token_val = os.environ.get("HF_TOKEN", None)
train_raw_path = hf_hub_download(
    repo_id="vsevolodpl/DEAR",
    filename="train_processed_markup.csv",
    repo_type="dataset",
    token=hf_token_val
)
test_raw_path = hf_hub_download(
    repo_id="vsevolodpl/DEAR",
    filename="test_processed_markup.csv",
    repo_type="dataset",
    token=hf_token_val
)

# Copy markups to working directory for K2b consumption
shutil.copy2(train_raw_path, OUT_BASE / "train_processed_markup.csv")
shutil.copy2(test_raw_path, OUT_BASE / "test_processed_markup.csv")

df_train = pd.read_csv(OUT_BASE / "train_processed_markup.csv")
df_test = pd.read_csv(OUT_BASE / "test_processed_markup.csv")

all_scenes = sorted(list(set(df_train['name'].unique()).union(set(df_test['name'].unique()))))
styles = ['original', 'tiff16_a', 'tiff16_b', 'tiff16_c', 'tiff16_d', 'tiff16_e']

print(f"Total Unique Scenes: {len(all_scenes)} (Train: {df_train['name'].nunique()}, Test: {df_test['name'].nunique()})")
print(f"Total Target Images: {len(all_scenes) * len(styles)} (6 styles per scene)")
"""),

    code("""# Cell 4: Multi-Threaded Streaming, Downscaling & Disk Space Monitoring
all_tasks = [(style, scene) for scene in all_scenes for style in styles]
print(f"Starting pipeline for {len(all_tasks)} images with {num_workers} worker threads...")

# Pre-create style directories
for s in styles:
    (IMG_DIR / s).mkdir(parents=True, exist_ok=True)

counter_lock = threading.Lock()
processed_counter = 0

def check_disk(count, total):
    if count % 500 == 0 or count == total:
        total_b, used_b, free_b = shutil.disk_usage("/kaggle/working")
        free_gb = free_b / (1024 ** 3)
        print(f"[{count}/{total}] Free disk space in /kaggle/working: {free_gb:.2f} GB")
        if free_gb < 1.0:
            raise RuntimeError(f"CRITICAL: Free disk space dropped below 1.0 GB ({free_gb:.2f} GB free). Aborting.")

def process_image(task):
    global processed_counter
    style, scene = task
    out_file = IMG_DIR / style / scene
    
    # 1. Resumable check: Already in working dir
    if out_file.exists() and out_file.stat().st_size > 0:
        with open(out_file, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        with counter_lock:
            processed_counter += 1
            check_disk(processed_counter, len(all_tasks))
        return {"style": style, "scene": scene, "ok": True, "error": "", "bytes": out_file.stat().st_size, "sha256": h}

    # 2. Resumable check: Available in previously mounted input
    prev_file = INPUT_PREV / "images_512" / style / scene
    if prev_file.exists() and prev_file.stat().st_size > 0:
        shutil.copy2(prev_file, out_file)
        with open(out_file, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        with counter_lock:
            processed_counter += 1
            check_disk(processed_counter, len(all_tasks))
        return {"style": style, "scene": scene, "ok": True, "error": "", "bytes": out_file.stat().st_size, "sha256": h}

    # 3. Download from HF Hub with retry & backoff
    hf_rel_path = f"images/{style}/{scene}"
    tmp_worker_dir = Path(f"/kaggle/temp/hf_raw_th_{threading.get_ident()}")
    tmp_worker_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = None
    last_err = ""
    for attempt in range(8):
        try:
            raw_path = hf_hub_download(
                repo_id="vsevolodpl/DEAR",
                filename=hf_rel_path,
                repo_type="dataset",
                local_dir=str(tmp_worker_dir),
                token=os.environ.get("HF_TOKEN", None)
            )
            break
        except Exception as e:
            last_err = str(e)
            is_429 = "429" in last_err or (hasattr(e, "response") and getattr(e.response, "status_code", None) == 429)
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                ra = e.response.headers.get("Retry-After")
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        pass
            if attempt < 7:
                if retry_after is not None:
                    sleep_s = retry_after + random.uniform(0.5, 2.0)
                elif is_429:
                    sleep_s = (2 ** attempt) * 1.5 + random.uniform(0.5, 2.0)
                else:
                    sleep_s = (2 ** attempt) * 0.5 + random.uniform(0.1, 1.0)
                time.sleep(sleep_s)
    
    if raw_path is None or not Path(raw_path).exists():
        with counter_lock:
            processed_counter += 1
            check_disk(processed_counter, len(all_tasks))
        return {"style": style, "scene": scene, "ok": False, "error": last_err[:200], "bytes": 0, "sha256": ""}

    try:
        # 4. Downscale long side to 512px, JPEG q92
        with Image.open(raw_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if max(w, h) > 512:
                scale = 512.0 / max(w, h)
                nw = max(1, int(round(w * scale)))
                nh = max(1, int(round(h * scale)))
                im = im.resize((nw, nh), Image.Resampling.LANCZOS)
            im.save(str(out_file), format="JPEG", quality=92)
            
        # 5. Delete raw downloaded file (resolving symlinks)
        real_p = Path(raw_path).resolve()
        if real_p.exists():
            try:
                real_p.unlink()
            except Exception:
                pass
        if Path(raw_path).exists():
            try:
                Path(raw_path).unlink()
            except Exception:
                pass
                
        # Clean local thread cache
        hf_cache_dir = Path(os.path.expanduser("~/.cache/huggingface"))
        if hf_cache_dir.exists():
            shutil.rmtree(hf_cache_dir, ignore_errors=True)

        with open(out_file, "rb") as f:
            f_bytes = out_file.stat().st_size
            sha = hashlib.sha256(f.read()).hexdigest()

        with counter_lock:
            processed_counter += 1
            check_disk(processed_counter, len(all_tasks))

        return {"style": style, "scene": scene, "ok": True, "error": "", "bytes": f_bytes, "sha256": sha}
    except Exception as e:
        with counter_lock:
            processed_counter += 1
            check_disk(processed_counter, len(all_tasks))
        return {"style": style, "scene": scene, "ok": False, "error": str(e)[:200], "bytes": 0, "sha256": ""}

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    manifest_records = list(tqdm(executor.map(process_image, all_tasks), total=len(all_tasks), desc="Processing DEAR 512px"))
"""),

    code("""# Cell 5: Generate manifest.csv & Fail Loudly if >1% Missing
df_manifest = pd.DataFrame(manifest_records)
manifest_path = OUT_BASE / "manifest.csv"
df_manifest.to_csv(manifest_path, index=False)

n_total = len(df_manifest)
n_ok = int(df_manifest['ok'].sum())
n_missing = n_total - n_ok
pct_missing = (n_missing / n_total) * 100.0

total_b, used_b, free_b = shutil.disk_usage("/kaggle/working")
runtime_sec = time.time() - start_time

print("="*70)
print(f"DEAR 512 MANIFEST SUMMARY")
print("="*70)
print(f"Total Images Expected : {n_total}")
print(f"Successfully Processed: {n_ok}")
print(f"Missing / Failed      : {n_missing} ({pct_missing:.2f}%)")
print(f"Manifest Path         : {manifest_path} ({os.path.getsize(manifest_path)} bytes)")
print(f"Working Directory Size: {sum(f.stat().st_size for f in IMG_DIR.glob('*/*')) / (1024**3):.2f} GB")
print(f"Remaining Free Disk   : {free_b / (1024**3):.2f} GB")
print(f"Total Runtime         : {runtime_sec:.1f}s ({runtime_sec/3600.0:.2f} hours)")
print("="*70)

if pct_missing > 1.0:
    raise RuntimeError(f"FATAL: Missing files {pct_missing:.2f}% exceeds 1.0% limit ({n_missing}/{n_total} files failed)!")
print("[SUCCESS] K2a dataset preprocessor finished within integrity tolerances.")
""")
]

# ==============================================================================
# K2b: aesfix-k2-dear-judges (GPU T4, single GPU)
# ==============================================================================
k2b_cells = [
    md("""# AesFix K2b: DEAR Multi-Judge Benchmark & AesFix-Pref v2 Model Selection
**Notebook**: `aesfix-k2-dear-judges`
**Hardware**: Single GPU (Tesla T4).
**Input**: Attached from `deepanshuop12/aesfix-dear-512` (zero image downloads).
**Role**:
1. Random scene-disjoint validation split from TRAIN (seed 0, 15%).
2. Multi-backbone embedding extraction (CLIP ViT-B/32, ViT-L/14, SigLIP so400m, DINOv2-base).
3. Train and select across 8 candidate Bradley-Terry heads on validation split.
4. Score ALL judges on the test split (1,283 scenes), INCLUDING `AesFix-Pref v1`.
5. Export:
   - `test_pair_scores.csv`
   - `per_image_scores.csv`
   - `dear_embeddings.npz`
   - `k2_full_results.json`
   - `aesfix_pref_v2_head.pt` (CPU tensors plus metadata)
   - `gpu_hours`: GPU notebook's runtime only."""),

    code(f"""# Cell 1: Environment & Embedded Checkpoint Setup
import os
import sys
import time
import json
import uuid
import base64
import random
import io
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Executing K2b on: {{device}} ({{gpu_name}})")
assert torch.cuda.is_available(), "FATAL: Single GPU T4 accelerator required for K2b!"

# Locate K2a dataset
INPUT_CANDIDATES = [
    Path("/kaggle/input/aesfix-dear-512"),
    Path("/kaggle/input/aesfix-dear-512/images_512"),
    Path("/kaggle/input/aesfix-dear-512/working/images_512")
]
IMG_DIR = None
for c in INPUT_CANDIDATES:
    if (c / "original").exists() or (c / "images_512" / "original").exists():
        IMG_DIR = c if (c / "original").exists() else c / "images_512"
        break

if IMG_DIR is None:
    # Try recursive search
    found = list(Path("/kaggle/input").glob("**/images_512"))
    if found:
        IMG_DIR = found[0]

assert IMG_DIR is not None and IMG_DIR.exists(), f"FATAL: K2a images_512 not found in /kaggle/input! Checked {{INPUT_CANDIDATES}}"
print(f"[Dataset] Verified K2a 512px dataset root at: {{IMG_DIR}}")

# Base64 embedded AesFix-Pref v1 head
V1_HEAD_B64 = "{v1_head_b64}"
v1_ckpt = None
if V1_HEAD_B64:
    v1_bytes = base64.b64decode(V1_HEAD_B64)
    v1_ckpt = torch.load(io.BytesIO(v1_bytes), map_location="cpu")
    print(f"[Model v1] Loaded embedded AesFix-Pref v1 checkpoint: {{v1_ckpt.get('encoder', 'clip-b32')}} (dim: {{v1_ckpt.get('feature_dim', 512)}})")
"""),

    code("""# Cell 2: Dependencies Installation
!pip install --no-cache-dir pyiqa==0.1.16 aesthetic-predictor-v2-5 transformers huggingface_hub scipy scikit-learn pandas tqdm timm
"""),

    code("""# Cell 3: Markups & Random Scene-Disjoint Validation Split (Seed 0, 15% of Train)
# Find markups from K2a input or download if needed
train_markup_path = Path("/kaggle/input/aesfix-dear-512/train_processed_markup.csv")
test_markup_path = Path("/kaggle/input/aesfix-dear-512/test_processed_markup.csv")

if not train_markup_path.exists():
    from huggingface_hub import hf_hub_download
    train_markup_path = Path(hf_hub_download(repo_id="vsevolodpl/DEAR", filename="train_processed_markup.csv", repo_type="dataset"))
    test_markup_path = Path(hf_hub_download(repo_id="vsevolodpl/DEAR", filename="test_processed_markup.csv", repo_type="dataset"))

df_train_raw = pd.read_csv(train_markup_path)
df_test = pd.read_csv(test_markup_path)

# Random scene-disjoint validation split from TRAIN only (seed 0, 15%)
all_train_scenes = sorted(df_train_raw['name'].unique().tolist())
rng = random.Random(0)
train_scenes_shuffled = list(all_train_scenes)
rng.shuffle(train_scenes_shuffled)

n_val = int(round(len(train_scenes_shuffled) * 0.15))
val_scenes = set(train_scenes_shuffled[:n_val])
train_scenes = set(train_scenes_shuffled[n_val:])
test_scenes = set(df_test['name'].unique().tolist())

# Verification of strict scene disjointness
assert len(train_scenes.intersection(val_scenes)) == 0, "Train and Val must be strictly scene-disjoint!"
assert len(train_scenes.intersection(test_scenes)) == 0, "Train and Test must be strictly scene-disjoint!"
assert len(val_scenes.intersection(test_scenes)) == 0, "Val and Test must be strictly scene-disjoint!"

df_train = df_train_raw[df_train_raw['name'].isin(train_scenes)].copy().reset_index(drop=True)
df_val = df_train_raw[df_train_raw['name'].isin(val_scenes)].copy().reset_index(drop=True)

print(f"Dataset Scene Partitioning (Seed 0, 15% Val):")
print(f"  - Train Split: {len(train_scenes)} scenes, {len(df_train)} pairs")
print(f"  - Val Split:   {len(val_scenes)} scenes, {len(df_val)} pairs (for model selection)")
print(f"  - Test Split:  {len(test_scenes)} scenes, {len(df_test)} pairs (evaluated ONCE)")
"""),

    code("""# Cell 4: Multi-Encoder Embedding Extraction on GPU
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

print("Initializing 4 frozen vision encoders on GPU...")
clip_b32_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).half().eval()
clip_b32_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

clip_l14_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).half().eval()
clip_l14_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

siglip_proc = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device).half().eval()

dinov2_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).half().eval()

# Gather all unique images available
unique_imgs = sorted(list(IMG_DIR.glob("*/*.jpeg")))
print(f"Extracting embeddings and luma for {len(unique_imgs)} images...")

b32_embs = {}
l14_embs = {}
siglip_embs = {}
dinov2_embs = {}
luma_cache = {}

BATCH_SIZE = 32
for i in tqdm(range(0, len(unique_imgs), BATCH_SIZE), desc="Feature extraction"):
    batch_paths = unique_imgs[i:i+BATCH_SIZE]
    batch_pils = [Image.open(p).convert("RGB") for p in batch_paths]
    
    with torch.no_grad():
        for idx, p in enumerate(batch_paths):
            arr = np.array(batch_pils[idx], dtype=np.float32) / 255.0
            luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
            luma_cache[p.name] = float(np.median(luma))

        # CLIP ViT-B/32
        in_b32 = clip_b32_proc(images=batch_pils, return_tensors="pt")["pixel_values"].to(device).half()
        v_b32 = clip_b32_model.vision_model(pixel_values=in_b32)[1]
        p_b32 = clip_b32_model.visual_projection(v_b32)
        p_b32 = p_b32 / p_b32.norm(p=2, dim=-1, keepdim=True)
        for idx, p in enumerate(batch_paths):
            b32_embs[p.as_posix()] = p_b32[idx].cpu().numpy().astype(np.float16)

        # CLIP ViT-L/14
        in_l14 = clip_l14_proc(images=batch_pils, return_tensors="pt")["pixel_values"].to(device).half()
        v_l14 = clip_l14_model.vision_model(pixel_values=in_l14)[1]
        p_l14 = clip_l14_model.visual_projection(v_l14)
        p_l14 = p_l14 / p_l14.norm(p=2, dim=-1, keepdim=True)
        for idx, p in enumerate(batch_paths):
            l14_embs[p.as_posix()] = p_l14[idx].cpu().numpy().astype(np.float16)

        # SigLIP so400m
        in_siglip = siglip_proc(images=batch_pils, return_tensors="pt")["pixel_values"].to(device).half()
        out_siglip = siglip_model.vision_model(pixel_values=in_siglip)[1]
        out_siglip = out_siglip / out_siglip.norm(p=2, dim=-1, keepdim=True)
        for idx, p in enumerate(batch_paths):
            siglip_embs[p.as_posix()] = out_siglip[idx].cpu().numpy().astype(np.float16)

        # DINOv2-base
        in_dino = dinov2_proc(images=batch_pils, return_tensors="pt")["pixel_values"].to(device).half()
        out_dino = dinov2_model(pixel_values=in_dino).last_hidden_state[:, 0, :]
        out_dino = out_dino / out_dino.norm(p=2, dim=-1, keepdim=True)
        for idx, p in enumerate(batch_paths):
            dinov2_embs[p.as_posix()] = out_dino[idx].cpu().numpy().astype(np.float16)

del clip_l14_model, siglip_model, dinov2_model
torch.cuda.empty_cache()

# Save dear_embeddings.npz
npz_path = Path("/kaggle/working/dear_embeddings.npz")
print(f"Saving dear_embeddings.npz to {npz_path}...")
img_keys = np.array(list(b32_embs.keys()))
np.savez_compressed(
    npz_path,
    image_paths=img_keys,
    clip_b32=np.array([b32_embs[k] for k in img_keys]),
    clip_l14=np.array([l14_embs[k] for k in img_keys]),
    siglip=np.array([siglip_embs[k] for k in img_keys]),
    dinov2=np.array([dinov2_embs[k] for k in img_keys])
)
print("dear_embeddings.npz size:", os.path.getsize(npz_path) / (1024*1024), "MB")
"""),

    code("""# Cell 5: Train and Select Across 8 Candidate Heads on Validation Split
encoder_dicts = {
    "clip_b32": (b32_embs, 512),
    "clip_l14": (l14_embs, 768),
    "siglip_so400m": (siglip_embs, 1152),
    "dinov2_base": (dinov2_embs, 768)
}

def build_split_pairs(df_split, embs_dict):
    X, y = [], []
    for _, row in df_split.iterrows():
        scene = row['name']
        left_style, right_style = row['left'], row['right']
        mos = float(row['mos'])
        if mos == 0.5:
            continue
        p_left = (IMG_DIR / left_style / scene).as_posix()
        p_right = (IMG_DIR / right_style / scene).as_posix()
        if p_left in embs_dict and p_right in embs_dict:
            diff = embs_dict[p_left].astype(np.float32) - embs_dict[p_right].astype(np.float32)
            label = 1.0 if mos > 0.5 else 0.0
            X.append(diff)
            y.append(label)
            # Antisymmetric augmentation
            X.append(-diff)
            y.append(1.0 - label)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

class LinearBT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1, bias=False)
    def forward(self, x):
        return self.fc(x)

class MLPBT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1, bias=False)
        )
    def forward(self, x):
        return self.net(x)

validation_table = []
candidate_models = {}

print("Training 8 candidate Bradley-Terry heads on Train split...")
for enc_name, (embs, dim) in encoder_dicts.items():
    X_tr, y_tr = build_split_pairs(df_train, embs)
    X_va, y_va = build_split_pairs(df_val, embs)
    
    loader_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    
    for arch in ["linear", "mlp"]:
        cand_id = f"{enc_name}_{arch}"
        head = LinearBT(dim).to(device) if arch == "linear" else MLPBT(dim).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        
        head.train()
        for epoch in range(12):
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = head(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        
        head.eval()
        with torch.no_grad():
            va_out = head(X_va.to(device))
            va_loss = criterion(va_out, y_va.to(device)).item()
            preds = (va_out.cpu().numpy() > 0).astype(float)
            val_acc = float(np.mean(preds == y_va.numpy()))
        
        validation_table.append({
            "candidate": cand_id,
            "encoder": enc_name,
            "architecture": arch,
            "feature_dim": dim,
            "val_accuracy": round(val_acc, 4),
            "val_loss": round(va_loss, 4)
        })
        candidate_models[cand_id] = head
        print(f"Candidate {cand_id:<22} | Val Accuracy: {val_acc:.4f} | Val Loss: {va_loss:.4f}")

df_val_results = pd.DataFrame(validation_table).sort_values(by="val_accuracy", ascending=False).reset_index(drop=True)
best_row = df_val_results.iloc[0]
best_candidate_id = best_row["candidate"]
best_encoder_name = best_row["encoder"]
best_head = candidate_models[best_candidate_id]
print(f"\\n[SELECTION] Chosen head: {best_candidate_id} (Val Acc: {best_row['val_accuracy']:.4f})")
"""),

    code("""# Cell 6: Score ALL Judges on Full Test Split (Including AesFix-Pref v1)
import pyiqa
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

print("Initializing all baseline judges...")
judges = {
    'pyiqa_musiq': pyiqa.create_metric('musiq', device=device, as_loss=False),
    'pyiqa_topiq_nr': pyiqa.create_metric('topiq_nr', device=device, as_loss=False),
    'pyiqa_liqe': pyiqa.create_metric('liqe', device=device, as_loss=False),
    'pyiqa_clip_iqa': pyiqa.create_metric('clipiqa', device=device, as_loss=False)
}

ap_model, ap_proc = convert_v2_5_from_siglip()
ap_model = ap_model.to(device).half().eval()

# RSinema
from huggingface_hub import hf_hub_download
rsinema_path = hf_hub_download("rsinema/aesthetic-scorer", "model.pt")
ckpt_rs = torch.load(rsinema_path, map_location=device)
clean_bb = {k.replace("backbone.", ""): v for k, v in ckpt_rs.items() if k.startswith("backbone.")}
clip_b32_model.vision_model.load_state_dict(clean_bb, strict=True)
rsinema_head = nn.Linear(768, 1).to(device).half()
rsinema_head.weight.data = ckpt_rs["aesthetic_head.0.weight"].half().to(device)
rsinema_head.bias.data = ckpt_rs["aesthetic_head.0.bias"].half().to(device)
rsinema_head.eval()

# Setup AesFix-Pref v1 Head
v1_head = nn.Linear(512, 1).to(device).half()
if v1_ckpt and "state_dict" in v1_ckpt:
    sd = v1_ckpt["state_dict"]
    v1_head.weight.data = sd["weight"].half().to(device)
    if "bias" in sd and sd["bias"] is not None:
        v1_head.bias.data = sd["bias"].half().to(device)
    else:
        v1_head.bias.data.zero_()
v1_head.eval()

test_unique_imgs = sorted(list(set((IMG_DIR / r['left'] / r['name']).as_posix() for _, r in df_test.iterrows()).union(
    set((IMG_DIR / r['right'] / r['name']).as_posix() for _, r in df_test.iterrows())
)))
print(f"Scoring {len(test_unique_imgs)} unique test images...")

scores_test = {k: {} for k in ['pyiqa_musiq', 'pyiqa_topiq_nr', 'pyiqa_liqe', 'pyiqa_clip_iqa', 'rsinema', 'apv2_5', 'aesfix_pref_v1', 'aesfix_pref_v2']}
per_image_rows = []

for p in tqdm(test_unique_imgs, desc="Scoring test images"):
    im_path = Path(p)
    style = im_path.parent.name
    scene = im_path.name
    luma_val = luma_cache.get(scene, 0.5)
    
    with torch.no_grad():
        s_musiq = float(judges['pyiqa_musiq'](p).item())
        s_topiq = float(judges['pyiqa_topiq_nr'](p).item())
        s_liqe = float(judges['pyiqa_liqe'](p).item())
        s_clipiqa = float(judges['pyiqa_clip_iqa'](p).item())
        
        im = Image.open(p).convert("RGB")
        in_ap = ap_proc(images=im, return_tensors="pt")["pixel_values"].to(device).half()
        s_ap = float(ap_model(pixel_values=in_ap).logits.item())
        
        in_rs = clip_b32_proc(images=im, return_tensors="pt")["pixel_values"].to(device).half()
        v_rs = clip_b32_model.vision_model(pixel_values=in_rs)[1]
        s_rsinema = float(rsinema_head(v_rs).item())
        
        # v1 score
        feat_b32 = torch.tensor(b32_embs[p], dtype=torch.float16, device=device).unsqueeze(0)
        s_v1 = float(v1_head(feat_b32).item())
        
        # v2 score (individual image score through chosen head)
        feat_chosen = torch.tensor(encoder_dicts[best_encoder_name][0][p], dtype=torch.float32, device=device).unsqueeze(0)
        s_v2 = float(best_head(feat_chosen).item())
        
        scores_test['pyiqa_musiq'][p] = s_musiq
        scores_test['pyiqa_topiq_nr'][p] = s_topiq
        scores_test['pyiqa_liqe'][p] = s_liqe
        scores_test['pyiqa_clip_iqa'][p] = s_clipiqa
        scores_test['apv2_5'][p] = s_ap
        scores_test['rsinema'][p] = s_rsinema
        scores_test['aesfix_pref_v1'][p] = s_v1
        scores_test['aesfix_pref_v2'][p] = s_v2
        
        per_image_rows.append({
            "style": style,
            "scene": scene,
            "luma": round(luma_val, 4),
            "score_pyiqa_musiq": round(s_musiq, 4),
            "score_pyiqa_topiq_nr": round(s_topiq, 4),
            "score_pyiqa_liqe": round(s_liqe, 4),
            "score_pyiqa_clip_iqa": round(s_clipiqa, 4),
            "score_apv2_5": round(s_ap, 4),
            "score_rsinema": round(s_rsinema, 4),
            "score_aesfix_pref_v1": round(s_v1, 4),
            "score_aesfix_pref_v2": round(s_v2, 4)
        })

df_per_image = pd.DataFrame(per_image_rows)
df_per_image.to_csv("/kaggle/working/per_image_scores.csv", index=False)
print(f"Saved per_image_scores.csv: {len(df_per_image)} rows.")
"""),

    code("""# Cell 7: Evaluate Test Pairs & Export test_pair_scores.csv
pair_rows = []
for _, row in df_test.iterrows():
    scene = row['name']
    left_style, right_style = row['left'], row['right']
    mos = float(row['mos'])
    p_l = (IMG_DIR / left_style / scene).as_posix()
    p_r = (IMG_DIR / right_style / scene).as_posix()
    
    if p_l not in scores_test['pyiqa_musiq'] or p_r not in scores_test['pyiqa_musiq']:
        continue
        
    diff_v1 = b32_embs[p_l].astype(np.float32) - b32_embs[p_r].astype(np.float32)
    pref_v1_logit = float(v1_head(torch.tensor(diff_v1, dtype=torch.float16, device=device).unsqueeze(0)).item())
    
    diff_v2 = encoder_dicts[best_encoder_name][0][p_l].astype(np.float32) - encoder_dicts[best_encoder_name][0][p_r].astype(np.float32)
    pref_v2_logit = float(best_head(torch.tensor(diff_v2, dtype=torch.float32, device=device).unsqueeze(0)).item())
    
    l_luma = luma_cache.get(Path(p_l).name, 0.5)
    r_luma = luma_cache.get(Path(p_r).name, 0.5)
    
    rec = {
        "scene": scene,
        "left": left_style,
        "right": right_style,
        "mos": mos,
        "pref_logit_v1": round(pref_v1_logit, 4),
        "pref_logit_v2": round(pref_v2_logit, 4),
        "left_median_luma": round(l_luma, 4),
        "right_median_luma": round(r_luma, 4)
    }
    for jname in ['pyiqa_musiq', 'pyiqa_topiq_nr', 'pyiqa_liqe', 'pyiqa_clip_iqa', 'rsinema', 'apv2_5']:
        rec[f"score_left_{jname}"] = round(scores_test[jname][p_l], 4)
        rec[f"score_right_{jname}"] = round(scores_test[jname][p_r], 4)
    pair_rows.append(rec)

df_pair_scores = pd.DataFrame(pair_rows)
df_pair_scores.to_csv("/kaggle/working/test_pair_scores.csv", index=False)
print(f"Saved test_pair_scores.csv: {len(df_pair_scores)} pairs.")
"""),

    code("""# Cell 8: Metrics, Bootstrap CIs & Full Results Export
filtered_test_pairs = df_pair_scores[df_pair_scores['mos'] != 0.5].reset_index(drop=True)
scenes_list = filtered_test_pairs['scene'].tolist()

def bootstrap_scene_ci(scene_ids, is_correct, n_resamples=2000, ci_level=0.95, seed=42):
    scene_map = defaultdict(list)
    for s_id, corr in zip(scene_ids, is_correct):
        scene_map[s_id].append(corr)
    unique_scenes = np.array(list(scene_map.keys()))
    n_scenes = len(unique_scenes)
    rng = np.random.default_rng(seed)
    boot_accs = np.zeros(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sample_scenes = rng.choice(unique_scenes, size=n_scenes, replace=True)
        corr = sum(sum(scene_map[s]) for s in sample_scenes)
        tot = sum(len(scene_map[s]) for s in sample_scenes)
        boot_accs[b] = corr / tot if tot > 0 else 0.0
    alpha = (1.0 - ci_level) / 2.0
    return float(np.mean(is_correct)), float(np.percentile(boot_accs, alpha * 100)), float(np.percentile(boot_accs, (1.0 - alpha) * 100))

def bootstrap_paired_diff_ci(scene_ids, corr_a, corr_b, n_resamples=2000, ci_level=0.95, seed=42):
    scene_map_a, scene_map_b = defaultdict(list), defaultdict(list)
    for s_id, a, b in zip(scene_ids, corr_a, corr_b):
        scene_map_a[s_id].append(a)
        scene_map_b[s_id].append(b)
    unique_scenes = np.array(list(scene_map_a.keys()))
    n_scenes = len(unique_scenes)
    rng = np.random.default_rng(seed)
    boot_diffs = np.zeros(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sample_scenes = rng.choice(unique_scenes, size=n_scenes, replace=True)
        corr_a_tot = sum(sum(scene_map_a[s]) for s in sample_scenes)
        corr_b_tot = sum(sum(scene_map_b[s]) for s in sample_scenes)
        tot = sum(len(scene_map_a[s]) for s in sample_scenes)
        if tot > 0:
            boot_diffs[b] = (corr_a_tot - corr_b_tot) / tot
    alpha = (1.0 - ci_level) / 2.0
    return float((sum(corr_a) - sum(corr_b)) / len(scene_ids)), float(np.percentile(boot_diffs, alpha * 100)), float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

all_judges = ['aesfix_pref_v1', 'aesfix_pref_v2', 'pyiqa_topiq_nr', 'pyiqa_musiq', 'pyiqa_liqe', 'pyiqa_clip_iqa', 'rsinema', 'apv2_5']
final_metrics = []
judge_correct_dict = {}

for j in all_judges:
    corr_list, deltas, margins = [], [], []
    for _, r in filtered_test_pairs.iterrows():
        human_prefers_left = r['mos'] > 0.5
        margin = r['mos'] - 0.5
        if j == 'aesfix_pref_v1':
            d = r['pref_logit_v1']
        elif j == 'aesfix_pref_v2':
            d = r['pref_logit_v2']
        else:
            d = r[f'score_left_{j}'] - r[f'score_right_{j}']
        judge_prefers_left = d > 0.0
        corr = (judge_prefers_left == human_prefers_left)
        corr_list.append(corr)
        deltas.append(d)
        margins.append(margin)
        
    judge_correct_dict[j] = corr_list
    mean_acc, ci_l, ci_h = bootstrap_scene_ci(scenes_list, corr_list)
    srcc_val, _ = spearmanr(deltas, margins)
    final_metrics.append({
        "judge": j,
        "pairwise_accuracy": round(mean_acc, 4),
        "ci_95": f"[{ci_l:.4f}, {ci_h:.4f}]",
        "ci_low": round(ci_l, 4),
        "ci_high": round(ci_h, 4),
        "spearman_signed_margin": round(float(srcc_val), 4)
    })

# High agreement accuracy (|2*mos - 1| >= 0.7)
df_ha = filtered_test_pairs[abs(2*filtered_test_pairs['mos'] - 1) >= 0.7].reset_index(drop=True)
ha_accs = {}
for j in all_judges:
    corr_ha = 0
    for _, r in df_ha.iterrows():
        if j == 'aesfix_pref_v1':
            d = r['pref_logit_v1']
        elif j == 'aesfix_pref_v2':
            d = r['pref_logit_v2']
        else:
            d = r[f'score_left_{j}'] - r[f'score_right_{j}']
        if (d > 0) == (r['mos'] > 0.5):
            corr_ha += 1
    ha_accs[j] = round(corr_ha / len(df_ha), 4) if len(df_ha) > 0 else 0.0

# Original vs edit breakdown
orig_pairs = filtered_test_pairs[(filtered_test_pairs['left'] == 'original') | (filtered_test_pairs['right'] == 'original')].reset_index(drop=True)
orig_accs = {}
for j in all_judges:
    corr_o = 0
    for _, r in orig_pairs.iterrows():
        if j == 'aesfix_pref_v1':
            d = r['pref_logit_v1']
        elif j == 'aesfix_pref_v2':
            d = r['pref_logit_v2']
        else:
            d = r[f'score_left_{j}'] - r[f'score_right_{j}']
        if (d > 0) == (r['mos'] > 0.5):
            corr_o += 1
    orig_accs[j] = round(corr_o / len(orig_pairs), 4) if len(orig_pairs) > 0 else 0.0

# Paired difference: v2 minus TopIQ
diff_mean, diff_l, diff_h = bootstrap_paired_diff_ci(scenes_list, judge_correct_dict['aesfix_pref_v2'], judge_correct_dict['pyiqa_topiq_nr'])

elapsed_sec = time.time() - start_time
gpu_hours_used = round(elapsed_sec / 3600.0, 4)

# Save aesfix_pref_v2_head.pt with CPU tensors
torch.save({
    "state_dict": {k: v.cpu() for k, v in best_head.state_dict().items()},
    "encoder_id": best_encoder_name,
    "head_architecture": best_row["architecture"],
    "feature_dim": encoder_dicts[best_encoder_name][1],
    "val_accuracy": best_row["val_accuracy"],
    "val_loss": best_row["val_loss"]
}, "/kaggle/working/aesfix_pref_v2_head.pt")

k2_results_payload = {
    "run_manifest": {
        "task": "K2b_dear_judges",
        "dataset": "DEAR (arXiv:2512.05209)",
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "test_scenes": len(test_scenes),
        "test_pairs": len(filtered_test_pairs),
        "runtime_seconds": round(elapsed_sec, 2),
        "gpu_hours": gpu_hours_used,
        "gpu_device": gpu_name
    },
    "validation_model_selection_table": validation_table,
    "chosen_model": {
        "candidate": best_candidate_id,
        "encoder": best_encoder_name,
        "val_accuracy": best_row["val_accuracy"]
    },
    "test_pairwise_metrics": final_metrics,
    "paired_difference_v2_minus_topiq": {
        "mean_diff": round(diff_mean, 4),
        "ci_95": f"[{diff_l:+.4f}, {diff_h:+.4f}]"
    },
    "high_agreement_accuracy": ha_accs,
    "original_vs_edit_accuracy": orig_accs
}

with open("/kaggle/working/k2_full_results.json", "w", encoding="utf-8") as f:
    json.dump(k2_results_payload, f, indent=2)

print("\\n" + "="*85)
print("K2b FINAL BENCHMARK SUMMARY (Single GPU T4):")
print("="*85)
print(pd.DataFrame(final_metrics).to_markdown(index=False))
print(f"\\nGPU Hours Consumed: {gpu_hours_used:.4f} (Runtime: {elapsed_sec:.1f}s)")
print("="*85)
""")
]

# Generate both notebooks
create_notebook(k2a_cells, "kaggle/k2a_dear/dear_512.ipynb", accelerator="NONE")
create_notebook(k2b_cells, "kaggle/k2_dear/dear_judges.ipynb", accelerator="GPU")
print("Successfully generated kaggle/k2a_dear/dear_512.ipynb and kaggle/k2_dear/dear_judges.ipynb!")
