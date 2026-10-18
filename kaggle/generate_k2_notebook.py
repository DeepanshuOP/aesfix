import json
from pathlib import Path

def create_notebook(cells, output_path: str):
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
            "accelerator": "GPU"
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

k2_full_cells = [
    md("""# AesFix K2-Full: Comprehensive DEAR Benchmark & AesFix-Pref v2 Model Selection
**Dataset**: DEAR (REPID / Beyond Distortions, arXiv:2512.05209)
**Split**: Full official split — 3,698 train scenes and 1,283 test scenes.
**Budget**: <= 5 GPU-hours on Kaggle Tesla T4.
**Objectives**:
1. Stream all 4,981 scenes at 512px; cache multi-backbone embeddings to `dear_embeddings.npz` (fp16).
2. Train Bradley-Terry preference heads across 4 frozen encoders:
   - CLIP ViT-B/32 (512-d)
   - CLIP ViT-L/14 (768-d)
   - SigLIP so400m (1152-d)
   - DINOv2-base (768-d)
   Each with both a Linear head and a 1-hidden-layer MLP (8 candidates total).
3. Perform model selection on a scene-disjoint validation split carved from TRAIN only (3,000 train / 698 val scenes).
4. Evaluate the winning candidate (AesFix-Pref v2) ONCE on the official TEST split (1,283 scenes, 18,995 pairs).
5. Report full B1-B4 metrics (signed Spearman, 2000-resample scene cluster bootstrap CIs, paired difference vs TopIQ-NR, original-vs-edit breakdown, high-agreement accuracy).
6. Save chosen head to `aesfix_pref_v2_head.pt` and full validation table."""),

    code("""# Cell 1: Environment & Secrets Setup
import os
import sys
import time
import json
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

start_time = time.time()

# Kaggle Secrets - HF_TOKEN
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("[Secrets] HF_TOKEN retrieved from Kaggle Secrets.")
except Exception as e:
    print(f"[Secrets] Note on secrets: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Executing on: {device} ({gpu_name})")
assert torch.cuda.is_available(), "T4 GPU accelerator must be enabled for K2-full!"
"""),

    code("""# Cell 2: Pinned Package Installation
!pip install --no-cache-dir pyiqa==0.1.16 aesthetic-predictor-v2-5 transformers huggingface_hub scipy scikit-learn pandas tqdm timm
"""),

    code("""# Cell 3: Verification of SDPA & FP16 Precision
import pyiqa
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"PyIQA version: {pyiqa.__version__}")
print(f"Transformers version: {transformers.__version__}")

if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    print("PyTorch SDPA attention enabled.")
"""),

    code("""# Cell 4: Download DEAR Official Markups & Carve Scene-Disjoint Validation Split
from huggingface_hub import hf_hub_download

BASE_DIR = Path("/kaggle/temp/dear")
BASE_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = BASE_DIR / "images_512"
IMG_DIR.mkdir(parents=True, exist_ok=True)

print("Fetching official train and test markups from vsevolodpl/DEAR...")
p_train = hf_hub_download(repo_id="vsevolodpl/DEAR", filename="train_processed_markup.csv", repo_type="dataset")
p_test = hf_hub_download(repo_id="vsevolodpl/DEAR", filename="test_processed_markup.csv", repo_type="dataset")

df_train_raw = pd.read_csv(p_train)
df_test = pd.read_csv(p_test)

total_train_scenes = sorted(df_train_raw['name'].unique().tolist())
total_test_scenes = sorted(df_test['name'].unique().tolist())
print(f"DEAR Official Split: {len(total_train_scenes)} train scenes, {len(total_test_scenes)} test scenes.")

# Carve scene-disjoint validation split from TRAIN only (3,000 train scenes, 698 val scenes)
train_scenes_list = total_train_scenes[:3000]
val_scenes_list = total_train_scenes[3000:]
test_scenes_list = total_test_scenes

train_scenes = set(train_scenes_list)
val_scenes = set(val_scenes_list)
test_scenes = set(test_scenes_list)

assert len(train_scenes.intersection(val_scenes)) == 0, "Train and Val must be strictly scene-disjoint!"
assert len(train_scenes.intersection(test_scenes)) == 0, "Train and Test must be strictly scene-disjoint!"
assert len(val_scenes.intersection(test_scenes)) == 0, "Val and Test must be strictly scene-disjoint!"

df_train = df_train_raw[df_train_raw['name'].isin(train_scenes)].copy().reset_index(drop=True)
df_val = df_train_raw[df_train_raw['name'].isin(val_scenes)].copy().reset_index(drop=True)

print(f"Dataset Partitions:")
print(f"  - Train Split: {len(train_scenes)} scenes, {len(df_train)} pairs")
print(f"  - Val Split:   {len(val_scenes)} scenes, {len(df_val)} pairs (for model selection)")
print(f"  - Test Split:  {len(test_scenes)} scenes, {len(df_test)} pairs (evaluated ONCE)")
"""),

    code("""# Cell 5: High-Speed Multi-Threaded Image Streaming (512px)
from tqdm.auto import tqdm

styles = ['original', 'tiff16_a', 'tiff16_b', 'tiff16_c', 'tiff16_d', 'tiff16_e']

# Gather all scenes needing extraction
all_required_scenes = sorted(list(train_scenes.union(val_scenes).union(test_scenes)))

download_tasks = []
for scene in all_required_scenes:
    for style in styles:
        out_path = IMG_DIR / style / scene
        if not out_path.exists():
            download_tasks.append((style, scene, out_path))

print(f"Total image files to stream and resize: {len(download_tasks)}")

def stream_and_downscale(task):
    style, scene, out_path = task
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filename = f"images/{style}/{scene}"
    try:
        raw_path = hf_hub_download(
            repo_id="vsevolodpl/DEAR",
            filename=filename,
            repo_type="dataset"
        )
        with Image.open(raw_path) as im:
            im = im.convert("RGB")
            im.thumbnail((512, 512), Image.Resampling.LANCZOS)
            im.save(str(out_path), quality=92)
        if os.path.exists(raw_path):
            os.unlink(raw_path)
        return True
    except Exception:
        return False

with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(tqdm(executor.map(stream_and_downscale, download_tasks), total=len(download_tasks), desc="Streaming DEAR images"))

success_count = sum(1 for r in results if r)
print(f"Successfully streamed: {success_count} / {len(download_tasks)} images.")
"""),

    code("""# Cell 6: Multi-Encoder Embedding Extraction & Embedding Caching
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

print("Initializing 4 frozen encoders on GPU...")

# 1. CLIP ViT-B/32
print("Loading CLIP ViT-B/32...")
clip_b32_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).half().eval()
clip_b32_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. CLIP ViT-L/14
print("Loading CLIP ViT-L/14...")
clip_l14_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).half().eval()
clip_l14_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 3. SigLIP so400m
print("Loading SigLIP so400m...")
siglip_proc = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device).half().eval()

# 4. DINOv2-base
print("Loading DINOv2-base...")
dinov2_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).half().eval()

# Unique image paths
unique_imgs = sorted(list(IMG_DIR.glob("*/*.jpeg")))
print(f"Extracting embeddings for {len(unique_imgs)} images across 4 backbones...")

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
        # Luma
        for idx, p in enumerate(batch_paths):
            arr = np.array(batch_pils[idx], dtype=np.float32) / 255.0
            luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
            luma_cache[p.as_posix()] = float(np.median(luma))

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

# Release GPU memory from heavy feature extractors
del clip_b32_model, clip_l14_model, siglip_model, dinov2_model
torch.cuda.empty_cache()

# Cache embeddings to notebook output
npz_path = Path("/kaggle/working/dear_embeddings.npz")
print(f"Caching embeddings to {npz_path}...")
img_keys = np.array(list(b32_embs.keys()))
np.savez_compressed(
    npz_path,
    image_paths=img_keys,
    clip_b32=np.array([b32_embs[k] for k in img_keys]),
    clip_l14=np.array([l14_embs[k] for k in img_keys]),
    siglip=np.array([siglip_embs[k] for k in img_keys]),
    dinov2=np.array([dinov2_embs[k] for k in img_keys])
)
print("Saved embeddings cache size:", os.path.getsize(npz_path) / (1024*1024), "MB")
"""),

    code("""# Cell 7: Model Selection on Validation Split (8 Candidates)
print("Building training and validation pairwise difference tensors...")

encoder_dicts = {
    "clip_b32": (b32_embs, 512),
    "clip_l14": (l14_embs, 768),
    "siglip_so400m": (siglip_embs, 1152),
    "dinov2_base": (dinov2_embs, 768)
}

def build_split_pairs(df_split, embs_dict):
    X = []
    y = []
    for _, row in df_split.iterrows():
        scene = row['name']
        left_style = row['left']
        right_style = row['right']
        mos = float(row['mos'])
        if mos == 0.5:
            continue
        p_left = (IMG_DIR / left_style / scene).as_posix()
        p_right = (IMG_DIR / right_style / scene).as_posix()
        if p_left in embs_dict and p_right in embs_dict:
            e_l = embs_dict[p_left].astype(np.float32)
            e_r = embs_dict[p_right].astype(np.float32)
            diff = e_l - e_r
            label = 1.0 if mos > 0.5 else 0.0
            X.append(diff)
            y.append(label)
            # Antisymmetric data augmentation
            X.append(-diff)
            y.append(1.0 - label)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

# Model definitions
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
            nn.Dropout(0.1),
            nn.Linear(256, 1, bias=False)
        )
    def forward(self, x):
        return self.net(x)

validation_table = []
candidate_models = {}

print("Training and evaluating 8 candidate architectures on scene-disjoint train/val splits...")
for enc_name, (embs, dim) in encoder_dicts.items():
    X_tr, y_tr = build_split_pairs(df_train, embs)
    X_va, y_va = build_split_pairs(df_val, embs)
    
    loader_tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    
    for arch in ["linear", "mlp"]:
        cand_id = f"{enc_name}_{arch}"
        head = LinearBT(dim).to(device) if arch == "linear" else MLPBT(dim).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Train head
        head.train()
        for epoch in range(12):
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = head(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation split
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
print("\n" + "="*70)
print("MODEL SELECTION TABLE (Scene-Disjoint Validation Split from TRAIN only):")
print("="*70)
print(df_val_results.to_markdown(index=False))
print("="*70)

# Select best model
best_row = df_val_results.iloc[0]
best_candidate_id = best_row["candidate"]
best_encoder_name = best_row["encoder"]
best_head = candidate_models[best_candidate_id]
print(f"\nCHOSEN WINNING MODEL: {best_candidate_id} (Val Acc: {best_row['val_accuracy']:.4f})")
print(f"This chosen model will now be evaluated on TEST once, and saved as AesFix-Pref v2.")
"""),

    code("""# Cell 8: Evaluate Zero-Shot Judges & Winning Model on Full Test Split (1,283 Scenes)
print(f"Evaluating zero-shot judges and {best_candidate_id} on full official test split ({len(df_test)} pairs)...")

# Load PyIQA metrics for test evaluation
judges = {}
judges['pyiqa_musiq'] = pyiqa.create_metric('musiq', device=device, as_loss=False)
judges['pyiqa_topiq_nr'] = pyiqa.create_metric('topiq_nr', device=device, as_loss=False)
judges['pyiqa_liqe'] = pyiqa.create_metric('liqe', device=device, as_loss=False)
judges['pyiqa_clip_iqa'] = pyiqa.create_metric('clipiqa', device=device, as_loss=False)

from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
ap_model, ap_proc = convert_v2_5_from_siglip()
ap_model = ap_model.to(device).half().eval()

# RSinema
clip_b32_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).half().eval()
rsinema_path = hf_hub_download("rsinema/aesthetic-scorer", "model.pt")
ckpt_rs = torch.load(rsinema_path, map_location=device)
clean_bb = {k.replace("backbone.", ""): v for k, v in ckpt_rs.items() if k.startswith("backbone.")}
clip_b32_model.vision_model.load_state_dict(clean_bb, strict=True)
rsinema_head = nn.Linear(768, 1).to(device).half()
rsinema_head.weight.data = ckpt_rs["aesthetic_head.0.weight"].half().to(device)
rsinema_head.bias.data = ckpt_rs["aesthetic_head.0.bias"].half().to(device)
rsinema_head.eval()

# Unique test images
test_unique_imgs = sorted(list(set((IMG_DIR / r['left'] / r['name']).as_posix() for _, r in df_test.iterrows()).union(
    set((IMG_DIR / r['right'] / r['name']).as_posix() for _, r in df_test.iterrows())
)))
print(f"Scoring {len(test_unique_imgs)} unique test images with zero-shot judges...")

scores_test = {k: {} for k in ['pyiqa_musiq', 'pyiqa_topiq_nr', 'pyiqa_liqe', 'pyiqa_clip_iqa', 'rsinema', 'apv2_5']}
for p in tqdm(test_unique_imgs, desc="Zero-shot test scoring"):
    with torch.no_grad():
        scores_test['pyiqa_musiq'][p] = float(judges['pyiqa_musiq'](p).item())
        scores_test['pyiqa_topiq_nr'][p] = float(judges['pyiqa_topiq_nr'](p).item())
        scores_test['pyiqa_liqe'][p] = float(judges['pyiqa_liqe'](p).item())
        scores_test['pyiqa_clip_iqa'][p] = float(judges['pyiqa_clip_iqa'](p).item())
        
        im = Image.open(p).convert("RGB")
        in_ap = ap_proc(images=im, return_tensors="pt")["pixel_values"].to(device).half()
        scores_test['apv2_5'][p] = float(ap_model(pixel_values=in_ap).logits.item())
        
        in_rs = clip_b32_proc(images=im, return_tensors="pt")["pixel_values"].to(device).half()
        v_rs = clip_b32_model.vision_model(pixel_values=in_rs)[1]
        scores_test['rsinema'][p] = float(rsinema_head(v_rs).item())

# Free models
del ap_model, clip_b32_model, rsinema_head
torch.cuda.empty_cache()

# Extract AesFix-Pref v2 logits for test pairs
chosen_embs = encoder_dicts[best_encoder_name][0]
test_pairs_data = []

best_head.eval()
with torch.no_grad():
    for _, row in df_test.iterrows():
        scene = row['name']
        left_style = row['left']
        right_style = row['right']
        mos = float(row['mos'])
        p_l = (IMG_DIR / left_style / scene).as_posix()
        p_r = (IMG_DIR / right_style / scene).as_posix()
        
        diff_pref = (chosen_embs[p_l] - chosen_embs[p_r]).astype(np.float32)
        pref_logit = float(best_head(torch.tensor(diff_pref, device=device).unsqueeze(0)).item())
        
        test_pairs_data.append({
            "scene": scene,
            "p_left": p_l,
            "p_right": p_r,
            "left_style": left_style,
            "right_style": right_style,
            "mos": mos,
            "pref_logit": pref_logit,
            "diff_luma": abs(luma_cache.get(p_l, 0.5) - luma_cache.get(p_r, 0.5))
        })

print(f"Computed test predictions for {len(test_pairs_data)} pairs.")
"""),

    code("""# Cell 9: Cluster Bootstrap CIs (2,000 resamples over scenes) & Full Metrics
print("Computing cluster bootstrap 95% CIs over 1,283 test scenes (2,000 resamples)...")

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
    ci_low = float(np.percentile(boot_accs, alpha * 100))
    ci_high = float(np.percentile(boot_accs, (1.0 - alpha) * 100))
    mean_acc = float(sum(is_correct) / len(is_correct))
    return mean_acc, ci_low, ci_high

def bootstrap_paired_diff_ci(scene_ids, corr_a, corr_b, n_resamples=2000, ci_level=0.95, seed=42):
    scene_map_a = defaultdict(list)
    scene_map_b = defaultdict(list)
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
    ci_low = float(np.percentile(boot_diffs, alpha * 100))
    ci_high = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))
    mean_diff = float((sum(corr_a) - sum(corr_b)) / len(scene_ids))
    return mean_diff, ci_low, ci_high

scenes_list = [r["scene"] for r in test_pairs_data if r["mos"] != 0.5]
filtered_test_pairs = [r for r in test_pairs_data if r["mos"] != 0.5]

judge_names = ['pyiqa_musiq', 'pyiqa_topiq_nr', 'pyiqa_liqe', 'pyiqa_clip_iqa', 'rsinema', 'apv2_5', 'aesfix_pref_v2']
final_test_table = []
judge_correct_dict = {}

for jname in judge_names:
    correct_list = []
    deltas = []
    signed_margins = []
    for r in filtered_test_pairs:
        mos = r["mos"]
        human_prefers_left = mos > 0.5
        signed_margin = mos - 0.5
        if jname == "aesfix_pref_v2":
            delta = r["pref_logit"]
        else:
            delta = scores_test[jname][r["p_left"]] - scores_test[jname][r["p_right"]]
        
        judge_prefers_left = delta > 0.0
        corr = (judge_prefers_left == human_prefers_left)
        correct_list.append(corr)
        deltas.append(delta)
        signed_margins.append(signed_margin)
    
    judge_correct_dict[jname] = correct_list
    mean_acc, ci_low, ci_high = bootstrap_scene_ci(scenes_list, correct_list, n_resamples=2000, seed=42)
    # Corrected Spearman: correlating signed delta with signed margin
    srcc_val, _ = spearmanr(deltas, signed_margins)
    
    final_test_table.append({
        "judge": jname,
        "pairwise_accuracy": round(mean_acc, 4),
        "ci_95": f"[{ci_low:.4f}, {ci_high:.4f}]",
        "spearman_signed_margin": round(float(srcc_val), 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4)
    })

# Paired difference: AesFix-Pref v2 - TopIQ-NR
diff_mean, diff_ci_low, diff_ci_high = bootstrap_paired_diff_ci(
    scenes_list, judge_correct_dict['aesfix_pref_v2'], judge_correct_dict['pyiqa_topiq_nr'], n_resamples=2000, seed=42
)
print(f"Paired Difference (AesFix-Pref v2 - TopIQ-NR): {diff_mean:+.4f} (95% CI: [{diff_ci_low:+.4f}, {diff_ci_high:+.4f}])")

# High-agreement pairs (|2*mos - 1| >= 0.7)
high_agreement_pairs = [r for r in filtered_test_pairs if abs(2*r["mos"] - 1) >= 0.7]
print(f"High-Agreement Pairs (|margin| >= 0.7): {len(high_agreement_pairs)} pairs ({len(high_agreement_pairs)/len(filtered_test_pairs)*100:.2f}%)")

high_agree_accs = {}
for jname in judge_names:
    corr_ha = 0
    for r in high_agreement_pairs:
        if jname == "aesfix_pref_v2":
            d = r["pref_logit"]
        else:
            d = scores_test[jname][r["p_left"]] - scores_test[jname][r["p_right"]]
        if (d > 0) == (r["mos"] > 0.5):
            corr_ha += 1
    high_agree_accs[jname] = round(corr_ha / len(high_agreement_pairs), 4)

df_final_test = pd.DataFrame(final_test_table)
print("\n" + "="*85)
print("FINAL TEST SPLIT BENCHMARK RESULTS (1,283 Scenes, 18,995 Pairs):")
print("="*85)
print(df_final_test.to_markdown(index=False))
print("="*85)
"""),

    code("""# Cell 10: Original vs Edit Breakdown & Artifact Export
orig_records = [r for r in filtered_test_pairs if r["left_style"] == "original" or r["right_style"] == "original"]
print(f"Original vs Edit pairs in test split: {len(orig_records)}")

# Human preference overall and per expert
exp_wins = defaultdict(int)
exp_totals = defaultdict(int)
overall_edit_wins = 0

for r in orig_records:
    if r["left_style"] == "original":
        edit_won = r["mos"] < 0.5
        exp = r["right_style"]
    else:
        edit_won = r["mos"] > 0.5
        exp = r["left_style"]
    if edit_won:
        overall_edit_wins += 1
        exp_wins[exp] += 1
    exp_totals[exp] += 1

print(f"Overall Human Edit Preference: {overall_edit_wins/len(orig_records)*100:.2f}%")
for exp in sorted(exp_totals.keys()):
    print(f"  {exp}: {exp_wins[exp]/exp_totals[exp]*100:.2f}% ({exp_wins[exp]}/{exp_totals[exp]})")

# Accuracy on original vs edit pairs
orig_accs = {}
for jname in judge_names:
    c_orig = 0
    for r in orig_records:
        if jname == "aesfix_pref_v2":
            d = r["pref_logit"]
        else:
            d = scores_test[jname][r["p_left"]] - scores_test[jname][r["p_right"]]
        if (d > 0) == (r["mos"] > 0.5):
            c_orig += 1
    orig_accs[jname] = round(c_orig / len(orig_records), 4)

elapsed_sec = time.time() - start_time
gpu_hours_used = round(elapsed_sec / 3600.0, 4)

# Save AesFix-Pref v2 weights with CPU tensors and complete metadata
v2_save_path = "/kaggle/working/aesfix_pref_v2_head.pt"
train_run_id = str(uuid.uuid4())
torch.save({
    "state_dict": {k: v.cpu() for k, v in best_head.state_dict().items()},
    "encoder_id": best_encoder_name,
    "feature_dim": encoder_dicts[best_encoder_name][1],
    "head_architecture": best_row["architecture"],
    "training_run_id": train_run_id,
    "candidate_id": best_candidate_id,
    "val_accuracy": best_row["val_accuracy"],
    "val_loss": best_row["val_loss"]
}, v2_save_path)
print(f"Saved chosen head to {v2_save_path} (run_id: {train_run_id}, CPU tensors)")

# Save full results payload
k2_full_payload = {
    "run_manifest": {
        "task": "K2_full_dear_judges",
        "dataset": "DEAR (Beyond Distortions / REPID, arXiv:2512.05209)",
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "test_scenes": len(test_scenes),
        "total_test_pairs": len(filtered_test_pairs),
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
    "test_pairwise_metrics": final_test_table,
    "paired_difference_pref_minus_topiq": {
        "mean_difference": round(diff_mean, 4),
        "ci_95": f"[{diff_ci_low:+.4f}, {diff_ci_high:+.4f}]",
        "ci_low": round(diff_ci_low, 4),
        "ci_high": round(diff_ci_high, 4)
    },
    "high_agreement_accuracy": high_agree_accs,
    "original_vs_edit": {
        "human_edit_preference_overall": round(overall_edit_wins / len(orig_records), 4),
        "human_edit_preference_by_expert": {
            exp: round(exp_wins[exp] / exp_totals[exp], 4) for exp in exp_totals
        },
        "judge_accuracies": orig_accs
    }
}

res_path = "/kaggle/working/k2_full_results.json"
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(k2_full_payload, f, indent=2)

print(f"Results exported to {res_path} ({os.path.getsize(res_path)} bytes)")
print(f"Total GPU-hours consumed: {gpu_hours_used:.4f} (Budget: <= 5.0 GPU-hours)")
""")
]

create_notebook(k2_full_cells, "kaggle/k2_dear/dear_judges.ipynb")
print("Successfully generated kaggle/k2_dear/dear_judges.ipynb for K2-full!")
