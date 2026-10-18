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

# Build K1 Notebook
k1_cells = [
    md("""# AesFix K1: KonIQ-10k Judge Evaluation & Reproduction Benchmark
**Objective**: Evaluate PyIQA metrics (MUSIQ, TOPIQ-NR, LIQE, CLIP-IQA), rsinema, and APv2.5 on the official KonIQ-10k test split.
Compute measured SRCC and PLCC against human Mean Opinion Scores (MOS) and compare with published literature benchmarks."""),

    code("""# Cell 1: Environment & Secrets Setup
import os
import sys
import time
import json
import urllib.request
import zipfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import torch

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

# Device verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Executing on: {device} ({gpu_name})")
assert torch.cuda.is_available(), "T4 GPU accelerator must be enabled for this notebook!"
"""),

    code("""# Cell 2: Pinned Package Installation
!pip install --no-cache-dir pyiqa==0.1.16 aesthetic-predictor-v2-5 transformers huggingface_hub scipy pandas
"""),

    code("""# Cell 3: Verification of SDPA & FP16 Precision
import pyiqa
import transformers

print(f"PyTorch version: {torch.__version__}")
print(f"PyIQA version: {pyiqa.__version__}")
print(f"Transformers version: {transformers.__version__}")

# Enable SDPA attention optimizations
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    print("PyTorch SDPA attention enabled.")
"""),

    code("""# Cell 4: Download KonIQ-10k Dataset from DaRUS & Official Split
DATA_DIR = Path("/kaggle/temp/koniq10k")
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = DATA_DIR / "512x384"

# 1. Download official splits CSV from subpic/koniq
split_url = "https://raw.githubusercontent.com/subpic/koniq/master/metadata/koniq10k_distributions_sets.csv"
split_csv_path = DATA_DIR / "koniq10k_distributions_sets.csv"
if not split_csv_path.exists():
    print("Downloading official KonIQ-10k split CSV...")
    urllib.request.urlretrieve(split_url, str(split_csv_path))

# 2. Download 512x384 images zip from DaRUS (ID: 72286)
zip_url = "https://darus.uni-stuttgart.de/api/access/datafile/72286"
zip_path = DATA_DIR / "koniq10k_512x384.zip"

if not IMG_DIR.exists() or len(list(IMG_DIR.glob("*.jpg"))) == 0:
    print("Downloading KonIQ-10k 512x384 images from DaRUS (~767 MB)...")
    urllib.request.urlretrieve(zip_url, str(zip_path))
    print("Extracting images...")
    with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
        zip_ref.extractall(str(DATA_DIR))
    if zip_path.exists():
        zip_path.unlink() # Free up disk immediately
    print(f"Extracted images to: {IMG_DIR}")

# 3. Load metadata and isolate official test split
df = pd.read_csv(split_csv_path)
print("Metadata columns:", df.columns.tolist())
test_df = df[df["set"] == "test"].copy().reset_index(drop=True)
print(f"Total dataset samples: {len(df)}")
print(f"Official test split samples: {len(test_df)}")

# Verify test image paths
test_df["full_path"] = test_df["image_name"].apply(lambda name: str(IMG_DIR / name))
missing = [p for p in test_df["full_path"] if not os.path.exists(p)]
print(f"Missing test images: {len(missing)}")
assert len(missing) == 0, f"Error: {len(missing)} images from test set missing!"
"""),

    code("""# Cell 5: Load Neural Judges in FP16 / Eval Mode
print("Initializing judges...")

judges = {}

# 1. MUSIQ
print("Loading MUSIQ...")
musiq = pyiqa.create_metric('musiq', device=device, as_loss=False)
judges['pyiqa_musiq'] = musiq

# 2. TOPIQ-NR
print("Loading TOPIQ-NR...")
topiq = pyiqa.create_metric('topiq_nr', device=device, as_loss=False)
judges['pyiqa_topiq_nr'] = topiq

# 3. LIQE
print("Loading LIQE...")
liqe = pyiqa.create_metric('liqe', device=device, as_loss=False)
judges['pyiqa_liqe'] = liqe

# 4. CLIP-IQA
print("Loading CLIP-IQA...")
clipiqa = pyiqa.create_metric('clipiqa', device=device, as_loss=False)
judges['pyiqa_clip_iqa'] = clipiqa

# 5. Aesthetic Predictor v2.5
print("Loading Aesthetic Predictor v2.5...")
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
ap_model, ap_processor = convert_v2_5_from_siglip()
ap_model = ap_model.to(device).half()
ap_model.eval()

# 6. RSinema Aesthetic Scorer
print("Loading RSinema Aesthetic Scorer...")
from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn

class RSinemaEvaluator:
    def __init__(self, device):
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).half()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download("rsinema/aesthetic-scorer", "model.pt")
        ckpt = torch.load(weights_path, map_location=device)
        clean_backbone = {k.replace("backbone.", ""): v for k, v in ckpt.items() if k.startswith("backbone.")}
        self.model.vision_model.load_state_dict(clean_backbone, strict=True)
        self.head = nn.Linear(768, 1).to(device).half()
        self.head.weight.data = ckpt["aesthetic_head.0.weight"].half().to(device)
        self.head.bias.data = ckpt["aesthetic_head.0.bias"].half().to(device)
        self.model.eval()
        self.head.eval()

    @torch.no_grad()
    def score(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device).half()
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs[1]
        scores = self.head(pooled).squeeze(-1)
        return scores.cpu().float().numpy().tolist()

rsinema = RSinemaEvaluator(device)

print("All judges loaded successfully onto GPU!")
"""),

    code("""# Cell 6: Test Split Inference Loop
from PIL import Image
from tqdm.auto import tqdm

image_paths = test_df["full_path"].tolist()
ground_truth_mos = test_df["MOS"].to_numpy()

results = {
    "pyiqa_musiq": [],
    "pyiqa_topiq_nr": [],
    "pyiqa_liqe": [],
    "pyiqa_clip_iqa": [],
    "rsinema": [],
    "apv2_5": []
}

print(f"Beginning inference on {len(image_paths)} official test images...")

# Batch evaluation
BATCH_SIZE = 16

for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
    batch_paths = image_paths[i:i+BATCH_SIZE]
    batch_pils = [Image.open(p).convert("RGB") for p in batch_paths]

    with torch.no_grad():
        # pyiqa metrics
        for p in batch_paths:
            results["pyiqa_musiq"].append(float(musiq(p).item()))
            results["pyiqa_topiq_nr"].append(float(topiq(p).item()))
            results["pyiqa_liqe"].append(float(liqe(p).item()))
            results["pyiqa_clip_iqa"].append(float(clipiqa(p).item()))

        # rsinema
        r_scores = rsinema.score(batch_pils)
        results["rsinema"].extend(r_scores)

        # APv2.5
        ap_inputs = ap_processor(images=batch_pils, return_tensors="pt")
        ap_pixel_values = ap_inputs["pixel_values"].to(device).half()
        ap_logits = ap_model(pixel_values=ap_pixel_values).logits.squeeze(-1)
        results["apv2_5"].extend(ap_logits.cpu().float().numpy().tolist())

print("Inference completed successfully.")
"""),

    code("""# Cell 7: Statistical Analysis - SRCC and PLCC vs MOS
metrics_summary = {}

# Literature published values on KonIQ-10k
published_values = {
    "pyiqa_musiq": {
        "published_srcc": 0.916,
        "published_plcc": 0.928,
        "citation": "Ke et al. MUSIQ (CVPR 2021)",
        "url": "https://arxiv.org/abs/2108.05997"
    },
    "pyiqa_topiq_nr": {
        "published_srcc": 0.894,
        "published_plcc": 0.907,
        "citation": "Chen et al. TOPIQ (IEEE TIP 2024)",
        "url": "https://arxiv.org/abs/2308.03060"
    },
    "pyiqa_liqe": {
        "published_srcc": 0.911,
        "published_plcc": 0.921,
        "citation": "Zhang et al. LIQE (CVPR 2023)",
        "url": "https://arxiv.org/abs/2303.14968"
    },
    "pyiqa_clip_iqa": {
        "published_srcc": 0.741,
        "published_plcc": 0.760,
        "citation": "Wang et al. CLIP-IQA (AAAI 2023)",
        "url": "https://arxiv.org/abs/2207.12396"
    },
    "rsinema": {
        "published_srcc": None,
        "published_plcc": None,
        "citation": "RSinema Aesthetic Scorer (Hugging Face)",
        "url": "https://huggingface.co/rsinema/aesthetic-scorer"
    },
    "apv2_5": {
        "published_srcc": None,
        "published_plcc": None,
        "citation": "Aesthetic Predictor v2.5 (discus0434)",
        "url": "https://huggingface.co/discus0434/aesthetic-predictor-v2-5"
    }
}

for name, preds in results.items():
    arr = np.array(preds)
    srcc, _ = spearmanr(arr, ground_truth_mos)
    plcc, _ = pearsonr(arr, ground_truth_mos)

    pub = published_values.get(name, {})
    metrics_summary[name] = {
        "measured_srcc": round(float(srcc), 4),
        "measured_plcc": round(float(plcc), 4),
        "published_srcc": pub.get("published_srcc"),
        "published_plcc": pub.get("published_plcc"),
        "citation": pub.get("citation"),
        "url": pub.get("url")
    }

elapsed_time = time.time() - start_time
gpu_hours = round(elapsed_time / 3600.0, 4)

run_manifest = {
    "task": "K1_judges_koniq",
    "dataset": "KonIQ-10k (DaRUS-2435)",
    "official_test_split_count": len(test_df),
    "gpu_device": gpu_name,
    "precision": "fp16",
    "attention": "sdpa",
    "runtime_seconds": round(elapsed_time, 2),
    "gpu_hours": gpu_hours,
    "packages": {
        "torch": torch.__version__,
        "pyiqa": pyiqa.__version__,
        "transformers": transformers.__version__
    }
}

output_payload = {
    "run_manifest": run_manifest,
    "metrics_summary": metrics_summary
}

# Display Table
print()
print("="*85)
print(f"{'Judge Model':<20} | {'Measured SRCC':<14} | {'Published SRCC':<14} | {'Measured PLCC':<14} | {'Citation'}")
print("-" * 85)
for judge, data in metrics_summary.items():
    pub_s = f"{data['published_srcc']:.3f}" if data['published_srcc'] is not None else "N/A"
    print(f"{judge:<20} | {data['measured_srcc']:<14.4f} | {pub_s:<14} | {data['measured_plcc']:<14.4f} | {data['citation']}")
print("="*85)
print(f"Total Runtime: {elapsed_time:.1f}s ({gpu_hours:.4f} GPU-hours used).")
"""),

    code("""# Cell 8: Save Artifacts for Ingestion
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_file = OUT_DIR / "koniq_results.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(output_payload, f, indent=2)

print(f"Results successfully saved to {out_file}")
print("Output file size:", os.path.getsize(out_file), "bytes")
""")
]

create_notebook(k1_cells, "kaggle/k1_koniq/judges_koniq.ipynb")
print("Saved kaggle/k1_koniq/judges_koniq.ipynb")
