"""Test judge parity between Kaggle reference implementation and local judges.

Why this test exists:
Verifies Task A4 requirements: local RSinema and AesFix-Pref judges must produce
identical scores (< 1e-3 tolerance, accounting for fp16 vs fp32 differences)
to the Kaggle notebook inference pipeline.
"""
import os
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
import pytest
from transformers import CLIPModel, CLIPProcessor

from aesfix.judges.ensemble import RsinemaAestheticJudge, AesFixPrefJudge, _find_hf_snapshot


def test_judge_parity_rsinema_and_aesfix_pref(tmp_path: Path) -> None:
    clip_snap = _find_hf_snapshot("openai/clip-vit-base-patch32")
    rsinema_snap = _find_hf_snapshot("rsinema/aesthetic-scorer")
    pref_head_path = Path("results/aesfix_pref_head.pt")

    if not clip_snap or not rsinema_snap or not pref_head_path.exists():
        pytest.skip("Required model checkpoints not present in local cache.")

    # Create synthetic test patterns to verify inference parity offline without network calls
    test_images = []
    for i, color in enumerate([(120, 140, 160), (30, 40, 50), (220, 200, 180), (100, 180, 120)]):
        img = Image.new("RGB", (256, 256), color=color)
        p = tmp_path / f"synth_parity_{i}.jpg"
        img.save(p, quality=95)
        test_images.append(p)

    # 1. Reference Kaggle Pipeline
    clip_model = CLIPModel.from_pretrained(clip_snap, local_files_only=True)
    clip_processor = CLIPProcessor.from_pretrained(clip_snap, local_files_only=True)
    clip_model.eval()

    rsinema_path = os.path.join(rsinema_snap, "model.pt")
    ckpt = torch.load(rsinema_path, map_location="cpu", weights_only=False)
    clean_backbone = {k.replace("backbone.", ""): v for k, v in ckpt.items() if k.startswith("backbone.")}
    clip_model.vision_model.load_state_dict(clean_backbone, strict=True)

    rsinema_head = nn.Linear(768, 1)
    rsinema_head.weight.data = ckpt["aesthetic_head.0.weight"].float()
    rsinema_head.bias.data = ckpt["aesthetic_head.0.bias"].float()
    rsinema_head.eval()

    pref_head = nn.Linear(512, 1)
    pref_ckpt = torch.load(pref_head_path, map_location="cpu", weights_only=False)
    pref_head.load_state_dict(pref_ckpt.get("state_dict", pref_ckpt))
    pref_head.eval()

    # 2. Local judges
    local_rsinema = RsinemaAestheticJudge()
    local_rsinema._ensure_loaded()

    local_pref = AesFixPrefJudge(weights_path=str(pref_head_path))
    local_pref._ensure_loaded()

    for p in test_images:
        pil_im = Image.open(p).convert("RGB")
        inputs = clip_processor(images=pil_im, return_tensors="pt")
        with torch.no_grad():
            v_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])[1]
            ref_r = float(rsinema_head(v_out).squeeze().item())
            proj = clip_model.visual_projection(v_out)
            proj_norm = proj / proj.norm(p=2, dim=-1, keepdim=True)
            ref_pref = float(pref_head(proj_norm).squeeze().item())

        loc_r = local_rsinema.evaluate(str(p)).raw_score
        loc_pref = local_pref.evaluate(str(p)).raw_score

        diff_r = abs(ref_r - loc_r)
        diff_pref = abs(ref_pref - loc_pref)

        # Tolerance check: max |difference| must be < 1e-3
        assert diff_r < 1e-3, f"RSinema parity failed: diff={diff_r} on {p.name}"
        assert diff_pref < 1e-3, f"AesFix-Pref parity failed: diff={diff_pref} on {p.name}"
