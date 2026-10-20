#!/usr/bin/env python3
"""Benchmark Heuristic vs Grid Proposers on Synthetic Cases and dark.jpg (S5).

Evaluates:
- Proposer="heuristic" (analytical histogram critique, k=1..4)
- Proposer="grid" (symmetric candidate set, k=15, best-of-k selection per SHIP arXiv:2608.21748)
across 500 synthetic cases (S4) with primary judges (v1, and v2 when available).
Reports recovered / worse / unchanged proportions with 95% Wilson score confidence intervals.
Evaluates dark.jpg separately.
"""
from __future__ import annotations

import argparse
import ast
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

from aesfix.agent.proposer import GridProposer, HeuristicProposer, ProposerDecision
from aesfix.executor import TierAExecutor
from aesfix.judges.ensemble import AesFixPrefJudge
from scripts.build_synth_benchmark_v2 import compute_lab_delta_e as calculate_cielab_delta_e

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark_proposers")


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculates Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054  # 95% normal quantile
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half_width = z * math.sqrt((p * (1.0 - p) + (z**2) / (4 * n)) / n) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return p, lower, upper


class V2JudgeWrapper:
    """Wrapper for AesFix-Pref v2 head."""
    name = "aesfix_pref_v2"

    def __init__(self, head_path: str, backbone_name: Optional[str] = None):
        self.head_path = head_path
        fp16_dir = Path(r"D:\aesfix-cache\models\siglip_so400m_vision_fp16")
        if backbone_name is not None:
            self.backbone_name = backbone_name
        elif fp16_dir.exists():
            self.backbone_name = str(fp16_dir)
        else:
            self.backbone_name = "google/siglip-so400m-patch14-384"
        self._head = None
        self._model = None
        self._processor = None
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._head is not None:
            return
        p = Path(self.head_path)
        if not p.exists():
            alt = Path(r"D:\aesfix-data\kaggle_runs\dear_full") / p.name
            if alt.exists():
                self.head_path = str(alt)
        ckpt = torch.load(self.head_path, map_location="cpu", weights_only=False)
        enc_id = str(ckpt.get("encoder_id", ""))
        fp16_dir = Path(r"D:\aesfix-cache\models\siglip_so400m_vision_fp16")
        if "siglip" in enc_id:
            if fp16_dir.exists():
                self.backbone_name = str(fp16_dir)
            else:
                self.backbone_name = "google/siglip-so400m-patch14-384"
        elif "clip_b32" in enc_id:
            self.backbone_name = "openai/clip-vit-base-patch32"
        elif "clip_l14" in enc_id:
            self.backbone_name = "openai/clip-vit-large-patch14"
        elif "dino" in enc_id:
            self.backbone_name = "facebook/dinov2-base"
        else:
            if fp16_dir.exists() and "siglip" in str(ckpt).lower():
                self.backbone_name = str(fp16_dir)
            else:
                self.backbone_name = ckpt.get("encoder_name", ckpt.get("backbone", self.backbone_name))
        
        feat_dim = int(ckpt.get("feature_dim", 1152))
        arch = str(ckpt.get("head_architecture", "mlp"))
        state_dict = ckpt.get("state_dict", ckpt)

        if arch == "mlp" or any(k.startswith("net.") for k in state_dict):
            self._head = torch.nn.Sequential(
                torch.nn.Linear(feat_dim, 256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 1, bias=False),
            )
            clean_sd = {k.replace("net.", ""): v for k, v in state_dict.items()}
            self._head.load_state_dict(clean_sd)
        else:
            self._head = torch.nn.Linear(feat_dim, 1, bias=False)
            clean_sd = {k.replace("fc.", ""): v for k, v in state_dict.items()}
            self._head.load_state_dict(clean_sd)
        self._head.eval()

        from transformers import AutoImageProcessor
        self._processor = AutoImageProcessor.from_pretrained(self.backbone_name)
        if "siglip" in self.backbone_name.lower():
            from transformers import SiglipVisionModel
            self._model = SiglipVisionModel.from_pretrained(self.backbone_name).eval()
        else:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(self.backbone_name).eval()

    def evaluate(self, image_path: str):
        with Image.open(image_path) as im:
            pil_img = im.convert("RGB")
        return type("Score", (), {"raw_score": self.evaluate_pil(pil_img), "name": self.name})()

    def evaluate_pil(self, pil_image: Image.Image) -> float:
        self._ensure_loaded()
        inputs = self._processor(images=pil_image.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            if "siglip" in self.backbone_name:
                out = self._model(pixel_values=inputs["pixel_values"])
                feat = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out[1]
            elif "clip" in self.backbone_name:
                feat = self._model.vision_model(pixel_values=inputs["pixel_values"])[1]
                if hasattr(self._model, "visual_projection"):
                    feat = self._model.visual_projection(feat)
            elif "dino" in self.backbone_name:
                feat = self._model(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 0, :]
            else:
                outputs = self._model(**inputs)
                feat = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]

            feat = feat / (torch.norm(feat, dim=-1, keepdim=True) + 1e-7)
            raw = float(self._head(feat.float()).squeeze().item())
        return round(raw, 4)

    def evaluate_batch_pil(self, pil_images: List[Image.Image]) -> List[float]:
        if not pil_images:
            return []
        self._ensure_loaded()
        inputs = self._processor(images=[im.convert("RGB") for im in pil_images], return_tensors="pt")
        with torch.no_grad():
            if "siglip" in self.backbone_name:
                out = self._model(pixel_values=inputs["pixel_values"])
                feat = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out[1]
            elif "clip" in self.backbone_name:
                feat = self._model.vision_model(pixel_values=inputs["pixel_values"])[1]
                if hasattr(self._model, "visual_projection"):
                    feat = self._model.visual_projection(feat)
            elif "dino" in self.backbone_name:
                feat = self._model(pixel_values=inputs["pixel_values"]).last_hidden_state[:, 0, :]
            else:
                outputs = self._model(**inputs)
                feat = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]

            feat = feat / (torch.norm(feat, dim=-1, keepdim=True) + 1e-7)
            logits = self._head(feat.float()).squeeze(-1)
            if logits.ndim == 0:
                return [round(float(logits.item()), 4)]
            return [round(float(v), 4) for v in logits.tolist()]


def run_agent_on_case(
    init_image: Image.Image,
    ref_image: Optional[Image.Image],
    proposer: Any,
    judge: Any,
    executor: TierAExecutor,
    max_steps: int = 3,
    tau: float = 0.5,
    noise_floor: float = 0.001,
    style_intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Runs closed loop agent with given proposer and primary judge."""
    current_img = init_image.copy()
    original_img = init_image.copy()

    total_k = 0
    decisions = []
    ops_applied = []

    if hasattr(judge, "evaluate_pil"):
        init_score = float(judge.evaluate_pil(current_img))
    else:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tp = tf.name
        current_img.save(tp, quality=95)
        try:
            init_score = float(judge.evaluate(tp).raw_score)
        finally:
            os.unlink(tp)
    curr_score = init_score

    for step in range(1, max_steps + 1):
        decision: ProposerDecision = proposer.propose(
            current_image=current_img,
            original_image=original_img,
            current_score=curr_score,
            noise_floor=noise_floor,
            primary_judge=judge,
            executor=executor,
            style_intent=style_intent,
        )
        total_k += decision.k
        decisions.append({
            "step": step,
            "k": decision.k,
            "gain": decision.best_gain,
            "op": decision.selected_proposal.op if decision.selected_proposal else None,
            "params": decision.selected_proposal.params if decision.selected_proposal else None,
            "stopping_reason": decision.stopping_reason,
        })

        if decision.selected_proposal is None:
            break

        # Execute selected proposal
        fix = decision.selected_proposal
        res = executor.execute_fixes(image_input=current_img, fixes=[fix])
        current_img = res.image
        if hasattr(judge, "evaluate_pil"):
            curr_score = float(judge.evaluate_pil(current_img))
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tp = tf.name
            current_img.save(tp, quality=95)
            try:
                curr_score = float(judge.evaluate(tp).raw_score)
            finally:
                os.unlink(tp)
        ops_applied.append(f"{fix.op}({fix.params})")

    # Metrics
    final_score = curr_score
    score_gain = final_score - init_score

    rec_cat = "unchanged"
    delta_e_init = 0.0
    delta_e_final = 0.0
    delta_delta_e = 0.0

    if ref_image is not None:
        delta_e_init = calculate_cielab_delta_e(init_image, ref_image)
        delta_e_final = calculate_cielab_delta_e(current_img, ref_image)
        delta_delta_e = delta_e_init - delta_e_final

        if delta_delta_e >= tau:
            rec_cat = "recovered"
        elif delta_delta_e <= -tau:
            rec_cat = "worse"
        else:
            rec_cat = "unchanged"

    return {
        "initial_score": round(init_score, 4),
        "final_score": round(final_score, 4),
        "score_gain": round(score_gain, 4),
        "total_k": total_k,
        "num_steps": len(ops_applied),
        "ops_applied": ops_applied,
        "decisions": decisions,
        "delta_e_init": round(delta_e_init, 4),
        "delta_e_final": round(delta_e_final, 4),
        "delta_delta_e": round(delta_delta_e, 4),
        "recovery_category": rec_cat,
        "final_image": current_img,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark edit proposers.")
    parser.add_argument("--cases-csv", type=str, default="results/synth_v2_cases.csv")
    parser.add_argument("--v2-head", type=str, default="results/aesfix_pref_v2_head.pt")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of cases (0 for all 500)")
    parser.add_argument("--output-json", type=str, default="results/proposer_benchmark_results.json")
    args = parser.parse_args()

    print("=== AesFix S5: Edit Proposer Benchmark (Heuristic vs Grid) ===")
    executor = TierAExecutor(device=torch.device("cpu"))
    
    # Load v1 judge
    print("Loading AesFix-Pref v1 judge...")
    judge_v1 = AesFixPrefJudge()
    judge_v1._ensure_loaded()

    # Check if v2 head exists
    judge_v2 = None
    v2_path = Path(args.v2_head)
    if not v2_path.exists():
        alt_v2 = Path(r"D:\aesfix-data\kaggle_runs\dear_full\aesfix_pref_v2_head.pt")
        if alt_v2.exists():
            v2_path = alt_v2

    if v2_path.exists():
        print(f"Loading AesFix-Pref v2 judge from {v2_path}...")
        judge_v2 = V2JudgeWrapper(str(v2_path))
    else:
        print("Note: v2 judge head not yet found. Benchmark will evaluate v1.")

    judges_to_test = [("v1", judge_v1)]
    if judge_v2 is not None:
        judges_to_test.append(("v2", judge_v2))

    # Evaluate dark.jpg separately
    print("\n--- Evaluating dark.jpg separately ---")
    dark_path = Path("dark.jpg")
    dark_img = Image.open(dark_path).convert("RGB")
    dark_results = {}

    for jname, judge in judges_to_test:
        for pname, proposer_cls in [("heuristic", HeuristicProposer), ("grid", GridProposer)]:
            proposer = proposer_cls()
            cfg_name = f"{pname}_{jname}"
            print(f"Running {cfg_name} on dark.jpg...")
            res = run_agent_on_case(dark_img, None, proposer, judge, executor, style_intent={"style_type": "low-key"})
            dark_results[cfg_name] = {
                "initial_score": res["initial_score"],
                "final_score": res["final_score"],
                "score_gain": res["score_gain"],
                "total_k": res["total_k"],
                "ops_applied": res["ops_applied"],
                "decisions": res["decisions"],
            }
            print(f"  Result: Gain={res['score_gain']:+.4f}, Ops={res['ops_applied']}, Total k={res['total_k']}")

    # Evaluate 500 cases
    cases_p = Path(args.cases_csv)
    if not cases_p.exists():
        print(f"Error: Cases CSV not found: {cases_p}")
        return

    df_cases = pd.read_csv(cases_p)
    if args.limit > 0:
        df_cases = df_cases.head(args.limit)
    n_cases = len(df_cases)
    print(f"\n--- Running Proposer Benchmark on {n_cases} Synthetic Cases ---")

    benchmark_records = {}

    for jname, judge in judges_to_test:
        for pname, proposer_cls in [("heuristic", HeuristicProposer), ("grid", GridProposer)]:
            cfg_key = f"{pname}_{jname}"
            proposer = proposer_cls()
            print(f"\nEvaluating configuration: {cfg_key} on {n_cases} cases...")
            
            case_results = []
            rec_counts = {"recovered": 0, "worse": 0, "unchanged": 0}
            total_k_sum = 0
            gains = []

            for idx, row in tqdm(df_cases.iterrows(), total=n_cases, desc=cfg_key):
                deg_p = Path(row["deg_path"])
                ref_p = Path(row["ref_path"])

                with Image.open(deg_p) as im:
                    init_img = im.convert("RGB")
                with Image.open(ref_p) as im:
                    ref_img = im.convert("RGB")

                case_out = run_agent_on_case(init_img, ref_img, proposer, judge, executor)
                cat = case_out["recovery_category"]
                rec_counts[cat] += 1
                total_k_sum += case_out["total_k"]
                gains.append(case_out["score_gain"])

                case_results.append({
                    "scene": row["scene"],
                    "deg_name": row["deg_name"],
                    "initial_score": case_out["initial_score"],
                    "final_score": case_out["final_score"],
                    "score_gain": case_out["score_gain"],
                    "delta_e_init": case_out["delta_e_init"],
                    "delta_e_final": case_out["delta_e_final"],
                    "delta_delta_e": case_out["delta_delta_e"],
                    "recovery_category": cat,
                    "total_k": case_out["total_k"],
                    "ops_applied": case_out["ops_applied"],
                })

            # Calculate Wilson intervals
            p_rec, l_rec, u_rec = wilson_ci(rec_counts["recovered"], n_cases)
            p_wor, l_wor, u_wor = wilson_ci(rec_counts["worse"], n_cases)
            p_unc, l_unc, u_unc = wilson_ci(rec_counts["unchanged"], n_cases)

            summary = {
                "proposer": pname,
                "primary_judge": jname,
                "n_cases": n_cases,
                "recovered_count": rec_counts["recovered"],
                "recovered_rate": round(p_rec, 4),
                "recovered_ci": [round(l_rec, 4), round(u_rec, 4)],
                "worse_count": rec_counts["worse"],
                "worse_rate": round(p_wor, 4),
                "worse_ci": [round(l_wor, 4), round(u_wor, 4)],
                "unchanged_count": rec_counts["unchanged"],
                "unchanged_rate": round(p_unc, 4),
                "unchanged_ci": [round(l_unc, 4), round(u_unc, 4)],
                "mean_score_gain": round(float(np.mean(gains)), 4),
                "mean_k_per_case": round(total_k_sum / n_cases, 2),
            }

            print(f"Summary for {cfg_key}:")
            print(f"  Recovered: {rec_counts['recovered']}/{n_cases} ({p_rec*100:.1f}%, 95% CI: [{l_rec*100:.1f}%, {u_rec*100:.1f}%])")
            print(f"  Worse:     {rec_counts['worse']}/{n_cases} ({p_wor*100:.1f}%, 95% CI: [{l_wor*100:.1f}%, {u_wor*100:.1f}%])")
            print(f"  Unchanged: {rec_counts['unchanged']}/{n_cases} ({p_unc*100:.1f}%, 95% CI: [{l_unc*100:.1f}%, {u_unc*100:.1f}%])")
            print(f"  Mean Gain: {summary['mean_score_gain']:+.4f}, Mean k: {summary['mean_k_per_case']}")

            benchmark_records[cfg_key] = {
                "summary": summary,
                "case_results": case_results,
            }

            # Incrementally save after each configuration
            out_p = Path(args.output_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            existing_data = {}
            if out_p.exists():
                try:
                    with open(out_p, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = {}
            if "dark_jpg" not in existing_data:
                existing_data["dark_jpg"] = {}
            existing_data["dark_jpg"].update(dark_results)
            if "synthetic_benchmark_500" not in existing_data:
                existing_data["synthetic_benchmark_500"] = {}
            existing_data["synthetic_benchmark_500"][cfg_key] = benchmark_records[cfg_key]

            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2)
            print(f"Incrementally saved results for {cfg_key} to {out_p}")


if __name__ == "__main__":
    main()
