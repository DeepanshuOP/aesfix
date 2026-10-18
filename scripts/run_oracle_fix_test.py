#!/usr/bin/env python3
"""Oracle-Fix Test and Agent Step Analysis on 8 Synthetic Degradations (J4).

Why this test is the most informative judge check:
1. Ground Truth Fix (Oracle):
   For each of the 8 synthetic images, we apply the EXACT inverse of the known degradation
   (e.g. +1.5 EV for the -1.5 EV image, -1.0 EV for +1.0 EV, contrast factor 1.6667 for 0.6,
   and inverse WB temp/tint shift for warm cast).
2. Per-Judge Measurement:
   For rsinema, MUSIQ, TopIQ-NR, and AesFix-Pref v1, report:
   - Baseline score on degraded image
   - Score on oracle-fixed image
   - Delta(oracle fix) in standardized units (Delta / epsilon)
   - Whether Delta > epsilon (did judge recognize ground-truth improvement?)
3. Agent E3 Execution (0 Gemini calls, purely deterministic heuristic critic + F3 caps):
   Per-step trace: image | proposed op + params | was it right direction | per-judge Delta/epsilon | reason
   Reveals whether 'unchanged' cases stem from critic omission or judge rejection.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyiqa
from aesfix.critic import HeuristicAspectCritic
from aesfix.executor import TierAExecutor
from aesfix.judges.ensemble import (
    AesFixPrefJudge,
    MusiqAestheticJudge,
    RsinemaAestheticJudge,
    compute_per_judge_noise_floors,
)
from scripts.benchmark_agent import ensure_synthetic_benchmark_suite


class TopiqNrJudgeWrapper:
    """Wrapper for pyiqa TopIQ-NR matching the BaseJudge evaluation contract."""
    name = "topiq_nr"

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._metric = pyiqa.create_metric("topiq_nr", device=self.device)

    def evaluate(self, image_path: str):
        val = float(self._metric(image_path).squeeze().item())
        return type("Score", (), {
            "name": self.name,
            "raw_score": round(val, 4),
            "normalized_score": round(val, 4),
            "aspects": None
        })()


def get_oracle_fix_op(deg_op: str, deg_params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Calculates the exact analytical inverse operation for each degradation."""
    if deg_op == "adjust_exposure":
        return "adjust_exposure", {"ev_delta": -float(deg_params["ev_delta"])}
    elif deg_op == "adjust_contrast":
        f = float(deg_params["factor"])
        return "adjust_contrast", {"factor": round(1.0 / f, 4)}
    elif deg_op == "adjust_white_balance":
        t = float(deg_params["temperature"])
        ti = float(deg_params["tint"])
        return "adjust_white_balance", {"temperature": -t, "tint": -ti}
    else:
        raise ValueError(f"Unknown degradation op: {deg_op}")


def is_direction_correct(deg_op: str, deg_params: Dict[str, Any], prop_op: str, prop_params: Dict[str, Any]) -> bool:
    """Verifies whether the proposed operation moves in the direction of the ground-truth inverse."""
    if prop_op != deg_op:
        return False
    if deg_op == "adjust_exposure":
        return (float(deg_params["ev_delta"]) * float(prop_params.get("ev_delta", 0.0))) < 0
    elif deg_op == "adjust_contrast":
        deg_factor = float(deg_params["factor"])
        prop_factor = float(prop_params.get("factor", 1.0))
        if deg_factor < 1.0:
            return prop_factor > 1.0
        else:
            return prop_factor < 1.0
    elif deg_op == "adjust_white_balance":
        dt = float(deg_params.get("temperature", 0.0))
        pt = float(prop_params.get("temperature", 0.0))
        return (dt * pt) <= 0
    return False


def run_oracle_fix_and_agent_test() -> Dict[str, Any]:
    output_dir = REPO_ROOT / "data" / "synthetic_benchmark"
    synth_cases = ensure_synthetic_benchmark_suite(output_dir)
    executor = TierAExecutor(device=torch.device("cpu"))

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

    # Part 1: Oracle Fix Evaluation
    print("\\n" + "="*80)
    print("PART 1: ORACLE FIX EVALUATION (Exact Analytical Inverse)")
    print("="*80)

    oracle_table: List[Dict[str, Any]] = []
    case_eps_cache: Dict[str, Dict[str, float]] = {}

    for case in synth_cases:
        deg_path = case["path"]
        deg_op = case["degradation_op"]
        deg_params = case["degradation_params"]
        inv_op, inv_params = get_oracle_fix_op(deg_op, deg_params)

        # Apply oracle fix to degraded image
        with Image.open(deg_path) as im:
            pil_deg = im.convert("RGB")
        t_deg = TF.to_tensor(pil_deg).unsqueeze(0)
        t_fix = executor.execute_op(t_deg, inv_op, inv_params)
        t_fix_clamped = torch.clamp(t_fix.squeeze(0), 0.0, 1.0)
        pil_fix = TF.to_pil_image(t_fix_clamped)

        oracle_path = output_dir / f"oracle_{case['name']}"
        pil_fix.save(oracle_path, quality=95)

        # Compute per-judge noise floors on the degraded image
        # Create an object wrapping judges for noise floor computation
        ensemble_obj = type("Obj", (), {"judges": list(judges.values())})()
        noise_floors = compute_per_judge_noise_floors(deg_path, ensemble_obj)
        case_eps_cache[case["name"]] = noise_floors

        case_record = {
            "image": case["name"],
            "label": case["label"],
            "degradation": f"{deg_op}({deg_params})",
            "oracle_fix": f"{inv_op}({inv_params})",
            "judges": {}
        }

        for j_key, j_obj in judges.items():
            eps = noise_floors.get(j_obj.name, 0.001)
            s_deg = float(j_obj.evaluate(str(deg_path)).raw_score)
            s_orc = float(j_obj.evaluate(str(oracle_path)).raw_score)
            delta = s_orc - s_deg
            delta_std = delta / eps if eps > 0 else 0.0
            passes_eps = delta > eps

            case_record["judges"][j_key] = {
                "score_degraded": round(s_deg, 4),
                "score_oracle": round(s_orc, 4),
                "delta": round(delta, 4),
                "epsilon": round(eps, 4),
                "standardized_delta": round(delta_std, 2),
                "passes_epsilon": passes_eps
            }

        oracle_table.append(case_record)

    # Part 2: Agent E3 Execution on the 8 Synthetic Images
    print("\\n" + "="*80)
    print("PART 2: AGENT E3 STEP TRACE (Heuristic Aspect Critic + F3 Caps)")
    print("="*80)

    critic = HeuristicAspectCritic()
    agent_traces: List[Dict[str, Any]] = []

    for case in synth_cases:
        deg_path = case["path"]
        deg_op = case["degradation_op"]
        deg_params = case["degradation_params"]

        with Image.open(deg_path) as im:
            current_pil = im.convert("RGB")
        current_path = output_dir / f"trace_{case['name']}"
        current_pil.save(current_path, quality=95)

        step_records: List[Dict[str, Any]] = []

        # Run up to 3 turns of the agent loop
        for step in range(1, 4):
            critique = critic.critique(str(current_path))
            if not critique.fixes:
                step_records.append({
                    "step": step,
                    "proposed_op": "none",
                    "params": {},
                    "is_right_direction": None,
                    "delta_per_judge": {},
                    "decision": "TERMINATED",
                    "reason": "Critic proposed no further fixes (leave it)"
                })
                break

            fix = critique.fixes[0]
            prop_op = fix.op
            prop_params = fix.params
            right_dir = is_direction_correct(deg_op, deg_params, prop_op, prop_params)

            # Apply candidate fix
            t_curr = TF.to_tensor(current_pil).unsqueeze(0)
            try:
                t_cand = executor.execute_op(t_curr, prop_op, prop_params)
                t_cand_clamped = torch.clamp(t_cand.squeeze(0), 0.0, 1.0)
                cand_pil = TF.to_pil_image(t_cand_clamped)
                cand_path = output_dir / f"cand_{case['name']}_step{step}.jpg"
                cand_pil.save(cand_path, quality=95)
            except Exception as e:
                step_records.append({
                    "step": step,
                    "proposed_op": prop_op,
                    "params": prop_params,
                    "is_right_direction": right_dir,
                    "delta_per_judge": {},
                    "decision": "REJECTED",
                    "reason": f"Execution error: {e}"
                })
                break

            # Judge candidate with AesFix-Pref v1 as primary acceptance judge
            eps_dict = case_eps_cache.get(case["name"], {})
            judge_deltas = {}
            accepted_by_primary = False

            for j_key, j_obj in judges.items():
                eps = eps_dict.get(j_obj.name, 0.001)
                s_curr = float(j_obj.evaluate(str(current_path)).raw_score)
                s_cand = float(j_obj.evaluate(str(cand_path)).raw_score)
                d = s_cand - s_curr
                d_std = round(d / eps, 2) if eps > 0 else 0.0
                judge_deltas[j_key] = {"delta": round(d, 4), "delta_std": d_std, "passes_eps": d > eps}

                if j_key == "aesfix_pref_v1":
                    accepted_by_primary = (d > eps)

            if accepted_by_primary:
                decision = "ACCEPTED"
                reason = f"AesFix-Pref delta (+{judge_deltas['aesfix_pref_v1']['delta']:.4f}) > epsilon ({eps_dict.get(judges['aesfix_pref_v1'].name, 0.001):.4f})"
                current_pil = cand_pil
                current_pil.save(current_path, quality=95)
            else:
                decision = "REJECTED"
                d_val = judge_deltas['aesfix_pref_v1']['delta']
                e_val = eps_dict.get(judges['aesfix_pref_v1'].name, 0.001)
                reason = f"Judge delta ({d_val:+.4f}) <= epsilon ({e_val:.4f})"

            step_records.append({
                "step": step,
                "proposed_op": prop_op,
                "params": prop_params,
                "is_right_direction": right_dir,
                "delta_per_judge": judge_deltas,
                "decision": decision,
                "reason": reason
            })

            if decision == "REJECTED":
                break

        agent_traces.append({
            "image": case["name"],
            "label": case["label"],
            "degradation": f"{deg_op}({deg_params})",
            "steps": step_records
        })

    payload = {
        "oracle_fix_results": oracle_table,
        "agent_e3_step_traces": agent_traces
    }

    out_file = REPO_ROOT / "results" / "oracle_fix_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\\nOracle fix and agent trace results saved to {out_file}")
    return payload


if __name__ == "__main__":
    run_oracle_fix_and_agent_test()
