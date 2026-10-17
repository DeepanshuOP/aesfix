#!/usr/bin/env python3
"""Benchmark AesFix single-pass pipeline vs multi-turn agent on legacy & synthetic suites (AQ6).

Evaluates:
1. 5 legacy images (blur.jpg, bright.jpg, dark.jpg, temp.jpg, test.jpg).
2. 8 synthetic images with known degradations (-1.5 EV, +1.0 EV, contrast 0.6, warm cast)
   built from reference photos using TierAExecutor.
3. Computes and records the empirical noise floor PER IMAGE.
4. Full-reference ground truth recovery metric: CIELAB mean |ΔE| distance to clean original.
   Verifies whether the result moved back toward the clean original ground truth.
5. Logs exactly one comprehensive run entry to results/ledger.jsonl.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.color import rgb2lab

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.agent.llm import LLMClient  # noqa: E402
from aesfix.agent.loop import AesFixAgent  # noqa: E402
from aesfix.critic import HeuristicAspectCritic  # noqa: E402
from aesfix.executor import TierAExecutor  # noqa: E402
from aesfix.executor.guard import IdentityGuard  # noqa: E402
from aesfix.judges.ensemble import (  # noqa: E402
    JudgeEnsemble,
    compute_noise_floor,
    create_default_ensemble,
)
from aesfix.ledger import DEFAULT_LEDGER_PATH, append_run  # noqa: E402

LEGACY_IMAGES = ["blur.jpg", "bright.jpg", "dark.jpg", "temp.jpg", "test.jpg"]


def compute_lab_delta(img_a: Image.Image, img_b: Image.Image) -> float:
    """Computes mean pixel Euclidean distance |ΔE| in CIELAB space between two images (AQ6)."""
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.Resampling.BILINEAR)
    arr_a = np.asarray(img_a.convert("RGB"), dtype=np.float32) / 255.0
    arr_b = np.asarray(img_b.convert("RGB"), dtype=np.float32) / 255.0
    lab_a = rgb2lab(arr_a)
    lab_b = rgb2lab(arr_b)
    delta_e = np.sqrt(np.sum((lab_a - lab_b) ** 2, axis=-1))
    return round(float(np.mean(delta_e)), 4)


def ensure_synthetic_benchmark_suite(output_dir: Path) -> List[Dict[str, Any]]:
    """Synthesizes 8 test images from clean photos with known degradations (AQ6).

    Degradations:
    - -1.5 EV underexposure
    - +1.0 EV overexposure
    - 0.6 contrast compression
    - warm cast (temperature +25, tint +5)

    Base clean photos:
    - test.jpg (reference portrait)
    - temp.jpg (scaled to 1024x1024)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    executor = TierAExecutor(device=torch.device("cpu"))

    clean_sources = {
        "test": REPO_ROOT / "test.jpg",
        "temp": REPO_ROOT / "temp.jpg",
    }

    degradation_specs = [
        {"id": "exp_m1_5", "op": "adjust_exposure", "params": {"ev_delta": -1.5}, "label": "-1.5 EV"},
        {"id": "exp_p1_0", "op": "adjust_exposure", "params": {"ev_delta": 1.0}, "label": "+1.0 EV"},
        {"id": "contrast_0_6", "op": "adjust_contrast", "params": {"factor": 0.6}, "label": "contrast 0.6"},
        {"id": "warm_cast", "op": "adjust_white_balance", "params": {"temperature": 25.0, "tint": 5.0}, "label": "warm cast"},
    ]

    synthetic_cases: List[Dict[str, Any]] = []

    for base_key, src_file in clean_sources.items():
        if not src_file.exists():
            continue

        with Image.open(src_file) as img:
            base_pil = img.convert("RGB")
            if max(base_pil.size) > 1024:
                base_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        clean_ref_path = output_dir / f"clean_{base_key}.jpg"
        if not clean_ref_path.exists():
            base_pil.save(clean_ref_path, quality=95)

        t_base = TF.to_tensor(base_pil).unsqueeze(0).to(torch.device("cpu"))

        for spec in degradation_specs:
            file_name = f"synth_{base_key}_{spec['id']}.jpg"
            out_path = output_dir / file_name

            if not out_path.exists():
                t_deg = executor.execute_op(t_base.clone(), spec["op"], spec["params"])
                t_deg_clamped = torch.clamp(t_deg.squeeze(0), 0.0, 1.0)
                deg_pil = TF.to_pil_image(t_deg_clamped.cpu())
                deg_pil.save(out_path, quality=95)

            synthetic_cases.append({
                "name": file_name,
                "path": out_path,
                "clean_path": clean_ref_path,
                "degradation_op": spec["op"],
                "degradation_params": spec["params"],
                "label": f"{base_key} ({spec['label']})",
            })

    # Copy the -1.5 EV synthetic image to web/public/samples/ for AQ7 UI sample
    web_sample_dir = REPO_ROOT / "web" / "public" / "samples"
    if web_sample_dir.exists():
        synth_m1_5 = output_dir / "synth_test_exp_m1_5.jpg"
        target_sample = web_sample_dir / "underexposed_synthetic.jpg"
        if synth_m1_5.exists() and not target_sample.exists():
            shutil.copy2(synth_m1_5, target_sample)

    return synthetic_cases


def run_single_pass_pipeline(
    image_path: Path,
    critic: HeuristicAspectCritic,
    executor: TierAExecutor,
    guard: IdentityGuard,
    ensemble: JudgeEnsemble,
    noise_floor: float,
) -> Tuple[float, str, List[str], Image.Image]:
    """Executes single-pass critic-executor pipeline on a single image with per-image noise floor."""
    baseline_eval = ensemble.evaluate(str(image_path))
    baseline_score = float(baseline_eval.overall_score)
    img_pil = Image.open(image_path).convert("RGB")

    critique = critic.critique(str(image_path))
    if not critique.fixes:
        return 0.0, "leave it", [], img_pil

    curr_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(torch.device("cpu"))
    executed_ops: List[str] = []

    for fix in critique.fixes:
        try:
            curr_tensor = executor.execute_op(curr_tensor, fix.op, fix.params)
            executed_ops.append(fix.op)
        except Exception:
            pass

    orig_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(torch.device("cpu"))
    guard_verdict = guard.evaluate(orig_tensor, curr_tensor)

    if not guard_verdict.is_valid:
        return 0.0, "leave it (guard rejected)", executed_ops, img_pil

    curr_tensor_clamped = torch.clamp(curr_tensor.squeeze(0), 0.0, 1.0)
    post_pil = TF.to_pil_image(curr_tensor_clamped.cpu())

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        post_pil.save(tmp_path, quality=95)
        post_eval = ensemble.evaluate(tmp_path)
        post_score = float(post_eval.overall_score)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    delta = round(post_score - baseline_score, 4)
    verdict = "measured improvement" if delta > noise_floor else "leave it"
    return delta, verdict, executed_ops, post_pil


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark single-pass pipeline vs agent on legacy and synthetic suites.")
    parser.add_argument("--provider", default=None, help="Agent LLM provider (gemini, openrouter, or heuristic).")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Agent LLM model id.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH, help="Path to ledger.jsonl.")
    parser.add_argument("--max-steps", type=int, default=2, help="Max agent steps per image.")
    parser.add_argument("--synthetic-dir", type=Path, default=REPO_ROOT / "data" / "synthetic_benchmark")
    args = parser.parse_args()

    provider = args.provider
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") else "heuristic"

    print(f"=== Starting AesFix Benchmark: 5 Legacy + 8 Synthetic ({provider} / {args.model}) ===")

    ensemble = create_default_ensemble()
    critic = HeuristicAspectCritic()
    executor = TierAExecutor(device=torch.device("cpu"))
    guard = IdentityGuard()

    llm_client = LLMClient(provider=provider, model_id=args.model)
    agent = AesFixAgent(
        llm_client=llm_client,
        critic=critic,
        ensemble=ensemble,
        guard=guard,
        executor=executor,
        max_steps=args.max_steps,
        consecutive_rejections_stop=2,
    )

    metrics: Dict[str, Any] = {}
    legacy_details: List[Dict[str, Any]] = []
    synthetic_details: List[Dict[str, Any]] = []

    sp_deltas: List[float] = []
    agent_deltas: List[float] = []
    total_agent_steps = 0
    total_agent_accepted = 0

    # -------------------------------------------------------------
    # 1. Benchmark 5 Legacy Images
    # -------------------------------------------------------------
    print("\n--- [Phase 1] Benchmarking 5 Legacy Images ---")
    for img_name in LEGACY_IMAGES:
        img_path = REPO_ROOT / img_name
        if not img_path.exists():
            print(f"Warning: Legacy image '{img_path}' not found, skipping.")
            continue

        base_stem = img_path.stem
        # Compute noise floor PER IMAGE (AQ6)
        noise_floor = round(compute_noise_floor(str(img_path), ensemble), 4)
        metrics[f"{base_stem}_noise_floor"] = noise_floor

        print(f"\nEvaluating Legacy: {img_name} (Noise floor ε={noise_floor:.4f})")

        sp_delta, sp_verdict, sp_ops, _ = run_single_pass_pipeline(
            img_path, critic, executor, guard, ensemble, noise_floor=noise_floor
        )
        sp_deltas.append(sp_delta)
        metrics[f"{base_stem}_sp_delta"] = sp_delta
        metrics[f"{base_stem}_sp_verdict"] = sp_verdict

        traj = agent.run(str(img_path))
        agent_delta = round(float(traj.total_gain), 4)
        agent_verdict = traj.verdict
        agent_deltas.append(agent_delta)
        total_agent_steps += traj.total_steps
        total_agent_accepted += len(traj.accepted_steps)

        metrics[f"{base_stem}_agent_delta"] = agent_delta
        metrics[f"{base_stem}_agent_verdict"] = agent_verdict
        metrics[f"{base_stem}_agent_steps"] = traj.total_steps
        metrics[f"{base_stem}_agent_accepted"] = len(traj.accepted_steps)

        print(f"  [Single-Pass] Δ={sp_delta:+.4f} | Verdict: {sp_verdict} | Ops: {sp_ops}")
        print(f"  [Agent]       Δ={agent_delta:+.4f} | Verdict: {agent_verdict} | Steps: {traj.total_steps} (acc: {len(traj.accepted_steps)})")

        legacy_details.append({
            "image": img_name,
            "noise_floor": noise_floor,
            "sp_delta": sp_delta,
            "sp_verdict": sp_verdict,
            "agent_delta": agent_delta,
            "agent_verdict": agent_verdict,
            "agent_steps": traj.total_steps,
            "agent_accepted": len(traj.accepted_steps),
        })

    # -------------------------------------------------------------
    # 2. Benchmark 8 Synthetic Images (with Ground Truth Recovery)
    # -------------------------------------------------------------
    print("\n--- [Phase 2] Synthesizing & Benchmarking 8 Synthetic Hard Cases (AQ6) ---")
    synthetic_cases = ensure_synthetic_benchmark_suite(args.synthetic_dir)

    synth_sp_deltas: List[float] = []
    synth_agent_deltas: List[float] = []
    agent_recovered_count = 0

    for case in synthetic_cases:
        case_name = case["name"]
        case_path = case["path"]
        clean_path = case["clean_path"]
        label = case["label"]

        clean_pil = Image.open(clean_path).convert("RGB")
        deg_pil = Image.open(case_path).convert("RGB")

        # Full-reference initial distance to clean original
        init_lab_dist = compute_lab_delta(deg_pil, clean_pil)

        # Compute noise floor PER IMAGE (AQ6)
        noise_floor = round(compute_noise_floor(str(case_path), ensemble), 4)
        metrics[f"{case_name}_noise_floor"] = noise_floor

        print(f"\nEvaluating Synthetic: {case_name} ({label}) (ε={noise_floor:.4f}, Init Lab dist={init_lab_dist:.2f})")

        # Single-pass
        sp_delta, sp_verdict, sp_ops, sp_img = run_single_pass_pipeline(
            case_path, critic, executor, guard, ensemble, noise_floor=noise_floor
        )
        synth_sp_deltas.append(sp_delta)
        sp_lab_dist = compute_lab_delta(sp_img, clean_pil)
        sp_lab_recovery = round(init_lab_dist - sp_lab_dist, 4)

        # Agent closed-loop
        traj = agent.run(str(case_path))
        agent_delta = round(float(traj.total_gain), 4)
        agent_verdict = traj.verdict
        synth_agent_deltas.append(agent_delta)
        total_agent_steps += traj.total_steps
        total_agent_accepted += len(traj.accepted_steps)

        agent_lab_dist = compute_lab_delta(traj.final_image, clean_pil)
        agent_lab_recovery = round(init_lab_dist - agent_lab_dist, 4)
        recovered_toward_clean = agent_lab_recovery > 0
        if recovered_toward_clean:
            agent_recovered_count += 1

        metrics[f"{case_name}_init_lab"] = init_lab_dist
        metrics[f"{case_name}_sp_delta"] = sp_delta
        metrics[f"{case_name}_sp_lab_recovery"] = sp_lab_recovery
        metrics[f"{case_name}_agent_delta"] = agent_delta
        metrics[f"{case_name}_agent_lab_recovery"] = agent_lab_recovery
        metrics[f"{case_name}_agent_recovered"] = recovered_toward_clean

        print(f"  [Single-Pass] Δ={sp_delta:+.4f} | Lab recovery: {sp_lab_recovery:+.2f} ({init_lab_dist:.2f} -> {sp_lab_dist:.2f})")
        print(f"  [Agent]       Δ={agent_delta:+.4f} | Lab recovery: {agent_lab_recovery:+.2f} ({init_lab_dist:.2f} -> {agent_lab_dist:.2f}) | Recovered: {recovered_toward_clean}")

        synthetic_details.append({
            "image": case_name,
            "label": label,
            "noise_floor": noise_floor,
            "init_lab": init_lab_dist,
            "sp_delta": sp_delta,
            "sp_lab_recovery": sp_lab_recovery,
            "agent_delta": agent_delta,
            "agent_lab_recovery": agent_lab_recovery,
            "agent_recovered": recovered_toward_clean,
            "agent_steps": traj.total_steps,
            "agent_accepted": len(traj.accepted_steps),
        })

    # Summary metrics across suites
    metrics["mean_legacy_sp_delta"] = round(float(np.mean(sp_deltas)), 4) if sp_deltas else 0.0
    metrics["mean_legacy_agent_delta"] = round(float(np.mean(agent_deltas)), 4) if agent_deltas else 0.0
    metrics["mean_synth_sp_delta"] = round(float(np.mean(synth_sp_deltas)), 4) if synth_sp_deltas else 0.0
    metrics["mean_synth_agent_delta"] = round(float(np.mean(synth_agent_deltas)), 4) if synth_agent_deltas else 0.0
    metrics["synthetic_recovery_rate"] = round(agent_recovered_count / len(synthetic_details), 4) if synthetic_details else 0.0
    metrics["total_agent_steps"] = total_agent_steps
    metrics["total_agent_accepted"] = total_agent_accepted

    est_cost = round(total_agent_steps * 0.0005, 4)

    config = {
        "benchmark": "legacy_5_plus_synth_8_agent_vs_sp",
        "agent_provider": provider,
        "agent_model": args.model,
        "num_legacy": len(legacy_details),
        "num_synthetic": len(synthetic_details),
    }
    manifest = {
        "legacy_images": [d["image"] for d in legacy_details],
        "synthetic_images": [d["image"] for d in synthetic_details],
        "total_count": len(legacy_details) + len(synthetic_details),
    }

    ledger_entry = append_run(
        metrics=metrics,
        config=config,
        data_manifest=manifest,
        cost_usd=est_cost,
        ledger_path=args.ledger,
    )

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETED. Run logged to ledger: {ledger_entry['run_id']}")
    print(f"Commit: {ledger_entry['commit']}")
    print(f"Legacy Mean Single-Pass Δ:    {metrics['mean_legacy_sp_delta']:+.4f}")
    print(f"Legacy Mean Agent Δ:          {metrics['mean_legacy_agent_delta']:+.4f}")
    print(f"Synthetic Mean Single-Pass Δ: {metrics['mean_synth_sp_delta']:+.4f}")
    print(f"Synthetic Mean Agent Δ:       {metrics['mean_synth_agent_delta']:+.4f}")
    print(f"Synthetic Clean Recovery Rate: {metrics['synthetic_recovery_rate'] * 100:.1f}% ({agent_recovered_count}/{len(synthetic_details)} moved toward clean original)")
    print("=" * 70)

    # Markdown Tables
    print("\n### 1. Legacy Images (5)")
    print("| Image | Noise Floor ε | SP Δ | SP Verdict | Agent Δ | Agent Verdict | Steps (Acc) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for d in legacy_details:
        print(f"| `{d['image']}` | {d['noise_floor']:.4f} | {d['sp_delta']:+.4f} | {d['sp_verdict']} | {d['agent_delta']:+.4f} | {d['agent_verdict']} | {d['agent_steps']} ({d['agent_accepted']}) |")

    print("\n### 2. Synthetic Test Images (8) — Ground Truth CIELAB Recovery")
    print("| Case | Degradation | Noise Floor ε | Init Lab Dist | SP Δ (Lab Recov) | Agent Δ (Lab Recov) | Toward Clean? |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for d in synthetic_details:
        toward = "YES" if d["agent_recovered"] else "NO"
        print(f"| `{d['image']}` | {d['label']} | {d['noise_floor']:.4f} | {d['init_lab']:.2f} | {d['sp_delta']:+.4f} ({d['sp_lab_recovery']:+.2f}) | {d['agent_delta']:+.4f} ({d['agent_lab_recovery']:+.2f}) | **{toward}** |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
