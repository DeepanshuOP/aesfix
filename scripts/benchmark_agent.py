#!/usr/bin/env python3
"""Benchmark AesFix single-pass pipeline vs multi-turn agent on legacy images.

Evaluates the 5 legacy images (blur.jpg, bright.jpg, dark.jpg, temp.jpg, test.jpg):
1. Single-pass pipeline: HeuristicAspectCritic + TierAExecutor + IdentityGuard + JudgeEnsemble.
2. Multi-turn Agent: Closed-loop AesFixAgent (Gemini / Heuristic) with guardrails and delta gating.
Logs exactly one run entry to results/ledger.jsonl with per-image verdicts and deltas.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.agent.llm import LLMClient
from aesfix.agent.loop import AesFixAgent
from aesfix.critic import HeuristicAspectCritic
from aesfix.executor import TierAExecutor
from aesfix.executor.guard import IdentityGuard
from aesfix.judges import JudgeEnsemble, create_default_ensemble
from aesfix.ledger import DEFAULT_LEDGER_PATH, append_run

LEGACY_IMAGES = ["blur.jpg", "bright.jpg", "dark.jpg", "temp.jpg", "test.jpg"]


def run_single_pass_pipeline(
    image_path: Path,
    critic: HeuristicAspectCritic,
    executor: TierAExecutor,
    guard: IdentityGuard,
    ensemble: JudgeEnsemble,
    noise_floor: float = 0.0037,
) -> Tuple[float, str, List[str]]:
    """Executes single-pass critic-executor pipeline on a single image."""
    baseline_eval = ensemble.evaluate(str(image_path))
    baseline_score = float(baseline_eval.overall_score)

    critique = critic.critique(str(image_path))
    if not critique.fixes:
        return 0.0, "leave it", []

    img_pil = Image.open(image_path).convert("RGB")
    # Execute fixes sequentially
    curr_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(torch.device("cpu"))
    executed_ops: List[str] = []

    for fix in critique.fixes:
        try:
            curr_tensor = executor.execute_op(curr_tensor, fix.op, fix.params)
            executed_ops.append(fix.op)
        except Exception:
            pass

    # Guard check
    orig_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(torch.device("cpu"))
    guard_verdict = guard.evaluate(orig_tensor, curr_tensor)

    if not guard_verdict.is_valid:
        return 0.0, "leave it (guard rejected)", executed_ops

    # Post score
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
    return delta, verdict, executed_ops


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark single-pass pipeline vs agent on legacy images.")
    parser.add_argument("--provider", default=None, help="Agent LLM provider (gemini or heuristic).")
    parser.add_argument("--model", default="gemini-3.8-flash", help="Agent LLM model id.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH, help="Path to ledger.jsonl.")
    parser.add_argument("--max-steps", type=int, default=2, help="Max agent steps per image.")
    args = parser.parse_args()

    provider = args.provider
    if not provider:
        provider = "gemini" if os.getenv("GEMINI_API_KEY") else "heuristic"

    print(f"=== Starting AesFix Benchmark: Single-Pass vs Agent ({provider} / {args.model}) ===")

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

    sp_deltas: List[float] = []
    agent_deltas: List[float] = []
    metrics: Dict[str, Any] = {}
    details: List[Dict[str, Any]] = []

    total_agent_steps = 0
    total_agent_accepted = 0

    for img_name in LEGACY_IMAGES:
        img_path = REPO_ROOT / img_name
        if not img_path.exists():
            print(f"Warning: Legacy image '{img_path}' not found, skipping.")
            continue

        base_stem = img_path.stem
        print(f"\n--- Benchmarking: {img_name} ---")

        # 1. Single-pass pipeline
        sp_delta, sp_verdict, sp_ops = run_single_pass_pipeline(
            img_path, critic, executor, guard, ensemble, noise_floor=0.0037
        )
        sp_deltas.append(sp_delta)
        metrics[f"{base_stem}_sp_delta"] = sp_delta
        metrics[f"{base_stem}_sp_verdict"] = sp_verdict
        print(f"  [Single-Pass] Δ={sp_delta:+.4f} | Verdict: {sp_verdict} | Ops: {sp_ops}")

        # 2. Agent closed loop
        traj = agent.run(str(img_path))
        agent_delta = round(float(traj.total_gain), 4)
        agent_verdict = traj.verdict
        agent_deltas.append(agent_delta)
        total_agent_steps += traj.total_steps
        total_agent_accepted += len(traj.accepted_steps)

        metrics[f"{base_stem}_agent_delta"] = agent_delta
        metrics[f"{base_stem}_agent_verdict"] = agent_verdict
        print(
            f"  [Agent]       Δ={agent_delta:+.4f} | Verdict: {agent_verdict} | "
            f"Steps: {traj.total_steps} (accepted: {len(traj.accepted_steps)})"
        )

        details.append({
            "image": img_name,
            "sp_delta": sp_delta,
            "sp_verdict": sp_verdict,
            "agent_delta": agent_delta,
            "agent_verdict": agent_verdict,
            "agent_steps": traj.total_steps,
            "agent_accepted": len(traj.accepted_steps),
        })

    # Summary metrics
    metrics["mean_single_pass_delta"] = round(float(np.mean(sp_deltas)), 4) if sp_deltas else 0.0
    metrics["mean_agent_delta"] = round(float(np.mean(agent_deltas)), 4) if agent_deltas else 0.0
    metrics["total_agent_steps"] = total_agent_steps
    metrics["total_agent_accepted"] = total_agent_accepted
    metrics["noise_floor"] = 0.0037

    # Estimated cost (approx. $0.0005 per agent call)
    est_cost = round(total_agent_steps * 0.0005, 4)

    # Log ONE ledger run with per-image verdicts and deltas (AGF5)
    config = {
        "benchmark": "legacy_5_single_pass_vs_agent",
        "agent_provider": provider,
        "agent_model": args.model,
        "images": [d["image"] for d in details],
    }
    manifest = {
        "benchmark": "legacy_5",
        "images": [d["image"] for d in details],
        "count": len(details),
    }

    ledger_entry = append_run(
        metrics=metrics,
        config=config,
        data_manifest=manifest,
        cost_usd=est_cost,
        ledger_path=args.ledger,
    )

    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETED. Run logged to ledger: {ledger_entry['run_id']}")
    print(f"Git commit: {ledger_entry['commit']}")
    print(f"Mean Single-Pass Δ: {metrics['mean_single_pass_delta']:+.4f}")
    print(f"Mean Agent Δ:       {metrics['mean_agent_delta']:+.4f}")
    print("=" * 60)

    # Print summary Markdown table
    print("\n### Legacy Benchmark Comparison Table\n")
    print("| Image | Single-Pass Δ | Single-Pass Verdict | Agent Δ | Agent Verdict | Agent Steps |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for d in details:
        print(
            f"| `{d['image']}` | {d['sp_delta']:+.4f} | {d['sp_verdict']} | "
            f"{d['agent_delta']:+.4f} | {d['agent_verdict']} | {d['agent_steps']} ({d['agent_accepted']} acc) |"
        )
    print(f"| **Mean** | **{metrics['mean_single_pass_delta']:+.4f}** | - | **{metrics['mean_agent_delta']:+.4f}** | - | {total_agent_steps} ({total_agent_accepted} acc) |\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
