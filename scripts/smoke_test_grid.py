#!/usr/bin/env python3
"""Smoke test for GridProposer on 8 synthetic images + dark.jpg (S5).

Reports numbers strictly as 'smoke test', not benchmark results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from aesfix.agent.proposer import GridProposer
from aesfix.executor import TierAExecutor
from aesfix.judges.ensemble import AesFixPrefJudge
from scripts.benchmark_agent import ensure_synthetic_benchmark_suite


def run_smoke_test():
    print("=== S5 Smoke Test: GridProposer (k=15) on 8 Synthetic Images + dark.jpg ===")
    out_dir = Path("tests/fixtures/synthetic_suite")
    synth_cases = ensure_synthetic_benchmark_suite(output_dir=out_dir)

    cases = [(c["name"], Path(c["path"])) for c in synth_cases]
    cases.append(("dark.jpg", Path("dark.jpg")))

    proposer = GridProposer()
    executor = TierAExecutor()
    judge = AesFixPrefJudge()
    noise_floor = 0.04

    results = []
    print(f"\n{'Image Name':<24} | {'Init Score':<10} | {'Decision / Op':<32} | {'Gain':<8} | {'k':<4} | {'Status'}")
    print("-" * 98)

    for name, path in cases:
        if not path.exists():
            print(f"File not found: {path}")
            continue
        im = Image.open(path).convert("RGB")
        init_score = judge.evaluate_pil(im)

        style_intent = {"style_type": "low-key"} if "dark" in name else None

        decision = proposer.propose(
            current_image=im,
            original_image=im,
            current_score=init_score,
            noise_floor=noise_floor,
            primary_judge=judge,
            executor=executor,
            style_intent=style_intent,
        )

        sel = decision.selected_proposal
        if sel is not None:
            op_str = f"{sel.op} {sel.params}"
            gain_str = f"{decision.best_gain:+.4f}"
            status = "ACCEPTED"
        else:
            op_str = "None (No admissible candidate)"
            gain_str = "0.0000"
            status = "STOP"

        print(f"{name:<24} | {init_score:<10.4f} | {op_str[:32]:<32} | {gain_str:<8} | {decision.k:<4} | {status}")
        results.append({
            "image": name,
            "init_score": init_score,
            "selected_op": sel.op if sel else None,
            "selected_params": sel.params if sel else None,
            "gain": decision.best_gain,
            "k": decision.k,
            "status": status,
        })

    with open("results/grid_smoke_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSmoke test complete. Results logged to results/grid_smoke_test_results.json (smoke test only).")


if __name__ == "__main__":
    run_smoke_test()
