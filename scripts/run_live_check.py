"""Live check script for verifying agent behavior on critical test photographs.

Tests anti-reward hacking on low-key images (dark.jpg) and positive control on
underexposed photographs (underexposed_synthetic.jpg).
"""
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.agent.llm import LLMClient  # noqa: E402
from aesfix.agent.loop import AesFixAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live checks on agent test photographs.")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openrouter", "heuristic"], help="LLM provider")
    parser.add_argument("--model-id", default=None, help="Explicit model identifier")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum steps per run")
    args = parser.parse_args()

    client = LLMClient(provider=args.provider, model_id=args.model_id)
    agent = AesFixAgent(llm_client=client, max_steps=args.max_steps, consecutive_rejections_stop=2)

    candidates = [
        REPO_ROOT / "tests" / "hard_cases" / "dark.jpg",
        REPO_ROOT / "dark.jpg",
        REPO_ROOT / "web" / "public" / "samples" / "underexposed_synthetic.jpg",
        REPO_ROOT / "data" / "synthetic_benchmark" / "synth_test_exp_m1_5.jpg",
    ]
    target_images = []
    seen_names = set()
    for p in candidates:
        if p.exists() and p.name not in seen_names:
            target_images.append(str(p))
            seen_names.add(p.name)

    print(f"=== Starting Live Check (Provider: {args.provider}) ===")
    for img_path in target_images:
        img_name = Path(img_path).name
        print("\n==========================================")
        print(f"RUNNING ON: {img_name}")
        print("==========================================")
        traj = agent.run(img_path)
        print(f"RESULT: {img_name}")
        print(f"Verdict: {traj.verdict}")
        print(f"Total steps: {traj.total_steps}")
        print(f"Accepted: {len(traj.accepted_steps)}, Rejected: {len(traj.rejected_steps)}")
        print(f"Total gain: {traj.total_gain:+.4f}")
        print(f"Accepted ops: {[s.proposal.op for s in traj.accepted_steps]}")
        for idx, s in enumerate(traj.steps, 1):
            print(f"  Step {idx}: {s.proposal.op} {s.proposal.params} -> {s.status} ({s.verdict_reason})")

    print("\nLive check complete.")


if __name__ == "__main__":
    main()
