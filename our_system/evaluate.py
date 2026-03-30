import json
import sys
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr


def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(results: list):
    our_scores      = []
    baseline_scores = []
    correct_labels  = 0
    total_with_gt   = 0

    label_map = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}

    for r in results:
        our    = r.get("fuzzy_score")
        base   = r.get("original_sentiment_score")
        if our is not None and base is not None:
            our_scores.append(float(our))
            baseline_scores.append(float(base))

        gt = r.get("ground_truth_quality")
        predicted = r.get("fuzzy_label")
        if gt and predicted:
            total_with_gt += 1
            if label_map.get(gt, "").upper() == predicted.upper():
                correct_labels += 1

    labels = [r.get("fuzzy_label") for r in results]
    high   = labels.count("HIGH")
    medium = labels.count("MEDIUM")
    low    = labels.count("LOW")
    total  = len(results)

    print("\n" + "="*55)
    print("  CritiQ — Evaluation Results")
    print("="*55)
    print(f"  Samples evaluated    : {total}")

    if len(our_scores) >= 3:
        srcc, _ = spearmanr(our_scores, baseline_scores)
        plcc, _ = pearsonr(our_scores, baseline_scores)
        print(f"  SRCC                 : {srcc:.4f}")
        print(f"  PLCC                 : {plcc:.4f}")
    else:
        print(f"  SRCC                 : N/A (need >= 3 samples)")
        srcc, plcc = 0.0, 0.0

    if total_with_gt > 0:
        accuracy = correct_labels / total_with_gt * 100
        print(f"  Label Accuracy       : {accuracy:.1f}% ({correct_labels}/{total_with_gt})")

    print(f"\n  Label Distribution:")
    print(f"    HIGH   : {high:3d} ({100*high/total:.1f}%)")
    print(f"    MEDIUM : {medium:3d} ({100*medium/total:.1f}%)")
    print(f"    LOW    : {low:3d} ({100*low/total:.1f}%)")

    print("\n  Comparison with RPCD baselines (from base paper):")
    print("  ┌─────────────────────┬────────┬────────┐")
    print("  │ Method              │  SRCC  │  PLCC  │")
    print("  ├─────────────────────┼────────┼────────┤")
    print("  │ NIMA                │ 0.1200 │ 0.1160 │")
    print("  │ ViT + Linear Probe  │ 0.1720 │ 0.1730 │")
    print("  │ AestheticViT        │ 0.2500 │ 0.2530 │")
    srcc_val = srcc if len(our_scores) >= 3 else 0.0
    plcc_val = plcc if len(our_scores) >= 3 else 0.0
    print(f"  │ CritiQ (Ours)       │ {srcc_val:6.4f} │ {plcc_val:6.4f} │")
    print("  └─────────────────────┴────────┴────────┘")

    print("\n  Sample Outputs (first 3):")
    for r in results[:3]:
        print(f"\n  [{r.get('fuzzy_label')}] {r.get('id', 'N/A')} | score: {r.get('fuzzy_score', 0):.4f} | gt: {r.get('ground_truth_quality', 'N/A')}")
        print(f"  {r.get('explanation', '')[:120]}...")
    print("="*55 + "\n")


if __name__ == "__main__":
    path = "results/critiq_output.json"
    if not os.path.exists(path):
        print(f"No results at {path}. Run pipeline first.")
        sys.exit(1)
    results = load_results(path)
    evaluate(results)
