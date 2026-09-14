import sys
import os
import argparse
import json
import time
try:
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai
except Exception:
    genai = None

from .our_system.fuzzy_system import build_fuzzy_system, evaluate_aesthetics
from .our_system.llm_extractor import extract_aesthetic_features, configure_gemini
from .our_system.aggregator import aggregate_critiques, identify_aspect_driver
from .our_system.explainer import generate_explanation
from .our_system.counterfactual_generator import generate_counterfactual
from .our_system.pipeline import run_pipeline, load_dataset, save_results
from .our_system.evaluate import load_results, evaluate
from .our_system.image_model import extract_image_features


def parse_args():
    parser = argparse.ArgumentParser(description="CritiQ - Hybrid Neuro-Fuzzy Aesthetic Evaluation")
    parser.add_argument("--mode", choices=["run", "eval", "single"], default="run")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/processed_info_test.json")
    parser.add_argument("--output", type=str, default="results/critiq_output.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--critique", type=str, default=None)
    parser.add_argument("--image", type=str, default=None,
                        help="Path to image file for image-based evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    user_critique = args.critique if hasattr(args, "critique") else None

    if args.mode == "eval":
        if not os.path.exists(args.output):
            print(f"ERROR: No results file found at {args.output}")
            print("Run pipeline first with: python main.py --mode run --api-key YOUR_KEY")
            sys.exit(1)
        results = load_results(args.output)
        evaluate(results)
        return

    api_key = args.api_key or "DUMMY"
    configure_gemini(api_key)
    model = None  # Zero API calls - rule-based system
    fuzzy_sim = build_fuzzy_system()

    if args.mode == "single":
        critique = args.critique

        # Handle image input
        if args.image and os.path.exists(args.image):
            print(f"\nImage mode: {args.image}")

            # ---- IMAGE FEATURES ----
            img_features = extract_image_features(args.image)

            # ---- TEXT FEATURES ----
            has_critique = bool(args.critique and args.critique.strip())
            if has_critique:
                text_features = extract_aesthetic_features(args.critique, model)
            else:
                text_features = None

            # DEBUG (you MUST see this now)
            print("DEBUG TEXT FEATURES:", text_features)
            print("DEBUG IMAGE FEATURES:", img_features)

            # ---- FUSION ----
            if has_critique and text_features:
                agg, disagreement, subjectivity = aggregate_critiques(
                    [text_features, img_features]
                )
            else:
                agg, disagreement, subjectivity = aggregate_critiques(
                    [img_features]
                )
            label, score = evaluate_aesthetics(
                fuzzy_sim,
                agg["composition"], agg["lighting"], agg["color"],
                agg["subject_clarity"], agg["technical"]
            )
            driver = identify_aspect_driver(agg, label)
            explanation = generate_explanation(label, score, agg, model)
            counterfactual = generate_counterfactual(label, agg, model)

            print("\n" + "="*55)
            print("  CritiQ — Image Analysis Result")
            print("="*55)
            print(f"  Image          : {args.image}")
            print(f"  Label          : {label}")
            print(f"  Confidence     : {score:.4f}")
            print(f"  Aspect Driver  : {driver}")
            print(f"\n  Feature Breakdown:")
            for k in ["composition","lighting","color","subject_clarity","technical"]:
                bar = "█" * int(agg[k] * 20)
                print(f"    {k:<18} {agg[k]:.2f}  {bar}")
            print(f"\n  Explanation:")
            print(f"  {explanation}")
            print(f"\n  To Improve:")
            print(f"  {counterfactual}")
            print("="*55 + "\n")
            return

        if critique is None:
            critique = input("Enter critique text: ").strip()

        print("\nProcessing single critique...")

        # Check if --critique is actually an image path
        if critique and os.path.exists(critique):
            print("  Detected image path input")
            
            features = extract_image_features(critique)
            agg, disagreement, subjectivity = aggregate_critiques([features])

        else:
            text_features = extract_aesthetic_features(critique, model)

            image_features = {
                "composition": 0.5,
                "lighting": 0.5,
                "color": 0.5,
                "subject_clarity": 0.5,
                "technical": 0.5,
                "modality": "image"
            }

            agg, disagreement, subjectivity = aggregate_critiques(
                [text_features, image_features]
            )

        label, score = evaluate_aesthetics(
            fuzzy_sim,
            agg["composition"],
            agg["lighting"],
            agg["color"],
            agg["subject_clarity"],
            agg["technical"]
        )
        driver = identify_aspect_driver(agg, label)
        explanation = generate_explanation(label, score, agg, model)
        time.sleep(1.5)
        counterfactual = generate_counterfactual(label, agg, model)

        print("\n" + "="*55)
        print("  CritiQ — Single Critique Result")
        print("="*55)
        print(f"  Label          : {label}")
        print(f"  Confidence     : {score:.4f}")
        print(f"  Aspect Driver  : {driver}")
        print(f"  Subjectivity   : {subjectivity}")
        print(f"\n  Feature Breakdown:")
        for k in ["composition", "lighting", "color", "subject_clarity", "technical"]:
            bar = "█" * int(agg[k] * 20)
            print(f"    {k:<18} {agg[k]:.2f}  {bar}")
        print(f"\n  Explanation:")
        print(f"  {explanation}")
        print(f"\n  To Improve:")
        print(f"  {counterfactual}")
        print("="*55 + "\n")
        return

    # mode == run
    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found at {args.data}")
        sys.exit(1)

    data = load_dataset(args.data)
    results = run_pipeline(data, model, fuzzy_sim, user_critique=user_critique)
    save_results(results, args.output)
    evaluate(results)


if __name__ == "__main__":
    main()