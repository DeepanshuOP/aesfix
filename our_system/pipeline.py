import json
import os
import sys
import time
import google.generativeai as genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from our_system.fuzzy_system import build_fuzzy_system, evaluate_aesthetics
from our_system.llm_extractor import extract_aesthetic_features, configure_gemini
from our_system.aggregator import aggregate_critiques, identify_aspect_driver
from our_system.explainer import generate_explanation
from our_system.counterfactual_generator import generate_counterfactual
from our_system.image_model import extract_image_features


def load_dataset(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(data: list, model, fuzzy_sim, limit: int = 30) -> list:
    results = []
    subset = data[:limit]
    total = len(subset)

    print(f"\nCritiQ Pipeline — Processing {total} samples\n")
    print("=" * 60)

    for i, item in enumerate(subset):
        item_id = item.get('id', item.get('image', f'sample_{i}'))
        print("-" * 40)
        print(f"Sample: {item_id}")

        comments = item.get("comments", item.get("comment", []))
        if isinstance(comments, str):
            comments = [comments]
        if not comments:
            comments = ["No critique available."]

        original_score = item.get("sentiment_score", item.get("score", None))

        # Check for image input
        image_path = item.get("image_path", item.get("image", None))

        if image_path and os.path.exists(str(image_path)):
            print(f"  Using image-based feature extraction: {image_path}")
            agg = extract_image_features(str(image_path))
            disagreement = 0.0
            subjectivity = "LOW"
            feature_list = [agg]
        else:
            print(f"  Extracting features from {len(comments)} critique(s)...")
            feature_list = []
            for comment in comments:
                features = extract_aesthetic_features(comment, model)
                feature_list.append(features)
                time.sleep(4)

            agg, disagreement, subjectivity = aggregate_critiques(feature_list)

        label, fuzzy_score = evaluate_aesthetics(
            fuzzy_sim,
            agg["composition"],
            agg["lighting"],
            agg["color"],
            agg["subject_clarity"],
            agg["technical"]
        )
        driver = identify_aspect_driver(agg, label)
        print(f"Label: {label}")
        print(f"Score: {fuzzy_score:.2f}\n")

        # Skip extra API calls
        try:
            explanation = generate_explanation(label, fuzzy_score, agg, model)
            time.sleep(4)
        except Exception:
            explanation = "Explanation unavailable."
            
        counterfactual = "Skipped to save API usage."

        result = {
            "index": i,
            "id": item.get("id", f"sample_{i}"),
            "image": item.get("image", ""),
            "n_critiques": len(comments),
            "original_sentiment_score": original_score,
            "ground_truth_quality": item.get("ground_truth_quality", "unknown"),
            "per_critique_features": feature_list,
            "aggregated_features": agg,
            "disagreement_score": disagreement,
            "subjectivity_flag": subjectivity,
            "fuzzy_label": label,
            "fuzzy_score": fuzzy_score,
            "aspect_driver": driver,
            "explanation": explanation,
            "counterfactual": counterfactual
        }

        results.append(result)
        
        print("Explanation:")
        print(f"{explanation}\n")
        print("Suggestions:")
        print(counterfactual)
        print("-" * 40)

    print("\n" + "=" * 60)
    return results


def save_results(results: list, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
