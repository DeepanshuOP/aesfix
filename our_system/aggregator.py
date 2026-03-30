import numpy as np
from typing import List, Dict, Tuple

FEATURE_KEYS = ["composition", "lighting", "color", "subject_clarity", "technical"]

def cosine_distance(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def compute_disagreement(feature_list: List[Dict]) -> float:
    if len(feature_list) <= 1:
        return 0.0
    vectors = [[f.get(k, 0.5) for k in FEATURE_KEYS] for f in feature_list]
    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(cosine_distance(vectors[i], vectors[j]))
    return round(float(np.mean(distances)), 4) if distances else 0.0

def subjectivity_label(disagreement: float) -> str:
    if disagreement < 0.15:
        return "LOW"
    elif disagreement < 0.35:
        return "MODERATE"
    else:
        return "HIGH"

def confidence_weighted_aggregate(feature_list: List[Dict]) -> Dict:
    if not feature_list:
        return {k: 0.5 for k in FEATURE_KEYS}
    aggregated = {}
    for key in FEATURE_KEYS:
        values = [f.get(key, 0.5) for f in feature_list]
        certainties = [abs(v - 0.5) * 2 for v in values]
        total_certainty = sum(certainties)
        if total_certainty == 0:
            aggregated[key] = round(float(np.mean(values)), 4)
        else:
            weighted = sum(v * c for v, c in zip(values, certainties))
            aggregated[key] = round(float(weighted / total_certainty), 4)
    aggregated["summary"] = feature_list[0].get("summary", "Aggregated from multiple critiques.")
    return aggregated

def identify_aspect_driver(features: Dict, label: str) -> str:
    scores = {k: features.get(k, 0.5) for k in FEATURE_KEYS}
    if label == "HIGH":
        driver = max(scores, key=scores.get)
        return f"{driver} (score: {scores[driver]:.2f})"
    elif label == "LOW":
        driver = min(scores, key=scores.get)
        return f"{driver} (score: {scores[driver]:.2f})"
    else:
        variance = {k: abs(v - 0.5) for k, v in scores.items()}
        driver = max(variance, key=variance.get)
        return f"{driver} (score: {scores[driver]:.2f})"

def aggregate_critiques(feature_list: List[Dict]) -> Tuple[Dict, float, str]:
    disagreement = compute_disagreement(feature_list)
    subjectivity = subjectivity_label(disagreement)
    aggregated = confidence_weighted_aggregate(feature_list)
    return aggregated, disagreement, subjectivity
