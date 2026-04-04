import cv2
import numpy as np
import os

def extract_image_features(image_path: str) -> dict:
    """
    Extract visual aesthetic features from an image using OpenCV.
    Returns same 5-feature dict format as llm_extractor.py
    so it can be fed directly into the fuzzy system.
    """
    default = {
        "composition": 0.5,
        "lighting": 0.5,
        "color": 0.5,
        "subject_clarity": 0.5,
        "technical": 0.5,
        "summary": "Extracted from image features."
    }

    if not image_path or not os.path.exists(image_path):
        return default

    try:
        img = cv2.imread(image_path)
        if img is None:
            return default

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = gray.shape

        # ── Lighting score ─────────────────────────────────────
        # Based on mean brightness — too dark or too bright = low score
        mean_brightness = float(np.mean(gray)) / 255.0
        # Ideal brightness is around 0.4-0.6
        lighting = 1.0 - abs(mean_brightness - 0.5) * 2
        lighting = float(np.clip(lighting, 0.01, 0.99))

        # ── Technical score (sharpness) ─────────────────────────
        # Laplacian variance — higher = sharper
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Normalize: 0-500 range typical, cap at 1000
        technical = float(np.clip(lap_var / 500.0, 0.01, 0.99))

        # ── Color score ─────────────────────────────────────────
        # Saturation variance — good color = adequate saturation
        sat_channel = hsv[:, :, 1]
        mean_sat = float(np.mean(sat_channel)) / 255.0
        color = float(np.clip(mean_sat * 1.5, 0.01, 0.99))

        # ── Composition proxy ────────────────────────────────────
        # Check if edges concentrate at rule-of-thirds intersections
        edges = cv2.Canny(gray, 50, 150)
        # Divide into 3x3 grid
        h3, w3 = h // 3, w // 3
        # Count edges in center vs overall
        center_edges = float(np.sum(edges[h3:2*h3, w3:2*w3]))
        total_edges  = float(np.sum(edges)) + 1e-6
        center_ratio = center_edges / total_edges
        # Good composition = edges NOT all in center (subject off-center)
        # Rule of thirds means subject at intersections, not dead center
        composition = float(np.clip(1.0 - abs(center_ratio - 0.25) * 2, 0.01, 0.99))

        # ── Subject clarity ──────────────────────────────────────
        # Contrast in the center region vs background
        center = gray[h3:2*h3, w3:2*w3]
        border = np.concatenate([
            gray[:h3, :].flatten(),
            gray[2*h3:, :].flatten(),
            gray[:, :w3].flatten(),
            gray[:, 2*w3:].flatten()
        ])
        center_std = float(np.std(center))
        border_std = float(np.std(border)) + 1e-6
        subject_clarity = float(np.clip(center_std / (border_std + center_std), 0.01, 0.99))

        return {
            "composition":     round(composition,     4),
            "lighting":        round(lighting,        4),
            "color":           round(color,           4),
            "subject_clarity": round(subject_clarity, 4),
            "technical":       round(technical,       4),
            "summary": f"Image analysis: brightness={mean_brightness:.2f}, sharpness={lap_var:.1f}, saturation={mean_sat:.2f}"
        }

    except Exception as e:
        print(f"  [Image Error] {e}")
        return default


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None

    if path and os.path.exists(path):
        features = extract_image_features(path)
        print("\nImage Feature Extraction Result:")
        for k, v in features.items():
            if k != "summary":
                bar = "█" * int(float(v) * 20)
                print(f"  {k:<18} {v:.4f}  {bar}")
        print(f"\n  Summary: {features['summary']}")
    else:
        print("Usage: python our_system/image_features.py path/to/image.jpg")
        print("\nRunning self-test with synthetic values...")

        # Self-test without real image
        test = extract_image_features("nonexistent.jpg")
        print(f"Fallback result: {test}")
        print("image_features.py OK")
