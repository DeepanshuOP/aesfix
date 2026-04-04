def generate_explanation(label: str, score: float, features: dict, model=None) -> str:
    """
    Template-based explanation generator.
    Zero API calls. Produces natural, specific explanations.
    """
    comp  = features.get("composition",     0.5)
    light = features.get("lighting",        0.5)
    color = features.get("color",           0.5)
    subj  = features.get("subject_clarity", 0.5)
    tech  = features.get("technical",       0.5)

    scores = {"composition": comp, "lighting": light,
              "color": color, "subject clarity": subj, "technical quality": tech}

    strongest = max(scores, key=scores.get)
    weakest   = min(scores, key=scores.get)
    strong_v  = scores[strongest]
    weak_v    = scores[weakest]

    if label == "HIGH":
        if strong_v >= 0.75:
            return (f"This image achieved a HIGH aesthetic rating with strong confidence ({score:.2f}). "
                    f"The standout quality is {strongest} ({strong_v:.2f}), which clearly elevates the overall impression. "
                    f"Supporting dimensions including {weakest} ({weak_v:.2f}) are also well executed, "
                    f"resulting in a cohesive and visually compelling photograph.")
        else:
            return (f"This image rates HIGH ({score:.2f}) due to consistently solid performance across all dimensions. "
                    f"No single feature dominates — the photograph succeeds through balance, "
                    f"with {strongest} ({strong_v:.2f}) leading and even {weakest} ({weak_v:.2f}) remaining acceptable.")

    elif label == "MEDIUM":
        if strong_v >= 0.7 and weak_v <= 0.4:
            return (f"This image receives a MEDIUM rating ({score:.2f}) because strong {strongest} ({strong_v:.2f}) "
                    f"is offset by weak {weakest} ({weak_v:.2f}). "
                    f"The technical inconsistency between dimensions prevents a higher rating — "
                    f"addressing {weakest} would significantly improve the overall result.")
        else:
            return (f"This image sits in the MEDIUM range ({score:.2f}) with adequate but unexceptional performance. "
                    f"{strongest.capitalize()} ({strong_v:.2f}) is the relative strength, "
                    f"while {weakest} ({weak_v:.2f}) holds the image back from a HIGH rating. "
                    f"More deliberate attention to the weaker dimensions would push this further.")

    else:  # LOW
        if weak_v <= 0.35:
            return (f"This image rates LOW ({score:.2f}) primarily due to significant issues with {weakest} ({weak_v:.2f}). "
                    f"Even the strongest dimension, {strongest} ({strong_v:.2f}), cannot compensate "
                    f"for these fundamental weaknesses. Core technical and compositional improvements are needed.")
        else:
            return (f"This image receives a LOW aesthetic rating ({score:.2f}). "
                    f"Multiple dimensions score below average, with {weakest} ({weak_v:.2f}) "
                    f"being the most significant concern. The photograph would benefit from "
                    f"foundational improvements in both technique and composition.")
