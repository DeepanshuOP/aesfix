IMPROVEMENT_TIPS = {
    "composition": [
        "Apply the rule of thirds by placing your subject at intersection points rather than dead center.",
        "Look for leading lines (roads, fences, rivers) that draw the eye toward your subject.",
        "Use foreground elements to add depth and frame your subject naturally.",
        "Try a different crop — eliminate distracting elements at the edges of the frame.",
    ],
    "lighting": [
        "Shoot during golden hour (1 hour after sunrise or before sunset) for warm, flattering light.",
        "Avoid harsh midday sun — use shade or wait for overcast conditions for softer light.",
        "Use exposure compensation (+/- EV) to prevent blown highlights or crushed shadows.",
        "For portraits, position your subject facing a window for soft, directional natural light.",
    ],
    "color": [
        "Check your white balance setting — Auto WB often introduces color casts indoors.",
        "Increase vibrance (not saturation) slightly in post-processing for more natural color pop.",
        "Try shooting in RAW format to give yourself full control over color grading in editing.",
        "Look for scenes with natural color contrast — complementary colors create visual interest.",
    ],
    "subject_clarity": [
        "Use a wider aperture (lower f-number like f/1.8 or f/2.8) to isolate your subject from the background.",
        "Ensure your autofocus point is on the subject's eye (for portraits) or the intended focal point.",
        "Increase your shutter speed to eliminate motion blur — use at least 1/focal length as a minimum.",
        "Move closer to your subject or use a longer focal length to fill the frame with your subject.",
    ],
    "technical": [
        "Lower your ISO to reduce grain — use a tripod or increase shutter speed compensation.",
        "Check your histogram to ensure no clipping on highlights or shadows before shooting.",
        "Clean your lens — dust and smudges reduce contrast and sharpness across the entire frame.",
        "Shoot in RAW format to preserve maximum detail for post-processing.",
    ]
}

def generate_counterfactual(label: str, features: dict, model=None) -> str:
    """
    Template-based counterfactual generator.
    Zero API calls. Gives specific, actionable advice.
    """
    if label == "HIGH":
        return ("This image already achieves a HIGH aesthetic rating. "
                "To maintain this quality: continue applying deliberate compositional choices, "
                "shoot in optimal lighting conditions, and review each shot critically before moving on.")

    import random
    scores = {k: features.get(k, 0.5)
              for k in ["composition", "lighting", "color", "subject_clarity", "technical"]}

    # Find 2 weakest dimensions
    sorted_dims = sorted(scores.items(), key=lambda x: x[1])
    weakest1_name, weakest1_val = sorted_dims[0]
    weakest2_name, weakest2_val = sorted_dims[1]

    tip1 = random.choice(IMPROVEMENT_TIPS[weakest1_name])
    tip2 = random.choice(IMPROVEMENT_TIPS[weakest2_name])

    if label == "LOW":
        return (f"To move from LOW toward HIGH, focus on two areas: "
                f"First, improve {weakest1_name} ({weakest1_val:.2f}) — {tip1} "
                f"Second, address {weakest2_name} ({weakest2_val:.2f}) — {tip2}")
    else:
        return (f"To push from MEDIUM to HIGH, the key improvement is {weakest1_name} ({weakest1_val:.2f}): "
                f"{tip1} "
                f"Additionally, {weakest2_name} ({weakest2_val:.2f}) could be strengthened: {tip2}")
