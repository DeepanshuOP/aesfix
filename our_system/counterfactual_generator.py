import google.generativeai as genai

def generate_counterfactual(label: str, features: dict, model) -> str:
    if label == "HIGH":
        return "This image already rates HIGH. To maintain quality, keep the same approach to composition and lighting."

    feature_lines = "\n".join([
        f"- {k.replace('_', ' ').title()}: {v:.2f}/1.0"
        for k, v in features.items()
        if k != "summary"
    ])

    prompt = f"""You are a photography coach AI.

A photo currently rates {label} in aesthetic quality.

Current feature scores:
{feature_lines}

Your task: Tell the photographer exactly what they need to change to reach a HIGH rating.

Rules:
- Be specific and actionable
- Focus on the 2 lowest-scoring features
- Give concrete photography techniques
- Write 2-3 sentences maximum
- Do not use bullet points
- Do not repeat the scores back to the user

Write the improvement suggestion now:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        weakest = min(
            {k: v for k, v in features.items() if k != "summary"},
            key=lambda k: features[k]
        )
        return f"To improve this image, focus primarily on {weakest.replace('_', ' ')} which is the weakest dimension. Consider seeking professional feedback on lighting and composition."
