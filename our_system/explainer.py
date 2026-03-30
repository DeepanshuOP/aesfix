import google.generativeai as genai

def generate_explanation(label: str, score: float, features: dict, model) -> str:
    prompt = f"""You are an expert photography critic AI.

A photo was evaluated by a hybrid neuro-fuzzy system and received this result:

Aesthetic Rating: {label} (confidence score: {score:.3f})

Feature breakdown:
- Composition:     {features.get('composition', 0.5):.2f} / 1.0
- Lighting:        {features.get('lighting', 0.5):.2f} / 1.0
- Color:           {features.get('color', 0.5):.2f} / 1.0
- Subject Clarity: {features.get('subject_clarity', 0.5):.2f} / 1.0
- Technical:       {features.get('technical', 0.5):.2f} / 1.0

Original critique summary: "{features.get('summary', 'N/A')}"

Write exactly 2-3 sentences explaining why this photo received a {label} rating.
Be specific about which features drove the result.
Write in plain English as if explaining to the photographer.
Do not use bullet points. Do not use markdown. Just plain prose."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"This photo received a {label} aesthetic rating with a confidence of {score:.2f}."
