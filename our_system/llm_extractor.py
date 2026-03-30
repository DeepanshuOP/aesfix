import google.generativeai as genai
import json
import os
import re

def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)

def extract_aesthetic_features(critique_text: str, model) -> dict:
    """
    Takes raw critique text, returns structured aesthetic feature scores.
    """
    if not critique_text or critique_text.strip() == "":
        return {
            "composition": 0.5,
            "lighting": 0.5,
            "color": 0.5,
            "subject_clarity": 0.5,
            "technical": 0.5,
            "summary": "No critique available."
        }

    prompt = f"""You are an expert photography critic AI.

Analyze this photo critique text and extract structured aesthetic scores.

Critique: "{critique_text}"

Return ONLY a valid JSON object with exactly these keys. No explanation, no markdown, no code blocks. Just raw JSON:
{{
  "composition": <float 0.0 to 1.0>,
  "lighting": <float 0.0 to 1.0>,
  "color": <float 0.0 to 1.0>,
  "subject_clarity": <float 0.0 to 1.0>,
  "technical": <float 0.0 to 1.0>,
  "summary": "<one sentence capturing the overall critique tone>"
}}

Scoring guide:
- 0.0 to 0.3 = poor / strongly criticized
- 0.4 to 0.6 = average / mixed feedback  
- 0.7 to 1.0 = good / praised

Base scores on what the critique actually says. If not mentioned, use 0.5."""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        
        # Strip markdown if Gemini wraps it
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        raw = raw.strip()
        
        features = json.loads(raw)
        
        # Validate and clamp all scores
        for key in ["composition", "lighting", "color", "subject_clarity", "technical"]:
            if key not in features:
                features[key] = 0.5
            features[key] = float(max(0.01, min(0.99, features[key])))
        
        if "summary" not in features:
            features["summary"] = "Critique analyzed."
            
        return features
        
    except Exception as e:
        print(f"  [LLM Error] {e} — using neutral defaults")
        return {
            "composition": 0.5,
            "lighting": 0.5,
            "color": 0.5,
            "subject_clarity": 0.5,
            "technical": 0.5,
            "summary": "Could not parse critique."
        }


if __name__ == "__main__":
    api_key = input("Enter your Gemini API key: ").strip()
    configure_gemini(api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    test_critiques = [
        "The composition is fantastic, rule of thirds used perfectly. Lighting is a bit flat though, could use more contrast. Colors are vivid and well saturated.",
        "This is terrible. The horizon is crooked, subject is blurry, and the exposure is way off. Very disappointing shot.",
        "Nice shot overall! The subject is sharp and well-placed. Background is a little busy but the lighting is beautiful. Colors feel natural.",
    ]
    
    print("\nTesting LLM Extractor...\n")
    for i, critique in enumerate(test_critiques):
        print(f"Test {i+1}: {critique[:60]}...")
        result = extract_aesthetic_features(critique, model)
        print(f"  Composition: {result['composition']}")
        print(f"  Lighting:    {result['lighting']}")
        print(f"  Color:       {result['color']}")
        print(f"  Summary:     {result['summary']}")
        print()
