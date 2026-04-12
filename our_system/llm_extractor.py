import re
import json
import os
import hashlib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# ── Keyword dictionaries for rule-based extraction ──────────
COMPOSITION_POSITIVE = ["rule of thirds", "composition", "framing", "leading lines",
    "symmetry", "balance", "well composed", "great framing", "diagonal",
    "negative space", "foreground", "layered", "depth", "centered well"]
COMPOSITION_NEGATIVE = ["cropped", "horizon crooked", "off center", "no focal",
    "confusing", "cluttered", "busy background", "poor framing", "tilted",
    "cut off", "wrong crop", "bad framing", "distracting background"]


LIGHTING_POSITIVE = ["golden hour", "well lit", "natural light", "beautiful light",
    "rim light", "soft light", "dramatic light", "catchlights", "exposed well",
    "balanced exposure", "good exposure", "window light", "backlit nicely"]
LIGHTING_NEGATIVE = ["too dark", "underexposed", "overexposed", "blown out",
    "flat lighting", "harsh", "shadows", "muddy", "dark", "bright spots",
    "poor lighting", "bad exposure", "blown highlights", "crushed blacks"]
# --- ADD THESE IMPORTANT KEYWORDS ---

COMPOSITION_POSITIVE += ["good composition", "nice composition"]
COMPOSITION_NEGATIVE += ["poor composition", "bad composition"]

LIGHTING_POSITIVE += ["good lighting", "nice lighting"]
LIGHTING_NEGATIVE += ["poor lighting", "bad lighting"]

COLOR_POSITIVE = ["vibrant", "vivid", "colorful", "color harmony", "rich colors",
    "well saturated", "natural colors", "accurate colors", "color grading",
    "warm tones", "cool tones", "cohesive palette", "good colors"]
COLOR_NEGATIVE = ["washed out", "oversaturated", "color cast", "green cast",
    "flat colors", "dull", "muted", "wrong white balance", "garish",
    "unnatural colors", "color grading hurts", "overdone"]

CLARITY_POSITIVE = ["sharp", "tack sharp", "in focus", "crisp", "clear subject",
    "well focused", "subject pops", "excellent detail", "good focus",
    "beautifully sharp", "perfectly focused"]
CLARITY_NEGATIVE = ["blurry", "out of focus", "soft", "motion blur", "unsharp",
    "camera shake", "missed focus", "not sharp", "blur", "hazy subject"]

TECHNICAL_POSITIVE = ["technically sound", "well exposed", "no noise", "clean",
    "technically flawless", "excellent quality", "high quality", "crisp",
    "technically excellent", "sharp throughout", "no grain"]
TECHNICAL_NEGATIVE = ["noisy", "grainy", "low quality", "artifacts", "pixelated",
    "chromatic aberration", "lens distortion", "overprocessed", "HDR effect",
    "heavy handed", "too much sharpening", "oversharpened"]
NEGATIONS = ["not", "no", "never", "isn't", "wasn't", "aren't"]
INTENSIFIERS = {
    "very": 1.2,
    "extremely": 1.4,
    "slightly": 0.8,
    "somewhat": 0.9
}


def _split_sentences(text: str):
    return re.split(r'[.,;]| but | however ', text.lower())

def _score_dimension(text: str, positives: list, negatives: list) -> float:
    sentences = _split_sentences(text)

    total_score = 0
    count = 0

    for sentence in sentences:
        pos_count = sum(1 for p in positives if p in sentence)
        neg_count = sum(1 for n in negatives if n in sentence)

        if pos_count == 0 and neg_count == 0:
            continue

        total = pos_count + neg_count
        raw = pos_count / total if total > 0 else 0.5

        score = 0.3 + (raw * 0.6)

        # 🔥 NEGATION HANDLING
        if any(neg in sentence for neg in NEGATIONS):
            score = 1 - score
            score = (score + 0.5) / 2   # soften inversion

        # 🔥 INTENSIFIER HANDLING
        for word, factor in INTENSIFIERS.items():
            if word in sentence:
                score = (score - 0.5) * factor + 0.5

        total_score += score
        count += 1

    if count == 0:
        return 0.5

    return round(float(total_score / count), 4)
def _sentiment_score(text: str) -> float:
    text_lower = text.lower()
    positive_words = ["great", "excellent", "beautiful", "fantastic", "perfect",
        "love", "amazing", "stunning", "wonderful", "superb", "good",
        "nice", "well done", "impressive", "sharp", "clean"]
    negative_words = ["terrible", "awful", "bad", "poor", "ugly", "boring",
        "disappointing", "wrong", "horrible", "worst", "failed",
        "problem", "issue", "unfortunately", "unfortunately"]
    pos = sum(1 for w in positive_words if w in text_lower)
    neg = sum(1 for w in negative_words if w in text_lower)
    if pos + neg == 0:
        return 0.5
        
    if any(neg in text_lower for neg in NEGATIONS):
        return round(float(1 - (pos / (pos + neg + 1e-6))), 4)
        
    return round(float(pos / (pos + neg)), 4)


CACHE_FILE = "cache/llm_cache.json"

def _load_cache() -> dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def _save_cache(cache: dict):
    os.makedirs("cache", exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def _cache_key(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def configure_gemini(api_key: str):
    """Keep this for compatibility — model used only as fallback now."""
    if api_key and api_key != "DUMMY":
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
        except:
            pass


def extract_aesthetic_features(critique_text: str, model=None) -> dict:
    """
    Extract aesthetic features using rule-based NLP.
    Zero API calls. Falls back to Gemini only if model provided
    AND rule-based returns mostly neutral scores.
    """
    if not critique_text or critique_text.strip() == "":
        return {
            "composition": 0.5, "lighting": 0.5, "color": 0.5,
            "subject_clarity": 0.5, "technical": 0.5,
            "summary": "No critique available."
        }

    # Check cache first
    cache = _load_cache()
    key = _cache_key(critique_text)
    if key in cache:
        return cache[key]

    # Rule-based extraction — zero API calls
    composition     = _score_dimension(critique_text, COMPOSITION_POSITIVE, COMPOSITION_NEGATIVE)
    lighting        = _score_dimension(critique_text, LIGHTING_POSITIVE,    LIGHTING_NEGATIVE)
    color           = _score_dimension(critique_text, COLOR_POSITIVE,       COLOR_NEGATIVE)
    subject_clarity = _score_dimension(critique_text, CLARITY_POSITIVE,     CLARITY_NEGATIVE)
    technical       = _score_dimension(critique_text, TECHNICAL_POSITIVE,   TECHNICAL_NEGATIVE)

    # 🔥 NEW: confidence based on whether feature deviates from neutral
    confidence = {
        key: abs(val - 0.5) * 2   # continuous confidence
        for key, val in {
            "composition": composition,
            "lighting": lighting,
            "color": color,
            "subject_clarity": subject_clarity,
            "technical": technical
        }.items()
    }
    # Check if everything is neutral (nothing was found)
    scores = [composition, lighting, color, subject_clarity, technical]
    all_neutral = all(s == 0.5 for s in scores)

    # If all neutral AND we have a model, use LLM as fallback
    if all_neutral and model is not None:
        try:
            import google.generativeai as genai
            prompt = f"""Extract aesthetic scores from this photo critique as JSON only.
            Critique: "{critique_text}"
            Return only: {{"composition":0.0-1.0,"lighting":0.0-1.0,"color":0.0-1.0,"subject_clarity":0.0-1.0,"technical":0.0-1.0,"summary":"one sentence"}}"""
            response = model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'```$', '', raw).strip()
            result = json.loads(raw)
            for k in ["composition","lighting","color","subject_clarity","technical"]:
                result[k] = float(max(0.01, min(0.99, result.get(k, 0.5))))
            cache[key] = result
            _save_cache(cache)
            return result
        except Exception as e:
            pass  # Fall through to rule-based result

    # Generate summary from sentiment
    overall = _sentiment_score(critique_text)
    tone = "positive" if overall > 0.6 else ("negative" if overall < 0.4 else "mixed")
    summary = f"Rule-based analysis: {tone} critique. Composition={composition:.2f}, Lighting={lighting:.2f}."

    result = {
        "composition":     composition,
        "lighting":        lighting,
        "color":           color,
        "subject_clarity": subject_clarity,
        "technical":       technical,
        "summary":         summary,
        "confidence":      confidence   
    }

    cache[key] = result
    _save_cache(cache)
    return result


if __name__ == "__main__":
    print("Testing rule-based extractor (zero API calls)...\n")
    tests = [
        "The composition is excellent, rule of thirds applied perfectly. Lighting is natural and flattering. Colors are vibrant.",
        "This is terrible. The horizon is crooked, subject is blurry, and the exposure is blown out.",
        "Nice shot overall! Subject is sharp and well placed. Background is a little busy but lighting is beautiful.",
    ]
    for i, t in enumerate(tests):
        r = extract_aesthetic_features(t)
        print(f"Test {i+1}: {t[:55]}...")
        for k in ["composition","lighting","color","subject_clarity","technical"]:
            bar = "█" * int(r[k] * 20)
            print(f"  {k:<18} {r[k]:.2f}  {bar}")
        print(f"  Summary: {r['summary']}\n")
