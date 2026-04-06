"""
CLIP Zero-Shot Aesthetic Feature Extractor
──────────────────────────────────────────
Uses CLIP ViT-B/32 to compute semantically meaningful aesthetic scores
for 5 dimensions: composition, lighting, color, subject_clarity, technical.

Architecture:
  1. Image → vision_model → visual_projection → normalized image embedding
  2. Text prompts → text_model → text_projection → normalized text embeddings
  3. Cosine similarity (image vs positive/negative prompts) → softmax → score

WHY THIS WORKS:
  CLIP's contrastive pre-training aligns images and text in a shared 512-dim
  embedding space. By comparing an image embedding against descriptive
  positive/negative text prompts for each aesthetic dimension, we get a
  probability score representing how well the image matches the positive
  description versus the negative one.

  We use the model's internal components directly (NOT get_image_features):
    - vision_model      → extracts visual features; pooler_output = CLS token
    - visual_projection → projects into shared embedding space (512-dim)
    - text_model        → extracts text features; pooler_output = EOS token
    - text_projection   → projects into shared embedding space (512-dim)
    - logit_scale       → learned temperature for cosine similarity calibration

  Text embeddings are computed ONCE and cached since prompts are constant.
  This makes subsequent image evaluations extremely fast.
"""

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os

# ── Module-level singletons (lazy-loaded) ────────────────────────
_processor = None
_model = None
_text_embeds_cache = None  # Computed once, reused for every image


# ── Zero-shot aesthetic prompts ──────────────────────────────────
# Each dimension has a (positive, negative) prompt pair.
# CLIP compares the image against both and returns a probability.
# Prompts are crafted to be descriptive and complementary.
AESTHETIC_PROMPTS = {
    "composition": (
        "a photograph with excellent composition, balanced framing, "
        "leading lines, and rule of thirds",
        "a photograph with poor composition, bad framing, cluttered "
        "layout, and distracting background"
    ),
    "lighting": (
        "a photograph with beautiful natural lighting, golden hour, "
        "well exposed, soft light, balanced exposure",
        "a photograph with harsh lighting, underexposed, overexposed, "
        "dark shadows, blown highlights, flat lighting"
    ),
    "color": (
        "a photograph with vibrant harmonious colors, rich color "
        "palette, warm tones, good white balance",
        "a photograph with washed out dull colors, poor white balance, "
        "color cast, oversaturated garish colors"
    ),
    "subject_clarity": (
        "a photograph with tack sharp focus, clear subject, crisp "
        "details, excellent depth of field, subject pops",
        "a photograph that is blurry, out of focus, soft, motion blur, "
        "camera shake, missed focus"
    ),
    "technical": (
        "a photograph with excellent technical quality, clean image, "
        "no noise, sharp throughout, well processed",
        "a photograph with poor technical quality, noisy, grainy, "
        "pixelated, artifacts, overprocessed, low quality"
    ),
}


def _load_model():
    """Lazy-load CLIP model and processor. Downloads ~350MB on first call."""
    global _processor, _model
    if _processor is None or _model is None:
        print("[CLIP] Loading openai/clip-vit-base-patch32...")
        _processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        _model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        _model.eval()
        print("[CLIP] Model loaded successfully.")


def _get_text_embeddings():
    """
    Compute and cache normalized text embeddings for all aesthetic prompts.
    Only computed once since prompts never change.

    Returns:
        dict: {dimension_name: Tensor of shape [2, 512]}
              Index 0 = positive prompt embedding
              Index 1 = negative prompt embedding
    """
    global _text_embeds_cache
    if _text_embeds_cache is not None:
        return _text_embeds_cache

    _text_embeds_cache = {}

    with torch.no_grad():
        for dim, (pos_prompt, neg_prompt) in AESTHETIC_PROMPTS.items():
            # Tokenize both prompts together as a batch
            text_inputs = _processor(
                text=[pos_prompt, neg_prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Forward: text_model → text_projection (into shared space)
            text_outputs = _model.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            text_embeds = _model.text_projection(text_outputs.pooler_output)

            # L2 normalize to unit sphere for cosine similarity
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            _text_embeds_cache[dim] = text_embeds  # Shape: [2, 512]

    print("[CLIP] Text embeddings cached for 5 aesthetic dimensions.")
    return _text_embeds_cache


def extract_image_features(image_path: str) -> dict:
    """
    Extract 5 aesthetic feature scores from an image using CLIP zero-shot
    classification against positive/negative aesthetic prompts.

    For each dimension:
      score = softmax(cos_sim(image, positive), cos_sim(image, negative))[0]

    Args:
        image_path: Path to the image file.

    Returns:
        dict with keys: composition, lighting, color,
        subject_clarity, technical — each a float in [0, 1].
        Higher = better aesthetic quality for that dimension.
    """
    default_scores = {
        "composition": 0.5,
        "lighting": 0.5,
        "color": 0.5,
        "subject_clarity": 0.5,
        "technical": 0.5,
    }

    try:
        if not image_path or not os.path.exists(image_path):
            print("[CLIP] Image path invalid or missing:", image_path)
            return default_scores

        _load_model()

        # ── 1. Load and preprocess image ─────────────────────────
        image = Image.open(image_path).convert("RGB")
        image_inputs = _processor(images=image, return_tensors="pt")

        # ── 2. Get image embedding: vision_model → visual_projection
        with torch.no_grad():
            vision_outputs = _model.vision_model(
                pixel_values=image_inputs["pixel_values"]
            )
            image_embeds = _model.visual_projection(
                vision_outputs.pooler_output
            )
            # L2 normalize to unit sphere
            image_embeds = image_embeds / image_embeds.norm(
                dim=-1, keepdim=True
            )

            # Learned temperature from contrastive pre-training
            logit_scale = _model.logit_scale.exp()

        # ── 3. Get cached text embeddings ────────────────────────
        text_embeddings = _get_text_embeddings()

        # ── 4. Compute zero-shot score for each dimension ────────
        scores = {}
        with torch.no_grad():
            for dim, text_embeds in text_embeddings.items():
                # Cosine similarity (already normalized) scaled by temperature
                # image_embeds: [1, 512]  ×  text_embeds.T: [512, 2]
                similarities = (image_embeds @ text_embeds.T) * logit_scale

                # Softmax over [positive_sim, negative_sim]
                probs = F.softmax(similarities.squeeze(0), dim=0)

                # Score = probability of matching the POSITIVE prompt
                scores[dim] = round(float(probs[0].item()), 4)

        print(f"[CLIP] Extracted: comp={scores['composition']:.3f}  "
              f"light={scores['lighting']:.3f}  "
              f"color={scores['color']:.3f}  "
              f"clarity={scores['subject_clarity']:.3f}  "
              f"tech={scores['technical']:.3f}")
        return scores

    except Exception as e:
        print(f"[IMAGE MODEL ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return default_scores


# ── Quick self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

    if os.path.exists(test_path):
        print(f"\nTesting CLIP aesthetic extraction on: {test_path}\n")
        result = extract_image_features(test_path)

        print("\n" + "=" * 55)
        print("  CLIP Zero-Shot Aesthetic Scores")
        print("=" * 55)
        for k, v in result.items():
            bar = "█" * int(v * 30)
            print(f"  {k:<18} {v:.4f}  {bar}")
        print("=" * 55)
    else:
        print(f"Image not found: {test_path}")