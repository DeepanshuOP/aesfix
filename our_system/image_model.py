import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# Lazy loading
processor = None
clip_model = None

def load_model():
    global processor, clip_model
    if processor is None:
        print("  [Image Model] Loading CLIP model...")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

# Prompts for each aesthetic dimension
PROMPTS = {
    "composition": (
        "a well composed professional photograph",
        "a poorly composed photograph"
    ),
    "lighting": (
        "a well lit photograph with good exposure",
        "a poorly lit dark or overexposed photograph"
    ),
    "color": (
        "a vibrant colorful photograph",
        "a dull washed out photograph"
    ),
    "subject_clarity": (
        "a sharp focused subject",
        "a blurry unfocused subject"
    ),
    "technical": (
        "a high quality clean photograph",
        "a noisy distorted low quality photograph"
    )
}

def score_dimension(image_embed, positive_text, negative_text):
    """Compute score using CLIP similarity"""
    inputs = processor(
        text=[positive_text, negative_text],
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        text_embeds = clip_model.get_text_features(**inputs)

    # Normalize
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # Similarity
    similarity = torch.matmul(image_embed, text_embeds.T)

    # Convert to probability
    probs = similarity.softmax(dim=-1)

    return probs[0][0].item()  # probability of positive

def extract_image_features(image_path: str) -> dict:
    if not image_path or not os.path.exists(image_path):
        return {
            "composition": 0.5, "lighting": 0.5, "color": 0.5,
            "subject_clarity": 0.5, "technical": 0.5,
            "summary": "Skipped (Image not found)."
        }

    try:
        load_model()

        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_embed = outputs.image_embeds

        scores = {}

        for dim, (pos, neg) in PROMPTS.items():
            score = score_dimension(image_embed, pos, neg)
            scores[dim] = round(float(score), 4)

        scores["summary"] = f"CLIP-based aesthetic analysis of {os.path.basename(image_path)} complete."

        return scores

    except Exception as e:
        print(f"  [Image Model Error] {e}")
        return {
            "composition": 0.5, "lighting": 0.5, "color": 0.5,
            "subject_clarity": 0.5, "technical": 0.5,
            "summary": f"Analytical failure: {str(e)[:50]}"
        }