import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

_processor = None
_model = None

def _load_model():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()

def extract_image_features(image_path: str) -> dict:
    default_scores = {
        "composition": 0.5,
        "lighting": 0.5,
        "color": 0.5,
        "subject_clarity": 0.5,
        "technical": 0.5
    }

    try:
        if not image_path or not os.path.exists(image_path):
            return default_scores

        _load_model()
        
        image = Image.open(image_path).convert("RGB")
        inputs = _processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            features = _model.get_image_features(pixel_values=inputs["pixel_values"])

            if not isinstance(features, torch.Tensor):
                features = features[0]

            features = features.detach()
            features = features / features.norm(dim=-1, keepdim=True)
            first_vector = features[0][:5]
            scores = torch.sigmoid(first_vector).flatten().tolist()
            
            return {
                "composition": float(scores[0]),
                "lighting": float(scores[1]),
                "color": float(scores[2]),
                "subject_clarity": float(scores[3]),
                "technical": float(scores[4])
            }
            
    except Exception as e:
        print("[IMAGE MODEL ERROR]:", e)
        return default_scores