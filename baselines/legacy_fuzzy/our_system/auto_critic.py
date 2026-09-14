from PIL import Image, ImageStat, ImageFilter
import numpy as np

def generate_auto_critique(image_path):
    img = Image.open(image_path).convert("RGB")

    # --- Brightness ---
    stat = ImageStat.Stat(img)
    brightness = sum(stat.mean) / 3

    # --- Sharpness ---
    sharpness = np.array(img.filter(ImageFilter.FIND_EDGES)).mean()

    # --- Contrast ---
    contrast = np.std(np.array(img))

    critique = []

    # Brightness rules
    if brightness < 80:
        critique.append("The image is too dark")
    elif brightness > 180:
        critique.append("The image is too bright")
    else:
        critique.append("The brightness is good")

    # Sharpness rules
    if sharpness < 10:
        critique.append("The image is blurry")
    else:
        critique.append("The image is sharp")

    # Contrast rules
    if contrast < 40:
        critique.append("The contrast is low")
    else:
        critique.append("The contrast is good")

    return ". ".join(critique)