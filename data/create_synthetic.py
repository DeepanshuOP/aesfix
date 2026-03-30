import json
import random
import math

random.seed(42)

# Structured critique pool with known aesthetic quality tiers
# Each critique has an associated ground truth quality level
critique_pool = {
    "high": [
        "The composition is superb — rule of thirds executed perfectly with the subject placed exactly at the intersection. Golden hour lighting creates beautiful rim light and the colors are rich without being oversaturated.",
        "Technically flawless. The depth of field is perfectly calibrated, bokeh is smooth and creamy, and the subject isolation is excellent. Skin tones are accurate and the catchlights in the eyes bring the portrait to life.",
        "Incredible use of leading lines drawing the eye directly to the subject. The exposure is spot on, shadow detail is preserved and highlights are not blown. Color harmony is natural and pleasing.",
        "The framing is inspired — negative space used intentionally to create tension. Lighting is dramatic and directional, giving strong three-dimensional depth. Technically sharp where it needs to be.",
        "Beautiful environmental portrait. The subject is sharp, background separation is perfect, and the ambient light is warm and flattering. The moment captured is genuine and emotionally resonant.",
        "Strong diagonal composition creates dynamic energy. The color palette is cohesive and the tonal range is excellent — deep blacks, clean whites, and well-preserved midtones throughout.",
        "The symmetry here is used masterfully. Reflections are perfectly aligned, exposure is balanced across the frame, and the color grading enhances without overpowering the natural scene.",
        "Exceptional macro work. The focus plane is exactly right, background falls off beautifully, and the natural lighting reveals stunning texture detail in the subject.",
    ],
    "medium": [
        "The composition is decent — rule of thirds is attempted but the subject placement feels slightly off. Lighting is adequate but flat, lacking the drama that would elevate this shot. Colors are natural.",
        "Good subject but the background is too busy and competes for attention. Exposure is correct and sharpness is acceptable. Could benefit from a wider aperture to separate subject from background.",
        "Nice moment captured but the framing is slightly tight on the left edge. Lighting is fine — overcast which gives even illumination but lacks character. Post-processing is clean and not overdone.",
        "The idea is strong but execution is inconsistent. Composition follows the rule of thirds loosely. Lighting is mixed — partly harsh direct sun creating unflattering shadows under the eyes.",
        "Reasonable portrait. The subject is reasonably sharp, lighting is from a window which is pleasant but the catchlights are at an odd angle. Background is slightly distracting but not terrible.",
        "The landscape framing is competent. Horizon is straight, foreground interest is present. Colors feel slightly washed out — could benefit from a small boost in contrast and clarity.",
        "Decent street photography. The moment is interesting but the composition could be tighter. Lighting is flat midday sun. The crop cuts the subject awkwardly at the knee.",
        "Good technical base — exposure and focus are correct. The composition feels generic though, placing the subject dead center without clear intent. More thoughtful framing would improve this significantly.",
    ],
    "low": [
        "The horizon is severely crooked which is immediately distracting. The exposure is blown out on the right third of the frame losing all highlight detail. Composition has no clear focal point.",
        "Very blurry — camera shake is evident throughout. The subject is completely unrecognizable. Exposure is underexposed by at least two stops making the image muddy and noisy.",
        "Poor framing with the subject's head cut off at the top. The background is chaotic with no separation from the subject. Harsh direct flash creates flat unflattering light and red-eye.",
        "The composition is confusing — it's unclear what the intended subject is. Everything is in focus which means nothing stands out. The color balance is very off, everything has a strong green cast.",
        "Heavily over-processed to the point where it looks artificial. The HDR effect is extreme — halos around every edge, colors are garish and oversaturated. The original scene must have been interesting but this processing ruins it.",
        "Significant motion blur on the subject makes this unusable. The background is sharp while the subject is blurred — the opposite of what's intended. Exposure compensation needed here.",
        "The image is severely underexposed — the subject is barely visible against the dark background. No detail can be recovered from shadows this deep. Needs flash or higher ISO.",
        "Framing cuts the subject at every wrong point — elbows and knees clipped. The background lamppost appears to grow from the subject's head. These are basic compositional errors.",
    ]
}

def generate_record(index, quality_tier, sentiment_range):
    tier_critiques = critique_pool[quality_tier]
    n_comments = random.randint(1, 4)
    
    # Sometimes mix in a critique from adjacent tier (simulates real disagreement)
    comments = []
    for _ in range(n_comments):
        if random.random() < 0.15 and quality_tier != "medium":
            # 15% chance of a dissenting opinion from adjacent tier
            adjacent = "medium"
            comments.append(random.choice(critique_pool[adjacent]))
        else:
            comments.append(random.choice(tier_critiques))
    
    sentiment = round(random.uniform(*sentiment_range), 4)
    
    return {
        "id": f"post_{index:04d}",
        "image": f"data/images/{quality_tier}/image_{index:04d}.jpg",
        "comments": comments,
        "sentiment_score": sentiment,
        "ground_truth_quality": quality_tier
    }

# Generate balanced dataset: equal thirds across quality tiers
records = []
idx = 0

# 200 total: ~67 high, ~67 medium, ~66 low
tiers = (
    [("high",   (0.70, 0.98))] * 67 +
    [("medium", (0.40, 0.69))] * 67 +
    [("low",    (0.05, 0.39))] * 66
)
random.shuffle(tiers)

for quality_tier, sentiment_range in tiers:
    records.append(generate_record(idx, quality_tier, sentiment_range))
    idx += 1

# Split 70/15/15
train      = records[:140]
validation = records[140:170]
test       = records[170:]

for split, data in [("train", train), ("validation", validation), ("test", test)]:
    path = f"data/processed_info_{split}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Created {path} — {len(data)} records")

# Print stats
print("\nDataset statistics:")
for split, data in [("train", train), ("validation", validation), ("test", test)]:
    high   = sum(1 for r in data if r["ground_truth_quality"] == "high")
    medium = sum(1 for r in data if r["ground_truth_quality"] == "medium")
    low    = sum(1 for r in data if r["ground_truth_quality"] == "low")
    print(f"  {split:12s}: HIGH={high}  MEDIUM={medium}  LOW={low}")

print("\nDone. Synthetic RPCD-style dataset created.")
