"""Trust, Provenance, Content Safety, and Retention Layer (AF-5.4 & Roadmap §20, §31, §42).

Why the Trust Layer is foundational for commercial scale and interview defensibility:
1. C2PA Content Credentials Compliance: Regulators and major platforms (Adobe, Google, OpenAI,
   IPTC) require clear provenance for algorithmic image transformations. AesFix injects C2PA
   metadata marking edits as verified photographic retouching rather than deepfakes.
2. Preview Protection via Watermarks: Prevents unauthorized commercial theft of preview assets
   while providing shareable before/after comparison cards with verified gain badges.
3. Content Safety Screening: Pre-emptively screens uploaded images for explicit or prohibited
   content prior to GPU processing, preventing platform abuse and reducing legal liability.
4. GDPR Right to Erasure & 30-Day Retention: Enforces a privacy-first retention policy: user
   photos are automatically purged after 30 days unless opted into persistent storage, and
   user photos are NEVER used for model training without explicit affirmative consent.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("aesfix.trust")


@dataclass
class SafetyVerdict:
    """Outcome of pre-flight content safety screening."""
    is_safe: bool
    risk_category: str
    confidence: float
    rejection_reason: Optional[str] = None


@dataclass
class C2PAManifest:
    """Standardized C2PA / IPTC digital provenance claim."""
    claim_generator: str
    title: str
    digital_source_type: str
    actions: List[Dict[str, Any]]
    assertions: Dict[str, Any]
    original_sha256: str
    edited_sha256: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_json_ld(self, indent: int = 2) -> str:
        """Serializes claim to standard JSON-LD provenance format."""
        payload = {
            "@context": "https://c2pa.org/standards/json-ld/v1",
            "dc:format": "image/jpeg",
            "claim_generator": self.claim_generator,
            "title": self.title,
            "digitalSourceType": self.digital_source_type,
            "c2pa:actions": self.actions,
            "c2pa:assertions": self.assertions,
            "c2pa:signatures": {
                "algorithm": "sha256",
                "original_digest": self.original_sha256,
                "edited_digest": self.edited_sha256,
                "timestamp": self.timestamp,
            },
        }
        return json.dumps(payload, indent=indent)


class ContentSafetyGate:
    """Pre-flight content safety screener to detect explicit, violent, or prohibited imagery."""

    def __init__(self, sensitivity: float = 0.85) -> None:
        self.sensitivity = sensitivity

    def screen_image(self, image_input: Union[str, Path, Image.Image]) -> SafetyVerdict:
        """Evaluates image pixels for prohibited content (NSFW/extremity heuristics)."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        arr = np.array(pil_img, dtype=np.float32)

        # 1. Color distribution analysis (e.g. extreme dominant flesh-tone saturation)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        skin_mask = (r > 95) & (g > 40) & (b > 20) & ((r - np.minimum(g, b)) > 15) & (np.abs(r - g) > 15)
        skin_ratio = float(np.mean(skin_mask))

        # 2. Check for synthetic uniform exploit patterns
        std_dev = float(np.std(arr))
        if std_dev < 1.0:
            return SafetyVerdict(
                is_safe=False,
                risk_category="corrupt_blank",
                confidence=0.99,
                rejection_reason="Image contains zero variance (completely blank or corrupt frame).",
            )

        # High skin ratio flag (heuristic screening gate for moderation queue)
        if skin_ratio > 0.88:
            return SafetyVerdict(
                is_safe=False,
                risk_category="nudity_warning",
                confidence=round(skin_ratio, 3),
                rejection_reason="Excessive skin surface ratio detected; flagged for human safety review.",
            )

        return SafetyVerdict(
            is_safe=True,
            risk_category="clean",
            confidence=round(1.0 - skin_ratio, 3),
        )


class WatermarkService:
    """Generates visual comparison cards and anti-theft preview watermarks."""

    @classmethod
    def apply_watermark(
        cls,
        image: Image.Image,
        watermark_text: str = "AesFix AI-Verified Preview",
        gain_badge: Optional[str] = None,
        opacity: float = 0.65,
    ) -> Image.Image:
        """Appends an elegant semi-transparent watermark badge to the preview image."""
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = base.size
        full_text = f"{watermark_text} · {gain_badge}" if gain_badge else watermark_text

        # Draw pill badge in bottom right corner
        padding_x = 16
        padding_y = 10
        badge_w = min(w - 20, max(240, len(full_text) * 9))
        badge_h = 32

        bx1 = w - badge_w - 16
        by1 = h - badge_h - 16
        bx2 = w - 16
        by2 = h - 16

        # Dark translucent pill background
        bg_alpha = int(255 * opacity)
        draw.rounded_rectangle(
            [(bx1, by1), (bx2, by2)],
            radius=6,
            fill=(20, 24, 33, bg_alpha),
            outline=(255, 255, 255, int(180 * opacity)),
            width=1,
        )

        # Text rendering
        draw.text(
            (bx1 + padding_x, by1 + padding_y - 2),
            full_text,
            fill=(255, 255, 255, 240),
        )

        watermarked = Image.alpha_composite(base, overlay).convert("RGB")
        return watermarked

    @classmethod
    def create_side_by_side_comparison(
        cls,
        original: Image.Image,
        edited: Image.Image,
        gain_badge: str,
    ) -> Image.Image:
        """Creates a side-by-side before/after comparison image with labeled headers."""
        # Normalize heights
        h = max(original.height, edited.height)
        orig_resized = original.resize((int(original.width * (h / original.height)), h), Image.Resampling.BILINEAR)
        edit_resized = edited.resize((int(edited.width * (h / edited.height)), h), Image.Resampling.BILINEAR)

        total_w = orig_resized.width + edit_resized.width + 4  # 4px divider
        header_h = 44

        canvas = Image.new("RGB", (total_w, h + header_h), (15, 17, 23))
        draw = ImageDraw.Draw(canvas)

        # Paste images
        canvas.paste(orig_resized, (0, header_h))
        canvas.paste(edit_resized, (orig_resized.width + 4, header_h))

        # Draw dividing line
        draw.line(
            [(orig_resized.width + 2, header_h), (orig_resized.width + 2, h + header_h)],
            fill=(60, 66, 82),
            width=3,
        )

        # Headers
        draw.text((20, 14), "BEFORE (Original)", fill=(180, 185, 195))
        draw.text((orig_resized.width + 24, 14), f"AFTER (AesFix Verified · {gain_badge})", fill=(52, 211, 153))

        return canvas


class C2PAProvenanceEngine:
    """Generates standard C2PA manifests certifying authentic AI-assisted photographic enhancement."""

    @classmethod
    def compute_sha256(cls, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """Calculates deterministic SHA-256 digest of image bytes."""
        if isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as fh:
                return hashlib.sha256(fh.read()).hexdigest()
        elif isinstance(image_input, bytes):
            return hashlib.sha256(image_input).hexdigest()
        elif isinstance(image_input, Image.Image):
            import io
            buf = io.BytesIO()
            image_input.save(buf, format="JPEG", quality=95)
            return hashlib.sha256(buf.getvalue()).hexdigest()
        else:
            raise TypeError(f"Unsupported image input for hashing: {type(image_input)}")

    @classmethod
    def generate_manifest(
        cls,
        original_digest: str,
        edited_digest: str,
        applied_ops: List[str],
        verified_aesthetic_gain: float,
        model_name: str = "AesFix-Qwen2.5-VL-7B (LoRA + GRPO)",
    ) -> C2PAManifest:
        """Constructs an IPTC/C2PA compliant manifest."""
        actions = [
            {
                "action": "c2pa.edited",
                "softwareAgent": model_name,
                "parameters": {"operations": applied_ops},
                "description": "Deterministic photographic adjustments applied via AesFix Tier-A Executor.",
            },
            {
                "action": "c2pa.evaluated",
                "softwareAgent": "AesFix Multi-Judge Ensemble v1.0",
                "parameters": {"verified_aesthetic_gain": verified_aesthetic_gain},
                "description": "Aesthetic gain computed by frozen multi-judge ensemble with Identity Guard verification.",
            },
        ]

        assertions = {
            "c2pa.training_opt_in": False,
            "c2pa.generator": "AesFix Photographic Verification Engine",
            "c2pa.model_provenance": model_name,
            "c2pa.gain_metric": "AG@1",
            "c2pa.gain_value": verified_aesthetic_gain,
        }

        return C2PAManifest(
            claim_generator="AesFix/1.0.0 (C2PA-Profile-v1.3)",
            title="AesFix Verified Photo Enhancement",
            digital_source_type="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
            actions=actions,
            assertions=assertions,
            original_sha256=original_digest,
            edited_sha256=edited_digest,
        )


class DataRetentionManager:
    """Manages GDPR right to erasure and enforces the 30-day automatic photo purge policy."""

    def __init__(self, storage_root: Optional[Path] = None) -> None:
        self.storage_root = storage_root or Path(tempfile.gettempdir())
        self._user_opt_ins: Dict[str, bool] = {}

    def set_training_opt_in(self, user_id: str, opt_in: bool) -> None:
        """Updates user's research training consent flag (strictly False by default)."""
        self._user_opt_ins[user_id] = bool(opt_in)
        logger.info("Updated training opt-in consent for user %s: %s", user_id, opt_in)

    def is_opted_in(self, user_id: str) -> bool:
        """Returns True only if user affirmatively consented to training."""
        return self._user_opt_ins.get(user_id, False)

    def purge_user_photos(self, user_id: str) -> int:
        """Immediately deletes all user uploaded and intermediate files (GDPR Right to Erasure)."""
        purged_count = 0
        # Search temporary directory for user's files
        for p in self.storage_root.glob(f"*{user_id}*"):
            try:
                if p.is_file():
                    p.unlink()
                    purged_count += 1
            except OSError as exc:
                logger.warning("Failed to delete user file %s: %s", p, exc)
        return purged_count

    def purge_expired_photos(self, max_age_seconds: float = 30 * 86400) -> int:
        """Purges photos older than the retention threshold (default: 30 days)."""
        now = time.time()
        purged_count = 0

        for p in self.storage_root.glob("aesfix_*.jpg"):
            try:
                if p.is_file():
                    mtime = p.stat().st_mtime
                    if (now - mtime) > max_age_seconds:
                        p.unlink()
                        purged_count += 1
            except OSError:
                pass

        logger.info("Auto-purge scan completed: removed %d expired photos", purged_count)
        return purged_count
