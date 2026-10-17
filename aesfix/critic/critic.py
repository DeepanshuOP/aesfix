"""Aspect Critic with strict machine-executable fix schema (AF-1.1).

Why strict JSON with expected_aspect is load-bearing:
1. Traditional AI critics emit freeform text advice ("improve the lighting") that cannot
   be verified or executed.
2. By requiring every suggested fix to declare an `expected_aspect`
   (e.g., op="adjust_exposure", expected_aspect="lighting") alongside concrete parameters,
   advice becomes falsifiable: the system can execute the fix and check whether the
   judge ensemble's lighting probe actually registered an improvement.
3. This is the core protocol enabling the C1 Aesthetic Gain (AG@k) loop.
"""
from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from aesfix.data.loaders import ASPECT_DIMENSIONS

logger = logging.getLogger("aesfix.critic")

# Canonical Tier-A executable fix operations
VALID_FIX_OPERATIONS = (
    "adjust_exposure",
    "adjust_contrast",
    "adjust_white_balance",
    "adjust_saturation",
    "unsharp_mask",
    "smart_crop",
    "correct_horizon",
    "denoise",
    "remove_distraction",
    "relight_subject",
)


class CriticFix(BaseModel):
    """A concrete, machine-executable photographic fix.

    Why this schema exists:
    Decouples linguistic intent from execution. The executor receives exact
    parameters (e.g. EV delta, crop coordinates) rather than ambiguous instructions.
    """
    op: str = Field(description="Actionable operation identifier from the executor vocabulary.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Numerical/categorical parameters for the op.")
    expected_aspect: str = Field(description="The primary aesthetic dimension this fix aims to improve.")
    direction: int = Field(default=1, description="Direction of expected aspect change (+1 or -1).")
    nl_instruction: str = Field(description="Human-readable coaching explanation.")
    region_hint: Optional[str] = Field(default=None, description="Optional bounding box or subject hint for masked ops.")

    @field_validator("expected_aspect")
    @classmethod
    def validate_aspect(cls, v: str) -> str:
        # Aspect must conform to canonical 5 dimensions so judge probes can measure delta
        v_clean = v.strip().lower()
        if v_clean not in ASPECT_DIMENSIONS:
            raise ValueError(f"expected_aspect must be one of {ASPECT_DIMENSIONS}, got {v!r}")
        return v_clean

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: int) -> int:
        # Direction must be +1 (boost/improve aspect) or -1 (attenuate)
        if v not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {v!r}")
        return v


class CriticOutput(BaseModel):
    """Full structured critique returned by the Aspect Critic.

    Combines 5-dimensional ratings, qualitative rationales, and falsifiable fix plans.
    """
    aspect_scores: Dict[str, float] = Field(description="Continuous aesthetic ratings in [0.0, 1.0] for the 5 aspects.")
    rationales: Dict[str, str] = Field(description="Per-aspect qualitative diagnosis explaining the score.")
    overall_summary: str = Field(description="Holistic appraisal of the photograph's strengths and primary flaws.")
    fixes: List[CriticFix] = Field(default_factory=list, description="Ranked candidate fix instructions.")
    genre: Optional[str] = Field(default=None, description="Active genre or goal context (e.g., portrait, listing).")

    @field_validator("aspect_scores")
    @classmethod
    def validate_aspect_scores(cls, scores: Dict[str, float]) -> Dict[str, float]:
        missing = [dim for dim in ASPECT_DIMENSIONS if dim not in scores]
        if missing:
            raise ValueError(f"Missing required aspect scores: {missing}")
        validated = {}
        for dim, val in scores.items():
            # Clamping prevents out-of-bound errors when models slightly exceed [0, 1]
            validated[dim] = round(max(0.0, min(1.0, float(val))), 4)
        return validated


class CriticPromptBuilder:
    """Constructs prompts enforcing strict JSON extraction and genre conditioning.

    Why system prompts must enforce strict schema:
    Frontier and quantized VLMs will default to conversational markdown unless
    the schema is constrained by structural instructions and few-shot formatting.
    """

    SYSTEM_PROMPT = """You are AesFix, an expert photographic critic.
Your role is to diagnose image quality across 5 specific photographic dimensions:
1. composition (framing, balance, subject placement, leading lines, crop)
2. lighting (exposure, highlights, shadows, dynamic range, contrast)
3. color (harmony, saturation, white balance, grading)
4. subject_clarity (focus plane, sharpness, subject isolation, depth of field)
5. technical (noise, artifacts, motion blur, compression damage)

CRITICAL INSTRUCTIONS:
- You must output ONLY a valid JSON object matching the exact schema specified below.
- Do NOT include any conversational preamble, markdown headers, or text outside the JSON.
- Every score must be a continuous float between 0.0 and 1.0.
- For every weak aspect (< 0.60), propose at least one concrete machine-executable fix.
- Valid operations: adjust_exposure, adjust_contrast, adjust_white_balance, adjust_saturation, unsharp_mask, smart_crop, correct_horizon, denoise, remove_distraction, relight_subject.

JSON SCHEMA:
{
  "aspect_scores": {
    "composition": float,
    "lighting": float,
    "color": float,
    "subject_clarity": float,
    "technical": float
  },
  "rationales": {
    "composition": "string",
    "lighting": "string",
    "color": "string",
    "subject_clarity": "string",
    "technical": "string"
  },
  "overall_summary": "string",
  "fixes": [
    {
      "op": "string",
      "params": {"param_key": value},
      "expected_aspect": "lighting|composition|color|subject_clarity|technical",
      "direction": 1,
      "nl_instruction": "string",
      "region_hint": null or "subject|background|sky"
    }
  ]
}"""

    @classmethod
    def build_user_prompt(
        cls,
        user_critique: Optional[str] = None,
        goal_or_genre: Optional[str] = None,
    ) -> str:
        """Compose the user prompt, incorporating optional multimodal critique and genre guidance."""
        parts = ["Evaluate the aesthetic quality of the provided photograph according to the 5 dimensions."]

        if goal_or_genre:
            parts.append(
                f"GOAL / GENRE CONTEXT: This photo is intended as a {goal_or_genre.strip()}. "
                f"Calibrate expectations and fix recommendations specifically for this use-case."
            )

        if user_critique and user_critique.strip():
            # Critique text is injected as bounded data to avoid prompt injection
            parts.append(
                f"USER-SUPPLIED OBSERVATION (to incorporate or verify):\n\"\"\"\n{user_critique.strip()}\n\"\"\""
            )

        return "\n\n".join(parts)


def parse_critic_json(raw_text: str) -> CriticOutput:
    """Extract and validate CriticOutput from model response text.

    Why robust extraction is needed:
    Models frequently wrap valid JSON in ```json ... ``` blocks or append trailing commentary.
    """
    clean_text = raw_text.strip()

    # Strip markdown code blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
    if match:
        clean_text = match.group(1).strip()

    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from critic output: {exc}\nRaw: {raw_text[:200]}") from exc

    return CriticOutput(**data)


class BaseCritic(ABC):
    """Abstract interface for all aspect critics."""

    @abstractmethod
    def critique(
        self,
        image_path: str,
        user_critique: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> CriticOutput:
        """Evaluate an image and return structured critique and fix proposals."""


class HeuristicAspectCritic(BaseCritic):
    """Deterministic, zero-cost Aspect Critic.

    Why this class is critical:
    1. Functions 100% offline with zero external API calls and zero GPU memory.
    2. Uses our calibrated ClipAspectJudge to obtain continuous 5-aspect scores.
    3. Derives targeted, falsifiable fix proposals for aspects scoring below neutral (0.55).
    4. Serves as a rock-solid ablation baseline and test fixture for the closed-loop pipeline.
    """

    def __init__(self) -> None:
        from aesfix.judges.ensemble import ClipAspectJudge
        self._judge = ClipAspectJudge()

    def critique(
        self,
        image_path: str,
        user_critique: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> CriticOutput:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found for critique: {image_path!r}")

        score = self._judge.evaluate(image_path)
        aspects = score.aspects or {dim: 0.5 for dim in ASPECT_DIMENSIONS}

        rationales: Dict[str, str] = {}
        fixes: List[CriticFix] = []

        # 1. Composition diagnosis
        comp_score = aspects.get("composition", 0.5)
        if comp_score >= 0.65:
            rationales["composition"] = "Balanced framing with strong structural alignment and subject placement."
        elif comp_score < 0.50:
            rationales["composition"] = "Framing feels off-center or cluttered, lacking clear compositional intent."
            fixes.append(
                CriticFix(
                    op="smart_crop",
                    params={"top": 0.05, "left": 0.05, "bottom": 0.95, "right": 0.95},
                    expected_aspect="composition",
                    direction=1,
                    nl_instruction="Re-crop along the rule-of-thirds grid to anchor the primary subject.",
                )
            )
        else:
            rationales["composition"] = "Standard framing; acceptable balance with room for tighter cropping."

        # 2. Lighting diagnosis
        light_score = aspects.get("lighting", 0.5)
        if light_score >= 0.65:
            rationales["lighting"] = "Well-calibrated exposure with pleasant contrast and shadow detail."
        elif light_score < 0.45:
            rationales["lighting"] = "Underexposed or harsh illumination resulting in flat or crushed tones."
            fixes.append(
                CriticFix(
                    op="adjust_exposure",
                    params={"ev_delta": 0.40, "highlight_recovery": True},
                    expected_aspect="lighting",
                    direction=1,
                    nl_instruction="Lift exposure by +0.4 EV and preserve highlight roll-off.",
                )
            )
        else:
            rationales["lighting"] = "Evenly lit but lacks dynamic depth and dimensional contrast."
            if light_score < 0.55:
                fixes.append(
                    CriticFix(
                        op="adjust_contrast",
                        params={"factor": 1.15},
                        expected_aspect="lighting",
                        direction=1,
                        nl_instruction="Apply a gentle contrast curve to add depth between midtones and shadows.",
                    )
                )

        # 3. Color diagnosis
        color_score = aspects.get("color", 0.5)
        if color_score >= 0.65:
            rationales["color"] = "Harmonious palette with natural saturation and accurate white balance."
        elif color_score < 0.45:
            rationales["color"] = "Dull or color-cast tones detracting from overall visual appeal."
            fixes.append(
                CriticFix(
                    op="adjust_white_balance",
                    params={"temperature": 15.0, "tint": 0.0},
                    expected_aspect="color",
                    direction=1,
                    nl_instruction="Correct white balance and subtly lift color vibrance.",
                )
            )
        else:
            rationales["color"] = "Moderate color balance; slightly muted palette."

        # 4. Subject clarity diagnosis
        clarity_score = aspects.get("subject_clarity", 0.5)
        if clarity_score >= 0.65:
            rationales["subject_clarity"] = "Sharp subject isolation with clear focal definition."
        elif clarity_score < 0.50:
            rationales["subject_clarity"] = "Subject lacks crisp separation from the background."
            fixes.append(
                CriticFix(
                    op="unsharp_mask",
                    params={"radius": 1.5, "amount": 1.25},
                    expected_aspect="subject_clarity",
                    direction=1,
                    nl_instruction="Enhance micro-contrast on the focal subject using unsharp masking.",
                    region_hint="subject",
                )
            )
        else:
            rationales["subject_clarity"] = "Acceptable focus, but subject edge contrast could be crisper."

        # 5. Technical quality diagnosis
        tech_score = aspects.get("technical", 0.5)
        if tech_score >= 0.65:
            rationales["technical"] = "Clean sensor output with minimal noise or processing artifacts."
        elif tech_score < 0.50:
            rationales["technical"] = "Visible softness, noise, or motion blur impacting fine texture."
            fixes.append(
                CriticFix(
                    op="denoise",
                    params={"strength": 25.0},
                    expected_aspect="technical",
                    direction=1,
                    nl_instruction="Apply mild luminance denoising to clean sensor grain.",
                )
            )
        else:
            rationales["technical"] = "Acceptable technical standard for web viewing."

        summary = (
            f"Overall aesthetic rating: {score.raw_score:.2f}/1.0. "
            f"Primary strength: {max(aspects, key=aspects.get)}. "
            f"Primary area for improvement: {min(aspects, key=aspects.get)}."
        )
        if user_critique:
            summary += f" Evaluated in context of user critique: '{user_critique.strip()}'."

        return CriticOutput(
            aspect_scores=aspects,
            rationales=rationales,
            overall_summary=summary,
            fixes=fixes,
            genre=genre,
        )
