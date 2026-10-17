"""Tool Schema Generator and Pydantic Validator for AesFix Agent (AG2).

Generates the canonical tool schema directly from the Tier-A visual executor
operation registry, strictly validates LLM action proposals using Pydantic,
and generates structured error feedback when a model violates constraints.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# Canonical aesthetic aspects supported by AesFix Closed Loop
VALID_ASPECTS: Set[str] = {
    "composition",
    "lighting",
    "color",
    "sharpness",
    "technical",
}

# The 12 canonical visual operators in TierAExecutor
VALID_OPS: Set[str] = {
    "adjust_exposure",
    "adjust_contrast",
    "adjust_highlights",
    "adjust_shadows",
    "adjust_white_balance",
    "adjust_saturation",
    "adjust_vibrance",
    "adjust_clarity",
    "unsharp_mask",
    "denoise",
    "smart_crop",
    "correct_horizon",
}

# Operator parameter specifications with canonical ranges
OP_SPECS: Dict[str, Dict[str, Any]] = {
    "adjust_exposure": {
        "params": {
            "ev_delta": {"type": "float", "min": -3.0, "max": 3.0, "default": 0.25, "desc": "Exposure delta in EV stops (-3.0 to +3.0)."}
        },
        "default_aspect": "lighting",
        "description": "Adjust exposure in linear-light domain with highlight shoulder preservation.",
    },
    "adjust_contrast": {
        "params": {
            "factor": {"type": "float", "min": 0.2, "max": 3.0, "default": 1.15, "desc": "Midtone pivot contrast factor (0.2 to 3.0). 1.0 is no-op."}
        },
        "default_aspect": "technical",
        "description": "Adjust global contrast pivoted symmetrically around midtone luminance 0.5.",
    },
    "adjust_highlights": {
        "params": {
            "highlights": {"type": "float", "min": -100.0, "max": 100.0, "default": -20.0, "desc": "Highlight recovery or boost (-100 to +100)."}
        },
        "default_aspect": "lighting",
        "description": "Targeted recovery or brightening of top luminance tones (>0.5).",
    },
    "adjust_shadows": {
        "params": {
            "shadows": {"type": "float", "min": -100.0, "max": 100.0, "default": 20.0, "desc": "Shadow lift or deepening (-100 to +100)."}
        },
        "default_aspect": "lighting",
        "description": "Targeted shadow recovery or deepening of dark tones (<0.5).",
    },
    "adjust_white_balance": {
        "params": {
            "temperature": {"type": "float", "min": -100.0, "max": 100.0, "default": 15.0, "desc": "Color temperature shift (-100 cool/blue to +100 warm/amber)."},
            "tint": {"type": "float", "min": -100.0, "max": 100.0, "default": 0.0, "desc": "Color tint shift (-100 green to +100 magenta)."},
        },
        "default_aspect": "color",
        "description": "Adjust white balance color temperature and green/magenta tint.",
    },
    "adjust_saturation": {
        "params": {
            "factor": {"type": "float", "min": 0.0, "max": 3.0, "default": 1.15, "desc": "Color saturation scale (0.0 monochrome to 3.0 hyper-vibrant)."}
        },
        "default_aspect": "color",
        "description": "Adjust chromatic saturation across HSV color space.",
    },
    "adjust_vibrance": {
        "params": {
            "vibrance": {"type": "float", "min": -100.0, "max": 100.0, "default": 25.0, "desc": "Perceptual saturation boosting muted colors (-100 to +100)."}
        },
        "default_aspect": "color",
        "description": "Intelligently boost saturation of muted pixels while protecting skin tones.",
    },
    "adjust_clarity": {
        "params": {
            "clarity": {"type": "float", "min": -100.0, "max": 100.0, "default": 20.0, "desc": "Midtone frequency contrast (-100 to +100)."}
        },
        "default_aspect": "sharpness",
        "description": "Enhance midtone local contrast and depth without clipping extremes.",
    },
    "unsharp_mask": {
        "params": {
            "amount": {"type": "float", "min": 0.0, "max": 150.0, "default": 40.0, "desc": "High-pass sharpening intensity (0.0 to 150.0)."},
            "radius": {"type": "float", "min": 0.5, "max": 5.0, "default": 1.2, "desc": "Filter radius in pixels (0.5 to 5.0)."},
        },
        "default_aspect": "sharpness",
        "description": "Apply high-pass edge sharpening with radius control.",
    },
    "denoise": {
        "params": {
            "strength": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.35, "desc": "Luminance noise suppression strength (0.0 to 1.0)."}
        },
        "default_aspect": "technical",
        "description": "Suppress high-frequency noise artifacts while preserving structural edges.",
    },
    "smart_crop": {
        "params": {
            "top": {"type": "float", "min": 0.0, "max": 0.9, "default": 0.05, "desc": "Normalized top margin (0.0 to 0.9)."},
            "left": {"type": "float", "min": 0.0, "max": 0.9, "default": 0.05, "desc": "Normalized left margin (0.0 to 0.9)."},
            "bottom": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.95, "desc": "Normalized bottom coordinate (0.1 to 1.0)."},
            "right": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.95, "desc": "Normalized right coordinate (0.1 to 1.0)."},
        },
        "default_aspect": "composition",
        "description": "Re-compose image boundaries along normalized bounding box coordinates.",
    },
    "correct_horizon": {
        "params": {
            "angle_degrees": {"type": "float", "min": -15.0, "max": 15.0, "default": 1.5, "desc": "Rotation in degrees (-15.0 to +15.0)."}
        },
        "default_aspect": "composition",
        "description": "Rotate image to correct tilted horizon lines with automatic border fill cropping.",
    },
}


class AgentProposal(BaseModel):
    """Pydantic validated action proposal emitted by the AesFix Agent."""
    op: str = Field(..., description="Canonical visual operation name.")
    params: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of parameters for the chosen operator.")
    aspect: str = Field(..., description="Targeted aesthetic aspect to improve.")
    direction: Literal[1, -1] = Field(1, description="Direction of expected score movement (+1 for improvement).")
    rationale: str = Field(..., min_length=6, description="Photographic explanation for proposing this specific edit.")

    @field_validator("op")
    @classmethod
    def validate_op_name(cls, v: str) -> str:
        op_norm = v.lower().strip()
        if op_norm not in VALID_OPS:
            raise ValueError(
                f"Invalid operator '{v}'. Supported operators are: {sorted(list(VALID_OPS))}"
            )
        return op_norm

    @field_validator("aspect")
    @classmethod
    def validate_aspect_name(cls, v: str) -> str:
        aspect_norm = v.lower().strip()
        # Map common synonyms to canonical aspects
        synonyms = {
            "light": "lighting",
            "exposure": "lighting",
            "illumination": "lighting",
            "colour": "color",
            "colors": "color",
            "clarity": "sharpness",
            "focus": "sharpness",
            "sharp": "sharpness",
            "tech": "technical",
            "quality": "technical",
            "comp": "composition",
            "framing": "composition",
        }
        aspect_norm = synonyms.get(aspect_norm, aspect_norm)
        if aspect_norm not in VALID_ASPECTS:
            raise ValueError(
                f"Invalid aspect '{v}'. Supported aspects are: {sorted(list(VALID_ASPECTS))}"
            )
        return aspect_norm

    @model_validator(mode="after")
    def validate_params(self) -> AgentProposal:
        """Validates operator parameters against canonical ranges and types."""
        op = self.op
        spec = OP_SPECS.get(op)
        if not spec:
            return self

        param_specs = spec["params"]
        validated_params: Dict[str, Any] = {}

        for p_name, p_rule in param_specs.items():
            # Allow parameter aliases (e.g. value, ev, temp, factor)
            val = self.params.get(p_name)
            if val is None:
                # Check common aliases
                if p_name == "ev_delta":
                    val = self.params.get("ev", self.params.get("value"))
                elif p_name == "factor":
                    val = self.params.get("contrast_factor", self.params.get("value"))
                elif p_name == "temperature":
                    val = self.params.get("temp", self.params.get("temperature_shift"))
                elif p_name == "highlights":
                    val = self.params.get("factor", self.params.get("value"))
                elif p_name == "shadows":
                    val = self.params.get("factor", self.params.get("value"))
                elif p_name == "vibrance":
                    val = self.params.get("factor", self.params.get("value"))
                elif p_name == "clarity":
                    val = self.params.get("factor", self.params.get("value"))
                elif p_name == "angle_degrees":
                    val = self.params.get("angle", self.params.get("degrees"))

            if val is None:
                # If optional with default
                if "default" in p_rule:
                    val = p_rule["default"]
                else:
                    raise ValueError(f"Missing required parameter '{p_name}' for operator '{op}'.")

            # Cast and check range
            try:
                num_val = float(val)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Parameter '{p_name}' for '{op}' must be numeric, got {val!r}"
                ) from err

            min_v = p_rule["min"]
            max_v = p_rule["max"]
            if num_val < min_v or num_val > max_v:
                raise ValueError(
                    f"Parameter '{p_name}={num_val}' for '{op}' exceeds allowed range [{min_v}, {max_v}]."
                )

            validated_params[p_name] = round(num_val, 4)

        # Operator-specific cross-parameter checks
        if op == "smart_crop":
            top = validated_params.get("top", 0.0)
            bottom = validated_params.get("bottom", 1.0)
            left = validated_params.get("left", 0.0)
            right = validated_params.get("right", 1.0)
            if top >= bottom:
                raise ValueError(f"smart_crop 'top' ({top}) must be strictly less than 'bottom' ({bottom}).")
            if left >= right:
                raise ValueError(f"smart_crop 'left' ({left}) must be strictly less than 'right' ({right}).")

        self.params = validated_params
        return self


def validate_proposal(
    raw_input: Union[str, Dict[str, Any]],
) -> Tuple[Optional[AgentProposal], Optional[str]]:
    """Validates raw string or dict proposal; returns (proposal, error_message)."""
    data: Dict[str, Any]
    if isinstance(raw_input, str):
        text = raw_input.strip()
        # Extract markdown json fences if present
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip() if end != -1 else text[start:].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip() if end != -1 else text[start:].strip()

        try:
            data = json.loads(text)
        except Exception as exc:
            return None, f"JSON parsing failed: {exc}. Please return valid JSON format."
    elif isinstance(raw_input, dict):
        data = raw_input
    else:
        return None, f"Expected dict or JSON string, got {type(raw_input)}"

    try:
        proposal = AgentProposal(**data)
        return proposal, None
    except Exception as exc:
        return None, str(exc)


def format_validation_error_feedback(error_message: str, raw_attempt: str) -> str:
    """Constructs constructive critique prompt for model to correct proposal."""
    return (
        f"Your proposed tool call was rejected due to schema validation errors:\n"
        f"ERROR: {error_message}\n\n"
        f"Your previous proposal:\n{raw_attempt[:300]}\n\n"
        f"Please re-plan and respond with a strictly valid JSON action proposal matching the tool schema."
    )


def generate_tool_schema_prompt() -> str:
    """Generates the system prompt tool description from OP_SPECS registry."""
    lines: List[str] = [
        "You are the AesFix Parametric Photo Retouching Agent.",
        "Your task is to analyze the photograph and propose ONE deterministic visual fix at each step.",
        "You MUST output valid JSON conforming strictly to the following tool schema:\n",
        "```json",
        "{",
        '  "op": "<operation_name>",',
        '  "params": { ... },',
        '  "aspect": "<lighting | composition | color | sharpness | technical>",',
        '  "direction": 1,',
        '  "rationale": "<reason for adjustment>"',
        "}",
        "```\n",
        "### Available Operators & Canonical Parameters:",
    ]

    for op_name, spec in OP_SPECS.items():
        lines.append(f"- `{op_name}`: {spec['description']}")
        for p_name, p_info in spec["params"].items():
            lines.append(f"    * `{p_name}`: {p_info['desc']}")

    lines.append("\nRULES:")
    lines.append("1. Only propose one operator per step.")
    lines.append("2. All parameters must stay strictly within the listed bounds.")
    lines.append("3. If an edit is rejected in previous turns, do NOT repeat the same operator.")
    lines.append("4. Always explain your photographic reasoning clearly in 'rationale'.")
    lines.append("5. Output ONLY JSON with no surrounding conversation.")

    return "\n".join(lines)
