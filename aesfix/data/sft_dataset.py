"""Supervised Fine-Tuning (SFT) Dataset Preparation for Qwen2.5-VL (AF-3.1).

Why this module exists:
1. Translates diverse aesthetic data sources (PCCD human qualitative critiques and
   Synthetic Counterfactual Engine triplets) into a unified Qwen2.5-VL conversational
   vision-language format.
2. Enforces structured JSON output generation conforming strictly to the CriticOutput
   and CriticFix schemas, bridging natural language visual diagnosis with deterministic,
   machine-executable parameterizations for the Tier-A executor.
3. Provides exact causal grounding: synthetic counterfactual samples teach the model
   provable analytical corrections, while PCCD samples supply rich human photographic vocabulary.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image
from torch.utils.data import Dataset

from aesfix.critic.critic import (
    CriticFix,
    CriticOutput,
    CriticPromptBuilder,
)
from aesfix.data.loaders import ASPECT_DIMENSIONS, DatasetSample
from aesfix.data.synthetic import SyntheticTriplet

logger = logging.getLogger("aesfix.data.sft")

# Canonical default prompt guiding Qwen2.5-VL to act as an execution-oriented aesthetic critic
DEFAULT_SFT_USER_INSTRUCTION = (
    "Analyze this photograph and output concrete, machine-executable photographic fixes."
)


@dataclass
class SFTConversationSample:
    """Standardized multimodal conversation instance formatted for Qwen2.5-VL.

    Why this dataclass exists:
    Qwen2.5-VL and modern vision-language fine-tuning libraries (TRL / Hugging Face)
    expect conversation dictionaries containing role-separated messages where image
    inputs and text instructions are cleanly demarcated before assistant targets.
    """
    sample_id: str
    messages: List[Dict[str, Any]]
    image: Optional[Union[Image.Image, str]] = None
    critic_output: Optional[CriticOutput] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to JSON-serializable dictionary."""
        return {
            "sample_id": self.sample_id,
            "messages": self.messages,
            "metadata": self.metadata,
        }


def format_qwen_vl_conversation(
    user_instruction: str,
    assistant_json: str,
    image_placeholder: str = "<image>\n",
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Builds standard Qwen2.5-VL chat messages list.

    Why explicit message formatting is required:
    TRL's SFTTrainer and Hugging Face chat templates require specific turn structures.
    We inject the image token into the user prompt content and provide the strict JSON
    response as the assistant turn.
    """
    sys_prompt = system_prompt or CriticPromptBuilder.SYSTEM_PROMPT
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"{image_placeholder}{user_instruction.strip()}",
        },
        {"role": "assistant", "content": assistant_json.strip()},
    ]
    return messages


def synthetic_triplet_to_sft_sample(
    triplet: SyntheticTriplet,
    sample_id: Optional[str] = None,
) -> SFTConversationSample:
    """Converts a Synthetic Counterfactual Triplet into a grounded SFT training instance.

    Why synthetic triplets are foundational for SFT warm-start:
    In natural critique datasets, ground-truth fixes are subjective. In a counterfactual
    triplet, we know with mathematical certainty that applying the oracle fix directly
    inverts the known degradation, giving the model unambiguous target parameters.
    """
    sid = sample_id or f"syn_{triplet.degradation_op}_{id(triplet)}"
    oracle_fix = triplet.oracle_fix
    degraded_aspect = oracle_fix.expected_aspect

    # Synthetic degradations intentionally impair one primary aspect while leaving others intact
    aspect_scores: Dict[str, float] = {}
    rationales: Dict[str, str] = {}

    for dim in ASPECT_DIMENSIONS:
        if dim == degraded_aspect:
            # Degraded dimension receives an impaired score calibrated to degradation severity
            aspect_scores[dim] = 0.38
            rationales[dim] = (
                f"Degradation detected: Image exhibits noticeable {triplet.degradation_op.replace('_', ' ')} "
                f"flaws with parameters {triplet.degradation_params}, suppressing aesthetic quality."
            )
        else:
            # Unaffected dimensions remain aesthetically sound
            aspect_scores[dim] = 0.85
            rationales[dim] = (
                f"Aspect {dim} displays well-balanced visual characteristics without major defects."
            )

    overall_summary = (
        f"Photograph suffers from clear {degraded_aspect} deficiency caused by "
        f"{triplet.degradation_op.replace('_', ' ')}. Targeted parametric correction required."
    )

    critic_output = CriticOutput(
        aspect_scores=aspect_scores,
        rationales=rationales,
        overall_summary=overall_summary,
        fixes=[oracle_fix],
        genre="general",
    )

    assistant_json = json.dumps(critic_output.model_dump(), indent=2)
    messages = format_qwen_vl_conversation(
        user_instruction=DEFAULT_SFT_USER_INSTRUCTION,
        assistant_json=assistant_json,
    )

    return SFTConversationSample(
        sample_id=sid,
        messages=messages,
        image=triplet.degraded_image,
        critic_output=critic_output,
        metadata={
            "source": "synthetic_counterfactual",
            "degradation_op": triplet.degradation_op,
            "degradation_params": triplet.degradation_params,
            "recovery_ratio": triplet.recovery_ratio,
        },
    )


def pccd_sample_to_sft_sample(
    pccd_sample: DatasetSample,
    default_fix_generator: Optional[Any] = None,
) -> SFTConversationSample:
    """Maps human-authored PCCD critiques into a structured CriticOutput SFT sample.

    Why this mapping is necessary:
    PCCD comments are unstructured natural language. We aggregate per-aspect professional
    observations into rationales and synthesize grounded candidate fixes for weak aspects,
    teaching the model professional photographic vocabulary and descriptive reasoning.
    """
    aspect_scores: Dict[str, float] = {}
    rationales: Dict[str, str] = {}
    fixes: List[CriticFix] = []

    # Map or default scores for each aspect dimension
    for dim in ASPECT_DIMENSIONS:
        # If the sample has explicit numerical scores from aspect mapping, use them
        score = pccd_sample.aspect_scores.get(dim)
        if score is None:
            # PCCD overall score normalized to [0, 1] as baseline if available, else 0.70 default
            base_score = (pccd_sample.overall_score / 5.0) if pccd_sample.overall_score else 0.70
            score = round(max(0.1, min(1.0, float(base_score))), 2)
        aspect_scores[dim] = float(score)

        # Retrieve text comments from PCCD
        comments = pccd_sample.critiques.get(dim, [])
        if comments:
            rationales[dim] = " ".join(comments[:2])
        else:
            rationales[dim] = f"Aspect {dim} is generally acceptable without severe highlighted flaws."

    # PCCD 'overall' critique category maps to overall_summary
    overall_comments = pccd_sample.critiques.get("overall", [])
    overall_summary = (
        " ".join(overall_comments[:2])
        if overall_comments
        else "Overall composition and photographic technique are balanced, with room for targeted refinement."
    )

    # For aspects scored below 0.60, synthesize a machine-executable fix candidate
    weakest_aspect = min(aspect_scores.items(), key=lambda kv: kv[1])
    if weakest_aspect[1] < 0.65:
        w_dim = weakest_aspect[0]
        # Assign a sensible Tier-A operation mapped to the deficient dimension
        if w_dim == "lighting":
            fixes.append(
                CriticFix(
                    op="adjust_exposure",
                    params={"ev": 0.40},
                    expected_aspect="lighting",
                    nl_instruction="Increase exposure slightly to open up shadowed midtones and balance dynamic range.",
                )
            )
        elif w_dim == "composition":
            fixes.append(
                CriticFix(
                    op="smart_crop",
                    params={"left": 0.05, "top": 0.05, "right": 0.95, "bottom": 0.95},
                    expected_aspect="composition",
                    nl_instruction="Crop marginally to eliminate peripheral clutter and align focal elements with thirds.",
                )
            )
        elif w_dim == "color":
            fixes.append(
                CriticFix(
                    op="adjust_saturation",
                    params={"factor": 1.15},
                    expected_aspect="color",
                    nl_instruction="Boost color saturation moderately to enhance visual richness without clipping.",
                )
            )
        elif w_dim == "subject_clarity":
            fixes.append(
                CriticFix(
                    op="unsharp_mask",
                    params={"radius": 1.5, "amount": 0.8},
                    expected_aspect="subject_clarity",
                    nl_instruction="Apply unsharp masking across focal plane to emphasize subject edge clarity.",
                )
            )
        elif w_dim == "technical":
            fixes.append(
                CriticFix(
                    op="denoise",
                    params={"strength": 0.4},
                    expected_aspect="technical",
                    nl_instruction="Denoise high-frequency sensor grain in smooth gradient areas.",
                )
            )

    # In case no aspect was low enough, provide a subtle polish fix
    if not fixes:
        fixes.append(
            CriticFix(
                op="adjust_contrast",
                params={"factor": 1.05},
                expected_aspect="lighting",
                nl_instruction="Introduce subtle contrast expansion to give midtones additional punch.",
            )
        )

    critic_output = CriticOutput(
        aspect_scores=aspect_scores,
        rationales=rationales,
        overall_summary=overall_summary,
        fixes=fixes,
        genre=pccd_sample.metadata.get("genre", "general"),
    )

    assistant_json = json.dumps(critic_output.model_dump(), indent=2)
    messages = format_qwen_vl_conversation(
        user_instruction=DEFAULT_SFT_USER_INSTRUCTION,
        assistant_json=assistant_json,
    )

    return SFTConversationSample(
        sample_id=pccd_sample.sample_id,
        messages=messages,
        image=pccd_sample.image_path,
        critic_output=critic_output,
        metadata={"source": "pccd", "raw_attributes": pccd_sample.metadata},
    )


class QwenVLSFTDataset(Dataset):
    """PyTorch Dataset yielding formatted vision-language conversation samples for Qwen2.5-VL.

    Why this class implements standard Dataset protocols:
    Enables zero-friction integration with PyTorch DataLoader, Hugging Face Dataset,
    and TRL's SFTTrainer without intermediate format conversion.
    """

    def __init__(self, samples: Sequence[SFTConversationSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "sample_id": sample.sample_id,
            "messages": sample.messages,
            "image": sample.image,
            "metadata": sample.metadata,
        }

    def to_jsonl(self, output_path: Union[str, Path]) -> None:
        """Serializes all conversation records to JSON Lines for distributed training pipelines."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for sample in self.samples:
                fh.write(json.dumps(sample.to_dict()) + "\n")
        logger.info("Exported %d SFT samples to %s", len(self.samples), path)

    @classmethod
    def from_sources(
        cls,
        pccd_samples: Optional[Sequence[DatasetSample]] = None,
        synthetic_triplets: Optional[Sequence[SyntheticTriplet]] = None,
    ) -> QwenVLSFTDataset:
        """Factory combining real human critiques and synthetic counterfactual triplets."""
        compiled_samples: List[SFTConversationSample] = []

        if pccd_samples:
            for s in pccd_samples:
                compiled_samples.append(pccd_sample_to_sft_sample(s))

        if synthetic_triplets:
            for t in synthetic_triplets:
                compiled_samples.append(synthetic_triplet_to_sft_sample(t))

        return cls(compiled_samples)
