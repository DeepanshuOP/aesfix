"""Unit tests for Qwen2.5-VL SFT Dataset & LoRA Pipeline (AF-3.1).

Why these tests are critical:
1. Verifies that both synthetic counterfactual triplets and human PCCD critiques
   translate into strict, machine-executable CriticOutput JSON schemas without syntax errors.
2. Validates that the vision-language data collator properly masks prompt tokens (-100)
   so loss backpropagation is strictly isolated to the assistant's structured JSON response.
3. Confirms PEFT LoRA configuration adheres exactly to target parameters (r=16, alpha=32,
   q_proj/v_proj) preventing catastrophic forgetting during VLM fine-tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from aesfix.critic.critic import CriticFix, parse_critic_json
from aesfix.data.loaders import DatasetSample
from aesfix.data.sft_dataset import (
    DEFAULT_SFT_USER_INSTRUCTION,
    QwenVLSFTDataset,
    format_qwen_vl_conversation,
    pccd_sample_to_sft_sample,
    synthetic_triplet_to_sft_sample,
)
from aesfix.data.synthetic import SyntheticTriplet
from aesfix.train.sft_config import (
    JSONParsingPenaltyValidator,
    QwenVLDataCollator,
    QwenVLLoRAConfig,
    SFTTrainingConfig,
    build_qwen_sft_pipeline,
)


@pytest.fixture
def dummy_synthetic_triplet() -> SyntheticTriplet:
    """Fixture providing an analytical counterfactual triplet."""
    img = Image.new("RGB", (64, 64), color=(100, 100, 100))
    oracle_fix = CriticFix(
        op="adjust_exposure",
        params={"ev": 0.60},
        expected_aspect="lighting",
        nl_instruction="Increase exposure by +0.60 EV to restore midtone fidelity.",
    )
    return SyntheticTriplet(
        degraded_image=img,
        oracle_fix=oracle_fix,
        target_image=img,
        degradation_op="adjust_exposure",
        degradation_params={"ev": -0.60},
        recovery_ratio=0.95,
        metadata={"seed": 42},
    )


@pytest.fixture
def dummy_pccd_sample() -> DatasetSample:
    """Fixture providing a human-authored PCCD critique sample."""
    return DatasetSample(
        sample_id="pccd_test_101",
        image_path="test_img.jpg",
        overall_score=3.2,
        aspect_scores={"lighting": 0.45, "composition": 0.80},
        critiques={
            "lighting": ["The shadows are muddy and lack separation from the dark background."],
            "composition": ["Pleasing balance along the rule of thirds."],
            "overall": ["A promising portrait that needs lighting adjustment."],
        },
        metadata={"genre": "portrait"},
    )


def test_format_qwen_vl_conversation() -> None:
    """Verifies standard 3-turn conversational structure formatted for Qwen2.5-VL."""
    messages = format_qwen_vl_conversation(
        user_instruction="Analyze photo",
        assistant_json='{"status": "ok"}',
    )

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "<image>" in messages[1]["content"]
    assert "Analyze photo" in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == '{"status": "ok"}'


def test_synthetic_triplet_to_sft_sample(dummy_synthetic_triplet: SyntheticTriplet) -> None:
    """Confirms synthetic triplet serialization produces valid CriticOutput JSON targets."""
    sample = synthetic_triplet_to_sft_sample(dummy_synthetic_triplet, sample_id="syn_sample_1")

    assert sample.sample_id == "syn_sample_1"
    assert sample.image is not None
    assert len(sample.messages) == 3

    # Extract assistant completion and parse as strict CriticOutput
    assistant_msg = sample.messages[2]["content"]
    critic_output = parse_critic_json(assistant_msg)

    # Why degraded aspect score check is important:
    # Verifies that the degraded dimension is numerically penalized while others stay sound
    assert critic_output.aspect_scores["lighting"] < 0.50
    assert critic_output.aspect_scores["composition"] > 0.70
    assert len(critic_output.fixes) == 1
    assert critic_output.fixes[0].op == "adjust_exposure"
    assert critic_output.fixes[0].params["ev"] == 0.60
    assert critic_output.fixes[0].expected_aspect == "lighting"


def test_pccd_sample_to_sft_sample(dummy_pccd_sample: DatasetSample) -> None:
    """Validates mapping of human qualitative critique into structured schema."""
    sample = pccd_sample_to_sft_sample(dummy_pccd_sample)

    assert sample.sample_id == "pccd_test_101"
    assistant_msg = sample.messages[2]["content"]
    critic_output = parse_critic_json(assistant_msg)

    # Lighting was low (0.45), so an exposure or lighting fix must be generated
    assert critic_output.aspect_scores["lighting"] == 0.45
    assert len(critic_output.fixes) >= 1
    assert any(fix.expected_aspect == "lighting" for fix in critic_output.fixes)
    assert "shadows are muddy" in critic_output.rationales["lighting"]


def test_qwen_vl_sft_dataset_operations(
    dummy_synthetic_triplet: SyntheticTriplet,
    dummy_pccd_sample: DatasetSample,
    tmp_path: Path,
) -> None:
    """Tests PyTorch Dataset container indexing and JSONL persistence."""
    syn_sample = synthetic_triplet_to_sft_sample(dummy_synthetic_triplet)
    pccd_sample = pccd_sample_to_sft_sample(dummy_pccd_sample)

    dataset = QwenVLSFTDataset([syn_sample, pccd_sample])
    assert len(dataset) == 2

    item = dataset[0]
    assert "messages" in item
    assert "sample_id" in item

    # Test factory creation from sources
    factory_ds = QwenVLSFTDataset.from_sources(
        pccd_samples=[dummy_pccd_sample],
        synthetic_triplets=[dummy_synthetic_triplet],
    )
    assert len(factory_ds) == 2

    # Test JSONL export format
    jsonl_file = tmp_path / "train_sft.jsonl"
    dataset.to_jsonl(jsonl_file)
    assert jsonl_file.exists()

    with open(jsonl_file, "r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]
    assert len(lines) == 2
    assert lines[0]["sample_id"] == syn_sample.sample_id
    assert lines[0]["messages"][1]["content"].startswith("<image>")


def test_lora_config_parameters() -> None:
    """Verifies LoRA configuration satisfies research hyperparameters for Qwen2.5-VL."""
    cfg = QwenVLLoRAConfig()
    assert cfg.r == 16
    assert cfg.lora_alpha == 32
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules

    peft_cfg = cfg.to_peft_config()
    assert peft_cfg.r == 16
    assert peft_cfg.lora_alpha == 32
    assert "q_proj" in peft_cfg.target_modules
    assert "v_proj" in peft_cfg.target_modules


def test_data_collator_text_and_token_masking() -> None:
    """Ensures prompt tokens are masked with -100 while assistant tokens receive loss."""
    collator = QwenVLDataCollator()

    # 1. Test text collation
    batch_text = [
        {
            "sample_id": "s1",
            "messages": [
                {"role": "user", "content": f"<image>\n{DEFAULT_SFT_USER_INSTRUCTION}"},
                {"role": "assistant", "content": '{"aspect_scores": {}}'},
            ],
        }
    ]
    text_collated = collator(batch_text)
    assert text_collated["batch_size"] == 1
    assert "user_prompts" in text_collated
    assert "assistant_completions" in text_collated

    # 2. Test tensor token masking
    # Sequence of 10 tokens where assistant starts at index 6
    seq = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=torch.long)
    masked_labels = collator.mask_prompt_tokens(seq, assistant_start_idx=6)

    # Why the first 6 tokens must equal -100:
    # PyTorch CrossEntropyLoss ignores -100, preventing gradient updates on prompt tokens
    assert (masked_labels[:6] == -100).all()
    assert (masked_labels[6:] == seq[6:]).all()

    # 3. Test batch token collation with padding
    token_batch = [
        {"input_ids": [1, 2, 3, 4], "assistant_start_idx": 2},
        {"input_ids": [1, 2, 3, 4, 5, 6], "assistant_start_idx": 3},
    ]
    collated_tensors = collator(token_batch)
    assert collated_tensors["input_ids"].shape == (2, 6)
    assert collated_tensors["labels"].shape == (2, 6)
    assert collated_tensors["labels"][0, :2].tolist() == [-100, -100]
    assert collated_tensors["labels"][0, 2:4].tolist() == [3, 4]
    # Padded tail should also be ignore_index (-100)
    assert collated_tensors["labels"][0, 4:].tolist() == [-100, -100]


def test_json_parsing_penalty_validator() -> None:
    """Verifies that the validator correctly distinguishes valid CriticOutput from malformed JSON."""
    validator = JSONParsingPenaltyValidator(penalty_weight=0.5)

    # Valid CriticOutput payload
    valid_payload = json.dumps({
        "aspect_scores": {
            "composition": 0.8,
            "lighting": 0.7,
            "color": 0.9,
            "subject_clarity": 0.85,
            "technical": 0.9,
        },
        "rationales": {
            "composition": "Well framed",
            "lighting": "Balanced",
            "color": "Vibrant",
            "subject_clarity": "Sharp",
            "technical": "Clean",
        },
        "overall_summary": "High quality photograph.",
        "fixes": [],
    })
    res_valid = validator.evaluate_completion(valid_payload)
    assert res_valid["is_valid_json"] is True
    assert res_valid["is_valid_critic_output"] is True
    assert res_valid["penalty"] == 0.0

    # Malformed JSON (syntax error)
    res_syntax_err = validator.evaluate_completion("{not valid json")
    assert res_syntax_err["is_valid_json"] is False
    assert res_syntax_err["penalty"] == 0.5

    # Valid JSON but missing required aspect scores schema
    invalid_schema = json.dumps({"aspect_scores": {"lighting": 0.5}})
    res_schema_err = validator.evaluate_completion(invalid_schema)
    assert res_schema_err["is_valid_json"] is True
    assert res_schema_err["is_valid_critic_output"] is False
    assert res_schema_err["penalty"] == 0.25

    # Test batch validity rate
    batch_stats = validator.compute_batch_validity_rate([valid_payload, "{not valid json"])
    assert batch_stats["validity_rate"] == 0.5
    assert batch_stats["num_samples"] == 2.0


def test_build_qwen_sft_pipeline() -> None:
    """Validates factory pipeline initialization."""
    pipeline = build_qwen_sft_pipeline()
    assert isinstance(pipeline["lora_config"], QwenVLLoRAConfig)
    assert isinstance(pipeline["training_config"], SFTTrainingConfig)
    assert isinstance(pipeline["validator"], JSONParsingPenaltyValidator)
    assert isinstance(pipeline["collator"], QwenVLDataCollator)
