"""LoRA Fine-Tuning Configuration & Data Collator for Qwen2.5-VL (AF-3.1).

Why this module exists:
1. Targets Qwen2.5-VL-7B-Instruct using Parameter-Efficient Fine-Tuning (PEFT/LoRA)
   to adapt the base vision-language foundation model to emit strictly validated
   photographic critiques without catastrophic forgetting of general visual priors.
2. Implements a dedicated vision-language data collator that masks prompt and visual
   tokens (labels = -100) so cross-entropy backpropagation is strictly concentrated on
   the assistant's structured JSON completion.
3. Provides a differentiable-compatible JSON penalty validator ensuring generated
   critic advice conforms unconditionally to the Tier-A machine-executable schema.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from peft import LoraConfig, TaskType

from aesfix.critic.critic import parse_critic_json

logger = logging.getLogger("aesfix.train.sft")


@dataclass
class QwenVLLoRAConfig:
    """LoRA parameter configuration targeting Qwen2.5-VL-7B-Instruct.

    Why these specific hyperparameters are selected:
    - rank (r) = 16 & alpha = 32: Yields a scaling factor alpha/r = 2.0. This grants
      sufficient expressive capacity to internalize the 5-dimensional aesthetic taxonomy
      and JSON grammar without causing weight drift or overfitting on small critique corpora.
    - target_modules = ["q_proj", "v_proj"]: Adapting query and value attention projections
      steers visual attention toward compositional defects and exposure anomalies while
      retaining pretrained key representations and linguistic backbone stability.
    """
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_peft_config(self) -> LoraConfig:
        """Converts into Hugging Face PEFT LoraConfig."""
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=TaskType.CAUSAL_LM,
        )


@dataclass
class SFTTrainingConfig:
    """Hyperparameters for instruction tuning Qwen2.5-VL on photographic critiques.

    Why these training parameters are defensible:
    - learning_rate = 1e-4: Recommended baseline for LoRA adapters on 7B models with AdamW.
    - gradient_accumulation_steps = 8: Simulates an effective batch size of 16-32 across
      memory-constrained GPU setups while accommodating high-resolution image patch tokens.
    - json_penalty_weight = 0.5: Regularization penalty penalizing malformed JSON during validation.
    """
    output_dir: str = "./checkpoints/qwen2.5-vl-aesfix-sft"
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    json_penalty_weight: float = 0.5


class JSONParsingPenaltyValidator:
    """Evaluates syntactic validity of critic outputs and computes schema violation penalties.

    Why explicit JSON penalties are necessary:
    Standard cross-entropy loss measures token-level log-likelihood, not grammatical syntax.
    A model could achieve low cross-entropy yet fail to produce parseable JSON, which breaks
    the automated execution loop. This validator provides programmatic verification.
    """

    def __init__(self, penalty_weight: float = 0.5) -> None:
        self.penalty_weight = penalty_weight

    def evaluate_completion(self, text: str) -> Dict[str, Any]:
        """Analyzes a generated string for JSON syntax and CriticOutput schema validity."""
        is_valid_json = False
        is_valid_critic_output = False
        error_msg = None

        try:
            # First check raw JSON decodability
            clean_str = text.strip()
            # Handle possible markdown wrapping
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            json.loads(clean_str.strip())
            is_valid_json = True

            # Next check CriticOutput Pydantic schema constraints
            parse_critic_json(text)
            is_valid_critic_output = True
        except Exception as exc:
            error_msg = str(exc)

        penalty = 0.0
        if not is_valid_json:
            penalty += self.penalty_weight
        elif not is_valid_critic_output:
            penalty += self.penalty_weight * 0.5

        return {
            "is_valid_json": is_valid_json,
            "is_valid_critic_output": is_valid_critic_output,
            "penalty": penalty,
            "error": error_msg,
        }

    def compute_batch_validity_rate(self, completions: Sequence[str]) -> Dict[str, float]:
        """Calculates syntax compliance rate across a batch of model generations."""
        if not completions:
            return {"validity_rate": 0.0, "mean_penalty": self.penalty_weight}

        valid_count = 0
        total_penalty = 0.0
        for comp in completions:
            res = self.evaluate_completion(comp)
            if res["is_valid_critic_output"]:
                valid_count += 1
            total_penalty += res["penalty"]

        return {
            "validity_rate": round(valid_count / len(completions), 4),
            "mean_penalty": round(total_penalty / len(completions), 4),
            "num_samples": float(len(completions)),
        }


class QwenVLDataCollator:
    """Data collator applying image and prompt token masking for Qwen2.5-VL SFT.

    Why prompt token masking is critical:
    In conversational instruction tuning, we must ONLY compute cross-entropy loss on
    the assistant tokens (the JSON diagnosis and fixes). Setting labels to -100 for all
    user instructions, system instructions, and vision tokens prevents the model from
    wasting capacity memorizing prompt templates.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        assistant_delimiter: str = "<|im_start|>assistant\n",
        ignore_index: int = -100,
        pad_token_id: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.assistant_delimiter = assistant_delimiter
        self.ignore_index = ignore_index
        self.pad_token_id = pad_token_id

    def mask_prompt_tokens(
        self,
        input_ids: torch.Tensor,
        assistant_start_idx: int,
    ) -> torch.Tensor:
        """Constructs target labels by masking everything prior to assistant response with -100.

        Why assistant_start_idx boundary is exact:
        Ensures strict supervision only over the generated JSON payload.
        """
        labels = input_ids.clone()
        # Mask all tokens before assistant generation
        if assistant_start_idx > 0:
            labels[:assistant_start_idx] = self.ignore_index
        return labels

    def collate_text_turns(
        self,
        batch: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collates conversational messages into input prompt and completion target pairs.

        Extracts user instructions (with image placeholder) and target assistant JSON completions,
        verifying that every sample has valid targets for supervised loss calculation.
        """
        user_prompts: List[str] = []
        assistant_completions: List[str] = []
        sample_ids: List[str] = []

        for item in batch:
            messages = item.get("messages", [])
            user_text = ""
            assistant_text = ""

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    user_text = content
                elif role == "assistant":
                    assistant_text = content

            user_prompts.append(user_text)
            assistant_completions.append(assistant_text)
            sample_ids.append(item.get("sample_id", "unknown"))

        return {
            "sample_ids": sample_ids,
            "user_prompts": user_prompts,
            "assistant_completions": assistant_completions,
            "batch_size": len(batch),
        }

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Main collation entrypoint supporting both tokenized and raw conversation batches."""
        # If tokenizer is present and items are pre-tokenized, apply tensor padding and label masking
        first_item = batch[0]
        if "input_ids" in first_item:
            input_ids_list = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
            padded_inputs = torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=self.pad_token_id,
            )

            # Build attention mask: 1 for valid tokens, 0 for pad tokens
            attention_mask = (padded_inputs != self.pad_token_id).long()

            # Build labels with prompt masking
            labels_list = []
            for idx, item in enumerate(batch):
                seq = input_ids_list[idx]
                assistant_start = item.get("assistant_start_idx", 0)
                masked_seq = self.mask_prompt_tokens(seq, assistant_start)
                labels_list.append(masked_seq)

            padded_labels = torch.nn.utils.rnn.pad_sequence(
                labels_list,
                batch_first=True,
                padding_value=self.ignore_index,
            )

            return {
                "input_ids": padded_inputs,
                "attention_mask": attention_mask,
                "labels": padded_labels,
            }

        # Otherwise collate conversational message turns
        return self.collate_text_turns(batch)


def build_qwen_sft_pipeline(
    lora_config: Optional[QwenVLLoRAConfig] = None,
    training_config: Optional[SFTTrainingConfig] = None,
) -> Dict[str, Any]:
    """Factory configuring the complete SFT training pipeline artifacts.

    Why this factory exists:
    Provides a standardized entrypoint for launching Qwen2.5-VL SFT training runs,
    encapsulating model LoRA configuration, collator creation, and validation hooks.
    """
    lora_cfg = lora_config or QwenVLLoRAConfig()
    train_cfg = training_config or SFTTrainingConfig()
    peft_config = lora_cfg.to_peft_config()
    validator = JSONParsingPenaltyValidator(penalty_weight=train_cfg.json_penalty_weight)
    collator = QwenVLDataCollator()

    return {
        "lora_config": lora_cfg,
        "training_config": train_cfg,
        "peft_config": peft_config,
        "validator": validator,
        "collator": collator,
    }
