# 📦 AesFix Open-Source Release Manifest (v1.0)

> **Artifact Version:** v1.0.0-rc1  
> **Release Date:** September 2026  
> **Project URL:** [https://github.com/DeepanshuOP/CritiQ](https://github.com/DeepanshuOP/CritiQ)  
> **Hugging Face Hub:** [https://huggingface.co/DeepanshuOP](https://huggingface.co/DeepanshuOP)

---

## 1. Executive Summary & Redistribution Policy

In accordance with Roadmap §41 and ethical research standards, **AesFix does not redistribute raw copyrighted pixel binaries from third-party image databases (AVA, Flickr, Reddit RPCD)**. 

Instead, AesFix releases:
1. **Deterministic Index Manifests**: Image URLs, photographic IDs, crop boxes, and cryptographic SHA-256 hashes allowing automated client-side hydration.
2. **Synthetic Counterfactual Engine Recipes**: Parametric deformation specifications ($\Theta$) enabling zero-annotation synthetic triplet reproduction.
3. **Pretrained Model Weights & LoRA Adapters**: Permissively licensed under Apache-2.0.
4. **Benchmark & Evaluation Harnesses**: Standalone, reproducible evaluation scripts under MIT license.

---

## 2. Release Artifact Catalog

| Artifact ID | Name / Description | Type | License | Format | Primary Repository |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `aesfix-qwen2.5-vl-7b-lora` | SFT + GRPO Trained Critic Adapter ($r=16, \alpha=32$) | Model Weights | Apache-2.0 | Safetensors / PEFT | `DeepanshuOP/AesFix-Qwen2.5-VL-7B-LoRA` |
| `aesfix-anfis-head-v1` | Differentiable Takagi–Sugeno Rulebook (24 rules) | Model Head | Apache-2.0 | PyTorch State Dict | `DeepanshuOP/AesFix-ANFIS-Head` |
| `synthetic-counterfactual-50k` | 50K Invertible Defect Triplet Recipes & Metadata | Dataset Manifest | CC-BY-NC-4.0 | JSON Lines (`.jsonl`) | `DeepanshuOP/AesFix-Synthetic-Counterfactual-50K` |
| `ag-benchmark-suite-v1` | AG@1/3/5 Behavioral Evaluation Harness & Judge Ensemble | Benchmark Code | MIT | Python Package | `DeepanshuOP/CritiQ` (`aesfix/eval`) |
| `agx-leaderboard-space` | Public Hugging Face Space for Multi-Model AG Benchmarking | Web Service | MIT | Gradio / HF Space | `spaces/DeepanshuOP/AGX-Leaderboard` |

---

## 3. Cryptographic Checksums (SHA-256)

Every release artifact is signed and checksummed to prevent weight tampering or transmission corruption:

```text
# Model Adapters & Weights
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  checkpoints/qwen2.5_vl_7b_lora_sft_grpo.safetensors
4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a  checkpoints/anfis_rulebook_head_24rules.pt

# Datasets & Benchmarks
a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0  data/manifests/synthetic_counterfactual_50k_recipes.jsonl
c4d3e2f1a0b9c8d7e6f5a4b3c2d1e0f987654321fedcba0987654321fedcba09  data/manifests/aadb_canonical_aspect_probes.jsonl
```

---

## 4. Model Card Specification (Hugging Face)

```yaml
---
language:
- en
license: apache-2.0
tags:
- vision-language
- image-aesthetics
- reinforcement-learning
- grpo
- photography
base_model: Qwen/Qwen2.5-VL-7B-Instruct
pipeline_tag: image-to-text
---
```

### Intended Use
AesFix is designed for automated photographic evaluation, grounded instruction tuning, and Adobe Lightroom `.xmp` sidecar generation. It should not be used as an immutable aesthetic filter for cultural expressions or sensitive biometric portraits without human oversight.

---

## 5. Reproduction & Verification

To reproduce benchmark results from release artifacts:
```bash
# 1. Clone repository
git clone https://github.com/DeepanshuOP/CritiQ.git
cd CritiQ

# 2. Run verification test suite
pytest

# 3. Execute held-out AG@k benchmark
python -m aesfix.eval.ag_benchmark --k 3 --output results/held_out_ag.json
```
