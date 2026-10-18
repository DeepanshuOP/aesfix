# Published Baselines: Literature Citations & KonIQ-10k Benchmarks (I6)

## Purpose & Scope

To ensure academic and professional defensibility during technical reviews and portfolio defense, this document establishes the provenance, exact literature citations, table/page citations, architectural backbones, and official KonIQ-10k performance figures for all external visual quality models evaluated in AesFix.

---

## 1. Summary of Published vs. Measured Performance on KonIQ-10k

| Model | Architecture / Backbone | Published Paper | Exact Citation Location | Published SRCC | Published PLCC | AesFix Measured SRCC | AesFix Measured PLCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MUSIQ** | **Multi-scale Transformer** *(not ResNet-50)* | Ke et al. (ICCV 2021) | Table 1, Page 6, Row "MUSIQ (KonIQ-10k)", Col "SRCC / PLCC" | **0.916** | **0.928** | 0.8699 | 0.8977 |
| **TOPIQ-NR** | ResNet-50 + CFAN (Coarse-to-Fine Attention) | Chen et al. (IEEE TIP 2024 / arXiv:2308.03060) | Table VIII (official split) [also Table V (random split): 0.926 / 0.939] | **0.928** | **0.941** | 0.9299 | 0.9436 |
| **LIQE** | CLIP ViT-B/16 Multi-task Quality Encoder | Zhang et al. (CVPR 2023 / arXiv:2303.14968) | Table 1 (median of 10 sessions) | **0.919** | **0.908** | 0.8967 | 0.9023 |
| **CLIP-IQA** | CLIP ViT-B/32 (Zero-shot antonym prompts) | Wang et al. (AAAI 2023 / arXiv:2207.12396) | Table 1 (Zero-shot) | **0.695** | **0.727** | 0.6421 | 0.7089 |
| **CLIP-IQA+** | CLIP ViT-B/32 (Prompt-tuned / antonyms) | Wang et al. (AAAI 2023 / arXiv:2207.12396) | Table 1 (CLIP-IQA+) | **0.895** | **0.909** | — | — |

*Note on AesFix Measured Numbers*: Evaluated on the official DaRUS 20% test split of KonIQ-10k ($N=2,015$ images) using `pyiqa==0.1.16` and `torch==2.11.0` on Kaggle T4 GPU (commit `73e1194`, run ID `7fc50d71-5d29-494b-9767-67444b213837`). Our measured TOPIQ-NR (0.930 / 0.944) compares directly to Table VIII (official split: 0.928 / 0.941). AesFix evaluated zero-shot CLIP-IQA, so the relevant baseline is 0.695 / 0.727 (not prompt-tuned CLIP-IQA+ at 0.895 / 0.909).

---

## 2. Detailed Per-Model Provenance

### 1. MUSIQ: Multi-scale Image Quality Transformer
- **Full Citation**: Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, Feng Yang. *"MUSIQ: Multi-scale Image Quality Transformer."* IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 5148-5157.
- **arXiv Link**: [arXiv:2108.05997](https://arxiv.org/abs/2108.05997)
- **Table / Location**: Table 1, Page 6.
- **Backbone Architecture Correction**:
  - **Correction**: MUSIQ uses a **Multi-scale Transformer**, NOT a ResNet-50.
  - **Mechanism**: The model processes patches sampled from multi-scale representations of the full-resolution input image (native resolution, fixed intermediate resolution, and thumbnail downsampling). It applies spatial position embeddings and explicit scale embeddings to handle variable aspect ratios and multi-scale image quality degradation without cropping or resizing distortion.
- **PyIQA Implementation**: Checkpoint `musiq_koniq_ckpt-e95806b9.pth` loaded via `pyiqa.create_metric('musiq')`.

### 2. TOPIQ: Top-Down Approach to Image Quality Assessment
- **Full Citation**: Chaofeng Chen, Jiadi Mo, Jingwen Hou, Haoning Wu, Liang Liao, Wenxiu Sun, Qiong Yan, Weisi Lin. *"TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment."* IEEE Transactions on Image Processing (TIP), vol. 33, 2024, pp. 1775-1788.
- **arXiv Link**: [arXiv:2308.03060](https://arxiv.org/abs/2308.03060)
- **Table / Location**: Table VIII (official split): SRCC 0.928, PLCC 0.941; Table V (random split): SRCC 0.926, PLCC 0.939. Compare measured 0.930 / 0.944 to Table VIII.
- **Backbone Architecture**: ResNet-50 backbone pre-trained on ImageNet, coupled with a Coarse-to-Fine Attention Network (CFAN) that propagates high-level semantic tokens down to lower-level feature maps to isolate local distortions.
- **PyIQA Implementation**: `pyiqa.create_metric('topiq_nr')`.

### 3. LIQE: Blind Image Quality Assessment via Language-Image Pre-training
- **Full Citation**: Weixia Zhang, Guangtao Zhai, Ying Wei, Xiaokang Yang, Kede Ma. *"Blind Image Quality Assessment via Language-Image Pre-training."* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 2252-2261.
- **arXiv Link**: [arXiv:2303.14968](https://arxiv.org/abs/2303.14968)
- **Table / Location**: Table 1: SRCC 0.919, PLCC 0.908 (median of 10 sessions).
- **Backbone Architecture**: CLIP ViT-B/16 visual encoder fine-tuned alongside a shared language encoder trained across multiple quality assessment tasks (synthetic distortion, authentic distortion, and scene attributes).
- **PyIQA Implementation**: `pyiqa.create_metric('liqe')`.

### 4. CLIP-IQA / CLIP-IQA+
- **Full Citation**: Jianyi Wang, Kelvin C.K. Chan, Chen Change Loy. *"Exploring CLIP for Assessing the Art and Quality of Images."* Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2023, pp. 2560-2568.
- **arXiv Link**: [arXiv:2207.12396](https://arxiv.org/abs/2207.12396)
- **Table / Location**:
  - CLIP-IQA (Zero-shot): Table 1, SRCC 0.695, PLCC 0.727.
  - CLIP-IQA+ (Prompt-tuned): Table 1, SRCC 0.895, PLCC 0.909.
- **Backbone Architecture**: OpenAI CLIP ViT-B/32. Computes the cosine similarity between the image embedding and paired antonym text prompts (e.g., `"Good photo."` vs. `"Bad photo."`), normalized via softmax. AesFix evaluated zero-shot CLIP-IQA.
- **PyIQA Implementation**: `pyiqa.create_metric('clip_iqa')`.

---

## 3. Dataset Attribution
- **KonIQ-10k**:
  - Citation: Hosu et al., *"KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment"*, IEEE TIP 2020.
  - Source: DaRUS DOI: [10.18419/darus-2435](https://doi.org/10.18419/darus-2435)
  - Split: Official 2,015 test samples.
- **DEAR (Dataset for Enhancement and Aesthetics Ranking)**:
  - Citation: Plohotnuk et al., *"DEAR: A Benchmark for Image Enhancement and Aesthetics Ranking"*, arXiv:2512.05209, Dec 2025.
  - Source: Hugging Face `vsevolodpl/DEAR`.

---

## 4. Values entered by Deepanshu after manual verification on 18 Oct 2026
All baseline metrics in this document were manually verified by opening the original paper PDFs on 18 Oct 2026:
- **TOPIQ (arXiv 2308.03060)**: Verified against Table VIII (official split: SRCC 0.928, PLCC 0.941) and Table V (random split: SRCC 0.926, PLCC 0.939). Measured 0.930 / 0.944 strictly compared against Table VIII.
- **LIQE (arXiv 2303.14968)**: Verified against Table 1 (median across 10 sessions: SRCC 0.919, PLCC 0.908).
- **CLIP-IQA (arXiv 2207.12396)**: Verified against Table 1 (Zero-shot: SRCC 0.695, PLCC 0.727; Prompt-tuned CLIP-IQA+: SRCC 0.895, PLCC 0.909). AesFix evaluates zero-shot CLIP-IQA.
- **MUSIQ (arXiv 2108.05997)**: Verified against Table 1 (KonIQ-10k: SRCC 0.916, PLCC 0.928).

*Policy rule*: Never write a table, page, or row reference without personally opening the source publication. Any unverified reference must be marked explicitly as "unverified".
