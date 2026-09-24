# AesFix: Closed-Loop Image Aesthetics via Grounded Critique and Edit Verification

**Deepanshu et al.**  
*Vellore Institute of Technology & The AesFix Research Consortium*  
`arXiv:2609.XXXXX [cs.CV, cs.AI]` — September 2026

---

### Abstract

Image Aesthetic Assessment (IAA) has historically operated as an open-loop prediction task: models predict scalar Mean Opinion Scores (MOS) or generate descriptive natural language critiques, but their advice is never executed or verified. Consequently, existing vision-language models (VLMs) frequently output persuasive, eloquent feedback that proves ineffective or counter-productive when translated into concrete photographic adjustments. 

In this work, we introduce **AesFix**, the first closed-loop framework for verifiable visual aesthetic enhancement. AesFix is built upon four foundational contributions:
1. **Aesthetic Gain at rank $k$ ($\text{AG}@k$)**: The first behavioral benchmark protocol for visual advice. Rather than evaluating passive text similarity, $\text{AG}@k$ executes a critic's top-$k$ machine-readable instructions via a deterministic editing engine and measures the verifiable aesthetic delta under an independent frozen judge ensemble, subject to an identity-preservation guard.
2. **The Frontier Negative Result**: Evaluating frontier VLMs (including GPT-4o and open-weight multimodal models) under $\text{AG}@k$ reveals that prompted general-purpose models default to ungrounded platitudes ("boost contrast", "warm slightly"), yielding near-zero or negative aesthetic gains ($\text{AG}@1 \le +0.015$). In contrast, AesFix's grounded aspect critic achieves $+0.082$ verified gain.
3. **Interpretable Differentiable ANFIS Head**: To resolve the opacity of deep scoring heads, we aggregate aspect evaluations through a differentiable Takagi–Sugeno neuro-fuzzy rulebook (15–30 printable rules). Our head matches unconstrained MLP predictive accuracy while guaranteeing monotonic compliance ($\ge 98.4\%$) and high deletion/insertion faithfulness.
4. **Synthetic Counterfactual Engine**: We release a zero-annotation-cost generator of 50K invertible defect triplets (degraded image, inverse edit, known-good reference) across 12 canonical operations, demonstrating $\ge 90\%$ signal recovery on controlled degradation baselines.

Finally, in a human-agreement mini-study across 100 before/after pairs with 3 independent raters, judge-measured gain correlates strongly with crowd preference (pairwise choice concordance of $76.0\%$, Spearman $\rho = +0.584$, Pearson $r = +0.618$), confirming that $\text{AG}@k$ reflects genuine human perceptual consensus.

---

## 1. Introduction

Automated visual aesthetic evaluation has advanced from hand-crafted photographic rule heuristics to deep regression networks (NIMA [1], TANet [2]) and multimodal vision-language models (AesExpert [3], Q-Align [4]). However, despite impressive Spearman rank correlation coefficients (SRCC) on static benchmarks such as AVA [5] and AADB [6], existing systems suffer from a systemic paradigm flaw: **they are passive observers, not active photographic critics.**

When an aesthetic assessment model scores an image $4.2 / 10$ or produces a paragraph describing "suboptimal lighting and lack of compositional balance," a photographer or automated pipeline cannot act upon this critique without translating it into concrete parameter adjustments. When contemporary frontier VLMs (e.g., GPT-4o, Claude 3.5 Sonnet) are prompted to emit editing advice, their suggestions almost universally collapse into safe, ungrounded generalities—such as recommending mild contrast and saturation boosts regardless of whether the photograph is already clipped or oversaturated. When these instructions are faithfully executed, they frequently degrade photographic fidelity or alter subject identity.

To bridge this chasm, we propose **AesFix**, a closed-loop system for grounded aesthetic critique and verified visual correction. AesFix transforms photographic critique into an actionable, falsifiable science by enforcing an end-to-end execution loop:
$$\text{Input Image } I_0 \xrightarrow{\text{Critic}} \text{Fixes } \{f_1, \dots, f_k\} \xrightarrow{\text{Executor}} \text{Edited Image } I_k \xrightarrow{\text{Guard \& Judge}} \text{Aesthetic Gain } \Delta S$$

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            AesFix Closed-Loop Pipeline                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  1. Aspect Critic: Diagnoses 5 calibrated dimensions (Lighting, Color,       │
│     Composition, Clarity, Technical) & emits schema-constrained CriticFix[]  │
│  2. Interpretable Head: Differentiable Takagi-Sugeno ANFIS (Printable Rules) │
│  3. Executor Engine: Tier-A (CPU deterministic Kornia/XMP) + Tier-B (SAM2)   │
│  4. Identity Guard: LPIPS + DINOv2 drift gate (Vetoes hallucinated edits)    │
│  5. Verification & AG@k: Frozen multi-judge ensemble measures true delta     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Related Work & Research Landscape

### 2.1 Multimodal Aesthetic Assessment & Reasoning
Recent benchmarks such as AesBench [7] and UNIAA [8] have shifted focus toward text-rich aesthetic critique. While models like AesCLIP [9] and Q-Align [4] predict multi-attribute ratings, their predictions are decoupled from machine execution. Concurrently, reasoning-centric reinforcement learning frameworks such as VisualQuality-R1 [10] apply RL to rank explanations against human textual rationales. AesFix diverges fundamentally from this lineage: **our reward is not textual agreement with a human critique, but the measurable aesthetic improvement achieved upon executing the critique.**

### 2.2 Differentiable Retouching & Filter Pipelines
White-box retouching algorithms (Exposure [11], DeepLPF [12], RSFNet [13], VeraRetouch [14]) have established the efficacy of parameterizing photographic adjustments as differentiable operators (exposure curves, 3D LUTs, bilateral filters). AesFix integrates these operations into a unified Tier-A executor, while pioneering an open Adobe Lightroom XMP sidecar engine that bridges computational photography models directly into professional creative workflows.

### 2.3 Interpretable-by-Design Quality Assessment
Post-hoc attribution methods (Grad-CAM, SHAP) often exhibit unfaithfulness in aesthetic scoring. Recent efforts such as concept-subspace decomposition [15] introduce interpretable bottlenecks. AesFix advances this front by introducing a differentiable Takagi–Sugeno fuzzy inference system (ANFIS) with trainable membership parameters and explicit, printable IF–THEN rule structures that enforce strict monotonic intervention guarantees.

---

## 3. The AesFix Architecture

### 3.1 Grounded Aspect Critic
The front-end critic evaluates an input image $I \in \mathbb{R}^{H \times W \times 3}$ across five canonical photographic dimensions:
$$\mathbf{a}(I) = \big[s_{\text{lighting}}, s_{\text{color}}, s_{\text{composition}}, s_{\text{clarity}}, s_{\text{technical}}\big] \in [0, 1]^5$$

For each identified flaw, the critic outputs a structured `CriticFix` tuple:
$$\mathcal{F} = \langle \text{op}, \Theta, \text{region\_hint}, \text{expected\_aspect}, \text{nl\_instruction} \rangle$$
where $\text{op} \in \mathcal{O}_{\text{Tier-A}} \cup \mathcal{O}_{\text{Tier-B}}$ belongs to a closed vocabulary of 12 canonical photographic adjustments (exposure, contrast, highlights, shadows, white balance, saturation, vibrance, clarity, unsharp mask, denoise, smart crop, horizon rotation), and $\Theta$ denotes bounded numerical parameters (e.g., $\Delta \text{EV} \in [-3.0, +3.0]$).

### 3.2 Interpretable Neuro-Fuzzy (ANFIS) Head
Rather than fusing aspect scores through an opaque multi-layer perceptron, AesFix employs a differentiable Takagi–Sugeno fuzzy head. For rule $r \in \{1, \dots, R\}$:
$$\mu_r(\mathbf{a}) = \prod_{j=1}^5 \exp\left(-\frac{(a_j - c_{rj})^2}{2\sigma_{rj}^2}\right), \quad w_r(\mathbf{a}) = \frac{\mu_r(\mathbf{a})}{\sum_{m=1}^R \mu_m(\mathbf{a})}$$
The final aesthetic score is computed as a rule-weighted linear combination:
$$S_{\text{overall}}(\mathbf{a}) = \sum_{r=1}^R w_r(\mathbf{a}) \left( \mathbf{p}_r^T \mathbf{a} + q_r \right)$$
Trained with an $L_1$ rule-consequent sparsity penalty $\lambda_{\text{sparse}} \sum_r \|\mathbf{p}_r\|_1$, the system prunes inactive connections, yielding a human-readable rulebook (e.g., *"IF lighting is LOW and technical is HIGH, THEN score is capped at 0.42"*).

### 3.3 Two-Tier Execution Engine & Adobe XMP Integration
- **Tier-A (Deterministic, CPU, $<10\text{ms}$)**: Implemented in PyTorch/Kornia tensor transformations. Round-trip invertible operations ensure zero generative hallucination. Every Tier-A edit automatically compiles into an ISO-compliant Adobe Lightroom `.xmp` sidecar file, allowing professional editors to inspect and modify non-destructive curves in Camera Raw.
- **Tier-A+ (Mask-Conditioned)**: Leverages SAM2 segmentation masks to apply Tier-A operations selectively (e.g., subject-isolated relighting).
- **Tier-B (Generative GPU fallback)**: Interfaces with diffusion inpainting pipelines (e.g., Qwen-Image-Edit, LaMa) strictly for object removal and non-linear restoration.

### 3.4 Identity Guard & Anti-Reward-Hacking Gating
To prevent models from "cheating" aesthetic judges by over-smoothing faces or hallucinating generic aesthetic textures, every edited image must pass an automated Identity Guard:
$$\mathcal{G}(I_0, I_k) = \mathbb{I}\left[\mathcal{D}_{\text{LPIPS}}(I_0, I_k) \le \tau_{\text{LPIPS}} \;\wedge\; \text{Sim}_{\text{DINOv2}}(I_0, I_k) \ge \tau_{\text{DINO}}\right]$$
If drift exceeds the threshold ($\tau_{\text{drift}} = 0.35$), the edit is vetoed, and the verified gain is clamped to zero: $\text{AG}@k = 0.0$.

---

## 4. The AG@k Benchmark & Synthetic Counterfactual Engine

### 4.1 AG@k Metric Definition
Given initial score $S(I_0)$ from a frozen judge ensemble, a ranked sequence of $k$ advice instructions $\{f_1, \dots, f_k\}$, and resulting edited image $I_k$, the verified Aesthetic Gain at rank $k$ is defined as:
$$\text{AG}@k(I_0) = \mathcal{G}(I_0, I_k) \cdot \max\Big(0, \; S(I_k) - S(I_0)\Big)$$
$$\text{AG}@1(I_0) = \mathcal{G}(I_0, I_1) \cdot \max\Big(0, \; S(I_1) - S(I_0)\Big)$$
Additionally, we measure the Aspect Verification Rate (AVR): the percentage of executed edits that produce a positive delta on their specifically declared aspect probe.

### 4.2 Synthetic Counterfactual Engine
To provide a rigorous sanity check and ground-truth benchmark without subjective annotation noise, we developed the Synthetic Counterfactual Engine. Starting with pristine images from AVA and FiveK, we inject known, invertible photographic degradations:
$$I_{\text{degraded}} = \mathcal{T}_\theta(I_{\text{pristine}}), \quad f^* = \mathcal{T}^{-1}_\theta$$
This yields 50,000 verified (degraded image, optimal inverse fix, ground-truth reference) triplets. When evaluated under $\text{AG}@1$, an oracle policy executing $f^*$ recovers $\ge 91.4\%$ of the introduced aesthetic degradation, confirming the metric's sensitivity.

---

## 5. Experimental Results

### 5.1 Interpretability & Faithfulness (Paper Table 1)
We compare our Takagi–Sugeno ANFIS head against standard black-box MLPs and legacy heuristic fuzzy controllers (Mamdani rules) on the AADB test split.

| Model / Architecture | # Rules / Params | SRCC (AADB) | Monotonic Compliance | Deletion AUC $\downarrow$ | Insertion AUC $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Unconstrained MLP Head | 16.4K params | 0.712 | 81.2% | 0.384 | 0.612 |
| Heuristic Mamdani (14 Rules) | 14 manual rules | 0.548 | 91.0% | 0.442 | 0.540 |
| **AesFix Takagi–Sugeno (Ours)** | **24 learned rules** | **0.708** | **98.4%** | **0.312** | **0.684** |

*Table 1: Faithfulness and interpretability benchmark. AesFix matches MLP correlation within 0.004 SRCC while delivering near-perfect monotonic consistency and superior attribution faithfulness.*

### 5.2 Frontier-Model Benchmark Comparison (Paper Table 2)
We benchmark four advice regimes across 1,000 synthetic and natural test samples under frozen multi-judge evaluation.

| Policy / Model | Backbone Architecture | AG@1 (Best) | AG@3 (Compound) | Guard Pass Rate | AVR % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Random Advice Control | Uniform Op Sampling | -0.0421 | -0.0812 | 84.2% | 22.4% |
| Prompted Frontier VLM (GPT-4o) | GPT-4o Zero-Shot JSON | +0.0118 | +0.0084 | 91.0% | 34.6% |
| Open Multimodal VLM (Qwen2-VL) | Qwen2-VL-7B Zero-Shot | +0.0142 | +0.0112 | 89.6% | 36.8% |
| **AesFix Grounded Critic (Ours)** | **AspectCritic-TS** | **+0.0824** | **+0.0965** | **96.8%** | **78.2%** |
| *Oracle Inversion (Ceiling)* | *Ground-Truth Inverse* | *+0.1640* | *+0.1820* | *99.4%* | *94.6%* |

*Table 2: The Frontier Negative Result. Prompted frontier VLMs produce negligible executed gain ($\le +0.015$), whereas AesFix's grounded diagnostic critic delivers $>5\times$ higher verified improvement.*

### 5.3 Human-Agreement Mini-Study (Paper Table 3)
To validate that judge-measured gain corresponds to human perceptual preference, we conducted a controlled study comprising 100 before/after image pairs evaluated by 3 independent human annotators on a 5-point Likert scale $[-2, +2]$.

| Metric Dimension | Value Reported | Statistical Significance |
| :--- | :---: | :---: |
| Sample Size ($N_{\text{pairs}} \times N_{\text{raters}}$) | 100 pairs $\times$ 3 raters | 300 human evaluations |
| **Pairwise Choice Concordance** | **76.0%** | Judge sign matches human majority vote |
| **Spearman Rank Correlation ($\rho$)** | **+0.584** | $p = 1.42 \times 10^{-10}$ |
| **Pearson Linear Correlation ($r$)** | **+0.618** | $p = 3.18 \times 10^{-11}$ |
| **Inter-Rater Agreement (Mean Pairwise)** | **0.684** | Substantial crowd consensus |

*Table 3: Human agreement mini-study validation. Strong rank and linear correlations establish that positive AG@k gains reliably mirror human aesthetic choices.*

---

## 6. Discussion, Limitations, & Conclusion

We have presented **AesFix**, the first closed-loop framework for verifiable visual aesthetic enhancement. By establishing the **$\text{AG}@k$** metric, documenting the **Frontier Negative Result**, providing an **interpretable neuro-fuzzy head**, and releasing the **Synthetic Counterfactual Engine**, this work transitions aesthetic quality assessment from passive description to actionable, verifiable science.

**Limitations**: Tier-A operations cannot correct semantic composition flaws requiring generative inpainting (e.g., removing background crowds), which remains gated by Tier-B latency. Furthermore, individual aesthetic preferences vary across cultural genres; future work will explore taste-conditioned personal judges.

**Reproducibility & Open Science**: All evaluation harnesses, synthetic generators, model weights, and the AGX leaderboard protocol are released under permissive open-source licenses at:  
`https://github.com/DeepanshuOP/CritiQ`

---

### References
- [1] Talebi, H., & Milanfar, P. (2018). *NIMA: Neural Image Assessment.* IEEE TIP.
- [2] He, S., et al. (2022). *TANet: Thinking Attribute Network for Image Aesthetics Assessment.* IJCAI.
- [3] Huang, Y., et al. (2024). *AesExpert: Towards Multi-modal Large Language Models for Image Aesthetics.* arXiv.
- [4] Zhang, Z., et al. (2024). *Q-Align: Teaching LMMs for Visual Scoring via Human-Aligned Prompts.* ICML.
- [5] Murray, N., et al. (2012). *AVA: A Large-scale Database for Aesthetic Visual Analysis.* CVPR.
- [6] Kong, S., et al. (2016). *Photo Aesthetics Ranking Network with Attributes.* ECCV.
- [7] Huang, Y., et al. (2024). *AesBench: An Extensive Benchmark for Multimodal Aesthetic Perception.* CVPR.
- [8] Zhai, Y., et al. (2024). *UNIAA: A Unified Multimodal Dataset for Aesthetics Assessment.* ECCV.
- [9] Sheng, K., et al. (2023). *AesCLIP: Multi-attribute Aesthetic Evaluation.* TPAMI.
- [10] Chen, L., et al. (2025). *VisualQuality-R1: Reinforcement Learning for Image Quality Assessment.* NeurIPS.
- [11] Hu, Y., et al. (2018). *Exposure: A White-Box Photo Post-Processing Framework.* ACM TOG.
- [12] Moran, C., et al. (2020). *DeepLPF: Deep Local Parametric Filters for Image Enhancement.* CVPR.
- [13] Wang, T., et al. (2023). *RSFNet: Region-Specific Filter Network for Photo Retouching.* ICCV.
- [14] Liu, X., et al. (2026). *VeraRetouch: Lightweight Fully-Differentiable Multi-Task Retouching.* CVPR.
- [15] Liu, Y., & Wagemans, J. (2026). *Concept-Subspace Decomposition for Interpretable Aesthetic Quality.* arXiv.
