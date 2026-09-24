# AGX Public Leaderboard — Methodology & Protocol Specification (v0.1.0)

**AesFix Research Consortium** — September 2026  
*Status: Immutable Protocol for Season 0 Evaluations*

---

## 1. Overview & Objective

The **AGX Public Leaderboard** provides the computer vision community with the first behavioral benchmark for visual advice. Traditional image quality assessment (IQA) and aesthetic assessment (IAA) benchmarks evaluate models on passive score predictions (e.g. SRCC against human MOS). In contrast, AGX evaluates whether a model's photographic advice actually **improves image quality** when executed on real photographs.

Every submitted model must take an input image and emit an ordered list of up to $k$ machine-executable parameter adjustments (`CriticFix`). These instructions are executed sequentially by the open-source **AesFix Tier-A Executor**, verified against identity drift by the **Identity Guard**, and scored by a **Frozen Multi-Judge Ensemble**.

---

## 2. Evaluation Protocol

```
Input Image I_0
      │
      ▼
Participant Model ───► Ranked Fixes [f_1, ..., f_k]
                              │
                              ▼
                      Tier-A Executor ───► Edited Image I_k
                              │
                              ▼
                       Identity Guard ───► Passed? (LPIPS <= 0.35, DINOv2 >= 0.65)
                              │                     │
                     Yes ─────┴───── No             │
                      │               │             │
                      ▼               ▼             ▼
              Frozen Judge      Gain Vetoed    AG@k = 0.0
              S(I_k) - S(I_0)   AG@k = 0.0
                      │
                      ▼
               Verified Gain AG@k
```

### 2.1 The AG@k Metric
For an image $I_0$, candidate advice sequence $\{f_1, \dots, f_k\}$, and resulting edited image $I_k$:
$$\text{AG}@k(I_0) = \mathcal{G}(I_0, I_k) \cdot \max\Big(0, \; S(I_k) - S(I_0)\Big)$$
where:
- $S(I) \in [0, 1]$ is the scalar score from the frozen judge ensemble.
- $\mathcal{G}(I_0, I_k) \in \{0, 1\}$ is the boolean Identity Guard verdict.

### 2.2 Aspect Verification Rate (AVR)
Every `CriticFix` specifies an `expected_aspect` ($\in \{\text{lighting}, \text{color}, \text{technical}, \text{composition}\}$). AVR measures the fraction of executed edits that register a positive directional delta on that specific aspect probe:
$$\text{AVR} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left[\Delta a_{\text{declared}}(I_0, I_{\text{edit}}) > 0\right]$$

---

## 3. Frozen Multi-Judge Ensemble

To prevent test-set contamination and reward hacking, all evaluations use a frozen ensemble of independent open-source aesthetic models:
1. **Aesthetic Predictor v2.5** (`discus0434/aesthetic-predictor-v2-5`) — Weight: 0.50
2. **LAION Aesthetic Scorer** (`RE-N-Y/laion-aesthetic`) — Weight: 0.50

Both models are evaluated in strict inference mode (`torch.no_grad()`) and normalized to $[0.0, 1.0]$.

---

## 4. Canonical Operations & Parameter Bounds

Participant models are restricted to the 12 canonical Tier-A photographic adjustments:
1. `adjust_exposure`: `ev_delta` $\in [-3.0, +3.0]$
2. `adjust_contrast`: `factor` $\in [0.1, 3.0]$ or `contrast` $\in [-100, +100]$
3. `adjust_highlights`: `highlights` $\in [-100, +100]$
4. `adjust_shadows`: `shadows` $\in [-100, +100]$
5. `adjust_white_balance`: `temperature` $\in [-100, +100]$, `tint` $\in [-100, +100]$
6. `adjust_saturation`: `factor` $\in [0.0, 3.0]$ or `saturation` $\in [-100, +100]$
7. `adjust_vibrance`: `vibrance` $\in [-100, +100]$
8. `adjust_clarity`: `clarity` $\in [-100, +100]$
9. `unsharp_mask`: `amount` $\in [0, 150]$, `radius` $\in [0.5, 5.0]$
10. `denoise`: `strength` $\in [0, 100]$
11. `smart_crop`: `crop_box` $[y_{\text{min}}, x_{\text{min}}, y_{\text{max}}, x_{\text{max}}]$ normalized
12. `correct_horizon`: `angle` $\in [-45.0, +45.0]$ degrees

---

## 5. Submission Schema

Submissions are accepted in JSON format:
```json
{
  "model_name": "MyCritic-8B",
  "organization": "Independent Lab",
  "backbone": "Qwen2.5-VL-7B-LoRA",
  "reproducibility_url": "https://github.com/myteam/mycritic",
  "predictions": [
    {
      "image_id": "test_0001",
      "fixes": [
        {
          "op": "adjust_exposure",
          "params": {"ev_delta": 0.45},
          "expected_aspect": "lighting",
          "nl_instruction": "Lift underexposed foreground by +0.45 EV."
        }
      ]
    }
  ]
}
```

---

## 6. Scientific Integrity & Versioning
1. **Immutability**: This document defines version `0.1.0`. The evaluation test set and judge parameters are frozen.
2. **Negative Results**: The leaderboard prominently features unguided controls (Random Advice) and frontier baseline negative results to benchmark progress honestly.
3. **Reproducibility**: Models submitted by authors must link to open-source code or API endpoints.
