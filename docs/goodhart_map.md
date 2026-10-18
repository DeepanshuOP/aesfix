# Empirical Goodhart Brightness Bias Analysis (I5)

## Executive Summary

When optimizing image aesthetic quality via reinforcement learning or search-based agent loops, optimizing directly against an unconstrained preference model or image quality assessment (IQA) score reliably triggers **Goodhart's Law**: *"When a measure becomes a target, it ceases to be a good measure."*

In aesthetic enhancement, this failure mode manifests most aggressively as **brightness gaming**: agents systematically brighten low-key and nighttime photographs, ruining intentional mood and artistic intent to exploit the judge's positive brightness preference.

This document records the empirical measurement of human vs. judge brightness preference on the DEAR test split, maps why and where benchmark datasets create this bias, and demonstrates how AesFix's structural guardrails (magnitude caps and style intent preservation) prevent this failure mode.

---

## 1. Empirical Measurements on DEAR Test Split

*Data source: `D:/aesfix-data/dear/test_processed_markup.csv` (18,995 test pairs across 1,233 scenes).*  
*Computed by: `scripts/compute_goodhart_bias.py` -> `results/goodhart_bias_metrics.json`.*

### (a) Human Brightness Preference Rate
Among DEAR test pairs where the lightness difference is perceptible ($|\Delta \text{luma}| > 0.15$ normalized to $[0, 1]$, $N=3,069$ pairs across 904 unique scenes):
- **Human Brighter Preference Rate**: **69.99%**
- **95% Bootstrap Confidence Interval** (clustered by scene, 5,000 resamples): **[67.37%, 72.43%]**

Crowd evaluators in general photo-editing benchmarks have a strong intrinsic prior favoring brighter renderings when all other visual attributes are held equal.

### (b) Extreme Scarcity of Dark / Low-Key Scenes in Benchmark Data
Analyzing the original scene luminance distribution across all 1,233 test scenes:
- 1st percentile ($p_{01}$): **0.2461**
- 5th percentile ($p_{05}$): **0.3433**
- Median ($p_{50}$): **0.4798**
- 95th percentile ($p_{95}$): **0.6117**
- **Scenes with original luminance $< 0.20$**: **Only 6 out of 1,233 scenes (0.49%)**

Across all 90 pairs evaluated on these 6 dark scenes:
- Human preference for the brighter rendering: **70.00%** (63 / 90)
- In pairwise comparisons directly against the unedited dark original: humans prefer the brighter edit **96.7%** (29 / 30)

**Root cause:** Photographic datasets like MIT-Adobe FiveK and AVA heavily sample daylight, outdoor, portrait, and travel scenes. Low-key photography (concerts, silhouettes, moody chiaroscuro, night streets) is almost completely absent ($<0.5\%$). Furthermore, when general crowd raters are presented with underexposed raw images vs. brightened edits, they treat low lightness as a technical defect rather than an intentional aesthetic choice.

### (c) Judge Bias Amplification Gap
When trained on crowd preference pairs, deep preference heads inherit and **amplify** this bias:
- Human brighter preference rate ($|\Delta \text{luma}| > 0.15$): **69.99%**
- AesFix-Pref (CLIP ViT-B/32 Linear) brighter preference rate: **80.21%**
- **Excess Judge Bias Gap**: **+10.22%**

The neural preference model turns a 70% human tendency into an 80% dominant signal, making exposure brightening the path of least resistance for any greedy search algorithm.

---

## 2. Experimental Ablation: Agent Loop Behavior

| Experiment | Setup | Behavior on `dark.jpg` (Low-Key Moody Portrait) | Exposure Shift Accepted | Final Gain Accepted | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1 (Critic-Only)** | Heuristic aspect critic proposes edits without judge scoring | Respects low luminance; attempts color balance and minor contrast adjustments | $0.00\text{ EV}$ | $0.00\text{ dB}$ | PASS (Preserved mood) |
| **E2 (Unconstrained Judge)** | Greedy search optimizing AesFix-Pref / TopIQ score with no caps | Exploits +10.22% brightness bias; continually accepts positive exposure lifts | $+0.40\text{ EV}$ | $+7.05\sigma$ | **FAIL (Goodhart Brightness Gaming)** |
| **E3 (F3 Guardrails)** | AesFix-Pref judge + strict magnitude caps ($\pm 0.6\text{ EV}$) | Proposes brightening, but capped before destructive blowout occurs | $\le +0.60\text{ EV}$ | $\le +2.0\sigma$ | PARTIAL (Limited destruction) |
| **E4 (Full Architecture)** | Judge + F3 Caps + F4 Style Intent Classification | Intent classifier detects low-key style; sets exposure budget to $0.00\text{ EV}$ | **$0.00\text{ EV}$** | **$0.00\text{ dB}$** | **PASS (Optimal preservation)** |

---

## 3. Case Study: `dark.jpg`

- **Visual Subject**: An intentionally underexposed, moody night silhouette with warm rim lighting and deep shadows (original luma $\approx 0.14$).
- **Unconstrained Reward Agent (E2)**:
  - Iteration 1: Proposes $+0.20\text{ EV}$. Judge score increases $+0.041$. Accepted.
  - Iteration 2: Proposes $+0.20\text{ EV}$. Judge score increases $+0.033$. Accepted.
  - Result: Lifted black levels to muddy gray, revealed sensor noise in shadows, and eliminated dramatic rim lighting contrast.
- **AesFix Architecture (E4)**:
  - Step 1: Pre-flight diagnosis identifies low-key aesthetic or user explicit goal `leave it`.
  - Step 2: Intent-aware exposure bounding suppresses positive EV lifts on low-key images.
  - Result: 0 exposure lifts accepted. Aesthetics and artistic intent preserved.

---

## 4. Key Takeaways for Model & System Design

1. **Reward Models Reflect Dataset Skew**: Data-driven aesthetic heads trained on public datasets cannot be trusted as solitary loss functions or acceptance criteria for creative tools.
2. **Structural Guardrails Are Non-Negotiable**: Domain-informed priors (parameter-specific magnitude caps, style intent classification, and signed physical critics) are required to prevent reward hacking.
3. **Defense in Interviews**: This empirical analysis provides concrete evidence for why AesFix combines neural preference models with explicit heuristic critics and certification bounds rather than running unconstrained end-to-end RL.
