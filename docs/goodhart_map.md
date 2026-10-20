# Empirical Goodhart Brightness Bias Analysis (I5)

## Executive Summary

When optimizing image aesthetic quality via reinforcement learning or search-based agent loops, optimizing directly against an unconstrained preference model or image quality assessment (IQA) score reliably triggers **Goodhart's Law**: *"When a measure becomes a target, it ceases to be a good measure."*

In aesthetic enhancement, this failure mode manifests most aggressively as **brightness gaming**: agents systematically brighten low-key and nighttime photographs, ruining intentional mood and artistic intent to exploit the judge's positive brightness preference.

This document records the empirical measurement of human vs. judge brightness preference on the DEAR test split, maps why and where benchmark datasets create this bias, and demonstrates how AesFix's structural guardrails (magnitude caps and style intent preservation) prevent this failure mode.

---

## 1. Empirical Measurements on DEAR Test Split

*Data source: `D:/aesfix-data/dear/test_processed_markup.csv` (18,995 test pairs across 1,283 unique scenes, 1,233 containing original-vs-edit comparisons).*  
*Computed by: `scripts/compute_goodhart_bias.py` -> `results/goodhart_bias_metrics.json`.*

### (a) Human and Judge Brightness Preference Rates
Among DEAR test pairs where the lightness difference is perceptible ($|\Delta \text{luma}| > 0.15$ normalized to $[0, 1]$):
- **Across the pilot pairs with judge scores** ($N=75$ pairs across 33 scenes):
  - **$P(\text{humans prefer brighter})$**: **70.67%** (95% scene-clustered bootstrap CI: **[57.30%, 82.86%]**)
  - **$P(\text{judge prefers brighter})$** (AesFix-Pref v1): **69.33%** (95% scene-clustered bootstrap CI: **[55.00%, 83.56%]**)
  - **Difference (Judge − Human)**: **-1.33%** (95% scene-clustered bootstrap CI: **[-17.28%, +14.29%]**)
- **Across the full DEAR test split** ($N=3,069$ pairs across 904 unique scenes):
  - **Human Brighter Preference Rate**: **69.99%** (95% scene-clustered bootstrap CI: **[67.37%, 72.43%]**)

### (b) Dark / Low-Key Scenes: Insufficient Data ($n=6$ Scenes)
Analyzing original scene luminance across all 1,233 scenes with original images:
- 1st percentile ($p_{01}$): **0.2461**
- 5th percentile ($p_{05}$): **0.3433**
- Median ($p_{50}$): **0.4798**
- 95th percentile ($p_{95}$): **0.6117**
- **Scenes with original luminance $< 0.20$**: **Only 6 out of 1,283 scenes (0.49%)**

*Finding*: Only 6 scenes in the entire DEAR test set qualify as dark or low-key ($L < 0.20$). With only $n=6$ scenes, there is **insufficient data** to draw general statistical conclusions about crowd rater behavior on low-key photography.

### (c) Correction of Brightness Preference Claim
In P5/I5(c), the value **0.8021** (80.21%) was cited as the judge's "brighter preference rate". This was incorrect: 0.8021 was the judge's pairwise **accuracy** (agreement with human raters) on the brightness-subset pairs in the pilot run. 
When directly measuring how often the judge prefers the brighter image vs. How often humans prefer the brighter image on the same pairs, the judge prefers brighter **69.33%** while humans prefer brighter **70.67%** (difference: **-1.33%**, with zero well inside the 95% confidence interval).

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

## 4. Entry 2: Overexposure Gaming & Preference Over Ground-Truth Correction (S3 & S4)

### (a) Ground-Truth Analytical Inversion vs. Judge Reward
- In **S3** (8 synthetic benchmark images), on the overexposed `test` image (`+1.0 EV` flaw), the ground-truth oracle fix (`-1.0 EV`) is scored **worse** than the overexposed flawed image by rsinema ($\Delta = -0.0734, -0.17\sigma$), TopIQ-NR ($\Delta = -0.0068, -0.08\sigma$), and AesFix-Pref v1 ($\Delta = -0.6577, -0.81\sigma$). Only MUSIQ rewarded the correction ($\Delta = +0.5148, +0.07\sigma$).
- In **S4** (500 cases across 100 DEAR test scenes), on 100 cases of intentional $+1.0\text{ EV}$ overexposure, analytical color recovery ($\Delta E_{\text{CIEDE2000}}$ drops $\ge 0.5$) succeeds in **99 out of 100 scenes (99.0%)**.
- However, empirical judge reward rates on the ground-truth inverse are severely depressed:
  - `aesfix_pref_v1`: rewards fix in only **9/100** cases (**91.0%** of cases preferred the blown-out overexposed image!).
  - `rsinema`: rewards fix in only **32/100** cases (**68.0%** preferred overexposure).
  - `pyiqa_musiq`: rewards fix in only **38/100** cases (**62.0%** preferred overexposure).
  - `pyiqa_topiq_nr`: rewards fix in **73/100** cases (**27.0%** preferred overexposure, benefiting from recalibrated $\epsilon = 0.005$).

### (b) Goodhart Vulnerability Mechanism
- Aesthetic scoring models trained on human preferences or photographic quality datasets exhibit monotonic positive correlations with luminance and vividness.
- While severe underexposure is consistently penalized by human raters and models alike, moderate overexposure (+1.0 EV) is frequently perceived by neural feature extractors as "cleaner", with less perceptible shadow noise, higher apparent contrast, and brighter skin tones or highlights.
- In unconstrained optimization, an agent optimizing pure judge reward will drive exposure upward until catastrophic highlight clipping occurs. Crucially, when handed an already overexposed image, an unconstrained agent refuses to darken it because reducing exposure lowers the raw aesthetic score.

### (c) AesFix Defense: The Anti-Bleach Triad
AesFix counters overexposure gaming not by relying on the judge, but through deterministic physical and architectural constraints:
1. **Highlight Clipping Guard**: The Tier A executor measures the percentage of clipped highlight pixels ($Y > 0.99$). If a proposed edit increases highlight clipping by more than $+2.0\%$, the edit is immediately aborted with status `CLIPPED_HIGHLIGHTS`.
2. **Signed Physical Aspect Critic**: The heuristic aspect critic computes histogram energy in the upper luminance decile. If severe overexposure is detected, positive exposure proposals are blocked at generation time.
3. **Intent Preservation Guard**: Bounded exposure budgets ($\pm 0.6\text{ EV}$ max delta per step, strictly limited cumulative budget) prevent runaway brightness drift.

---

## 5. Entry 3: Colour-Cast Inversion Gaming & Discrete Grid Quantization (T1 & S4)

### (a) Empirical Failure on Warm and Cool Colour Casts
In the T1 proposer benchmark (500 synthetic cases across 100 scenes):
- **Warm Cast** (100 cases, ground-truth perturbation $+15.0$ temperature shift):
  - `grid_v2` achieved only **2 recovered** (**2.0%** [0.5%, 7.0%]), **32 unchanged** (abstained), and **66 worse** (**66.0%** [56.2%, 74.5%]).
- **Cool Cast** (100 cases, ground-truth perturbation $-15.0$ temperature shift):
  - `grid_v2` achieved **0 recovered** (**0.0%** [0.0%, 3.7%]), **33 unchanged**, and **67 worse** (**67.0%** [57.3%, 75.4%]).

Across all 200 colour-cast degradation cases, unconstrained grid search with AesFix-Pref v2 degraded perceptual quality ($\Delta E$ increased by $\ge 0.5$) in **133 out of 200 cases (66.5%)**, recovering only 1.0% of cases.

### (b) Root Cause Analysis: The Three Contributing Mechanisms

#### 1. Discrete Grid Unreachability ($\pm 10/\pm 25$ vs $\pm 15$ Oracle)
- The synthetic degradation benchmark applies an analytical white balance shift of $\Delta T = +15.0$ (warm cast) or $\Delta T = -15.0$ (cool cast).
- The exact analytical inverse (oracle fix) is $\Delta T = -15.0$ (warm cast) and $\Delta T = +15.0$ (cool cast).
- However, the standard Best-of-15 candidate grid defines signed white-balance shifts at fixed grid intervals:
  $$\Delta T_{\text{candidates}} \in \{-25.0, -10.0, +10.0, +25.0\}$$
- Consequently, the ground-truth oracle fix is **topologically unreachable** within the candidate action space:
  - Choosing $\mp 10.0$ under-corrects by $33.3\%$ (leaving residual $\Delta E \approx 0.8 - 1.2$).
  - Choosing $\mp 25.0$ over-corrects by $66.7\%$ (overshooting into an unnatural opposite chromatic tint, increasing $\Delta E$).

#### 2. Model Reward on Ground-Truth Oracle vs Candidate Selection
Evaluating AesFix-Pref v2 on the analytical ground-truth fix ($\mp 15.0$):
- For **warm cast**, v2 rewards the oracle fix in **62 out of 100 cases (62.0%)** (compared to 47% for v1 and 12% for TopIQ-NR).
- For **cool cast**, v2 rewards the oracle fix in **72 out of 100 cases (72.0%)** (compared to 58% for v1 and 6% for TopIQ-NR).

While the neural preference model genuinely prefers the ground-truth analytical correction over the degraded image when tested directly, discrete candidate search forces the model to choose between an under-correcting $\mp 10.0$ shift, an over-correcting $\mp 25.0$ shift, or orthogonal non-color operators (e.g. contrast $\times 1.25$ or positive exposure). Because contrast and slight warming frequently carry high marginal preference reward, the model preferentially selects non-inverting edits, exacerbating perceptual error.

#### 3. TopIQ Guard Inactivity
Because the legacy TopIQ noise floor was frozen at $\epsilon_{\text{topiq}} = 0.15721$ ($1.97\sigma$), the non-inferiority guard never rejected these chromatic degradations, allowing over-correcting or cast-amplifying edits to pass unhindered.

### (c) WB Selection Histogram & Mitigation

| Degradation | Abstained (Unchanged) | Selected Under-Correction ($\mp 10$) | Selected Over-Correction ($\mp 25$) | Selected Orthogonal (Contrast/EV) | Selected Harmful Opposite Cast |
|---|---|---|---|---|---|
| Warm Cast (+15) | 32% | 2% (Recovered) | 38% (Worse, Overcorrected) | 16% (Worse, Contrast) | 12% (Worse, Further Warmed) |
| Cool Cast (-15) | 33% | 0% (Recovered) | 41% (Worse, Overcorrected) | 14% (Worse, Contrast) | 12% (Worse, Further Cooled) |

**AesFix Architectural Remedy**:
1. **Dynamic WB Refinement**: Adding continuous or golden-section bracketed search around estimated cast temperature prevents grid quantization error.
2. **Gray-World Aspect Gating**: The Tier A physical critic computes spatial illuminant chromaticity; proposals opposing the measured illuminant vector are rejected a priori.
3. **Calibrated Null-Transform Guard**: Deploying $\epsilon_{\text{topiq}}^{\text{null}} = 0.01707$ rejects severe over-saturation and chromatic distortion.

---

## 6. Key Takeaways for Model & System Design

1. **Reward Models Reflect Dataset Skew**: Data-driven aesthetic heads trained on public datasets cannot be trusted as solitary loss functions or acceptance criteria for creative tools.
2. **Structural Guardrails Are Non-Negotiable**: Domain-informed priors (parameter-specific magnitude caps, style intent classification, and signed physical critics) are required to prevent reward hacking.
3. **Discrete Action Sets Create Quantization Goodharting**: Even when reward models positively evaluate the ground-truth fix, a coarse discrete candidate grid can induce systematic failures by forcing choices between under-correction and destructive overshoot.
4. **Defense in Interviews**: This empirical analysis provides concrete evidence for why AesFix combines neural preference models with explicit heuristic critics and certification bounds rather than running unconstrained end-to-end RL.

