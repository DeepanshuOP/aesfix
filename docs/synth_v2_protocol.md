# Pre-Registered Protocol: Synthetic Benchmark v2 (Statistical Power Upgrade)

**Author**: Deepanshu <Deepanshuop@gmail.com>  
**Status**: Pre-registered (Frozen before execution)  
**Date**: October 2026  
**Repository**: `aesfix`

---

## 1. Motivation & Scientific Objective

The pilot synthetic benchmark (J4 in P6) evaluated only 8 synthetic cases constructed from 2 test photographs. While diagnostically illuminating (revealing Goodhart bias where judges preferred overexposure over ground truth), 8 cases lack statistical power to construct rigorous confidence intervals or draw definitive conclusions across judge architectures and edit proposers.

Synthetic Benchmark v2 scales the evaluation to **500 controlled test cases** across **100 independent DEAR test scenes**.

---

## 2. Dataset & Sampling Protocol

1. **Source Population**: The official DEAR test split (`test_processed_markup.csv`), containing 1,283 unique photographic scenes.
2. **Reference Rendering**: Expert C (`tiff16_c`), documented as the most human-preferred tone curve and color rendering in the DEAR human preference annotations.
3. **Sampling Mechanism**: Exactly 100 unique scene IDs selected using a fixed deterministic pseudo-random seed (`seed = 42`).
4. **Reference Storage**: 512px downscaled JPEG images stored at `D:\aesfix-data\synth_refs\`.

---

## 3. Degradation Operators & Analytical Oracles

For each of the 100 reference images, 5 distinct photographic degradations are deterministically applied, creating a total of $100 \times 5 = 500$ evaluation cases:

| Case # | Degradation Operator | Parameters | Ground-Truth Oracle Inverse | Inversion Parameters |
|---|---|---|---|---|
| **D1** | Underexposure (`adjust_exposure`) | `ev_delta: -1.5` | `adjust_exposure` | `ev_delta: +1.5` |
| **D2** | Overexposure (`adjust_exposure`) | `ev_delta: +1.0` | `adjust_exposure` | `ev_delta: -1.0` |
| **D3** | Low Contrast (`adjust_contrast`) | `factor: 0.6` | `adjust_contrast` | `factor: 1.6667` ($1.0 / 0.6$) |
| **D4** | Warm Color Cast (`adjust_white_balance`) | `temperature: +15.0, tint: 0.0` | `adjust_white_balance` | `temperature: -15.0, tint: 0.0` |
| **D5** | Cool Color Cast (`adjust_white_balance`) | `temperature: -15.0, tint: 0.0` | `adjust_white_balance` | `temperature: +15.0, tint: 0.0` |

---

## 4. Frozen Evaluation Metrics & Decision Rules

To prevent post-hoc metric hacking, evaluation thresholds and categorization rules are frozen prior to running experiments:

### 4.1 Ground-Truth Distance Metric
The primary objective distance metric is CIELAB $\Delta E$ (Euclidean distance in $L^*a^*b^*$ color space) between an image and the pristine reference $I_{ref}$:
$$\Delta E(I) = \|\text{Lab}(I) - \text{Lab}(I_{ref})\|_2$$

Let $\Delta E_{deg} = \Delta E(I_{deg})$ and $\Delta E_{out} = \Delta E(I_{out})$.

### 4.2 Categorical Outcomes
With margin threshold $\tau = 0.5$ (imperceptible color difference bound):
1. **Recovered**: $\Delta E_{out} < \Delta E_{deg} - \tau$ (image moved strictly closer to ground truth reference).
2. **Worse**: $\Delta E_{out} > \Delta E_{deg} + \tau$ (image moved strictly further from ground truth reference).
3. **Unchanged**: $|\Delta E_{out} - \Delta E_{deg}| \le \tau$ (no significant change in ground truth distance).

### 4.3 Statistical Reporting
All recovery and failure rates are reported with exact 95% Wilson score confidence intervals:
$$\hat{p} \pm \frac{z}{1 + z^2/n}\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}$$
where $n = 500$ and $z = 1.96$.

---

## 5. Judge Scoring & Standardized Units

On the Oracle-Fix test, judges evaluate both degraded image $I_{deg}$ and oracle-reconstructed image $I_{orc}$.
- **Raw Delta**: $\Delta_{raw} = S(I_{orc}) - S(I_{deg})$.
- **Standardized Delta**:
  $$\Delta_{std} = \frac{\Delta_{raw}}{\sigma_{DEAR}}$$
  where $\sigma_{DEAR}$ is the empirical standard deviation of the judge across the DEAR test set (NOT per-image $\epsilon$, avoiding uninterpretable division by tiny numbers).
- **Reward Condition**: The judge rewards the oracle fix if $\Delta_{raw} > \epsilon_{judge}$, where $\epsilon_{judge}$ is the judge's noise floor.

### 5.1 Pre-Registered Noise Floor Selection Rule
To ensure noise floor filtering operates strictly at fine-grained sub-perceptual perturbation scales without rejecting genuine aesthetic improvements, every candidate judge noise floor $\epsilon_j$ must satisfy the pre-registered selection rule:
$$\epsilon_j < 0.5 \sigma_{\text{DEAR}}$$
where $\sigma_{\text{DEAR}}$ is the empirical standard deviation of the judge's score distribution across the DEAR test split (e.g. $\sigma_{\text{topiq}} = 0.0796 \implies \epsilon_{\text{topiq}} < 0.0398$; $\sigma_{v1} = 0.9870 \implies \epsilon_{v1} < 0.4935$). Any noise floor exceeding $0.5\sigma_{\text{DEAR}}$ represents a macroscopic half-standard-deviation aesthetic shift and is disqualified as an over-conservative threshold that disables guardrails (such as the legacy $\epsilon_{\text{topiq}} = 0.157 \approx 1.97\sigma$, which caused the TopIQ non-inferiority guard to never fire).

### 5.2 Pre-Registered Null-Transform Noise Floor ($\epsilon^{\text{null}}$)
In addition to compression and resize perturbations, an empirical 'null-transform $\epsilon$' is defined by evaluating content-preserving spatial and compression operations:
1. Horizontal reflection (`flip`)
2. 1–2px translation with reflect padding (`translate_reflect`)
3. Same-pipeline JPEG q92 re-encode (`q92_reencode`)

The null-transform noise floor $\epsilon_j^{\text{null}}$ is computed as the 95th percentile of absolute score deltas $|\Delta_j| = |S_j(I_{\text{null}}) - S_j(I)|$ across 50 disjoint DEAR train images, verified to satisfy $\epsilon_j^{\text{null}} < 0.5 \sigma_{\text{DEAR}}$, and frozen in `config/judge_epsilon.json`.

---

## 6. Proposer Policies (S5)

Two agent proposal policies will be compared across all 500 cases with both v1 and v2 primary judges:
1. **Heuristic Proposer**: Single-step heuristic critic analyzing luminance histograms and color statistics.
2. **Grid Proposer**: Symmetric candidate evaluation (Best-of-$k$ selection across exposure $\pm0.3, \pm0.7, \pm1.2, \pm1.6$ EV; contrast $0.8, 1.25, 1.6$; signed WB $\pm10, \pm25$ temperature) filtered by guard rails (intent preservation, clipping budget, non-inferiority). Candidate count $k$ is logged per decision to calibrate certification under SHIP (arXiv 2608.21748).
