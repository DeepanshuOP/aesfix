# Tuning Disclosure: Hand-Tuned Heuristics and Intent Detector Values

**Author**: Deepanshu <Deepanshuop@gmail.com>  
**Status**: Measured / Implemented-Unvalidated (Hand-Tuned on Test Cases)

---

## 1. Explicit Disclosure of Post-Observation Parameter Tuning

All heuristic thresholds, operational caps, and statistical parameters in `config/agent.yaml`, `aesfix/agent/loop.py`, and the style intent detector (`_detect_style_intent`) were modified **after** observing failure modes on test images (specifically `dark.jpg` and the 8 synthetic test images `synth_test_*.jpg` and `synth_temp_*.jpg`).

**These parameters are hand-tuned on specific test cases and must not be presented as validated.** They are empirical heuristic rules that have not undergone external validation across an independent photographic distribution.

---

## 2. Parameter Transition Log (Old → New)

| Component | File / Location | Old Value | New Value | Reason for Modification & Trigger Image |
| :--- | :--- | :--- | :--- | :--- |
| **Exposure Cap (Ordinary)** | `config/agent.yaml` (`loop.op_magnitude_caps.adjust_exposure`) | `0.6` EV | `2.0` EV | `0.6` EV prevented positive control recovery on severe underexposure (`synth_test_exp_m1_5.jpg`). Changed after inspecting synthetic underexposure failure. |
| **Exposure Cap (Intentional)** | `config/agent.yaml` (`loop.op_magnitude_caps.adjust_exposure_intentional`) | None (`0.6` default) | `0.6` EV | Restricted exposure lifts on intentional low-key images (`dark.jpg`) to prevent aesthetic degradation. |
| **Highlight Clipping Threshold** | `config/agent.yaml` (`overprocessing.max_highlight_clipping_growth`) | None | `0.02` (2%) | Added after observing blown-out specular highlights on `dark.jpg` when exposure was lifted. |
| **Black Level Lift Threshold** | `config/agent.yaml` (`overprocessing.max_black_level_lift_low_key`) | None | `0.03` (3%) | Added after observing washed-out shadows on `dark.jpg` when shadows were lifted. |
| **Saturation Clipping Threshold** | `config/agent.yaml` (`overprocessing.max_saturation_clipping_growth`) | None | `0.05` (5%) | Added after observing synthetic color oversaturation on `synth_test_warm_cast.jpg`. |
| **Low-Key Intent Mean Luma** | `aesfix/agent/loop.py:365` (`_detect_style_intent`) | None | `mean_luma < 0.15` | Designed specifically while examining `dark.jpg` (mean luma 0.088) to trigger intentional low-key classification. |
| **Low-Key Intent 5th Percentile** | `aesfix/agent/loop.py:365` (`_detect_style_intent`) | None | `p5_luma < 0.02` | Tuned directly on `dark.jpg` to detect deep crushed blacks alongside low mean luminance. |
| **High-Key Intent Mean Luma** | `aesfix/agent/loop.py:369` (`_detect_style_intent`) | None | `mean_luma > 0.80` | Added to prevent unintended darkening of high-key photographs. |
| **High-Key Intent 95th Percentile** | `aesfix/agent/loop.py:369` (`_detect_style_intent`) | None | `p95_luma > 0.95` | Added to confirm high-key tonal distribution. |
| **Exposure Heuristic Delta** | `aesfix/agent/loop.py:808` (`HeuristicProposalProvider`) | Constant `+0.5` EV | $\text{clip}(\log_2(0.45/\text{median}), -2.0, 2.0)$ | Replaced hardcoded step after observing that +0.5 EV under-corrected -1.5 EV underexposure. Target 0.45 median was selected by inspecting `test.jpg` median. |
| **Multi-Judge Rule** | `aesfix/agent/loop.py:875` | Strict Consensus ($\Delta_j > \varepsilon_j \ \forall j$) | Non-inferiority ($\Delta_{\text{primary}} > \varepsilon_{\text{primary}}$ and $\Delta_j \ge -\varepsilon_j$) | Strict consensus caused MUSIQ to veto valid exposure corrections on `synth_test_exp_m1_5.jpg` (MUSIQ registered $\Delta = +0.012 < \varepsilon = 0.029$). Relaxed to non-inferiority to unblock positive controls. |
| **Sharpness Metric Normalization** | `aesfix/agent/loop.py:335` | Replicated `subject_clarity` | $1 / (1 + \exp(-\text{var}/500.0 + 2.0))$ | Gradient variance sigmoid was calibrated after viewing unsharp mask outputs on `synth_test_contrast_0_6.jpg` and `blur.jpg`. |

---

## 3. Status and Research Rigor

- **Intent Features ("high dynamic range, bright specular highlights, steep histogram contrast")**: These visual descriptors were derived by manually examining `dark.jpg` and noting the presence of deep shadows alongside specular key lights. They were not learned from a training set or validated on a benchmark.
- **Classification**: All heuristics above are categorized as **Hand-Tuned Heuristic Rules (Implemented-Unvalidated)**. They must not be cited as generalizable research findings.
