# Multi-Judge Non-Inferiority Rule

## 1. Problem Definition: The Multi-Judge Veto Failure
In photographic enhancement systems governed by multi-judge ensembles (e.g. CLIP-based semantic aesthetics + patch-based technical IQA), requiring every independent judge family to simultaneously register a positive improvement ($\Delta_j > \varepsilon_j$) produces systematic false rejections on valid tonal edits.

Specifically:
- **KonIQ / MUSIQ (Technical Quality)** is trained on general image quality impairments (compression artifacts, severe blur, noise). On clean photographs suffering solely from midtone underexposure (e.g. $-1.5$ EV), lifting exposure by $+1.0$ EV dramatically improves aesthetic perception and semantic clarity, yet MUSIQ barely registers a shift ($\Delta_{\text{musiq}} \approx +0.006$ to $+0.02$, well within or near its noise floor $\varepsilon_{\text{musiq}} \approx 0.029$).
- Under strict consensus ("all families must improve by $> \varepsilon$"), MUSIQ vetoes virtually every legitimate lighting correction, forcing the agent into an erroneous "leave it" verdict on severe underexposure (positive control failure).

## 2. Mathematical Formulation of Non-Inferiority
AesFix implements a formal non-inferiority framework adapted from clinical trial safety protocols:

1. **Primary Judge Superiority**:
   The primary judge (responsible for aesthetic alignment, such as `rsinema` or `AesFix-Pref`) must demonstrate measured superiority exceeding its empirical noise floor:
   $$\Delta_{\text{primary}} > \varepsilon_{\text{primary}}$$
   where $\Delta_{\text{primary}} = \text{Score}_{\text{candidate}} - \text{Score}_{\text{baseline}}$, and $\varepsilon$ is computed across 4 non-destructive image perturbations (JPEG $q90$, JPEG $q95$, $+2\%$ scale, $-2\%$ scale).

2. **Secondary Family Non-Inferiority**:
   Every secondary judge family $F \neq F_{\text{primary}}$ (e.g. the non-CLIP patch-transformer family represented by MUSIQ) must not suffer a statistically meaningful degradation exceeding its own noise floor:
   $$\Delta_j \ge -\varepsilon_j \quad \iff \quad \frac{\Delta_j}{\varepsilon_j} \ge -1.0 \quad \forall j \in F$$

3. **Dimensionless Ratio Metric ($\Delta / \varepsilon$)**:
   Because judges operate on fundamentally different scales (e.g. rsinema $[1, 5]$ vs MUSIQ $[0, 100]$ vs CLIP $[-1, 1]$), raw scores are **never averaged** across judges. Instead:
   - Overall $\Delta$ is defined strictly as $\Delta_{\text{primary}}$.
   - Cross-judge impacts are reported as the dimensionless ratio:
     $$\text{Ratio}_j = \frac{\Delta_j}{\varepsilon_j}$$
   - Any secondary judge exhibiting $\text{Ratio}_j < -1.0$ triggers an immediate step rejection for structural/technical degradation.

## 3. UI Display & Auditability
On every step card in the AesFix Studio UI:
- Primary metric displays $\Delta_{\text{primary}}$ alongside its noise floor $\varepsilon_{\text{primary}}$.
- A dedicated badge strip displays $\text{Ratio}_j = \Delta_j / \varepsilon_j$ for each participating judge:
  - Green badge: $\Delta_j / \varepsilon_j > +1.0$ (superiority confirmed across multiple families)
  - Neutral badge: $-1.0 \le \Delta_j / \varepsilon_j \le +1.0$ (non-inferiority satisfied; within noise envelope)
  - Red badge: $\Delta_j / \varepsilon_j < -1.0$ (veto; significant collateral defect detected)
