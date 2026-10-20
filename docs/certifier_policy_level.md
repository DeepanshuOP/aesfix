# Policy-Level Certifier on Synthetic Data (T3)

**Author:** Deepanshu <Deepanshuop@gmail.com>  
**Status:** Pre-registered Protocol Specification & Methodology  
**Reference:** SHIP (*Statistical Guarantees for Image Harmonization & Processing*, arXiv:2608.21748); LTT (*Learn-Then-Test*, Angelopoulos et al., 2021)

---

## 1. Motivation: The Candidate vs. Policy Calibration Gap

In multi-candidate post-processing pipelines, an edit proposer generates $k$ candidate modifications $\{c_1, \dots, c_k\}$ for an input image $X$. A judge model evaluates each candidate and selects the proposal with the highest predicted aesthetic gain:
$$c^* = \arg\max_{c \in \{c_1, \dots, c_k\}} s(c)$$

A common statistical fallacy in post-processing safety systems is **candidate-level calibration**: calibrating a safety threshold $\lambda$ using single, randomly drawn candidate edits ($k=1$), and then deploying that threshold to filter the output of a best-of-$k$ policy ($k \gg 1$).

Under best-of-$k$ selection, the distribution of realized candidate scores shifts systematically upwards due to extreme-value selection (the optimizer's curse / Goodhart's law). Consequently:
1. **Candidate-Level LTT (Method A)** calibrates $\lambda$ on single random candidates. When applied to best-of-$k$ outputs, the selection bias causes empirical test harm rates to systematically violate the nominal risk level $\alpha$.
2. **Naive Thresholding (Method C)** uses the judge noise floor $s > \epsilon$. This offers no statistical guarantee whatsoever and experiences catastrophic violation rates under multi-candidate optimization.
3. **Policy-Level LTT (Method B)** applies risk control directly to the *composite deployed policy*—treating the entire selection operator as a single stochastic mapping $X \mapsto c^*$. Under exchangeability of scenes, Policy-Level LTT guarantees that the realized test harm rate remains bounded by $\alpha$ with probability at least $1 - \delta$.

---

## 2. Formal Problem Formulation

### 2.1 Harm Metric and Pre-Registered Threshold
Given an input degraded image $X_{\text{deg}}$, its ground-truth reference $X_{\text{ref}}$, and the final edited output $X_{\text{final}}$:
$$\Delta E_{\text{initial}} = \Delta E_{00}(X_{\text{deg}}, X_{\text{ref}})$$
$$\Delta E_{\text{final}} = \Delta E_{00}(X_{\text{final}}, X_{\text{ref}})$$

The binary harm indicator $Y \in \{0, 1\}$ is defined per the pre-registered protocol:
$$Y = \mathbb{I}\left\{ \Delta E_{\text{final}} > \Delta E_{\text{initial}} + \tau \right\}$$
where $\tau = 0.5$ CIELAB $\Delta E$. An edit is classified as harmful if it degrades ground-truth perceptual color accuracy by more than 0.5 $\Delta E$ units.

### 2.2 Release Rule
The system deploys a release filter governed by the judge gain $s = \text{Judge}(X_{\text{final}}) - \text{Judge}(X_{\text{deg}})$:
$$A_\lambda(X) = \begin{cases} 1 & \text{if } s \ge \lambda \text{ (release edit)} \\ 0 & \text{if } s < \lambda \text{ (abstain; retain original } X_{\text{deg}}\text{)} \end{cases}$$

If the system abstains ($A=0$), the image remains untouched ($X_{\text{final}} = X_{\text{deg}}$), guaranteeing zero additional harm ($Y = 0$).

The False Inclusion Rate (FIR), or realized harm rate, is:
$$\text{FIR}(\lambda) = \mathbb{E}\left[ Y \cdot A_\lambda(X) \right] = \mathbb{P}\left(\Delta E_{\text{final}} > \Delta E_{\text{initial}} + \tau \land s \ge \lambda\right)$$

### 2.3 Statistical Guarantee (Learn-Then-Test)
Given user-specified risk tolerance $\alpha \in \{0.05, 0.10, 0.20\}$ and failure probability $\delta = 0.10$, LTT inverts a concentration bound (Bentkus / Hoeffding inequality) on calibration data $D_{\text{cal}}$ to determine the minimum threshold $\hat{\lambda}$ such that:
$$\mathbb{P}\left( \text{FIR}(\hat{\lambda}) \le \alpha \right) \ge 1 - \delta$$

---

## 3. Experimental Protocol

1. **Scene-Level Splitting:** The 100 scenes from DEAR synthetic benchmark v2 are split 50/50 into calibration ($N_{\text{cal}} = 50$ scenes, 250 cases) and test ($N_{\text{test}} = 50$ scenes, 250 cases) across 20 independent seeds. Splitting is strictly performed at the scene level to prevent data leakage across degradation types.
2. **Evaluated Methods:**
   - **Method A (Candidate-Level LTT):** $\lambda$ calibrated using random single candidates ($k=1$). Deployed on best-of-15 outputs.
   - **Method B (Policy-Level LTT):** $\lambda$ calibrated directly on the best-of-15 policy outputs. Deployed on best-of-15 outputs.
   - **Method C (Naive Threshold):** Deploys fixed threshold $\lambda = \epsilon_{v2}$ (the frozen empirical noise floor).
3. **Reporting:**
   - Realized harm rate on test split: mean, maximum, and empirical violation rate $\mathbb{P}(\text{harm} > \alpha)$ across the 20 seeds.
   - Release rate (fraction of cases where $s \ge \hat{\lambda}$).

---

## 4. Key Assumptions

1. **Scene Exchangeability:**
   The validity of the statistical guarantee rests on the assumption that scenes in the calibration set and deployment test set are exchangeable draws from the target population:
   $$(X^{(1)}, Y^{(1)}), \dots, (X^{(n)}, Y^{(n)}) \sim \mathcal{P}_{\text{scene}} \quad \text{exchangeable}$$
   Clustering by scene is strictly respected during all calibration and testing steps.
2. **Fixed Operator Repertoire:**
   The candidate generation distribution must remain invariant between calibration and deployment. If new candidate operators (e.g., local curves, neural inpainting) are introduced, $\hat{\lambda}$ must be re-calibrated.

---

## 5. Limitations

1. **Synthetic $\Delta E$ as a Perceptual Proxy:**
   In synthetic benchmark v2, ground truth is defined by the expert render $X_{\text{ref}}$, and harm is measured by CIELAB $\Delta E$. While $\Delta E$ captures pixel-level and perceptual color fidelity, human preference in aesthetic photography may tolerate or even prefer mild global deviations (e.g., warmer color temperatures or higher contrast) that increase $\Delta E$.
2. **Goodhart Divergence:**
   As demonstrated in Goodhart Entry 2 and Task T4, overexposure degradations (+1 EV) exhibit significant highlight clipping (84% of cases have $>1\%$ highlight clipping). While TopIQ-NR and v2 capture aspects of this degradation, pure single-metric selection risks overoptimizing surrogate signals without corresponding perceptual gain.
