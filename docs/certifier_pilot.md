# Certifier Pilot C1: Learn-Then-Test False Improvement Rate (FIR) Control

> **Mandatory Dataset Label**: *pilot — expert rendering vs original (DEAR/FiveK), not agent edits*.

## 1. Problem Formulation & Research Question

When automated photographic post-processing pipelines propose edits, releasing a degraded or human-inferior edit damages user trust. In benchmark evaluations, aesthetic judges (including CLIP, TopIQ, and AesFix-Pref) frequently predict positive improvements ($s > 0$) on edits that human evaluators reject ($MOS \le 0.5$).

### The Research Question
On photographic original-vs-edit pairs, can we decide **"release the edit"** ($D=1$) versus **"leave the original"** ($D=0$) such that the fraction of released edits that humans did NOT prefer—the **False Improvement Rate (FIR)**—is strictly bounded at $\alpha$ with probability at least $1 - \delta$?

$$\mathbb{P}_{S_{\text{eval}}}\left( \text{FIR}(\hat{\lambda}) \le \alpha \right) \ge 1 - \delta$$

Where:
- $s = \text{AesFix-Pref}(\text{edit}) - \text{AesFix-Pref}(\text{original})$
- $y \in \{0, 1\}$: $y=1$ if human majority preferred the edit ($MOS > 0.5$); $y=0$ if humans preferred the original or tied ($MOS \le 0.5$).
- Decision rule: Release edit if $s \ge \lambda$; abstain / retain original if $s < \lambda$.
- Realized False Improvement Rate:
  $$\text{FIR}(\lambda) = \frac{\sum_{i: s_i \ge \lambda} (1 - y_i)}{\sum_{i: s_i \ge \lambda} 1}$$
  (Defined as $0.0$ if no edits are released).

---

## 2. Methodology: Learn-Then-Test (LTT)

We adopt the **Learn-Then-Test (LTT)** risk control framework (Angelopoulos, Bates, Candès, Jordan, Wainwright, *JMLR 2021* / *Harvard Data Science Review*).

### 2.1 Candidate Threshold Grid & Fixed-Sequence Testing
We define a discrete grid of candidate thresholds ordered monotonically from most conservative to most liberal:

$$\Lambda = (\lambda_1, \lambda_2, \dots, \lambda_M), \quad \text{where } \lambda_1 > \lambda_2 > \dots > \lambda_M$$

For each candidate threshold $\lambda_m$, we formulate the null hypothesis:

$$H_{0, m}: \mathbb{E}[\text{FIR}(\lambda_m)] > \alpha \quad \text{vs} \quad H_{1, m}: \mathbb{E}[\text{FIR}(\lambda_m)] \le \alpha$$

### 2.2 Clopper–Pearson Exact Binomial p-Values
On the calibration split of $n_{\text{calib}}$ pairs, let:
- $n(\lambda) = \sum_{i \in \text{calib}} \mathbf{1}\{s_i \ge \lambda\}$ be the number of released edits.
- $k(\lambda) = \sum_{i \in \text{calib}} \mathbf{1}\{s_i \ge \lambda, y_i = 0\}$ be the number of false improvements.

If $n(\lambda) = 0$, the release set is empty ($0$ false improvements), yielding $p(\lambda) = 0.0$.
For $n(\lambda) > 0$, under $H_{0, m}$, $k \sim \text{Binomial}(n(\lambda), p)$ with $p > \alpha$. The exact one-sided Clopper–Pearson binomial p-value is:

$$p_m = \sum_{j=0}^{k(\lambda_m)} \binom{n(\lambda_m)}{j} \alpha^j (1 - \alpha)^{n(\lambda_m) - j} = I_{1 - \alpha}(n - k, k + 1)$$

### 2.3 Statistical Guarantee via Fixed-Sequence Testing
Fixed-sequence testing evaluates hypotheses in the predetermined order $m = 1, 2, \dots, M$:
1. If $p_m \le \delta$, reject $H_{0, m}$ (threshold $\lambda_m$ is certified safe at level $\alpha$). Proceed to $m + 1$.
2. At the first index $m^*$ where $p_{m^*} > \delta$, **STOP immediately**.
3. Select $\hat{\lambda} = \lambda_{m^* - 1}$ (the most liberal certified threshold).
4. If even $m=1$ fails, select $\hat{\lambda} = \text{None}$ ("abstain everywhere").

**Theorem (Maurer et al., 1995; Angelopoulos et al., 2021)**: Because testing halts at the first unrejected null, the family-wise error rate is bounded by $\delta$ without any Bonferroni multiplicity correction.

---

## 3. Experimental Protocol

- **Dataset**: DEAR test split scenes only. Neither the judge encoder nor the preference head was ever exposed to these scenes.
- **Scene-Disjoint Splitting**: Because multiple expert renderings of the same raw image share visual content, pair outcomes within a scene are statistically dependent. Splitting is strictly performed **at the scene level**:
  - 50% scenes $\to$ Calibration Split
  - 50% scenes $\to$ Evaluation Split
- **Random Seeds**: 20 independent scene-disjoint partition seeds ($0 \dots 19$).
- **Target Risk Levels**: $\alpha \in \{0.05, 0.10, 0.20\}$ with failure probability $\delta = 0.10$.
- **Baselines**:
  - *Naive $s > 0$*: Releases any edit predicted to improve aesthetic preference.
  - *Naive $s > \varepsilon$*: Releases any edit where predicted delta exceeds the empirical noise floor ($\varepsilon = 0.04$).

---

## 4. Assumptions & Theoretical Limits

1. **Exchangeability Across Scenes**:
   The guarantee assumes test scenes are exchangeable with calibration scenes drawn from the same underlying distribution $\mathcal{P}_{\text{scenes}}$. Cross-dataset distribution shifts (e.g. smartphone casual photos vs DSLR RAW scenes) may violate exchangeability and require recalibration.
2. **Cluster Dependence**:
   Individual pairs within a scene share lighting and composition. The binomial model conditions on the total releases $n$; while conservative, finite-sample scene clustering increases variance in realized evaluation FIR.
3. **Conservative Slack**:
   Clopper–Pearson provides exact non-asymptotic coverage for any sample size, but discrete integer counts introduce slight conservatism (realized FIR is typically well below $\alpha$).
4. **Scope**:
   All pilot experiments evaluate human expert renderings vs original camera renders (DEAR FiveK). Autonomous agent edits generated via local iterative toolchains have distinct artifact profiles and must be separately calibrated.
