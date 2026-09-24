"""Human-Agreement Mini-Study and Preference Correlation Harness (AF-2.7).

Why human-agreement evaluation is critical to validating AesFix:
1. Metric Validity: An automated benchmark (AG@k) is only meaningful if positive judge gains
   correlate with human perceptual preference. Without human validation, optimization against
   frozen judges risks reward hacking without aesthetic merit.
2. Inter-Rater Reliability: Aesthetic judgments are subjective. By sampling 3 independent raters
   per pair across 100 pairs, we can measure both majority-vote concordance and inter-rater
   reliability (Fleiss' Kappa / pairwise agreement) to isolate high-confidence edits.
3. Correlation Analysis: Computes both rank correlation (Spearman rho) and linear correlation
   (Pearson r) between judge-measured aesthetic gains (Delta S = S_post - S_orig) and mean
   human preference ratings on a 5-point Likert scale [-2: Original Much Better, +2: Edited Much Better].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import stats

from aesfix.eval.ag_benchmark import AGEvaluationHarness


@dataclass
class HumanPairRating:
    """Individual human rater preference evaluation for an image pair.

    Why this dataclass:
    Enables tracking of individual annotator IDs, enabling inter-rater variance analysis
    and outlier annotator filtering.
    """
    pair_id: str
    rater_id: str
    preference_score: int  # -2: Original Much Better, -1: Orig Better, 0: Equal, +1: Edit Better, +2: Edit Much Better
    confidence: float = 1.0  # Optional self-reported rater confidence [0.0 - 1.0]
    aspect_focus: Optional[str] = None  # e.g. "lighting", "color", "technical"


@dataclass
class EvaluatedPairResult:
    """Aggregate evaluation result for a before/after image pair.

    Why this dataclass:
    Pairs objective judge deltas directly alongside crowd annotator consensus.
    """
    pair_id: str
    judge_original_score: float
    judge_edited_score: float
    judge_gain: float  # S_post - S_orig
    human_ratings: List[int]
    human_mean_preference: float
    human_majority_choice: int  # -1 (Original), 0 (Tie), +1 (Edited)
    is_concordant: bool  # sign(judge_gain) matches sign(human_majority_choice)
    dominant_aspect: str = "general"


@dataclass
class HumanAgreementReport:
    """Statistical summary report of the human agreement study (AF-2.7).

    Why this report structure:
    Directly reports the Paper §14 empirical validation DoD:
    judge-gain <-> human-preference correlation.
    """
    total_pairs: int
    num_raters: int
    pairwise_concordance_rate: float  # Fraction where judge sign matches human majority
    spearman_rho: float
    spearman_pvalue: float
    pearson_r: float
    pearson_pvalue: float
    inter_rater_agreement: float  # Mean pairwise Cohen's kappa or percent agreement
    pair_results: List[EvaluatedPairResult] = field(default_factory=list)
    aspect_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Formats the human agreement study into academic publication markdown."""
        lines = [
            "### Table 3: Human-Agreement Mini-Study Results (AF-2.7)",
            "",
            f"- **Sample Size**: {self.total_pairs} pairs, {self.num_raters} independent raters per pair",
            f"- **Pairwise Choice Concordance**: **{self.pairwise_concordance_rate*100:.1f}%** (Judge gain sign agrees with human majority)",
            f"- **Spearman Rank Correlation (rho)**: **{self.spearman_rho:+.4f}** (p = {self.spearman_pvalue:.2e})",
            f"- **Pearson Linear Correlation (r)**: **{self.pearson_r:+.4f}** (p = {self.pearson_pvalue:.2e})",
            f"- **Inter-Rater Agreement (Kappa)**: **{self.inter_rater_agreement:.3f}**",
            "",
            "| Aspect Dimension | Pairs Evaluated | Concordance Rate | Spearman rho |",
            "| :--- | :---: | :---: | :---: |",
        ]
        for asp, metrics in self.aspect_breakdown.items():
            lines.append(
                f"| {asp.capitalize():<20} | {int(metrics.get('count', 0)):<15} | "
                f"{metrics.get('concordance', 0.0)*100:5.1f}% | {metrics.get('spearman_rho', 0.0):+.4f} |"
            )
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Serializes the report metrics to JSON."""
        summary = {
            "total_pairs": self.total_pairs,
            "num_raters": self.num_raters,
            "pairwise_concordance_rate": self.pairwise_concordance_rate,
            "spearman_rho": self.spearman_rho,
            "spearman_pvalue": self.spearman_pvalue,
            "pearson_r": self.pearson_r,
            "pearson_pvalue": self.pearson_pvalue,
            "inter_rater_agreement": self.inter_rater_agreement,
            "aspect_breakdown": self.aspect_breakdown,
        }
        return json.dumps(summary, indent=indent)


class HumanAgreementStudy:
    """Orchestrates human agreement evaluation over paired before/after images."""

    def __init__(
        self,
        harness: Optional[AGEvaluationHarness] = None,
        scorer_fn: Optional[Callable[[Image.Image], Tuple[float, Dict[str, float]]]] = None,
    ) -> None:
        """Initializes the human agreement study with scoring backend."""
        self.harness = harness or AGEvaluationHarness()
        self._scorer_fn = scorer_fn or self.harness._scorer_fn

    def evaluate_pair_dataset(
        self,
        pairs: Sequence[Dict[str, Any]],
    ) -> HumanAgreementReport:
        """Evaluates an empirical dataset of image pairs and crowd ratings.

        Expected schema per pair dict:
        - "pair_id": str
        - "original_image": Image.Image or str or Path
        - "edited_image": Image.Image or str or Path
        - "ratings": List[HumanPairRating] or List[int]
        - "dominant_aspect": Optional[str] (e.g. "lighting", "color", "technical")
        """
        evaluated_pairs: List[EvaluatedPairResult] = []
        all_judge_gains: List[float] = []
        all_human_means: List[float] = []
        concordance_hits: int = 0

        # Rater matrix for inter-rater agreement computation
        rater_ratings_matrix: List[List[int]] = []

        aspect_groups: Dict[str, List[Tuple[float, float, bool]]] = {}

        for item in pairs:
            pair_id = str(item.get("pair_id", f"pair_{len(evaluated_pairs)}"))
            orig_in = item["original_image"]
            edit_in = item["edited_image"]
            raw_ratings = item["ratings"]
            dominant_aspect = str(item.get("dominant_aspect", "general"))

            pil_orig = orig_in if isinstance(orig_in, Image.Image) else Image.open(orig_in).convert("RGB")
            pil_edit = edit_in if isinstance(edit_in, Image.Image) else Image.open(edit_in).convert("RGB")

            # 1. Score with judge
            s_orig, _ = self._scorer_fn(pil_orig)
            s_edit, _ = self._scorer_fn(pil_edit)
            judge_gain = round(s_edit - s_orig, 4)

            # 2. Extract numeric ratings
            numeric_ratings: List[int] = []
            for r in raw_ratings:
                if isinstance(r, HumanPairRating):
                    numeric_ratings.append(r.preference_score)
                elif isinstance(r, (int, float)):
                    numeric_ratings.append(int(r))
                elif isinstance(r, dict) and "preference_score" in r:
                    numeric_ratings.append(int(r["preference_score"]))

            if not numeric_ratings:
                continue

            mean_pref = float(np.mean(numeric_ratings))
            # Majority choice: +1 if edit preferred, -1 if orig preferred, 0 if tied
            pos_votes = sum(1 for v in numeric_ratings if v > 0)
            neg_votes = sum(1 for v in numeric_ratings if v < 0)
            if pos_votes > neg_votes:
                majority = 1
            elif neg_votes > pos_votes:
                majority = -1
            else:
                majority = 0

            # Concordance: signs agree or both near zero
            judge_sign = 1 if judge_gain > 0.005 else (-1 if judge_gain < -0.005 else 0)
            is_concordant = (judge_sign == majority) or (judge_sign == 0 and majority == 0)

            if is_concordant:
                concordance_hits += 1

            all_judge_gains.append(judge_gain)
            all_human_means.append(mean_pref)
            rater_ratings_matrix.append(numeric_ratings)

            evaluated_pairs.append(EvaluatedPairResult(
                pair_id=pair_id,
                judge_original_score=round(s_orig, 4),
                judge_edited_score=round(s_edit, 4),
                judge_gain=judge_gain,
                human_ratings=numeric_ratings,
                human_mean_preference=round(mean_pref, 3),
                human_majority_choice=majority,
                is_concordant=is_concordant,
                dominant_aspect=dominant_aspect,
            ))

            if dominant_aspect not in aspect_groups:
                aspect_groups[dominant_aspect] = []
            aspect_groups[dominant_aspect].append((judge_gain, mean_pref, is_concordant))

        n_pairs = len(evaluated_pairs)
        if n_pairs == 0:
            raise ValueError("No valid pairs evaluated in human agreement study.")

        concordance_rate = concordance_hits / n_pairs

        # 3. Compute Spearman and Pearson correlations
        if len(set(all_judge_gains)) > 1 and len(set(all_human_means)) > 1:
            srcc_res = stats.spearmanr(all_judge_gains, all_human_means)
            spearman_rho = float(srcc_res.statistic)
            spearman_pval = float(srcc_res.pvalue)

            plcc_res = stats.pearsonr(all_judge_gains, all_human_means)
            pearson_r = float(plcc_res.statistic)
            pearson_pval = float(plcc_res.pvalue)
        else:
            spearman_rho, spearman_pval = 0.0, 1.0
            pearson_r, pearson_pval = 0.0, 1.0

        # 4. Inter-rater agreement (pairwise percent agreement among raters)
        inter_rater_kappa = self._compute_mean_inter_rater_agreement(rater_ratings_matrix)

        # 5. Aspect-specific breakdown
        aspect_breakdown: Dict[str, Dict[str, float]] = {}
        for asp, tuples in aspect_groups.items():
            gains = [t[0] for t in tuples]
            means = [t[1] for t in tuples]
            conc_count = sum(1 for t in tuples if t[2])
            asp_rho = 0.0
            if len(gains) > 2 and len(set(gains)) > 1 and len(set(means)) > 1:
                asp_rho = float(stats.spearmanr(gains, means).statistic)
            aspect_breakdown[asp] = {
                "count": float(len(tuples)),
                "concordance": conc_count / len(tuples),
                "spearman_rho": asp_rho,
            }

        num_raters = len(rater_ratings_matrix[0]) if rater_ratings_matrix else 0

        return HumanAgreementReport(
            total_pairs=n_pairs,
            num_raters=num_raters,
            pairwise_concordance_rate=round(concordance_rate, 4),
            spearman_rho=round(spearman_rho, 4),
            spearman_pvalue=spearman_pval,
            pearson_r=round(pearson_r, 4),
            pearson_pvalue=pearson_pval,
            inter_rater_agreement=round(inter_rater_kappa, 4),
            pair_results=evaluated_pairs,
            aspect_breakdown=aspect_breakdown,
        )

    @staticmethod
    def _compute_mean_inter_rater_agreement(matrix: List[List[int]]) -> float:
        """Computes mean pairwise agreement across all rater combinations.

        Why: Provides an intuitive, standardized metric of crowd consensus on aesthetic choices.
        """
        if not matrix or len(matrix[0]) < 2:
            return 1.0

        total_comparisons = 0
        matching_comparisons = 0

        for ratings in matrix:
            n = len(ratings)
            for i in range(n):
                for j in range(i + 1, n):
                    total_comparisons += 1
                    # Agreement on preference direction (+1, 0, -1)
                    s_i = 1 if ratings[i] > 0 else (-1 if ratings[i] < 0 else 0)
                    s_j = 1 if ratings[j] > 0 else (-1 if ratings[j] < 0 else 0)
                    if s_i == s_j:
                        matching_comparisons += 1

        return matching_comparisons / max(1, total_comparisons)

    @classmethod
    def generate_synthetic_benchmark_pairs(
        cls,
        num_pairs: int = 100,
        num_raters: int = 3,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """Generates a statistically calibrated 100-pair x 3-rater dataset (AF-2.7 DoD).

        Why this generator exists:
        Allows complete, deterministic reproducibility of the human study protocol
        and validation of statistical harnesses prior to running costly crowd annotations.
        """
        rng = np.random.default_rng(seed)
        pairs: List[Dict[str, Any]] = []
        aspects = ["lighting", "technical", "color", "composition"]

        for i in range(num_pairs):
            pair_id = f"af27_pair_{i:03d}"
            aspect = aspects[i % len(aspects)]

            # Base image: 64x64 synthetic RGB patch
            base_lum = int(rng.uniform(30, 220))
            orig_arr = np.clip(rng.normal(base_lum, 15.0, (64, 64, 3)), 0, 255).astype(np.uint8)
            orig_img = Image.fromarray(orig_arr, mode="RGB")

            # True underlying quality delta: 70% positive improvements, 20% neutral, 10% regressions
            u = rng.uniform(0.0, 1.0)
            if u < 0.70:
                true_delta = rng.uniform(0.08, 0.40)
            elif u < 0.90:
                true_delta = rng.uniform(-0.04, 0.04)
            else:
                true_delta = rng.uniform(-0.35, -0.08)

            # Create edited image matching delta
            edit_arr = orig_arr.astype(np.float32)
            if aspect == "lighting":
                edit_arr += true_delta * 80.0
            elif aspect == "technical":
                edit_arr = edit_arr * (1.0 + true_delta * 0.5)
            else:
                edit_arr[:, :, 0] += true_delta * 40.0
            edit_img = Image.fromarray(np.clip(edit_arr, 0, 255).astype(np.uint8), mode="RGB")

            # Simulate 3 human ratings on Likert scale [-2, +2] with Gaussian perceptual noise
            ratings: List[HumanPairRating] = []
            for r in range(num_raters):
                rater_id = f"rater_{r+1}"
                # Human perception combines true delta with subjective rater noise
                perceived = true_delta * 5.0 + rng.normal(0.0, 0.6)
                # Discretize to -2, -1, 0, +1, +2
                if perceived > 1.2:
                    score = 2
                elif perceived > 0.3:
                    score = 1
                elif perceived < -1.2:
                    score = -2
                elif perceived < -0.3:
                    score = -1
                else:
                    score = 0

                ratings.append(HumanPairRating(
                    pair_id=pair_id,
                    rater_id=rater_id,
                    preference_score=score,
                    confidence=round(float(rng.uniform(0.7, 1.0)), 2),
                    aspect_focus=aspect,
                ))

            pairs.append({
                "pair_id": pair_id,
                "original_image": orig_img,
                "edited_image": edit_img,
                "ratings": ratings,
                "dominant_aspect": aspect,
            })

        return pairs
