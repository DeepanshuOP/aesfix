"""Taste Fingerprint, Pairwise Preference Calibration, and Personalization Engine (AF-6.3 & Roadmap §21, §25 F4.4).

Why the Taste Fingerprint is a foundational moonshot for AesFix:
1. Resolves the Subjectivity Paradox: Purely objective aesthetic scoring works well for
   blatant technical errors (blur, sensor noise, underexposure), but high-end photography
   diverges sharply on stylistic preference (e.g., moody low-key vs vibrant commercial).
2. The Product as RLHF Flywheel: Users ranking ~20 pairwise selections during onboarding
   or picking variants from the WhatIf-Grid train a personalized Bradley-Terry preference head.
3. Honesty via Divergence Tracking: AesFix explicitly reports divergence between personal
   gain (AG_user@k) and global consensus (AG_global@k) — warning photographers when an edit
   helps universal appeal but fights their personal artistic signature.
4. Creator Economy & Taste-Packs: Allows photographers to package their calibrated aesthetic
   weights into exportable, licensable "Taste-Packs" (the Imagen Talent-profile play).
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aesfix.critic.critic import CriticFix
from aesfix.data.loaders import ASPECT_DIMENSIONS

logger = logging.getLogger("aesfix.taste")

# Prior baseline aspect distribution grounded in the PARA benchmark
DEFAULT_ASPECT_PRIORS: Dict[str, float] = {
    "composition": 0.20,
    "lighting": 0.25,
    "color": 0.20,
    "subject_clarity": 0.20,
    "technical": 0.15,
}


@dataclass
class PairwiseComparison:
    """Individual pairwise preference observation (e.g. from onboarding or WhatIf-grid)."""
    image_a_path: str
    image_b_path: str
    preferred: str  # 'A', 'B', or 'TIE'
    user_id: str = "default_user"
    aspect_diff: Optional[Dict[str, float]] = None


@dataclass
class TastePack:
    """Exportable, shareable creator aesthetic profile with calibrated weights and op biases."""
    pack_id: str
    name: str
    creator: str
    description: str
    aspect_weights: Dict[str, float]
    target_adjustments: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Taste-Pack to clean JSON dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TastePack:
        """Instantiates Taste-Pack from dictionary representation."""
        return cls(**data)


# Curated Built-In Taste Packs
BUILTIN_TASTE_PACKS: Dict[str, TastePack] = {
    "cinematic_moody": TastePack(
        pack_id="pack_cinematic_moody",
        name="Cinematic Moody",
        creator="AesFix Curators",
        description="Deep shadows, high subject isolation, muted secondary colors, and filmic tonal contrast.",
        aspect_weights={
            "composition": 0.25,
            "lighting": 0.30,
            "color": 0.15,
            "subject_clarity": 0.20,
            "technical": 0.10,
        },
        target_adjustments={"ev_delta": -0.20, "contrast_factor": 1.25, "saturation_factor": 0.85},
    ),
    "high_key_commercial": TastePack(
        pack_id="pack_high_key_commercial",
        name="High-Key Commercial",
        creator="AesFix Curators",
        description="Luminous even exposure, vibrant color fidelity, and razor-sharp edge definition for products.",
        aspect_weights={
            "composition": 0.20,
            "lighting": 0.25,
            "color": 0.25,
            "subject_clarity": 0.15,
            "technical": 0.15,
        },
        target_adjustments={"ev_delta": 0.25, "contrast_factor": 1.05, "saturation_factor": 1.15},
    ),
    "minimalist_editorial": TastePack(
        pack_id="pack_minimalist_editorial",
        name="Minimalist Editorial",
        creator="AesFix Curators",
        description="Severe compositional geometry, neutral white balance, and quiet negative space.",
        aspect_weights={
            "composition": 0.40,
            "lighting": 0.20,
            "color": 0.15,
            "subject_clarity": 0.15,
            "technical": 0.10,
        },
        target_adjustments={"rule": "rule_of_thirds", "tightness": 0.80},
    ),
}


class TasteFingerprint:
    """Personal aesthetic preference model parameterized over the 5 canonical dimensions."""

    def __init__(
        self,
        user_id: str = "default_user",
        aspect_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.user_id = user_id
        # Normalize initial weights to guarantee convex combination
        raw_weights = aspect_weights or DEFAULT_ASPECT_PRIORS.copy()
        self.aspect_weights = self._normalize_weights(raw_weights)
        self.history: List[PairwiseComparison] = []

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
        """Ensures all aspect weights are non-negative and sum to 1.0."""
        clamped = {k: max(0.01, float(v)) for k, v in weights.items()}
        total = sum(clamped.values())
        return {k: round(v / total, 4) for k, v in clamped.items()}

    @property
    def confidence_score(self) -> float:
        """Measures preference calibration certainty as an asymptotic function of sample count.

        Why: Informs users how firmly their taste profile is anchored vs still relying on generic priors.
        """
        n = len(self.history)
        # Sigmoid curve: 0 pairs = 0.10 confidence, 20 pairs ~= 0.85 confidence
        return round(1.0 / (1.0 + math.exp(-0.2 * (n - 10))), 3)

    def score_aspects(self, aspects: Dict[str, float]) -> float:
        """Computes the personalized continuous aesthetic rating S_user(I) in [0.0, 1.0]."""
        score = sum(self.aspect_weights.get(k, 0.2) * aspects.get(k, 0.5) for k in ASPECT_DIMENSIONS)
        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def compute_personal_gain(
        self,
        original_aspects: Dict[str, float],
        edited_aspects: Dict[str, float],
        global_gain: float,
    ) -> Tuple[float, float, str]:
        """Calculates personalized aesthetic gain and tracks divergence from global consensus.

        Returns:
            Tuple of (personal_gain, divergence, alignment_badge).
        """
        orig_personal = self.score_aspects(original_aspects)
        edit_personal = self.score_aspects(edited_aspects)
        personal_gain = round(max(0.0, edit_personal - orig_personal), 4)

        # Divergence: positive means edit satisfies user MORE than average; negative means user is dissatisfied
        divergence = round(personal_gain - global_gain, 4)

        if divergence > 0.03:
            badge = f"✨ Highly Aligns With Your Taste (+{personal_gain:.3f} Personal Gain)"
        elif divergence < -0.03:
            badge = f"⚠️ Fights Your Taste Signature (Diverges by {divergence:.3f} from global gain)"
        else:
            badge = f"✅ Harmonious (+{personal_gain:.3f} Personal Gain, Balanced with Global)"

        return personal_gain, divergence, badge

    def fit_from_pairwise_comparisons(
        self,
        comparisons: Sequence[PairwiseComparison],
        regularization_lambda: float = 2.0,
    ) -> Dict[str, float]:
        """Fits personal aspect weights via regularized Bradley-Terry gradient descent.

        Why L2 Bayesian regularization against prior is essential:
        With small onboarding batches (~10-20 pairs), unconstrained optimization will
        collapse to extreme one-hot weights (e.g. 100% lighting, 0% composition).
        Regularizing toward PARA priors guarantees well-conditioned, robust weights.
        """
        if not comparisons:
            return self.aspect_weights

        # Deferred import avoids circular dependency between aesfix.judges and aesfix.api.demo
        from aesfix.api.demo import get_demo_components

        critic, _, _ = get_demo_components()
        diff_vectors: List[np.ndarray] = []
        labels: List[float] = []

        for comp in comparisons:
            # 1. Obtain aspect ratings for image A and image B
            if comp.aspect_diff is not None:
                # Precomputed differential for speed
                delta = np.array([comp.aspect_diff.get(d, 0.0) for d in ASPECT_DIMENSIONS], dtype=np.float32)
            else:
                if not os.path.exists(comp.image_a_path) or not os.path.exists(comp.image_b_path):
                    continue
                score_a = critic.critique(comp.image_a_path).aspect_scores
                score_b = critic.critique(comp.image_b_path).aspect_scores
                delta = np.array([score_a.get(d, 0.5) - score_b.get(d, 0.5) for d in ASPECT_DIMENSIONS], dtype=np.float32)

            if comp.preferred == "A":
                diff_vectors.append(delta)
                labels.append(1.0)
            elif comp.preferred == "B":
                diff_vectors.append(-delta)
                labels.append(1.0)
            elif comp.preferred == "TIE":
                # Ties provide symmetric zero-pull constraint
                diff_vectors.append(delta)
                labels.append(0.5)

        if not diff_vectors:
            return self.aspect_weights

        X = np.stack(diff_vectors, axis=0)  # Shape: [N, 5]
        y = np.array(labels, dtype=np.float32)  # Shape: [N]

        # Prior anchor
        w_prior = np.array([DEFAULT_ASPECT_PRIORS[d] for d in ASPECT_DIMENSIONS], dtype=np.float32)
        w = np.copy(w_prior)

        # Gradient descent with Ridge penalty toward prior
        lr = 0.05
        epochs = 100
        for _ in range(epochs):
            # Bradley-Terry probability P(A > B) = sigmoid(X @ w)
            logits = np.clip(X @ w, -10.0, 10.0)
            preds = 1.0 / (1.0 + np.exp(-logits))

            # Binary cross-entropy gradient + L2 regularization toward w_prior
            grad = X.T @ (preds - y) / len(y) + regularization_lambda * (w - w_prior)
            w -= lr * grad
            w = np.maximum(w, 0.01)  # Non-negative weights

        # Normalize to probability simplex
        w = w / np.sum(w)
        updated_weights = {dim: round(float(w[i]), 4) for i, dim in enumerate(ASPECT_DIMENSIONS)}

        self.aspect_weights = self._normalize_weights(updated_weights)
        self.history.extend(comparisons)
        return self.aspect_weights

    def export_taste_pack(
        self,
        pack_name: str,
        creator: str,
        description: str,
    ) -> TastePack:
        """Packages calibrated fingerprint into an exportable, licensable Taste-Pack."""
        pack_id = f"pack_{self.user_id}_{pack_name.lower().replace(' ', '_')}"
        return TastePack(
            pack_id=pack_id,
            name=pack_name,
            creator=creator,
            description=description,
            aspect_weights=self.aspect_weights.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes fingerprint state for user profile persistence."""
        return {
            "user_id": self.user_id,
            "aspect_weights": self.aspect_weights,
            "sample_count": len(self.history),
            "confidence_score": self.confidence_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TasteFingerprint:
        """Reconstructs user Taste Fingerprint from stored JSON data."""
        fp = cls(user_id=data.get("user_id", "default_user"), aspect_weights=data.get("aspect_weights"))
        return fp


class TastePersonalizer:
    """Ranks and filters candidate photographic edits conditioned on personal taste."""

    @staticmethod
    def personalize_fix_ranking(
        candidate_fixes: Sequence[CriticFix],
        original_aspects: Dict[str, float],
        fingerprint: TasteFingerprint,
        global_weight: float = 0.5,
    ) -> List[Tuple[CriticFix, float, float, str]]:
        """Ranks machine-executable candidate fixes balancing personal utility and global consensus.

        Returns:
            List of Tuples: (fix, blended_utility, personal_gain, alignment_badge), sorted descending.
        """
        ranked_fixes: List[Tuple[CriticFix, float, float, str]] = []

        for fix in candidate_fixes:
            # Model expected post-fix aspect delta based on the targeted aspect
            expected_aspect = fix.expected_aspect or "lighting"
            simulated_aspects = original_aspects.copy()

            # Fixes generally provide an expected ~0.15 delta on their primary dimension
            simulated_aspects[expected_aspect] = min(1.0, simulated_aspects.get(expected_aspect, 0.5) + 0.15)

            # Global score uses equal weight average; personal score uses fingerprint
            global_baseline = sum(original_aspects.values()) / 5.0
            global_post = sum(simulated_aspects.values()) / 5.0
            global_gain = round(max(0.0, global_post - global_baseline), 4)

            personal_gain, divergence, badge = fingerprint.compute_personal_gain(
                original_aspects=original_aspects,
                edited_aspects=simulated_aspects,
                global_gain=global_gain,
            )

            # Blended utility balances user preference with objective quality standards
            blended_utility = round(
                (1.0 - global_weight) * personal_gain + global_weight * global_gain,
                4,
            )
            ranked_fixes.append((fix, blended_utility, personal_gain, badge))

        # Sort descending by blended utility
        ranked_fixes.sort(key=lambda item: item[1], reverse=True)
        return ranked_fixes
