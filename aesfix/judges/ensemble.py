"""Frozen Judge Ensemble for multi-perspective aesthetic evaluation.

Why an ensemble is necessary:
Individual aesthetic scorers suffer from strong idiosyncratic biases:
- Prompt-based zero-shot CLIP aligns with semantic attributes (composition, lighting)
  but can over-reward colorful or sharp images regardless of photographic merit.
- Continuous regression models (e.g. aesthetic-predictor-v2-5, LAION) capture general
  visual appeal but fail to isolate specific technical defects like blur or motion.
- Running a frozen, disjoint ensemble and calculating cross-judge disagreement
  allows AesFix to (a) prevent reward hacking during reinforcement learning and
  (b) expose honest uncertainty badges to users when judges disagree on taste.
"""
from __future__ import annotations

import logging
import os
import statistics as stats
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from PIL import Image

logger = logging.getLogger("aesfix.judges")


class JudgeEvaluationError(RuntimeError):
    """Raised when an individual judge or the entire ensemble fails fatally."""


@dataclass(frozen=True)
class JudgeScore:
    """Score output from a single judge.

    Scores are normalized to [0.0, 1.0] to enable fair cross-model comparison
    and disagreement variance computation, while retaining raw scores for auditability.
    """
    name: str
    raw_score: float
    normalized_score: float
    aspects: Optional[Dict[str, float]] = None


@dataclass
class JudgeResult:
    """Consolidated evaluation from the judge ensemble.

    Disagreement (variance across normalized scores) acts as the primary signal
    for aesthetic uncertainty, flagging subjective images where judges diverge.
    """
    image_path: str
    overall_score: float
    judge_scores: Dict[str, JudgeScore] = field(default_factory=dict)
    disagreement: float = 0.0
    is_uncertain: bool = False
    aspects: Dict[str, float] = field(default_factory=dict)


class BaseJudge(ABC):
    """Abstract interface defining the contract for all aesthetic judges."""

    name: str

    @abstractmethod
    def evaluate(self, image_path: str) -> JudgeScore:
        """Evaluate an image file and return raw + normalized scores."""

    @staticmethod
    def _validate_image_path(image_path: str) -> None:
        # Fail loudly on missing files to prevent downstream components from
        # hallucinating defaults or operating on corrupted inputs.
        if not image_path or not os.path.exists(image_path):
            raise JudgeEvaluationError(f"Image not found or unreadable: {image_path!r}")


class ClipAspectJudge(BaseJudge):
    """Multi-attribute zero-shot CLIP judge.

    Why this judge is kept in the ensemble:
    It provides deterministic, zero-cost, locally cached feature extraction
    across the 5 core photographic aspects (composition, lighting, color,
    subject clarity, and technical quality) without requiring GPU instances.
    """

    name = "clip_aspect"

    def __init__(self) -> None:
        # Import dynamically so module import doesn't stall if CLIP isn't needed immediately
        from baselines.legacy_fuzzy.our_system.image_model import (
            extract_image_features,
        )
        self._extract_fn = extract_image_features

    def evaluate(self, image_path: str) -> JudgeScore:
        self._validate_image_path(image_path)
        try:
            features = self._extract_fn(image_path)
        except Exception as exc:
            raise JudgeEvaluationError(f"CLIP feature extraction failed on {image_path}: {exc}") from exc

        aspect_keys = ["composition", "lighting", "color", "subject_clarity", "technical"]
        aspects = {k: float(features[k]) for k in aspect_keys if k in features}

        # Equal weighting across 5 aspects represents the balanced aesthetic baseline
        overall = float(stats.fmean(aspects.values())) if aspects else 0.5

        return JudgeScore(
            name=self.name,
            raw_score=round(overall, 4),
            normalized_score=round(max(0.0, min(1.0, overall)), 4),
            aspects=aspects,
        )


class AestheticPredictorV25Judge(BaseJudge):
    """SigLIP-based Aesthetic Predictor V2.5 judge (discus0434).

    Why this judge is included:
    Trained on expert human curation ratings, it provides a strong 1-10 scalar
    evaluation of modern visual appeal. Runs efficiently on CPU or GPU.
    """

    name = "aesthetic_predictor_v2_5"

    def __init__(self, model_id: str = "discus0434/aesthetic-predictor-v2-5") -> None:
        self.model_id = model_id
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from aesthetic_predictor_v2_5 import (
                AestheticPredictorV2_5Model,
                AestheticPredictorV2_5Processor,
            )
            import torch

            # Load in eval mode to prevent gradient tracking during reward/verification passes
            self._model = AestheticPredictorV2_5Model.from_pretrained(self.model_id)
            self._processor = AestheticPredictorV2_5Processor.from_pretrained(self.model_id)
            self._model.eval()
            self._torch = torch
            logger.info("Loaded AestheticPredictorV2_5Model from %s", self.model_id)
        except Exception as exc:
            raise JudgeEvaluationError(
                f"Failed to load aesthetic-predictor-v2-5 ({self.model_id}): {exc}"
            ) from exc

    def evaluate(self, image_path: str) -> JudgeScore:
        self._validate_image_path(image_path)
        self._ensure_loaded()

        try:
            pil_image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=pil_image, return_tensors="pt")
            with self._torch.no_grad():
                outputs = self._model(**inputs)
                raw_val = float(outputs.logits.squeeze().item())

            # AestheticPredictorV2.5 is trained on a 1.0 to 10.0 target range.
            # Normalization projects [1, 10] into [0.0, 1.0] for ensemble aggregation.
            normalized = max(0.0, min(1.0, (raw_val - 1.0) / 9.0))

            return JudgeScore(
                name=self.name,
                raw_score=round(raw_val, 4),
                normalized_score=round(normalized, 4),
            )
        except Exception as exc:
            raise JudgeEvaluationError(f"AestheticPredictor inference failed on {image_path}: {exc}") from exc


class ImscoreLaionJudge(BaseJudge):
    """LAION Aesthetic Scorer via imscore.

    Why this judge is included:
    LAION-Aesthetic is the standard baseline used across open-source diffusion
    and quality curation pipelines. imscore provides a clean, self-contained implementation.
    """

    name = "imscore_laion"

    def __init__(self, model_id: str = "RE-N-Y/laion-aesthetic") -> None:
        self.model_id = model_id
        self._scorer = None

    def _ensure_loaded(self) -> None:
        if self._scorer is not None:
            return
        try:
            from imscore.aesthetic.model import LAIONAestheticScorer
            import torch

            self._scorer = LAIONAestheticScorer.from_pretrained(self.model_id)
            self._scorer.eval()
            self._torch = torch
            logger.info("Loaded LAIONAestheticScorer from %s", self.model_id)
        except Exception as exc:
            raise JudgeEvaluationError(
                f"Failed to load imscore LAION scorer ({self.model_id}): {exc}"
            ) from exc

    def evaluate(self, image_path: str) -> JudgeScore:
        self._validate_image_path(image_path)
        self._ensure_loaded()

        try:
            import torchvision.transforms as T
            pil_image = Image.open(image_path).convert("RGB")
            # LAION scorer expects normalized tensor [0, 1]
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            tensor_img = transform(pil_image).unsqueeze(0)

            with self._torch.no_grad():
                raw_val = float(self._scorer(tensor_img).squeeze().item())

            # LAION aesthetic scores typically range from ~1.0 to ~10.0
            normalized = max(0.0, min(1.0, (raw_val - 1.0) / 9.0))

            return JudgeScore(
                name=self.name,
                raw_score=round(raw_val, 4),
                normalized_score=round(normalized, 4),
            )
        except Exception as exc:
            raise JudgeEvaluationError(f"Imscore LAION inference failed on {image_path}: {exc}") from exc


class JudgeEnsemble:
    """Ensemble orchestrator aggregating diverse frozen judges.

    Why trimmed mean + disagreement tracking:
    - Trimmed mean reduces susceptibility to single-model outliers or adversarial shortcuts.
    - Disagreement (population variance of normalized scores) signals high aesthetic subjectivity,
      preventing the system from falsely projecting high certainty when experts disagree.
    """

    def __init__(
        self,
        judges: Sequence[BaseJudge],
        disagreement_threshold: float = 0.04,
    ) -> None:
        if not judges:
            raise ValueError("JudgeEnsemble requires at least one judge.")
        self.judges = list(judges)
        self.disagreement_threshold = disagreement_threshold

    def evaluate(self, image_path: str) -> JudgeResult:
        """Evaluate an image across all judges in the ensemble."""
        scores_by_name: Dict[str, JudgeScore] = {}
        collected_aspects: Dict[str, float] = {}

        for judge in self.judges:
            try:
                score = judge.evaluate(image_path)
                scores_by_name[judge.name] = score
                if score.aspects:
                    collected_aspects.update(score.aspects)
            except Exception as exc:
                # Log non-fatal judge failure to enable degraded operation if one model fails,
                # but fail completely if no judges succeed.
                logger.warning("Judge %s failed on %s: %s", judge.name, image_path, exc)

        if not scores_by_name:
            raise JudgeEvaluationError(f"All judges failed to evaluate image: {image_path}")

        normalized_vals = [s.normalized_score for s in scores_by_name.values()]

        # Compute trimmed mean if 3 or more judges are present, else plain mean
        if len(normalized_vals) >= 3:
            sorted_vals = sorted(normalized_vals)
            # Trim highest and lowest to resist single-judge exploitation
            trimmed = sorted_vals[1:-1]
            overall = float(stats.fmean(trimmed))
        else:
            overall = float(stats.fmean(normalized_vals))

        # Population variance quantifies inter-judge disagreement
        disagreement = float(stats.pvariance(normalized_vals)) if len(normalized_vals) > 1 else 0.0
        is_uncertain = disagreement >= self.disagreement_threshold

        return JudgeResult(
            image_path=image_path,
            overall_score=round(overall, 4),
            judge_scores=scores_by_name,
            disagreement=round(disagreement, 4),
            is_uncertain=is_uncertain,
            aspects=collected_aspects,
        )

    def evaluate_batch(self, image_paths: Sequence[str]) -> List[JudgeResult]:
        """Evaluate a batch of images sequentially."""
        return [self.evaluate(p) for p in image_paths]

    def rank_images(self, image_paths: Sequence[str]) -> List[JudgeResult]:
        """Rank images from highest to lowest aesthetic score."""
        results = self.evaluate_batch(image_paths)
        return sorted(results, key=lambda r: r.overall_score, reverse=True)


def create_default_ensemble(
    include_heavy_models: bool = False,
    disagreement_threshold: float = 0.04,
) -> JudgeEnsemble:
    """Factory creating the standard judge ensemble.

    By default, initializes with ClipAspectJudge (zero-cost, fast). Heavy models
    (apv2.5, imscore_laion) are enabled when explicit or when weights are cached.
    """
    judges: List[BaseJudge] = [ClipAspectJudge()]

    if include_heavy_models:
        try:
            judges.append(AestheticPredictorV25Judge())
        except Exception as exc:
            logger.warning("Could not attach AestheticPredictorV25Judge: %s", exc)

        try:
            judges.append(ImscoreLaionJudge())
        except Exception as exc:
            logger.warning("Could not attach ImscoreLaionJudge: %s", exc)

    return JudgeEnsemble(judges=judges, disagreement_threshold=disagreement_threshold)
