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

import io
import logging
import os
import statistics as stats
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
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
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            from aesthetic_predictor_v2_5 import (  # type: ignore[import-untyped,import-not-found]
                AestheticPredictorV2_5Model,
                AestheticPredictorV2_5Processor,
            )

            # Load in eval mode to prevent gradient tracking during reward/verification passes
            model = AestheticPredictorV2_5Model.from_pretrained(self.model_id)
            model.eval()
            self._model = model
            self._processor = AestheticPredictorV2_5Processor.from_pretrained(self.model_id)
            logger.info("Loaded AestheticPredictorV2_5Model from %s", self.model_id)
        except Exception as exc:
            raise JudgeEvaluationError(
                f"Failed to load aesthetic-predictor-v2-5 ({self.model_id}): {exc}"
            ) from exc

    def evaluate(self, image_path: str) -> JudgeScore:
        self._validate_image_path(image_path)
        self._ensure_loaded()
        if self._model is None or self._processor is None:
            raise JudgeEvaluationError(f"AestheticPredictor not initialized for {self.model_id}")

        try:
            pil_image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
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
        self._scorer: Optional[Any] = None

    def _ensure_loaded(self) -> None:
        if self._scorer is not None:
            return
        try:
            from imscore.aesthetic.model import LAIONAestheticScorer  # type: ignore[import-untyped,import-not-found]

            scorer = LAIONAestheticScorer.from_pretrained(self.model_id)
            scorer.eval()
            self._scorer = scorer
            logger.info("Loaded LAIONAestheticScorer from %s", self.model_id)
        except Exception as exc:
            raise JudgeEvaluationError(
                f"Failed to load imscore LAION scorer ({self.model_id}): {exc}"
            ) from exc

    def evaluate(self, image_path: str) -> JudgeScore:
        self._validate_image_path(image_path)
        self._ensure_loaded()
        if self._scorer is None:
            raise JudgeEvaluationError(f"Imscore scorer not initialized for {self.model_id}")

        try:
            import torchvision.transforms as T  # type: ignore[import-untyped,import-not-found]
            pil_image = Image.open(image_path).convert("RGB")
            # LAION scorer expects normalized tensor [0, 1]
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
            ])
            t_out = transform(pil_image)
            tensor_img = t_out.unsqueeze(0) if isinstance(t_out, torch.Tensor) else torch.as_tensor(t_out).unsqueeze(0)

            with torch.no_grad():
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


def _find_hf_snapshot(repo_id: str) -> Optional[str]:
    """Finds an existing snapshot directory for a model in local HF caches."""
    cache_roots = []
    if "HF_HOME" in os.environ:
        cache_roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    cache_roots.extend([
        r"D:\aesfix-cache\hf\hub",
        os.path.expanduser("~/.cache/huggingface/hub"),
    ])
    repo_folder = "models--" + repo_id.replace("/", "--")
    for root in cache_roots:
        snaps = os.path.join(root, repo_folder, "snapshots")
        if os.path.exists(snaps):
            subdirs = [os.path.join(snaps, d) for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d))]
            if subdirs:
                return subdirs[0]
    return None


class RsinemaAestheticJudge(BaseJudge):
    """Independent aesthetic regression judge (rsinema/aesthetic-scorer).

    Why this judge is required (Task D3.a):
    The critic proposes fixes based on zero-shot CLIP aspect text prompts.
    To avoid self-referential bias and Goodhart reward-hacking, gain verification
    must use an independent judge trained specifically on aesthetic ratings.
    """

    name = "rsinema/aesthetic-scorer"

    def __init__(self, model_id: str = "rsinema/aesthetic-scorer") -> None:
        self.model_id = model_id
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._head_weight: Optional[torch.Tensor] = None
        self._head_bias: Optional[torch.Tensor] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        rsinema_snap = _find_hf_snapshot(self.model_id)
        clip_snap = _find_hf_snapshot("openai/clip-vit-base-patch32")

        if not rsinema_snap or not clip_snap:
            raise JudgeEvaluationError(
                f"Missing local cache snapshots for {self.model_id} or openai/clip-vit-base-patch32"
            )

        try:
            from transformers import CLIPProcessor, CLIPVisionModel

            pt_path = os.path.join(rsinema_snap, "model.pt")
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Weight file not found at {pt_path}")

            state = torch.load(pt_path, map_location="cpu", weights_only=False)

            vision = CLIPVisionModel.from_pretrained(clip_snap, local_files_only=True)
            vision.eval()
            proc = CLIPProcessor.from_pretrained(clip_snap, local_files_only=True)

            clean_backbone = {}
            for k, v in state.items():
                if k.startswith("backbone."):
                    clean_backbone[k.replace("backbone.", "")] = v

            vision.vision_model.load_state_dict(clean_backbone, strict=False)

            self._model = vision
            self._processor = proc
            self._head_weight = state["aesthetic_head.0.weight"]
            self._head_bias = state["aesthetic_head.0.bias"]
            logger.info("Loaded RsinemaAestheticJudge from %s", rsinema_snap)
        except Exception as exc:
            raise JudgeEvaluationError(f"Failed to initialize {self.model_id}: {exc}") from exc

    def evaluate(self, image_path: str) -> JudgeScore:
        self._validate_image_path(image_path)
        self._ensure_loaded()
        if self._model is None or self._processor is None:
            raise JudgeEvaluationError(f"Model not initialized for {self.model_id}")

        try:
            pil_image = Image.open(image_path).convert("RGB")
            inputs = self._processor(images=pil_image, return_tensors="pt")
            with torch.no_grad():
                out = self._model(**inputs)
                cls_token = out.pooler_output
                raw_score = (cls_token @ self._head_weight.T + self._head_bias).item()

            norm_score = max(0.0, min(1.0, (raw_score - 1.0) / 9.0))
            return JudgeScore(
                name=self.name,
                raw_score=round(raw_score, 4),
                normalized_score=round(norm_score, 4),
            )
        except Exception as exc:
            raise JudgeEvaluationError(f"Rsinema inference failed on {image_path}: {exc}") from exc


def compute_noise_floor(
    image_input: Union[str, Path, Image.Image],
    judge_or_ensemble: Any,
) -> float:
    """Computes the per-image measurement noise floor ε (Task D3.b).

    Perturbs the ORIGINAL image across 4 transformations:
    1. JPEG re-encode (quality=90)
    2. JPEG re-encode (quality=95)
    3. Bilinear resize +2%
    4. Bilinear resize -2%

    ε = max |Δ| over these 4 perturbations.
    """
    if isinstance(image_input, (str, Path)):
        orig = Image.open(str(image_input)).convert("RGB")
    else:
        orig = image_input.convert("RGB")

    def _eval(img: Image.Image) -> float:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            img.save(tf.name, format="PNG")
            temp_path = tf.name
        try:
            res = judge_or_ensemble.evaluate(temp_path)
            if hasattr(res, "overall_score"):
                return float(res.overall_score)
            elif hasattr(res, "normalized_score"):
                return float(res.normalized_score)
            return float(res)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    baseline_score = _eval(orig)

    # 1. JPEG q90
    buf90 = io.BytesIO()
    orig.save(buf90, format="JPEG", quality=90)
    buf90.seek(0)
    s_q90 = _eval(Image.open(buf90))

    # 2. JPEG q95
    buf95 = io.BytesIO()
    orig.save(buf95, format="JPEG", quality=95)
    buf95.seek(0)
    s_q95 = _eval(Image.open(buf95))

    # 3. +2% resize
    w, h = orig.size
    w_up, h_up = max(1, int(round(w * 1.02))), max(1, int(round(h * 1.02)))
    img_up = orig.resize((w_up, h_up), Image.Resampling.BILINEAR)
    s_up = _eval(img_up)

    # 4. -2% resize
    w_down, h_down = max(1, int(round(w * 0.98))), max(1, int(round(h * 0.98)))
    img_down = orig.resize((w_down, h_down), Image.Resampling.BILINEAR)
    s_down = _eval(img_down)

    deltas = [
        abs(s_q90 - baseline_score),
        abs(s_q95 - baseline_score),
        abs(s_up - baseline_score),
        abs(s_down - baseline_score),
    ]
    epsilon = round(float(max(deltas)), 4)
    # Guarantee at least 0.001 noise floor to prevent false claims on degenerate uniform inputs
    return max(0.0010, epsilon)


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

    Why (Task D3.a): Enforces judge independence so that the gain judge is not the
    critic's own zero-shot CLIP aspect probe. Attempts to load the independent
    rsinema/aesthetic-scorer from local cache. If unavailable, falls back to
    ClipAspectJudge labelled explicitly as a same-family non-independent judge.
    """
    judges: List[BaseJudge] = []
    try:
        rsinema = RsinemaAestheticJudge()
        rsinema._ensure_loaded()
        judges.append(rsinema)
        logger.info("Ensemble initialized with independent judge: %s", rsinema.name)
    except Exception as exc:
        logger.warning("Could not load independent judge rsinema/aesthetic-scorer: %s", exc)
        clip_fallback = ClipAspectJudge()
        clip_fallback.name = "clip_aspect (same-family judge, not independent)"
        judges.append(clip_fallback)

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
