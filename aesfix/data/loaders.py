"""Standardized Dataset Loaders & Evaluation Metrics for AesFix (AF-0.3).

Why standardized loaders with canonical aspect mapping are necessary:
AADB (10k images, 11 attributes) and PCCD (~4.3k images, aspect critiques) use
differing taxonomies from each other and from modern VLM evaluation benches.
By mapping their raw attributes and critique sections into AesFix's 5 core
photographic dimensions (composition, lighting, color, subject clarity, technical):
1. AADB provides continuous numerical ground truth for training/evaluating aspect probes.
2. PCCD provides gold-standard professional textual critiques for SFT warm-starting.
3. Both feed into a unified evaluation harness calculating SRCC and PLCC correlations.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import statistics as stats
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger("aesfix.data")

# Core aspect taxonomy matching AADB attribute subsets and PCCD critique categories
ASPECT_DIMENSIONS = ("composition", "lighting", "color", "subject_clarity", "technical")


@dataclass
class DatasetSample:
    """Universal sample representation across all aesthetic datasets.

    Enables uniform consumption by critics, fuzzy heads, and evaluation harnesses
    regardless of source dataset schema.
    """
    sample_id: str
    image_path: Optional[str] = None
    overall_score: Optional[float] = None
    aspect_scores: Dict[str, float] = field(default_factory=dict)
    critiques: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_correlation_metrics(
    predictions: Sequence[float],
    ground_truth: Sequence[float],
) -> Dict[str, float]:
    """Calculate SRCC, PLCC, MAE, and MSE between predicted and target scores.

    Why both SRCC and PLCC:
    - Spearman (SRCC) measures monotonic ranking agreement — critical for sorting
      photo candidate variants and evaluating recommendation quality.
    - Pearson (PLCC) measures linear score calibration — necessary for absolute rating.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match ground truth count ({len(ground_truth)})"
        )
    if len(predictions) < 3:
        raise ValueError("At least 3 paired samples are required to compute correlation metrics.")

    preds = [float(p) for p in predictions]
    targets = [float(t) for t in ground_truth]

    # Handle zero-variance degeneracies gracefully to prevent NaN crashes
    if len(set(preds)) == 1 or len(set(targets)) == 1:
        return {"srcc": 0.0, "plcc": 0.0, "mae": 0.0, "mse": 0.0, "num_samples": float(len(preds))}

    srcc, _ = spearmanr(preds, targets)
    plcc, _ = pearsonr(preds, targets)

    differences = [abs(p - t) for p, t in zip(preds, targets)]
    mae = float(stats.fmean(differences))
    mse = float(stats.fmean([d ** 2 for d in differences]))

    return {
        "srcc": round(float(srcc), 4),
        "plcc": round(float(plcc), 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "num_samples": float(len(preds)),
    }


class AADBLoader:
    """Loader for the Aesthetics and Attributes Database (Kong et al., ECCV 2016).

    Why this mapping is used:
    AADB annotates 11 specific visual attributes. We project these into our 5 dimensions:
    - composition <- rule_of_thirds, balancing_elements, symmetry
    - lighting    <- light
    - color       <- color_harmony, vivid_color
    - clarity     <- depth_of_field, object_emphasis
    - technical   <- motion_blur (inverted, since motion blur degrades technical quality)
    """

    AADB_RAW_ATTRIBUTES = (
        "balancing_elements", "color_harmony", "content", "do_depth_of_field",
        "light", "motion_blur", "repetition", "rule_of_thirds", "symmetry", "vivid_color"
    )

    def __init__(self, dataset_dir: Optional[str] = None) -> None:
        self.dataset_dir = dataset_dir

    @staticmethod
    def map_attributes_to_aspects(raw_attributes: Dict[str, float]) -> Dict[str, float]:
        """Project 11 AADB attributes into the 5 canonical photographic aspects."""
        aspects: Dict[str, float] = {}

        # Composition incorporates structural arrangement attributes
        comp_keys = [k for k in ("rule_of_thirds", "balancing_elements", "symmetry") if k in raw_attributes]
        if comp_keys:
            aspects["composition"] = round(stats.fmean([raw_attributes[k] for k in comp_keys]), 4)

        if "light" in raw_attributes:
            aspects["lighting"] = round(float(raw_attributes["light"]), 4)

        color_keys = [k for k in ("color_harmony", "vivid_color") if k in raw_attributes]
        if color_keys:
            aspects["color"] = round(stats.fmean([raw_attributes[k] for k in color_keys]), 4)

        clarity_keys = [k for k in ("do_depth_of_field", "content") if k in raw_attributes]
        if clarity_keys:
            aspects["subject_clarity"] = round(stats.fmean([raw_attributes[k] for k in clarity_keys]), 4)

        # Invert motion_blur: higher motion blur reduces technical rating
        if "motion_blur" in raw_attributes:
            aspects["technical"] = round(max(0.0, 1.0 - float(raw_attributes["motion_blur"])), 4)

        return aspects

    def load_from_json(self, json_path: str) -> List[DatasetSample]:
        """Load AADB samples from pre-processed JSON."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"AADB annotation file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        samples: List[DatasetSample] = []
        for item in data:
            raw_attrs = item.get("attributes", {})
            mapped_aspects = self.map_attributes_to_aspects(raw_attrs)

            samples.append(
                DatasetSample(
                    sample_id=str(item.get("id") or item.get("image_name", "")),
                    image_path=item.get("image_path"),
                    overall_score=item.get("score") or item.get("overall_score"),
                    aspect_scores=mapped_aspects,
                    metadata={"raw_attributes": raw_attrs},
                )
            )
        return samples

    def load_from_csv(self, csv_path: str) -> List[DatasetSample]:
        """Load AADB samples from raw CSV table."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"AADB CSV not found: {csv_path}")

        samples: List[DatasetSample] = []
        with open(csv_path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw_attrs = {}
                for k, v in row.items():
                    if k in self.AADB_RAW_ATTRIBUTES:
                        try:
                            raw_attrs[k] = float(v)
                        except (ValueError, TypeError):
                            continue

                mapped_aspects = self.map_attributes_to_aspects(raw_attrs)
                score_str = row.get("score") or row.get("overall_score")
                overall = float(score_str) if score_str is not None else None

                img_path = row.get("image_path") or row.get("image_name")
                if img_path and self.dataset_dir:
                    img_path = os.path.join(self.dataset_dir, img_path)

                samples.append(
                    DatasetSample(
                        sample_id=str(row.get("id") or row.get("image_name", "")),
                        image_path=img_path,
                        overall_score=overall,
                        aspect_scores=mapped_aspects,
                        metadata={"raw_attributes": raw_attrs},
                    )
                )
        return samples


class PCCDLoader:
    """Loader for the Photo Critique Caption Dataset (Chang et al., 2017).

    Why this mapping is used:
    PCCD critiques are written by professional photographers categorized by sections:
    - "Composition and Framing"        -> composition
    - "Use of Camera and Lighting"     -> lighting
    - "Color and Tone"                -> color
    - "Subject of the Photo"          -> subject_clarity
    - "General Impression"            -> overall summary / guidance
    """

    PCCD_CATEGORY_MAP = {
        "Composition and Framing": "composition",
        "Use of Camera and Lighting": "lighting",
        "Color and Tone": "color",
        "Subject of the Photo": "subject_clarity",
        "General Impression": "overall",
    }

    def __init__(self, dataset_dir: Optional[str] = None) -> None:
        self.dataset_dir = dataset_dir

    def load_from_json(self, json_path: str) -> List[DatasetSample]:
        """Load PCCD structured critique comments from JSON."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"PCCD file not found: {json_path}")

        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        samples: List[DatasetSample] = []
        for item in data:
            categorized_critiques: Dict[str, List[str]] = {aspect: [] for aspect in ASPECT_DIMENSIONS}
            categorized_critiques["overall"] = []

            # Handle both dictionary-of-sections and list-of-comments schemas
            raw_sections = item.get("critiques") or item.get("comments") or {}
            if isinstance(raw_sections, dict):
                for cat, text in raw_sections.items():
                    target_aspect = self.PCCD_CATEGORY_MAP.get(cat, "overall")
                    if isinstance(text, list):
                        categorized_critiques.setdefault(target_aspect, []).extend(text)
                    elif isinstance(text, str) and text.strip():
                        categorized_critiques.setdefault(target_aspect, []).append(text.strip())
            elif isinstance(raw_sections, list):
                # Generic critique list mapped to overall
                categorized_critiques["overall"].extend([str(t) for t in raw_sections])

            img_path = item.get("image_path") or item.get("image")
            if img_path and self.dataset_dir:
                img_path = os.path.join(self.dataset_dir, img_path)

            samples.append(
                DatasetSample(
                    sample_id=str(item.get("id", "")),
                    image_path=img_path,
                    overall_score=item.get("score") or item.get("sentiment_score"),
                    critiques=categorized_critiques,
                    metadata=item.get("metadata", {}),
                )
            )
        return samples


class EvaluationHarness:
    """Benchmark evaluation harness running model predictors against dataset ground truth."""

    def __init__(self, samples: Sequence[DatasetSample]) -> None:
        self.samples = list(samples)

    def evaluate_model(
        self,
        predictor_fn: Callable[[DatasetSample], Tuple[float, Optional[Dict[str, float]]]],
    ) -> Dict[str, Any]:
        """Evaluate a predictor function yielding (overall_score, Optional[aspect_scores]).

        Reports overall SRCC/PLCC plus per-aspect correlations whenever aspect ground truth exists.
        """
        predicted_scores: List[float] = []
        target_scores: List[float] = []
        aspect_preds: Dict[str, List[float]] = {dim: [] for dim in ASPECT_DIMENSIONS}
        aspect_targets: Dict[str, List[float]] = {dim: [] for dim in ASPECT_DIMENSIONS}

        for sample in self.samples:
            if sample.overall_score is None:
                continue

            pred_overall, pred_aspects = predictor_fn(sample)
            predicted_scores.append(float(pred_overall))
            target_scores.append(float(sample.overall_score))

            if pred_aspects:
                for dim in ASPECT_DIMENSIONS:
                    if dim in pred_aspects and dim in sample.aspect_scores:
                        aspect_preds[dim].append(float(pred_aspects[dim]))
                        aspect_targets[dim].append(float(sample.aspect_scores[dim]))

        results: Dict[str, Any] = {
            "overall": compute_correlation_metrics(predicted_scores, target_scores)
        }

        # Calculate per-aspect correlations if sufficient samples exist
        aspect_results: Dict[str, Dict[str, float]] = {}
        for dim in ASPECT_DIMENSIONS:
            if len(aspect_preds[dim]) >= 3:
                aspect_results[dim] = compute_correlation_metrics(aspect_preds[dim], aspect_targets[dim])
        results["aspects"] = aspect_results

        return results
