"""Tests for the frozen Judge Ensemble (AF-0.4).

Why these tests are structured this way:
1. Unit tests mock the BaseJudge to verify aggregation logic (trimmed mean vs plain mean)
   and disagreement-variance thresholding without depending on model weights or GPU.
2. Integration tests run on the 5 calibration images (blur/dark/bright/temp/test.jpg)
   to ensure the roadmap Definition of Done (DoD) is strictly met:
   - blur.jpg has the lowest technical/clarity score
   - test.jpg (reference image) scores strictly higher overall than degraded variants
   - Missing image paths raise JudgeEvaluationError loudly instead of returning defaults
"""
import os
import pytest

from aesfix.judges import (
    BaseJudge,
    ClipAspectJudge,
    JudgeEnsemble,
    JudgeEvaluationError,
    JudgeScore,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BLUR_JPG = os.path.join(_PROJECT_ROOT, "blur.jpg")
_DARK_JPG = os.path.join(_PROJECT_ROOT, "dark.jpg")
_BRIGHT_JPG = os.path.join(_PROJECT_ROOT, "bright.jpg")
_TEMP_JPG = os.path.join(_PROJECT_ROOT, "temp.jpg")
_TEST_JPG = os.path.join(_PROJECT_ROOT, "test.jpg")


class _MockJudge(BaseJudge):
    """Deterministic mock judge for unit testing ensemble aggregation mathematics."""

    def __init__(self, name: str, return_score: float, aspects: dict = None):
        self.name = name
        self._return_score = return_score
        self._aspects = aspects or {}

    def evaluate(self, image_path: str) -> JudgeScore:
        self._validate_image_path(image_path)
        return JudgeScore(
            name=self.name,
            raw_score=self._return_score,
            normalized_score=max(0.0, min(1.0, self._return_score)),
            aspects=self._aspects,
        )


def test_missing_image_path_raises_loudly():
    ensemble = JudgeEnsemble(judges=[_MockJudge("dummy", 0.5)])
    with pytest.raises(JudgeEvaluationError):
        ensemble.evaluate("non_existent_file_xyz.jpg")


def test_ensemble_trimmed_mean_resists_outliers():
    """Verify that with >=3 judges, the highest and lowest scores are trimmed.

    Scores: [0.10, 0.50, 0.60, 0.95]
    Trimming 0.10 and 0.95 leaves [0.50, 0.60] -> mean = 0.55.
    Un-trimmed plain mean would be 0.5375.
    """
    judges = [
        _MockJudge("j1", 0.10),
        _MockJudge("j2", 0.50),
        _MockJudge("j3", 0.60),
        _MockJudge("j4", 0.95),
    ]
    ensemble = JudgeEnsemble(judges=judges)
    result = ensemble.evaluate(_TEST_JPG)

    assert result.overall_score == pytest.approx(0.55, abs=1e-3)


def test_ensemble_disagreement_triggers_uncertainty():
    """Conflicting judges should produce high disagreement variance and set is_uncertain=True."""
    divergent_judges = [
        _MockJudge("optimist", 0.90),
        _MockJudge("pessimist", 0.20),
    ]
    ensemble = JudgeEnsemble(judges=divergent_judges, disagreement_threshold=0.04)
    result = ensemble.evaluate(_TEST_JPG)

    # Variance between 0.9 and 0.2 is (0.35^2 + (-0.35)^2)/2 = 0.1225 > 0.04
    assert result.disagreement > 0.04
    assert result.is_uncertain is True

    consensus_judges = [
        _MockJudge("r1", 0.70),
        _MockJudge("r2", 0.72),
    ]
    ensemble_consensus = JudgeEnsemble(judges=consensus_judges, disagreement_threshold=0.04)
    result_consensus = ensemble_consensus.evaluate(_TEST_JPG)

    assert result_consensus.disagreement < 0.04
    assert result_consensus.is_uncertain is False


def test_af04_calibration_images_ordering_dod():
    """DoD: Verify sane ordering across the 5 calibration images using ClipAspectJudge.

    Expected constraints:
    1. blur.jpg has low clarity and technical score compared to test.jpg.
    2. test.jpg (reference) achieves higher overall quality than degraded images.
    3. dark.jpg has lower lighting score than test.jpg.
    """
    judge = ClipAspectJudge()
    ensemble = JudgeEnsemble(judges=[judge])

    res_test = ensemble.evaluate(_TEST_JPG)
    res_blur = ensemble.evaluate(_BLUR_JPG)
    res_dark = ensemble.evaluate(_DARK_JPG)
    res_bright = ensemble.evaluate(_BRIGHT_JPG)

    # 1. Subject clarity and technical quality of blur must be significantly lower than test.jpg
    assert res_blur.aspects["subject_clarity"] < res_test.aspects["subject_clarity"]
    assert res_blur.aspects["technical"] < res_test.aspects["technical"]

    # 2. Lighting of dark.jpg must be lower than test.jpg and bright.jpg
    assert res_dark.aspects["lighting"] < res_test.aspects["lighting"]
    assert res_dark.aspects["lighting"] < res_bright.aspects["lighting"]

    # 3. Overall aesthetic score of test.jpg must exceed blur and dark variants
    assert res_test.overall_score > res_blur.overall_score
    assert res_test.overall_score > res_dark.overall_score

    # 4. Image ranking utility orders test.jpg above degraded variants
    ranked = ensemble.rank_images([_BLUR_JPG, _DARK_JPG, _TEST_JPG])
    assert ranked[0].image_path == _TEST_JPG
