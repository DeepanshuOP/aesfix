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
    RsinemaAestheticJudge,
)

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_BLUR_JPG = os.path.join(_PROJECT_ROOT, "blur.jpg")
_DARK_JPG = os.path.join(_PROJECT_ROOT, "dark.jpg")
_BRIGHT_JPG = os.path.join(_PROJECT_ROOT, "bright.jpg")
_TEMP_JPG = os.path.join(_PROJECT_ROOT, "temp.jpg")
_TEST_JPG = os.path.join(_PROJECT_ROOT, "test.jpg")


class _MockJudge(BaseJudge):
    """Deterministic mock judge for unit testing ensemble aggregation mathematics."""

    def __init__(self, name: str, return_score: float, aspects: dict = None, family: str = "clip"):
        self.name = name
        self.family = family
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


def test_rsinema_aesthetic_judge_loads_zero_key_counts_and_all_heads():
    """Task D3b: Verify RsinemaAestheticJudge backbone key counts, heads, output scale and label."""
    judge = RsinemaAestheticJudge()
    judge._ensure_loaded()

    # 1. Label verification
    expected_label = (
        "fine-tuned CLIP ViT-B/32 aesthetic judge "
        "(same backbone family as the critic — partially independent)"
    )
    assert judge.name == expected_label

    # 2. All 7 heads loaded
    assert set(judge._heads.keys()) == {
        "aesthetic",
        "quality",
        "composition",
        "light",
        "color",
        "dof",
        "content",
    }

    # 3. Evaluate image
    score = judge.evaluate(_TEST_JPG)

    # 4. Raw score reported directly without (raw-1)/9 mapping
    assert score.raw_score == score.normalized_score
    assert score.name == expected_label

    # 5. Named per-aspect scores labelled 'unvalidated'
    expected_aspect_names = {
        "quality (unvalidated)",
        "composition (unvalidated)",
        "light (unvalidated)",
        "color (unvalidated)",
        "dof (unvalidated)",
        "content (unvalidated)",
    }
    assert score.aspects is not None
    assert set(score.aspects.keys()) == expected_aspect_names
    for v in score.aspects.values():
        assert isinstance(v, float)


def test_rsinema_aesthetic_judge_fails_if_key_counts_nonzero(monkeypatch):
    """Task D3b: Verify loading fails loudly if backbone key counts are non-zero."""
    import torch

    judge = RsinemaAestheticJudge()
    original_torch_load = torch.load

    def mocked_torch_load(*args, **kwargs):
        state = original_torch_load(*args, **kwargs)
        # Inject an unexpected key into backbone to simulate mismatched key mapping
        corrupted_state = dict(state)
        corrupted_state["backbone.unexpected_tensor_key"] = torch.zeros(1)
        return corrupted_state

    monkeypatch.setattr(torch, "load", mocked_torch_load)
    with pytest.raises(JudgeEvaluationError, match="Non-zero key counts loading backbone"):
        judge._ensure_loaded()


def test_musiq_aesthetic_judge_loads_on_cpu():
    """AQ3: Verify MUSIQ loads on CPU with exact id 'musiq-koniq' and 'non-clip' family."""
    from aesfix.judges.ensemble import MusiqAestheticJudge

    judge = MusiqAestheticJudge(device="cpu")
    judge._ensure_loaded()

    assert judge.family == "non-clip"
    assert judge.exact_id == "musiq-koniq"
    assert "non-CLIP" in judge.name

    # Evaluate test.jpg
    score = judge.evaluate(_TEST_JPG)
    assert 0.0 <= score.raw_score <= 100.0
    assert 0.0 <= score.normalized_score <= 1.0
    # Normalized score should be raw / 100.0
    assert score.normalized_score == pytest.approx(score.raw_score / 100.0, abs=1e-3)


def test_compute_per_judge_noise_floors():
    """AQ3: Verify compute_per_judge_noise_floors returns noise floor >= 0.001 per judge."""
    from aesfix.judges.ensemble import compute_per_judge_noise_floors

    j1 = _MockJudge("j_clip", 0.60, family="clip")
    j2 = _MockJudge("j_nonclip", 0.70, family="non-clip")
    ensemble = JudgeEnsemble(judges=[j1, j2])

    floors = compute_per_judge_noise_floors(_TEST_JPG, ensemble)
    assert "j_clip" in floors
    assert "j_nonclip" in floors
    assert floors["j_clip"] >= 0.0010
    assert floors["j_nonclip"] >= 0.0010


def test_multi_judge_family_agreement_rules():
    """AQ3: Verify multi-judge agreement requires >= 2 independent families to agree."""
    from aesfix.agent.loop import AesFixAgent
    class _LocalFakeLLM:
        def __init__(self, responses):
            self.responses = list(responses)
            self.provider = "fake"
            self.model_id = "fake"
        def generate(self, *args, **kwargs):
            from aesfix.agent.llm import LLMResponse
            return LLMResponse(content=self.responses.pop(0), provider="fake", model_id="fake")

    import json
    from unittest.mock import patch
    from aesfix.executor.guard import GuardVerdict

    prop = json.dumps({
        "op": "adjust_exposure",
        "params": {"ev_delta": 0.25},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "Exposure lift test.",
    })
    fake_llm = _LocalFakeLLM([prop])

    # Two judges from two independent families
    j_clip = _MockJudge("clip_judge", 0.50, family="clip")
    j_nonclip = _MockJudge("nonclip_judge", 0.50, family="non-clip")
    ensemble = JudgeEnsemble(judges=[j_clip, j_nonclip])

    agent = AesFixAgent(llm_client=fake_llm, max_steps=1)
    agent.ensemble = ensemble

    # Case 1: Disagreement between families (CLIP improves by 0.05 > 0.01, but non-CLIP improves by 0.00 <= 0.01)
    def mock_extract(path, ens, crt):
        if "in.jpg" in path or "initial" in path:
            return ({"lighting": 0.50}, 0.50)
        return ({"lighting": 0.55}, 0.55)

    with patch.object(agent, "_extract_aspect_scores", side_effect=mock_extract), \
         patch.object(ensemble, "evaluate") as mock_eval, \
         patch("aesfix.agent.loop.compute_noise_floor", return_value=0.01), \
         patch("aesfix.agent.loop.compute_per_judge_noise_floors", return_value={"clip_judge": 0.01, "nonclip_judge": 0.01}), \
         patch.object(agent.guard, "evaluate", return_value=GuardVerdict(
             is_valid=True, composite_drift=0.02, feature_drift=0.01, structural_drift=0.01,
             threshold=0.35, backend_name="rgb_layout_stat", guard_backend="SSIM + rgb_layout_stat",
         )):
        # Candidate eval returns improvement for clip_judge only
        from aesfix.judges.ensemble import JudgeResult, JudgeScore
        mock_base = JudgeResult(
            image_path="base", overall_score=0.50,
            judge_scores={"clip_judge": JudgeScore("clip_judge", 0.50, 0.50), "nonclip_judge": JudgeScore("nonclip_judge", 0.50, 0.50)}
        )
        mock_cand_disagree = JudgeResult(
            image_path="cand", overall_score=0.53,
            judge_scores={"clip_judge": JudgeScore("clip_judge", 0.55, 0.55), "nonclip_judge": JudgeScore("nonclip_judge", 0.50, 0.50)}
        )
        mock_eval.side_effect = [mock_base, mock_cand_disagree]

        traj_disagree = agent.run(_TEST_JPG)

    assert len(traj_disagree.steps) == 1
    assert traj_disagree.steps[0].status == "rejected"
    assert "Multi-judge agreement rejected" in traj_disagree.steps[0].verdict_reason
    assert traj_disagree.steps[0].judge_deltas["clip_judge"] == 0.05
    assert traj_disagree.steps[0].judge_deltas["nonclip_judge"] == 0.0

    # Case 2: Agreement between families (both CLIP and non-CLIP improve by > 0.01)
    fake_llm_agree = _LocalFakeLLM([prop])
    agent_agree = AesFixAgent(llm_client=fake_llm_agree, max_steps=1)
    agent_agree.ensemble = ensemble

    with patch.object(agent_agree, "_extract_aspect_scores", side_effect=mock_extract), \
         patch.object(ensemble, "evaluate") as mock_eval_agree, \
         patch("aesfix.agent.loop.compute_noise_floor", return_value=0.01), \
         patch("aesfix.agent.loop.compute_per_judge_noise_floors", return_value={"clip_judge": 0.01, "nonclip_judge": 0.01}), \
         patch.object(agent_agree.guard, "evaluate", return_value=GuardVerdict(
             is_valid=True, composite_drift=0.02, feature_drift=0.01, structural_drift=0.01,
             threshold=0.35, backend_name="rgb_layout_stat", guard_backend="SSIM + rgb_layout_stat",
         )):
        mock_cand_agree = JudgeResult(
            image_path="cand", overall_score=0.55,
            judge_scores={"clip_judge": JudgeScore("clip_judge", 0.55, 0.55), "nonclip_judge": JudgeScore("nonclip_judge", 0.54, 0.54)}
        )
        mock_eval_agree.side_effect = [mock_base, mock_cand_agree]
        traj_agree = agent_agree.run(_TEST_JPG)

    assert len(traj_agree.steps) == 1
    assert traj_agree.steps[0].status == "accepted"
    assert traj_agree.steps[0].judge_deltas["clip_judge"] == 0.05
    assert traj_agree.steps[0].judge_deltas["nonclip_judge"] == 0.04


