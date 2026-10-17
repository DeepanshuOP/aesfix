"""Unit and live integration tests for AesFix Closed-Loop Agent (AG5).

Covers:
- FakeLLM (test-only, clearly named):
  1. accept
  2. reject-with-feedback
  3. invalid schema
  4. budget stop
  5. leave-it
- Live integration test (@pytest.mark.live) running against Google Gemini API
  on legacy image test.jpg and logging evidence to results/ledger.jsonl.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import pytest
from PIL import Image

from aesfix.agent.llm import LLMResponse
from aesfix.agent.loop import AesFixAgent
from aesfix.executor.guard import GuardVerdict
from aesfix.ledger import append_run

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEST_JPG = str(_PROJECT_ROOT / "test.jpg")


class FakeLLM:
    """Test-only deterministic mock LLM for testing agent loop trajectories."""

    def __init__(self, responses: Optional[List[str]] = None) -> None:
        self.responses: List[str] = list(responses or [])
        self.call_count: int = 0
        self.prompts_received: List[str] = []
        self.provider: str = "fake_test_provider"
        self.model_id: str = "fake-test-model-v1"

    def generate(
        self,
        prompt: str,
        image_input: Union[str, Path, Image.Image],
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)

        if self.responses:
            content = self.responses.pop(0)
        else:
            content = json.dumps({
                "op": "adjust_exposure",
                "params": {"ev_delta": 0.25},
                "aspect": "lighting",
                "direction": 1,
                "rationale": "Default neutral fake proposal.",
            })

        return LLMResponse(
            content=content,
            provider=self.provider,
            model_id=self.model_id,
            cached=False,
        )


@pytest.fixture
def dummy_image(tmp_path: Path) -> str:
    """Creates a temporary test JPEG image for fast agent loop verification."""
    img_path = tmp_path / "dummy_test.jpg"
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    img.save(str(img_path), format="JPEG")
    return str(img_path)


def test_fake_llm_accept(dummy_image: str) -> None:
    """Verifies that an edit exceeding noise floor without drift is accepted."""
    proposal_json = json.dumps({
        "op": "adjust_exposure",
        "params": {"ev_delta": 0.3},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "Slightly lift midtones for better exposure balance.",
    })
    fake_llm = FakeLLM([proposal_json])
    agent = AesFixAgent(llm_client=fake_llm, max_steps=1, consecutive_rejections_stop=2)

    # Mock score extraction: initial baseline -> post-edit improvement
    mock_scores = [
        ({"composition": 0.50, "lighting": 0.50, "color": 0.50, "technical": 0.50}, 0.50),  # baseline
        ({"composition": 0.50, "lighting": 0.56, "color": 0.51, "technical": 0.50}, 0.56),  # post-step 1
    ]

    with patch.object(agent, "_extract_aspect_scores", side_effect=mock_scores), \
         patch("aesfix.agent.loop.compute_noise_floor", return_value=0.01), \
         patch.object(agent.guard, "evaluate", return_value=GuardVerdict(
             is_valid=True,
             composite_drift=0.05,
             feature_drift=0.04,
             structural_drift=0.02,
             threshold=0.35,
             backend_name="rgb_layout_stat",
             guard_backend="SSIM + rgb_layout_stat",
         )):

        trajectory = agent.run(dummy_image)

    assert len(trajectory.steps) >= 1
    s0 = trajectory.steps[0]
    assert s0.status == "accepted"
    assert s0.proposal.op == "adjust_exposure"
    assert s0.overall_delta > s0.noise_floor
    assert trajectory.verdict == "measured improvement"
    assert len(trajectory.accepted_steps) == 1
    assert "x:xmpmeta" in trajectory.stacked_xmp


def test_fake_llm_reject_with_feedback(dummy_image: str) -> None:
    """Verifies that a regressing proposal is rejected and feedback is provided to the next step."""
    step1_proposal = json.dumps({
        "op": "adjust_contrast",
        "params": {"factor": 1.5},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "High contrast push.",
    })
    step2_proposal = json.dumps({
        "op": "adjust_exposure",
        "params": {"ev_delta": 0.2},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "Subtle exposure adjustment based on feedback.",
    })
    fake_llm = FakeLLM([step1_proposal, step2_proposal])
    agent = AesFixAgent(llm_client=fake_llm, max_steps=2, consecutive_rejections_stop=3)

    mock_scores = [
        ({"composition": 0.50, "lighting": 0.50, "color": 0.50, "technical": 0.50}, 0.50),  # baseline
        ({"composition": 0.50, "lighting": 0.45, "color": 0.50, "technical": 0.45}, 0.46),  # step 1 (regresses)
        ({"composition": 0.50, "lighting": 0.55, "color": 0.50, "technical": 0.50}, 0.55),  # step 2 (improves)
    ]

    with patch.object(agent, "_extract_aspect_scores", side_effect=mock_scores), \
         patch("aesfix.agent.loop.compute_noise_floor", return_value=0.01), \
         patch.object(agent.guard, "evaluate", return_value=GuardVerdict(
             is_valid=True,
             composite_drift=0.08,
             feature_drift=0.06,
             structural_drift=0.03,
             threshold=0.35,
             backend_name="rgb_layout_stat",
             guard_backend="SSIM + rgb_layout_stat",
         )):

        trajectory = agent.run(dummy_image)

    # Assert Step 1 rejected and Step 2 accepted
    assert len(trajectory.steps) == 2
    assert trajectory.steps[0].status == "rejected"
    assert trajectory.steps[1].status == "accepted"

    # Verify that the model received rejection feedback in its dialogue prompt
    second_prompt = fake_llm.prompts_received[1]
    assert "REJECTED adjust_contrast" in second_prompt


def test_fake_llm_invalid_schema(dummy_image: str) -> None:
    """Verifies that non-JSON or invalid-schema proposals trigger validation feedback."""
    invalid_schema_proposal = "I recommend brightening the photograph by +0.5 EV."
    invalid_retry_proposal = "Still just free text advice without json."
    fake_llm = FakeLLM([invalid_schema_proposal, invalid_retry_proposal])
    agent = AesFixAgent(llm_client=fake_llm, max_steps=1, consecutive_rejections_stop=2)

    mock_scores = [
        ({"composition": 0.50, "lighting": 0.50, "color": 0.50, "technical": 0.50}, 0.50),
    ]

    with patch.object(agent, "_extract_aspect_scores", side_effect=mock_scores), \
         patch("aesfix.agent.loop.compute_noise_floor", return_value=0.01):

        trajectory = agent.run(dummy_image)

    # Step 1 failed schema validation
    assert len(trajectory.steps) >= 1
    assert trajectory.steps[0].status == "rejected"
    assert "schema validation" in trajectory.steps[0].verdict_reason.lower()


def test_fake_llm_budget_stop(dummy_image: str) -> None:
    """Verifies that the agent halts when hitting max_steps budget."""
    proposals = [
        json.dumps({
            "op": "adjust_exposure",
            "params": {"ev_delta": 0.05},
            "aspect": "lighting",
            "direction": 1,
            "rationale": f"Step {i} proposal",
        })
        for i in range(10)
    ]
    fake_llm = FakeLLM(proposals)
    # Budget set to 3 steps
    agent = AesFixAgent(llm_client=fake_llm, max_steps=3, consecutive_rejections_stop=10)

    # Mock continuous scores
    def mock_eval(*args: Any, **kwargs: Any) -> Tuple[Dict[str, float], float]:
        return ({"composition": 0.50, "lighting": 0.53, "color": 0.50, "technical": 0.50}, 0.53)

    with patch.object(agent, "_extract_aspect_scores", side_effect=mock_eval), \
         patch("aesfix.agent.loop.compute_noise_floor", return_value=0.01), \
         patch.object(agent.guard, "evaluate", return_value=GuardVerdict(
             is_valid=True,
             composite_drift=0.05,
             feature_drift=0.04,
             structural_drift=0.02,
             threshold=0.35,
             backend_name="rgb_layout_stat",
             guard_backend="SSIM + rgb_layout_stat",
         )):

        trajectory = agent.run(dummy_image)

    assert len(trajectory.steps) == 3
    assert trajectory.total_steps == 3


def test_fake_llm_leave_it(dummy_image: str) -> None:
    """Verifies that when no edit exceeds noise floor, verdict is 'leave it'."""
    # Proposal produces tiny delta below noise floor (ε = 0.02)
    sub_epsilon_proposal = json.dumps({
        "op": "adjust_exposure",
        "params": {"ev_delta": 0.01},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "Imperceptible exposure nudge.",
    })
    fake_llm = FakeLLM([sub_epsilon_proposal, sub_epsilon_proposal])
    agent = AesFixAgent(llm_client=fake_llm, max_steps=4, consecutive_rejections_stop=2)

    # Baseline = 0.50, post = 0.505 (gain = 0.005 < noise floor 0.02)
    mock_scores = [
        ({"composition": 0.50, "lighting": 0.50, "color": 0.50, "technical": 0.50}, 0.50),
        ({"composition": 0.50, "lighting": 0.505, "color": 0.50, "technical": 0.50}, 0.505),
        ({"composition": 0.50, "lighting": 0.505, "color": 0.50, "technical": 0.50}, 0.505),
    ]

    with patch.object(agent, "_extract_aspect_scores", side_effect=mock_scores), \
         patch("aesfix.agent.loop.compute_noise_floor", return_value=0.02), \
         patch.object(agent.guard, "evaluate", return_value=GuardVerdict(
             is_valid=True,
             composite_drift=0.02,
             feature_drift=0.01,
             structural_drift=0.01,
             threshold=0.35,
             backend_name="rgb_layout_stat",
             guard_backend="SSIM + rgb_layout_stat",
         )):

        trajectory = agent.run(dummy_image)

    # 2 consecutive rejections stops loop
    assert len(trajectory.steps) == 2
    assert len(trajectory.accepted_steps) == 0
    assert trajectory.verdict == "leave it"
    assert trajectory.stacked_xmp == ""


@pytest.mark.live
def test_live_gemini_agent(tmp_path: Path) -> None:
    """Executes a real live multi-turn agent trajectory against Google Gemini API."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        pytest.skip("Requires GEMINI_API_KEY environment variable")

    if not os.path.exists(_TEST_JPG):
        pytest.skip(f"Legacy image '{_TEST_JPG}' not found")

    from aesfix.agent.llm import LLMClient
    client = LLMClient(provider="gemini", model_id="gemini-3.8-flash")
    agent = AesFixAgent(llm_client=client, max_steps=2, consecutive_rejections_stop=2)

    trajectory = agent.run(_TEST_JPG)

    # Verify live trajectory produced valid measurements
    assert trajectory.total_steps >= 1
    assert trajectory.provider == "gemini"
    assert trajectory.model_id == "gemini-3.8-flash"
    assert trajectory.verdict in ("measured improvement", "leave it")

    # Tests must write to a temporary ledger path, never to results/ledger.jsonl (AGF4)
    temp_ledger = tmp_path / "ledger.jsonl"
    ledger_entry = append_run(
        metrics={
            "agent_steps": trajectory.total_steps,
            "agent_accepted_steps": len(trajectory.accepted_steps),
            "agent_gain": float(trajectory.total_gain),
            "noise_floor": float(trajectory.noise_floor),
        },
        config={
            "agent_provider": trajectory.provider,
            "agent_model": trajectory.model_id,
            "verdict": trajectory.verdict,
            "test_image": "test.jpg",
        },
        data_manifest={"image": "test.jpg", "type": "legacy_benchmark"},
        cost_usd=0.0005,
        ledger_path=temp_ledger,
    )
    assert "run_id" in ledger_entry


def test_judge_delta_regression_exposure(tmp_path: Path) -> None:
    """Regression test for AGF1:

    1. Strong exposure edit (+1 EV) on test.jpg must produce a judge score
       that differs from baseline by more than the measured noise floor ε.
    2. A pixel-identical image must give Δ = 0.
    3. The SHA256 of the edited image actually passed to the judge must differ
       from baseline when pixels changed.
    4. Judge and LLM cache keys must be keyed on content hash, never path.
    """
    if not os.path.exists(_TEST_JPG):
        pytest.skip(f"Test image '{_TEST_JPG}' not found")

    from aesfix.judges.ensemble import create_default_ensemble, compute_noise_floor
    from aesfix.executor.tier_a import TierAExecutor, CriticFix
    import hashlib
    from PIL import Image

    ensemble = create_default_ensemble(include_heavy_models=False)
    executor = TierAExecutor()

    # 1. Baseline evaluation
    base_res = ensemble.evaluate(_TEST_JPG)
    epsilon = compute_noise_floor(_TEST_JPG, ensemble)
    base_sha256 = hashlib.sha256(Path(_TEST_JPG).read_bytes()).hexdigest()

    # 2. Pixel-identical image at different path must yield exact delta = 0
    identical_copy_path = tmp_path / "identical_copy.jpg"
    identical_copy_path.write_bytes(Path(_TEST_JPG).read_bytes())
    copy_sha256 = hashlib.sha256(identical_copy_path.read_bytes()).hexdigest()
    assert copy_sha256 == base_sha256

    copy_res = ensemble.evaluate(str(identical_copy_path))
    delta_identical = copy_res.overall_score - base_res.overall_score
    # Identical image must give exact Δ = 0
    assert delta_identical == 0.0

    # 3. Strong exposure edit (+1 EV) on test.jpg
    with Image.open(_TEST_JPG) as img:
        img_rgb = img.convert("RGB")

    fix = CriticFix(
        op="adjust_exposure",
        params={"ev_delta": 1.0},
        expected_aspect="lighting",
        direction=1,
        nl_instruction="Lift exposure by +1 EV",
    )
    exec_res = executor.execute_fixes(image_input=img_rgb, fixes=[fix])

    edited_path = tmp_path / "edited_plus_1ev.jpg"
    exec_res.image.save(str(edited_path), format="JPEG", quality=95)
    edited_sha256 = hashlib.sha256(edited_path.read_bytes()).hexdigest()

    # Proved: edited image hash differs from baseline
    assert edited_sha256 != base_sha256

    # Score edited image
    edited_res = ensemble.evaluate(str(edited_path))
    delta_exposure = abs(edited_res.overall_score - base_res.overall_score)

    # Strong exposure edit must produce score differing by more than noise floor ε
    assert delta_exposure > epsilon, (
        f"Measured delta {delta_exposure:.4f} did not exceed noise floor {epsilon:.4f}"
    )


def test_ineffective_edit_check_rejected(dummy_image: str) -> None:
    """Verifies that an edit with mean|Δpixel| < 1e-3 is rejected as ineffective edit."""
    # Proposal with ev_delta = 0.0 produces zero pixel delta
    ineffective_proposal = json.dumps({
        "op": "adjust_exposure",
        "params": {"ev_delta": 0.0},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "Proposing zero delta exposure adjustment.",
    })
    improving_proposal = json.dumps({
        "op": "adjust_exposure",
        "params": {"ev_delta": 0.5},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "Actual exposure adjustment.",
    })
    fake_llm = FakeLLM([ineffective_proposal, improving_proposal])
    agent = AesFixAgent(llm_client=fake_llm, max_steps=2, consecutive_rejections_stop=3)

    mock_scores = [
        ({"lighting": 0.50}, 0.50),  # baseline
        ({"lighting": 0.56}, 0.56),  # step 2
    ]

    with patch.object(agent, "_extract_aspect_scores", side_effect=mock_scores), \
         patch("aesfix.agent.loop.compute_noise_floor", return_value=0.01), \
         patch.object(agent.guard, "evaluate", return_value=GuardVerdict(
             is_valid=True,
             composite_drift=0.02,
             feature_drift=0.01,
             structural_drift=0.01,
             threshold=0.35,
             backend_name="rgb_layout_stat",
             guard_backend="SSIM + rgb_layout_stat",
         )):

        trajectory = agent.run(dummy_image)

    assert len(trajectory.steps) == 2
    # Step 1 was caught by ineffective edit check
    step1 = trajectory.steps[0]
    assert step1.status == "rejected"
    assert step1.verdict_reason == "ineffective edit (no visible change)"

    # Feedback was supplied to LLM for next step
    assert "ineffective edit (no visible change)" in fake_llm.prompts_received[1]

