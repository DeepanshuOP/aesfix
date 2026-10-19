"""Unit tests for Edit Proposers (Heuristic & Grid Best-of-k) - S5."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aesfix.agent.proposer import GridProposer, HeuristicProposer, ProposerDecision
from aesfix.executor import TierAExecutor


class MockResult:
    def __init__(self, score: float):
        self.raw_score = score
    def __float__(self):
        return self.raw_score


class DummyJudge:
    """Deterministic dummy judge for testing proposers."""
    name = "dummy_judge"

    def __init__(self, target_luma: float = 0.5):
        self.target_luma = target_luma

    def evaluate_pil(self, pil_image: Image.Image) -> float:
        arr = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        luma = float(np.mean(0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]))
        score = 5.0 - 10.0 * ((luma - self.target_luma) ** 2)
        return round(score, 4)

    def evaluate_batch_pil(self, pil_images: list[Image.Image]) -> list[float]:
        return [self.evaluate_pil(im) for im in pil_images]

    def evaluate(self, image_path: str) -> MockResult:
        im = Image.open(image_path)
        return MockResult(self.evaluate_pil(im))


def test_grid_proposer_candidate_set():
    """Verify that GridProposer specifies exactly 15 symmetric candidates."""
    proposer = GridProposer()
    assert len(proposer.candidate_specs) == 15

    ops = [spec[0] for spec in proposer.candidate_specs]
    assert ops.count("adjust_exposure") == 8
    assert ops.count("adjust_contrast") == 3
    assert ops.count("adjust_white_balance") == 4

    exp_deltas = sorted([spec[1]["ev_delta"] for spec in proposer.candidate_specs if spec[0] == "adjust_exposure"])
    assert exp_deltas == [-1.6, -1.2, -0.7, -0.3, 0.3, 0.7, 1.2, 1.6]

    contrast_factors = sorted([spec[1]["factor"] for spec in proposer.candidate_specs if spec[0] == "adjust_contrast"])
    assert contrast_factors == [0.8, 1.25, 1.6]

    wb_temps = sorted([spec[1]["temperature"] for spec in proposer.candidate_specs if spec[0] == "adjust_white_balance"])
    assert wb_temps == [-25.0, -10.0, 10.0, 25.0]


def test_grid_proposer_selection():
    """Verify best-of-k selection and decision logging (k=15)."""
    proposer = GridProposer()
    executor = TierAExecutor()
    judge = DummyJudge(target_luma=0.6)

    dark_img = Image.new("RGB", (64, 64), (50, 50, 50))
    curr_score = judge.evaluate_pil(dark_img)

    decision = proposer.propose(
        current_image=dark_img,
        original_image=dark_img,
        current_score=curr_score,
        noise_floor=0.04,
        primary_judge=judge,
        executor=executor,
    )

    assert isinstance(decision, ProposerDecision)
    assert decision.k == 15
    assert len(decision.evaluated_candidates) == 15
    assert decision.selected_proposal is not None
    assert decision.best_gain > 0.04
    assert decision.selected_proposal.op == "adjust_exposure"
    assert decision.selected_proposal.params["ev_delta"] > 0


def test_grid_proposer_intent_guard_low_key():
    """Verify that intent guard prevents positive exposure lifts on low-key images."""
    proposer = GridProposer()
    executor = TierAExecutor()
    judge = DummyJudge(target_luma=0.7)

    dark_img = Image.new("RGB", (64, 64), (30, 30, 30))
    curr_score = judge.evaluate_pil(dark_img)

    style_intent = {"style": "low_key", "is_low_key": True}

    decision = proposer.propose(
        current_image=dark_img,
        original_image=dark_img,
        current_score=curr_score,
        noise_floor=0.04,
        primary_judge=judge,
        executor=executor,
        style_intent=style_intent,
    )

    if decision.selected_proposal is not None:
        if decision.selected_proposal.op == "adjust_exposure":
            assert decision.selected_proposal.params["ev_delta"] <= 0


def test_heuristic_proposer():
    """Verify HeuristicProposer runs and reports variable k."""
    proposer = HeuristicProposer()
    executor = TierAExecutor()
    judge = DummyJudge(target_luma=0.5)

    dark_img = Image.new("RGB", (64, 64), (40, 40, 40))
    curr_score = judge.evaluate_pil(dark_img)

    decision = proposer.propose(
        current_image=dark_img,
        original_image=dark_img,
        current_score=curr_score,
        noise_floor=0.04,
        primary_judge=judge,
        executor=executor,
    )

    assert isinstance(decision, ProposerDecision)
    assert decision.k >= 1
