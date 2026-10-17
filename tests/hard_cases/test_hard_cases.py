"""Hard-case regression test suite for AesFix (AQ1).

Hard-cases:
1. dark.jpg: labelled "intentional low-key — exposure lift is NOT an improvement"
2. test.jpg: labelled "already good — expect leave it"

Regression Invariant:
When an agent or LLM proposes greedy repeated +0.3 EV exposure lifts on dark.jpg,
the anti-greedy and over-processing mechanisms MUST stop the loop before 4 stacked
exposure lifts are accepted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from aesfix.agent.llm import LLMResponse
from aesfix.agent.loop import AesFixAgent

_DIR = Path(__file__).resolve().parent

HARD_CASES: Dict[str, Dict[str, Any]] = {
    "dark.jpg": {
        "path": _DIR / "dark.jpg",
        "label": "intentional low-key — exposure lift is NOT an improvement",
        "expected_behaviour": "Reject stacked exposure lifts; preserve low-key photographic mood.",
    },
    "test.jpg": {
        "path": _DIR / "test.jpg",
        "label": "already good — expect leave it",
        "expected_behaviour": "Leave as is without degrading photographic balance.",
    },
}


class GreedyExposureFakeLLM:
    """Mock LLM proposing repeated +0.3 EV exposure lifts indefinitely."""

    def __init__(self, ev_delta: float = 0.3) -> None:
        self.ev_delta = ev_delta
        self.call_count: int = 0
        self.prompts_received: List[str] = []
        self.provider: str = "fake_test_provider"
        self.model_id: str = "greedy-exposure-mock-v1"

    def generate(
        self,
        prompt: str,
        image_input: Union[str, Path, Image.Image],
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)
        proposal = {
            "op": "adjust_exposure",
            "params": {"ev_delta": self.ev_delta},
            "aspect": "lighting",
            "direction": 1,
            "rationale": f"Greedy exposure lift attempt #{self.call_count}.",
        }
        return LLMResponse(
            content=json.dumps(proposal),
            provider=self.provider,
            model_id=self.model_id,
            cached=False,
        )


def test_hard_case_labels_registered() -> None:
    """Verifies that dark.jpg and test.jpg are present and correctly labelled in hard_cases."""
    for case_name, meta in HARD_CASES.items():
        case_path = meta["path"]
        assert case_path.exists(), f"Hard case image {case_name} missing at {case_path}"
        assert len(meta["label"]) > 0, f"Hard case {case_name} must have non-empty label"

    assert (
        HARD_CASES["dark.jpg"]["label"]
        == "intentional low-key — exposure lift is NOT an improvement"
    )
    assert HARD_CASES["test.jpg"]["label"] == "already good — expect leave it"


def test_dark_jpg_regression_prevents_stacked_exposure_lifts() -> None:
    """Asserts that running agent on dark.jpg with repeated +0.3 EV proposals stops.

    Regression test for AQ1:
    A greedy model attempting 4 consecutive +0.3 EV lifts on intentional low-key photo
    must NOT produce 4 stacked exposure lifts.
    """
    dark_jpg_path = str(HARD_CASES["dark.jpg"]["path"])
    fake_llm = GreedyExposureFakeLLM(ev_delta=0.3)

    # Allow up to 6 steps to ensure the loop has full budget to attempt stacking
    agent = AesFixAgent(
        llm_client=fake_llm,
        max_steps=6,
        consecutive_rejections_stop=2,
    )

    trajectory = agent.run(dark_jpg_path)

    accepted_exposure_lifts = [
        s for s in trajectory.accepted_steps if s.proposal.op == "adjust_exposure"
    ]

    # Core assertion of AQ1: NOT 4 stacked exposure lifts
    assert len(accepted_exposure_lifts) < 4, (
        f"Reward hacking detected: {len(accepted_exposure_lifts)} stacked exposure lifts were accepted!"
    )
    # Further bounded by AQ2 op quota (max 2 per run)
    assert len(accepted_exposure_lifts) <= 2, (
        f"Exceeded op quota: {len(accepted_exposure_lifts)} > 2 accepted exposure lifts"
    )

    # Cumulative EV gain must not exceed ±0.6 EV
    cumulative_ev = sum(
        float(s.proposal.params.get("ev_delta", 0.0)) for s in accepted_exposure_lifts
    )
    assert round(cumulative_ev, 2) <= 0.60, (
        f"Cumulative exposure {cumulative_ev:+.2f} EV exceeded 0.60 EV cap"
    )
