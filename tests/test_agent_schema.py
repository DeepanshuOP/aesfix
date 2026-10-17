"""Unit tests for Agent Tool Schema and Canonical Parameter Enforcement (AGF2).

Ensures that deprecated aliases (rule, tightness, contrast_factor, temperature_shift, etc.)
and unknown parameters are strictly rejected with actionable feedback naming canonical parameters,
and validates that the system prompt specifies unit, range, and example magnitude for each parameter.
"""
import pytest
from aesfix.agent.schema import (
    AgentProposal,
    generate_tool_schema_prompt,
    validate_proposal,
)


def test_reject_crop_deprecated_aliases() -> None:
    """Verifies that 'rule' and 'tightness' are rejected for smart_crop with canonical naming."""
    proposal_dict = {
        "op": "smart_crop",
        "params": {"rule": "rule_of_thirds", "tightness": 0.85},
        "aspect": "composition",
        "direction": 1,
        "rationale": "Re-compose along rule of thirds.",
    }
    proposal, err = validate_proposal(proposal_dict)
    assert proposal is None
    assert err is not None
    assert "Deprecated parameter alias" in err
    assert "top, left, bottom, right" in err


def test_reject_contrast_factor_alias() -> None:
    """Verifies that 'contrast_factor' is rejected for adjust_contrast in favor of 'factor'."""
    proposal_dict = {
        "op": "adjust_contrast",
        "params": {"contrast_factor": 1.25},
        "aspect": "technical",
        "direction": 1,
        "rationale": "Increase contrast.",
    }
    proposal, err = validate_proposal(proposal_dict)
    assert proposal is None
    assert err is not None
    assert "contrast_factor" in err
    assert "factor" in err


def test_reject_temperature_shift_alias() -> None:
    """Verifies that 'temperature_shift' and 'temp' are rejected for adjust_white_balance."""
    for alias in ["temperature_shift", "temp"]:
        proposal_dict = {
            "op": "adjust_white_balance",
            "params": {alias: 20.0},
            "aspect": "color",
            "direction": 1,
            "rationale": "Warm up scene.",
        }
        proposal, err = validate_proposal(proposal_dict)
        assert proposal is None
        assert err is not None
        assert alias in err
        assert "temperature" in err


def test_reject_ev_and_generic_value_alias() -> None:
    """Verifies that 'ev' and generic 'value' are rejected for adjust_exposure."""
    for alias in ["ev", "value"]:
        proposal_dict = {
            "op": "adjust_exposure",
            "params": {alias: 0.5},
            "aspect": "lighting",
            "direction": 1,
            "rationale": "Brighten photograph.",
        }
        proposal, err = validate_proposal(proposal_dict)
        assert proposal is None
        assert err is not None
        assert alias in err
        assert "ev_delta" in err


def test_reject_unknown_param() -> None:
    """Verifies that unknown parameters trigger explicit feedback listing canonical params."""
    proposal_dict = {
        "op": "adjust_exposure",
        "params": {"ev_delta": 0.5, "non_existent_param": 42},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "Brighten photograph.",
    }
    proposal, err = validate_proposal(proposal_dict)
    assert proposal is None
    assert err is not None
    assert "Unknown parameter 'non_existent_param'" in err
    assert "ev_delta" in err


def test_accept_canonical_proposals() -> None:
    """Verifies that canonical proposals accept cleanly without error."""
    # 1. adjust_exposure
    p1, err1 = validate_proposal({
        "op": "adjust_exposure",
        "params": {"ev_delta": 0.33},
        "aspect": "lighting",
        "direction": 1,
        "rationale": "Subtle exposure compensation.",
    })
    assert err1 is None
    assert p1 is not None
    assert p1.params["ev_delta"] == 0.33

    # 2. smart_crop
    p2, err2 = validate_proposal({
        "op": "smart_crop",
        "params": {"top": 0.05, "left": 0.05, "bottom": 0.95, "right": 0.95},
        "aspect": "composition",
        "direction": 1,
        "rationale": "Crop out peripheral distractions.",
    })
    assert err2 is None
    assert p2 is not None
    assert p2.params["top"] == 0.05

    # 3. adjust_white_balance
    p3, err3 = validate_proposal({
        "op": "adjust_white_balance",
        "params": {"temperature": 15.0, "tint": 0.0},
        "aspect": "color",
        "direction": 1,
        "rationale": "Warming up cool shadow tones.",
    })
    assert err3 is None
    assert p3 is not None
    assert p3.params["temperature"] == 15.0


def test_tool_schema_prompt_contains_units_ranges_and_examples() -> None:
    """Verifies that the prompt shown to the LLM details unit, range, and example magnitude."""
    prompt = generate_tool_schema_prompt()
    assert "unit: EV stops" in prompt
    assert "range: -2.0..2.0" in prompt
    assert "example magnitude: 0.33" in prompt
    assert "unit: value -100..100" in prompt
    assert "range: -100.0..100.0" in prompt
    assert "top" in prompt and "0.0..0.9" in prompt
    assert "factor" in prompt
    assert "temperature" in prompt
