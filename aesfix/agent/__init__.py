"""AesFix Multi-Turn Parametric Retouching Agent.

Provides vision-language closed-loop retouching with per-aspect verification,
tool schema generation, and deterministic acceptance criteria.
"""
from __future__ import annotations

from aesfix.agent.llm import LLMClient, LLMResponse
from aesfix.agent.loop import AesFixAgent, AgentStep, AgentTrajectory
from aesfix.agent.schema import (
    AgentProposal,
    VALID_ASPECTS,
    VALID_OPS,
    format_validation_error_feedback,
    generate_tool_schema_prompt,
    validate_proposal,
)

__all__ = [
    "LLMClient",
    "LLMResponse",
    "AesFixAgent",
    "AgentStep",
    "AgentTrajectory",
    "AgentProposal",
    "VALID_ASPECTS",
    "VALID_OPS",
    "validate_proposal",
    "format_validation_error_feedback",
    "generate_tool_schema_prompt",
    "BaseProposer",
    "GridProposer",
    "HeuristicProposer",
]
from aesfix.agent.proposer import BaseProposer, GridProposer, HeuristicProposer
