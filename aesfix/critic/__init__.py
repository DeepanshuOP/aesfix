"""Aspect Critic package for grounded aesthetic diagnosis and executable advice.

Why this package exists:
Implements the first stage of the closed loop: diagnosing photographic quality across
5 dimensions and proposing falsifiable, machine-executable fixes (CriticFix) that can be
applied by the executor and verified by the judge ensemble.
"""
from .critic import (
    VALID_FIX_OPERATIONS,
    BaseCritic,
    CriticFix,
    CriticOutput,
    CriticPromptBuilder,
    HeuristicAspectCritic,
    parse_critic_json,
)
from .live_coach import (
    LiveShootingCoach,
    NudgeSeverity,
    ShutterRecommendation,
    ViewfinderFrameTelemetry,
    ViewfinderNudge,
)

__all__ = [
    "VALID_FIX_OPERATIONS",
    "BaseCritic",
    "CriticFix",
    "CriticOutput",
    "CriticPromptBuilder",
    "HeuristicAspectCritic",
    "parse_critic_json",
    "LiveShootingCoach",
    "NudgeSeverity",
    "ShutterRecommendation",
    "ViewfinderFrameTelemetry",
    "ViewfinderNudge",
]
