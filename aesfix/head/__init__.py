"""Interpretable neuro-fuzzy aggregation head package.

Why this package exists:
Implements the C3 interpretability contribution: a differentiable Takagi-Sugeno neuro-fuzzy
network (ANFIS) that aggregates 5 continuous aspect ratings into an overall aesthetic score
via an inspectable, human-auditable rulebook rather than a black-box MLP.
"""
from .anfis import (
    ANFISHead,
    FiredRule,
    GaussianMembership,
    RulebookTrace,
    train_anfis_epoch,
)

__all__ = [
    "ANFISHead",
    "FiredRule",
    "GaussianMembership",
    "RulebookTrace",
    "train_anfis_epoch",
]
