"""Evaluation and benchmarking suite for AesFix.

Why this package exists:
Implements quantitative verification for both empirical performance (SRCC/PLCC)
and scientific integrity (monotonic compliance, deletion/insertion curves, and Paper Table 1).
"""
from .faithfulness import (
    FaithfulnessMetrics,
    benchmark_table1_interpretability,
    evaluate_deletion_insertion,
    evaluate_monotonic_compliance,
)

__all__ = [
    "FaithfulnessMetrics",
    "benchmark_table1_interpretability",
    "evaluate_deletion_insertion",
    "evaluate_monotonic_compliance",
]
