"""Evaluation and benchmarking suite for AesFix.

Why this package exists:
Implements quantitative verification for both empirical performance (SRCC/PLCC)
and scientific integrity (monotonic compliance, deletion/insertion curves, and Paper Table 1).
"""
from .ag_benchmark import (
    AGBenchmarkReport,
    AGEvaluationHarness,
    AGSampleResult,
)
from .faithfulness import (
    FaithfulnessMetrics,
    benchmark_table1_interpretability,
    evaluate_deletion_insertion,
    evaluate_monotonic_compliance,
)
from .frontier_benchmark import (
    FrontierBenchmarkSuite,
    FrontierComparisonTable,
    make_aesfix_grounded_critic_policy,
    make_frontier_vlm_generic_policy,
    make_random_advice_policy,
)
from .human_study import (
    EvaluatedPairResult,
    HumanAgreementReport,
    HumanAgreementStudy,
    HumanPairRating,
)
from .leaderboard import (
    LEADERBOARD_VERSION,
    AGXLeaderboard,
    LeaderboardEntry,
)

__all__ = [
    "FaithfulnessMetrics",
    "benchmark_table1_interpretability",
    "evaluate_deletion_insertion",
    "evaluate_monotonic_compliance",
    "AGEvaluationHarness",
    "AGBenchmarkReport",
    "AGSampleResult",
    "FrontierBenchmarkSuite",
    "FrontierComparisonTable",
    "make_random_advice_policy",
    "make_frontier_vlm_generic_policy",
    "make_aesfix_grounded_critic_policy",
    "HumanPairRating",
    "EvaluatedPairResult",
    "HumanAgreementReport",
    "HumanAgreementStudy",
    "AGXLeaderboard",
    "LeaderboardEntry",
    "LEADERBOARD_VERSION",
]



