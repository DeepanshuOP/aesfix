"""Standardized dataset loaders and evaluation harness for AesFix.

Why this package is structured this way:
Decouples raw external dataset formats (AADB attributes, PCCD critiques) from core model
components via a normalized DatasetSample abstraction with canonical 5-aspect mapping.
"""
from .loaders import (
    ASPECT_DIMENSIONS,
    AADBLoader,
    DatasetSample,
    EvaluationHarness,
    PCCDLoader,
    compute_correlation_metrics,
)

__all__ = [
    "ASPECT_DIMENSIONS",
    "AADBLoader",
    "DatasetSample",
    "EvaluationHarness",
    "PCCDLoader",
    "compute_correlation_metrics",
]
