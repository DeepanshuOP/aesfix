"""API and demo interface package for AesFix.

Why this package exists:
Provides user-facing and application-level endpoints for the AesFix diagnostic pipeline,
including the Gradio v2 interactive demo (AF-1.4) and FastAPI service routes.
"""
from .demo import (
    PipelineResult,
    build_demo_app,
    run_aesfix_pipeline,
)

__all__ = [
    "PipelineResult",
    "build_demo_app",
    "run_aesfix_pipeline",
]
