"""Production FastAPI Application Gateway and Router Consolidation (AF-5.7 & Roadmap §28, §42).

Why this application gateway is the core operational entry point:
1. Unified Domain Architecture: Integrates all decoupled sub-services:
   - Observability & Metrics (/health, /ready, /version, /metrics, /api/v1/observability)
   - SaaS Auth, Billing & Tiers (/api/v1/billing)
   - Closed-Loop Pipeline & Real-Time SSE (/api/v1/pipeline)
   - Goal Modes & Shoot-Level Batch Triage (/api/v1/batch)
   - Trust, C2PA Provenance, and GDPR Purge (/api/v1/trust)
   - Beta Cohort Management & Funnel Analytics (/api/v1/beta)
2. Production Security & CORS: Configures Cross-Origin Resource Sharing for the Next.js 15
   frontend (https://aesfix.app and local development).
3. Request Correlation & Context: Injects X-Request-ID headers for distributed tracing across
   API, queue, and inference workers.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from aesfix.api.routes_batch import router as batch_router
from aesfix.api.routes_beta import router as beta_router
from aesfix.api.routes_billing import router as billing_router
from aesfix.api.routes_observability import router as observability_router
from aesfix.api.routes_pipeline import router as pipeline_router
from aesfix.api.routes_trust import router as trust_router

logger = logging.getLogger("aesfix.app")


def create_app() -> FastAPI:
    """Initializes and configures the production AesFix FastAPI application."""
    app = FastAPI(
        title="AesFix API — Closed-Loop Image Aesthetics & Edit Verification",
        description=(
            "The photo critic that proves its advice works. "
            "Evaluates photos across 5 aesthetic dimensions, computes interpretable neuro-fuzzy "
            "rulebook traces, executes deterministic photographic adjustments, tests against "
            "Identity Guard drift, and reports verified Aesthetic Gain (AG@k) with Lightroom XMP sidecars."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. CORS Configuration for Next.js frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://aesfix.app",
            "https://www.aesfix.app",
            "http://localhost:3000",
            "http://localhost:3001",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Correlation ID and Request Timing Middleware
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next: Any) -> Response:
        req_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:10]}")
        start_time = time.perf_counter()

        response: Response = await call_next(request)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        return response

    # 3. Mount Sub-Routers
    app.include_router(observability_router)
    app.include_router(billing_router)
    app.include_router(pipeline_router)
    app.include_router(batch_router)
    app.include_router(trust_router)
    app.include_router(beta_router)

    # 4. Root Welcome Endpoint
    @app.get("/", tags=["root"])
    def root_welcome() -> Dict[str, Any]:
        """Root API metadata, research preprint link, and documentation gateway."""
        return {
            "name": "AesFix API",
            "tagline": "The photo critic that proves its advice works.",
            "version": "1.0.0",
            "status": "operational",
            "paper_preprint": "docs/paper/arxiv_preprint_draft.md",
            "agx_leaderboard": "https://huggingface.co/spaces/aesfix/agx-leaderboard",
            "docs": "/docs",
        }

    return app


# Application singleton for ASGI servers (uvicorn aesfix.api.app:app)
app = create_app()
