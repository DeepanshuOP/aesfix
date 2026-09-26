"""FastAPI Observability, Health, Prometheus Metrics, and COGS Routes (AF-5.5 & Roadmap §42).

Why these endpoints are organized here:
1. Standard Root Probes: Exposes /health, /ready, /version, and /metrics at the root
   level where standard Kubernetes, Docker, and Prometheus agents expect them.
2. Business Margin Tracking: /api/v1/observability/cogs exposes real-time gross margin health,
   ensuring operations teams can verify unit economics against our target >85% margin envelope.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Response
from pydantic import BaseModel

from aesfix.api.observability import CostTracker, HealthService, MetricsCollector

logger = logging.getLogger("aesfix.api.routes_observability")

router = APIRouter(tags=["observability"])

# Shared observability singletons
metrics_collector = MetricsCollector()
cost_tracker = CostTracker()


class RecordRevenueRequest(BaseModel):
    """Payload to record revenue from checkout webhooks."""
    amount_cents: float


@router.get("/health")
def get_liveness() -> Dict[str, Any]:
    """Kubernetes liveness probe: confirms the process is responsive."""
    metrics_collector.record_request("/health", 200)
    return HealthService.check_liveness()


@router.get("/ready")
def get_readiness() -> Dict[str, Any]:
    """Kubernetes readiness probe: confirms tensor and ML execution pipelines are ready."""
    res = HealthService.check_readiness()
    code = 200 if res["status"] == "ready" else 503
    metrics_collector.record_request("/ready", code)
    return res


@router.get("/version")
def get_version() -> Dict[str, Any]:
    """Returns deployment version and metadata."""
    metrics_collector.record_request("/version", 200)
    return HealthService.get_version_info()


@router.get("/metrics")
def get_prometheus_metrics() -> Response:
    """Prometheus scrape endpoint providing standard plain-text telemetry."""
    metrics_collector.record_request("/metrics", 200)
    body = metrics_collector.generate_prometheus_text()
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get("/api/v1/observability/cogs")
def get_unit_economics_dashboard() -> Dict[str, Any]:
    """Returns real-time COGS, revenue, and gross margin telemetry (Roadmap §10)."""
    metrics_collector.record_request("/api/v1/observability/cogs", 200)
    summary = cost_tracker.get_summary()
    return {
        "status": "healthy",
        "unit_economics": {
            "total_processed_photos": summary.total_processed_photos,
            "total_cogs_cents": summary.total_cogs_cents,
            "total_cogs_usd": round(summary.total_cogs_cents / 100.0, 4),
            "estimated_revenue_cents": summary.estimated_revenue_cents,
            "estimated_revenue_usd": round(summary.estimated_revenue_cents / 100.0, 2),
            "gross_margin_percentage": f"{summary.gross_margin_percentage:.1f}%",
            "margin_status": "EXCELLENT" if summary.gross_margin_percentage >= 85.0 else "REVIEW_MARGIN",
            "avg_cogs_per_photo_cents": summary.avg_cogs_per_photo_cents,
            "cogs_by_tier": summary.cogs_by_tier,
        },
    }


@router.post("/api/v1/observability/record-revenue")
def record_subscription_revenue(req: RecordRevenueRequest) -> Dict[str, Any]:
    """Internal hook to credit revenue to telemetry dashboards upon completed checkout."""
    cost_tracker.record_revenue(req.amount_cents)
    metrics_collector.record_request("/api/v1/observability/record-revenue", 200)
    return {"status": "recorded", "amount_cents": req.amount_cents}
