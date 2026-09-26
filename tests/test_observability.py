"""Unit and integration tests for Observability, Health Probes, Prometheus Metrics, and COGS (AF-5.5).

Why these tests are critical for production operability and interview defensibility:
1. Production Readiness Invariants: Asserts that /health, /ready, and /version probes return
   correct HTTP statuses and detect tensor engine operational readiness.
2. Unit Economics Verification: Validates that hardware cost modeling for Tier-A execution,
   VLM inference, and judge evaluations proves a >85% gross margin at $12/mo Pro pricing.
3. Prometheus Format Compliance: Ensures metrics strictly adhere to the Prometheus text
   exposition format (HELP, TYPE, counters, labels, and quantiles).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aesfix.api.observability import (
    CostTracker,
    HealthService,
    MetricsCollector,
)
from aesfix.api.routes_observability import router as obs_router


def test_health_service_probes() -> None:
    """Verifies that liveness, readiness, and version services return valid diagnostic data."""
    # 1. Liveness
    live = HealthService.check_liveness()
    assert live["status"] == "healthy"
    assert "timestamp" in live

    # 2. Readiness
    ready = HealthService.check_readiness()
    assert ready["status"] == "ready"
    assert ready["components"]["tensor_engine"] == "operational"
    assert "device" in ready["components"]

    # 3. Version info
    ver = HealthService.get_version_info()
    assert ver["version"] == "1.0.0"
    assert ver["service"] == "aesfix-api"
    assert "github.com" in ver["documentation"]


def test_cost_tracker_and_gross_margin_target() -> None:
    """Verifies per-photo COGS calculation and enforces >85% gross margin target (Roadmap §10)."""
    tracker = CostTracker()

    # Credit one Pro subscription ($12.00 = 1200 cents)
    tracker.record_revenue(1200.0)

    # Simulate processing full monthly quota of 300 photos for a Pro user
    for i in range(300):
        # Deterministic Tier-A edit: 25ms CPU time, 0 Tier-B, 2 judges
        cost_rep = tracker.compute_job_cogs(
            job_id=f"job_{i}",
            user_tier="PRO",
            tier_a_duration_ms=25.0,
            tier_b_duration_sec=0.0,
            num_judges=2,
        )
        assert cost_rep.total_cogs_cents > 0.0

    summary = tracker.get_summary()
    assert summary.total_processed_photos == 300
    assert summary.estimated_revenue_cents == 1200.0

    # Total COGS for 300 Tier-A photos should be less than $1.50 (150 cents)
    assert summary.total_cogs_cents < 150.0

    # Roadmap §10 / §45 strict requirement: gross margin must exceed 85%
    assert summary.gross_margin_percentage >= 85.0
    assert summary.avg_cogs_per_photo_cents > 0.0
    assert summary.cogs_by_tier["PRO"] > 0.0


def test_metrics_collector_prometheus_exposition() -> None:
    """Verifies Prometheus metrics aggregation, label formatting, and text generation."""
    collector = MetricsCollector()

    # Record sample traffic
    collector.record_request("/health", 200)
    collector.record_request("/health", 200)
    collector.record_request("/api/v1/pipeline/submit", 200)
    collector.record_request("/api/v1/pipeline/submit", 401)

    collector.record_critique(goal="listing", gain=0.084, guard_passed=True, latency_sec=1.85)
    collector.record_critique(goal="portrait", gain=0.000, guard_passed=False, latency_sec=2.10)

    prom_text = collector.generate_prometheus_text()

    # Metric type headers
    assert "# TYPE aesfix_http_requests_total counter" in prom_text
    assert "# TYPE aesfix_critiques_total counter" in prom_text
    assert "# TYPE aesfix_verified_aesthetic_gain_sum counter" in prom_text
    assert "# TYPE aesfix_identity_guard_verdicts_total counter" in prom_text

    # Sample values
    assert 'aesfix_http_requests_total{endpoint="/health",status="200"} 2' in prom_text
    assert 'aesfix_critiques_total{goal="listing"} 1' in prom_text
    assert 'aesfix_identity_guard_verdicts_total{verdict="approved"} 1' in prom_text
    assert 'aesfix_identity_guard_verdicts_total{verdict="vetoed"} 1' in prom_text
    assert "aesfix_pipeline_latency_seconds" in prom_text


def test_fastapi_observability_endpoints_integration() -> None:
    """Tests FastAPI root probes, Prometheus scraping, and COGS dashboard."""
    app = FastAPI()
    app.include_router(obs_router)
    client = TestClient(app)

    # 1. /health probe
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # 2. /ready probe
    resp_ready = client.get("/ready")
    assert resp_ready.status_code == 200
    assert resp_ready.json()["status"] == "ready"

    # 3. /version endpoint
    resp_ver = client.get("/version")
    assert resp_ver.status_code == 200
    assert resp_ver.json()["version"] == "1.0.0"

    # 4. /metrics Prometheus endpoint
    resp_metrics = client.get("/metrics")
    assert resp_metrics.status_code == 200
    assert "text/plain" in resp_metrics.headers["content-type"]
    assert "aesfix_http_requests_total" in resp_metrics.text

    # 5. Record revenue hook
    resp_rev = client.post(
        "/api/v1/observability/record-revenue",
        json={"amount_cents": 1200.0},
    )
    assert resp_rev.status_code == 200
    assert resp_rev.json()["status"] == "recorded"

    # 6. COGS dashboard
    resp_cogs = client.get("/api/v1/observability/cogs")
    assert resp_cogs.status_code == 200
    cogs_data = resp_cogs.json()
    assert cogs_data["status"] == "healthy"
    assert "gross_margin_percentage" in cogs_data["unit_economics"]
