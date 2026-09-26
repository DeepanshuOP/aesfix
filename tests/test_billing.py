"""Comprehensive unit tests for SaaS Auth, Billing, Tiers, and Stripe Webhooks (AF-5.1).

Why these tests are critical for commercial readiness and interview defensibility:
1. Hard Quota Enforcements: Validates that users cannot exceed their monthly allowances or
   trigger expensive generative Tier-B edits without an active Pro/Studio subscription or credit pack.
2. Webhook Idempotency: Asserts that duplicate Stripe webhook deliveries are safely ignored,
   preventing double-crediting of packs or corrupted billing state.
3. Cryptographic Signature Validation: Validates that HMAC SHA-256 signatures and timestamp
   drift tolerances prevent forgery and replay attacks.
4. End-to-End API Integration: Uses FastAPI TestClient to test authentication headers, quota
   pre-flight checks, and checkout session generation.
"""
from __future__ import annotations

import json
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aesfix.api.billing import (
    PACK_CONFIGS,
    TIER_CONFIGS,
    PackType,
    QuotaManager,
    StripeBillingService,
    SubscriptionStatus,
    SupabaseAuthService,
    TierLevel,
)
from aesfix.api.routes_billing import router


def test_tier_entitlements_and_pricing() -> None:
    """Verifies that all tiers strictly conform to Roadmap §10 specifications."""
    free = TIER_CONFIGS[TierLevel.FREE]
    assert free.monthly_critique_limit == 5
    assert free.tier_b_monthly_limit == 0
    assert free.price_cents == 0
    assert free.raw_support is False

    pro = TIER_CONFIGS[TierLevel.PRO]
    assert pro.monthly_critique_limit == 300
    assert pro.tier_b_monthly_limit == 50
    assert pro.price_cents == 1200  # $12/month
    assert pro.tier_a_unlimited is True

    studio = TIER_CONFIGS[TierLevel.STUDIO]
    assert studio.monthly_critique_limit == 2000
    assert studio.tier_b_monthly_limit == 300
    assert studio.price_cents == 3900  # $39/month
    assert studio.raw_support is True
    assert studio.priority_gpu is True


def test_quota_manager_critique_consumption_and_pack_credit() -> None:
    """Verifies critique quota deduction, blocking on exhaustion, and pack expansion."""
    qm = QuotaManager()
    user_id = "test_user_free"

    # Default initial state is FREE with 5 credits
    assert qm.get_remaining_critiques(user_id) == 5

    # Consume all 5 free credits
    for i in range(5):
        can_run, _ = qm.can_critique(user_id)
        assert can_run is True
        assert qm.consume_critique(user_id) is True
        assert qm.get_remaining_critiques(user_id) == 4 - i

    # 6th attempt should be blocked
    can_run, reason = qm.can_critique(user_id)
    assert can_run is False
    assert "quota exceeded" in reason.lower()
    assert qm.consume_critique(user_id) is False

    # Credit Dating Profile Pack (+20 critiques)
    qm.credit_pack(user_id, PackType.DATING_PROFILE)
    assert qm.get_remaining_critiques(user_id) == 20

    # User can critique again
    assert qm.can_critique(user_id)[0] is True
    assert qm.consume_critique(user_id) is True
    assert qm.get_remaining_critiques(user_id) == 19


def test_quota_manager_tier_b_gating() -> None:
    """Verifies that generative Tier-B edits are restricted to Pro/Studio or pack holders."""
    qm = QuotaManager()
    user_id = "test_user_tier_b"

    # FREE user cannot run Tier-B
    can_b, reason = qm.can_execute_tier_b(user_id)
    assert can_b is False
    assert "restricted" in reason.lower()
    assert qm.consume_tier_b(user_id) is False

    # Upgrade user to PRO
    user = qm.get_or_create_user(user_id)
    user.tier = TierLevel.PRO
    assert qm.get_remaining_tier_b(user_id) == 50

    # Consume 1 Tier-B
    assert qm.consume_tier_b(user_id) is True
    assert qm.get_remaining_tier_b(user_id) == 49

    # Reset cycle restores allowance
    qm.reset_monthly_cycle(user_id)
    assert qm.get_remaining_tier_b(user_id) == 50


def test_supabase_auth_service() -> None:
    """Verifies Supabase JWT token claim parsing and validation."""
    auth = SupabaseAuthService(jwt_secret="test_secret")

    # Mock token
    claims = auth.verify_token("Bearer mock-user-12345:photographer@example.com")
    assert claims.user_id == "mock-user-12345"
    assert claims.email == "photographer@example.com"
    assert claims.role == "authenticated"

    # Missing token raises ValueError
    with pytest.raises(ValueError, match="Empty or missing"):
        auth.verify_token("")


def test_stripe_signature_verification_and_replay_protection() -> None:
    """Verifies HMAC SHA-256 signature calculation and timestamp tolerance checks."""
    billing = StripeBillingService(webhook_secret="whsec_unit_test")
    raw_payload = b'{"id": "evt_test", "type": "ping"}'
    now = int(time.time())

    # Valid signature passes
    valid_sig = billing.generate_signature(raw_payload, timestamp=now)
    assert billing.verify_webhook_signature(raw_payload, valid_sig) is True

    # Tampered payload fails
    tampered_payload = b'{"id": "evt_test", "type": "malicious"}'
    assert billing.verify_webhook_signature(tampered_payload, valid_sig) is False

    # Expired timestamp (> 300s) fails replay protection
    old_timestamp = now - 600
    expired_sig = billing.generate_signature(raw_payload, timestamp=old_timestamp)
    assert billing.verify_webhook_signature(raw_payload, expired_sig) is False


def test_stripe_webhook_checkout_subscription_flow() -> None:
    """Tests upgrading user from FREE to PRO via checkout.session.completed webhook."""
    qm = QuotaManager()
    billing = StripeBillingService(webhook_secret="whsec_unit_test", quota_mgr=qm)
    user_id = "user_upgrade_99"

    # Pre-condition: user is FREE
    user = qm.get_or_create_user(user_id)
    assert user.tier == TierLevel.FREE
    assert user.subscription_status == SubscriptionStatus.INACTIVE

    event_payload = {
        "id": "evt_checkout_123",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test_session_123",
                "client_reference_id": user_id,
                "customer": "cus_stripe_123",
                "subscription": "sub_stripe_abc",
                "mode": "subscription",
                "metadata": {"tier": "PRO"},
            }
        },
    }
    raw_bytes = json.dumps(event_payload).encode("utf-8")
    sig = billing.generate_signature(raw_bytes, timestamp=int(time.time()))

    res = billing.process_webhook_event(raw_bytes, sig)
    assert res["status"] == "success"

    # Post-condition: user is PRO and ACTIVE
    assert user.tier == TierLevel.PRO
    assert user.subscription_status == SubscriptionStatus.ACTIVE
    assert user.stripe_customer_id == "cus_stripe_123"
    assert user.stripe_subscription_id == "sub_stripe_abc"
    assert qm.get_remaining_critiques(user_id) == 300


def test_stripe_webhook_one_off_pack_flow() -> None:
    """Tests crediting one-off packs via checkout.session.completed."""
    qm = QuotaManager()
    billing = StripeBillingService(webhook_secret="whsec_unit_test", quota_mgr=qm)
    user_id = "user_pack_buyer"

    event_payload = {
        "id": "evt_pack_checkout_456",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_pack_session_456",
                "client_reference_id": user_id,
                "mode": "payment",
                "metadata": {"pack_type": "LISTING_PACK"},
            }
        },
    }
    raw_bytes = json.dumps(event_payload).encode("utf-8")
    sig = billing.generate_signature(raw_bytes, timestamp=int(time.time()))

    billing.process_webhook_event(raw_bytes, sig)
    user = qm.get_or_create_user(user_id)
    pack_cfg = PACK_CONFIGS[PackType.LISTING_PACK]
    assert user.bonus_critique_credits == pack_cfg["critique_credits"]
    assert user.bonus_tier_b_credits == pack_cfg["tier_b_credits"]


def test_stripe_webhook_subscription_lifecycle_and_idempotency() -> None:
    """Verifies status updates, cancellations, and duplicate event prevention."""
    qm = QuotaManager()
    billing = StripeBillingService(webhook_secret="whsec_unit_test", quota_mgr=qm)
    user_id = "user_lifecycle_001"
    user = qm.get_or_create_user(user_id)
    user.tier = TierLevel.STUDIO
    user.stripe_subscription_id = "sub_lifecycle_001"
    user.subscription_status = SubscriptionStatus.ACTIVE

    # 1. Past due event
    event_past_due = {
        "id": "evt_sub_updated_001",
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_lifecycle_001",
                "status": "past_due",
                "metadata": {"user_id": user_id},
            }
        },
    }
    raw_bytes = json.dumps(event_past_due).encode("utf-8")
    sig = billing.generate_signature(raw_bytes, timestamp=int(time.time()))

    res1 = billing.process_webhook_event(raw_bytes, sig)
    assert res1["status"] == "success"
    assert user.subscription_status == SubscriptionStatus.PAST_DUE

    # 2. Idempotency test: send same event again
    res2 = billing.process_webhook_event(raw_bytes, sig)
    assert res2["status"] == "ignored_duplicate"

    # 3. Cancellation event
    event_cancel = {
        "id": "evt_sub_deleted_002",
        "type": "customer.subscription.deleted",
        "data": {
            "object": {
                "id": "sub_lifecycle_001",
                "metadata": {"user_id": user_id},
            }
        },
    }
    raw_cancel = json.dumps(event_cancel).encode("utf-8")
    sig_cancel = billing.generate_signature(raw_cancel, timestamp=int(time.time()))

    billing.process_webhook_event(raw_cancel, sig_cancel)
    assert user.tier == TierLevel.FREE
    assert user.subscription_status == SubscriptionStatus.CANCELED


def test_fastapi_billing_endpoints() -> None:
    """Tests FastAPI router endpoints via TestClient."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    auth_header = {"Authorization": "Bearer mock-user-client-1:client@aesfix.app"}

    # 1. Unauthenticated request to /me returns 401
    resp_unauth = client.get("/api/v1/billing/me")
    assert resp_unauth.status_code == 401

    # 2. Authenticated request to /me returns 200 and Free account defaults
    resp_me = client.get("/api/v1/billing/me", headers=auth_header)
    assert resp_me.status_code == 200
    data = resp_me.json()
    assert data["user_id"] == "mock-user-client-1"
    assert data["tier"] == "FREE"
    assert data["remaining_critiques"] == 5

    # 3. Pre-flight quota check
    resp_quota = client.post(
        "/api/v1/billing/verify-quota",
        json={"action": "critique"},
        headers=auth_header,
    )
    assert resp_quota.status_code == 200
    assert resp_quota.json()["allowed"] is True

    # 4. Initiate checkout session for PRO tier
    resp_checkout = client.post(
        "/api/v1/billing/checkout",
        json={"tier": "PRO"},
        headers=auth_header,
    )
    assert resp_checkout.status_code == 200
    checkout_data = resp_checkout.json()
    assert "cs_sub_pro_" in checkout_data["session_id"]
    assert checkout_data["price_cents"] == 1200

    # 5. Customer portal
    resp_portal = client.post("/api/v1/billing/portal", headers=auth_header)
    assert resp_portal.status_code == 200
    assert "billing.stripe.com" in resp_portal.json()["portal_url"]
