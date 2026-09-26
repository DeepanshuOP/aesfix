"""FastAPI Billing & Auth API Routes (AF-5.1 & Roadmap §42).

Why this router is organized under /api/v1/billing:
1. RESTful Contract: Exposes clean interfaces for frontend client state (/me, /verify-quota),
   payment initiation (/checkout, /portal), and backend webhook ingestion (/webhook).
2. Defense in Depth: Protects all user endpoints with Supabase JWT bearer token verification.
3. Stripe Webhook Security: Extracts raw byte payloads before JSON parsing to preserve cryptographic
   HMAC signatures against whitespace/formatting tampering.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from aesfix.api.billing import (
    PACK_CONFIGS,
    TIER_CONFIGS,
    AuthClaims,
    PackType,
    QuotaManager,
    StripeBillingService,
    SupabaseAuthService,
    TierLevel,
)

logger = logging.getLogger("aesfix.api.routes_billing")

# Singleton services for the API router instance
auth_service = SupabaseAuthService()
quota_manager = QuotaManager()
billing_service = StripeBillingService(quota_mgr=quota_manager)

router = APIRouter(prefix="/api/v1/billing", tags=["billing"])


# --- Request & Response Models ---

class CheckoutSessionRequest(BaseModel):
    """Payload to initiate a Stripe checkout session."""
    tier: Optional[TierLevel] = Field(default=None, description="Target subscription tier (PRO or STUDIO)")
    pack_type: Optional[PackType] = Field(default=None, description="One-off credit pack type")
    success_url: str = Field(default="https://aesfix.app/dashboard?session_id={CHECKOUT_SESSION_ID}")
    cancel_url: str = Field(default="https://aesfix.app/pricing")


class CheckoutSessionResponse(BaseModel):
    """Checkout session URL and metadata."""
    session_id: str
    checkout_url: str
    mode: str
    price_cents: int


class QuotaCheckRequest(BaseModel):
    """Payload to verify action eligibility before running inference."""
    action: str = Field(description="'critique' or 'tier_b'")


class QuotaCheckResponse(BaseModel):
    """Quota verification decision."""
    allowed: bool
    remaining: int
    reason: str


class UserAccountResponse(BaseModel):
    """User account details, entitlements, and remaining quota."""
    user_id: str
    email: str
    tier: str
    subscription_status: str
    remaining_critiques: int
    remaining_tier_b: int
    bonus_critique_credits: int
    bonus_tier_b_credits: int
    entitlements: Dict[str, Any]


# --- Authentication Dependency ---

def get_current_user(authorization: Optional[str] = Header(None)) -> AuthClaims:
    """Extracts and verifies Supabase JWT Bearer token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        claims = auth_service.verify_token(authorization)
        return claims
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# --- API Endpoints ---

@router.get("/me", response_model=UserAccountResponse)
def get_user_account(current_user: AuthClaims = Depends(get_current_user)) -> UserAccountResponse:
    """Returns the authenticated user's profile, tier entitlements, and remaining quotas."""
    user = quota_manager.get_or_create_user(current_user.user_id, current_user.email)
    entitlements = TIER_CONFIGS[user.tier]

    return UserAccountResponse(
        user_id=user.user_id,
        email=user.email,
        tier=user.tier.value,
        subscription_status=user.subscription_status.value,
        remaining_critiques=quota_manager.get_remaining_critiques(user.user_id),
        remaining_tier_b=quota_manager.get_remaining_tier_b(user.user_id),
        bonus_critique_credits=user.bonus_critique_credits,
        bonus_tier_b_credits=user.bonus_tier_b_credits,
        entitlements={
            "monthly_critique_limit": entitlements.monthly_critique_limit,
            "tier_a_unlimited": entitlements.tier_a_unlimited,
            "tier_b_monthly_limit": entitlements.tier_b_monthly_limit,
            "max_batch_size": entitlements.max_batch_size,
            "raw_support": entitlements.raw_support,
            "priority_gpu": entitlements.priority_gpu,
            "price_cents": entitlements.price_cents,
        },
    )


@router.post("/verify-quota", response_model=QuotaCheckResponse)
def verify_quota(
    req: QuotaCheckRequest,
    current_user: AuthClaims = Depends(get_current_user),
) -> QuotaCheckResponse:
    """Pre-flight check before executing an image critique or generative edit."""
    if req.action == "critique":
        allowed, reason = quota_manager.can_critique(current_user.user_id)
        remaining = quota_manager.get_remaining_critiques(current_user.user_id)
    elif req.action == "tier_b":
        allowed, reason = quota_manager.can_execute_tier_b(current_user.user_id)
        remaining = quota_manager.get_remaining_tier_b(current_user.user_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown action: '{req.action}'. Expected 'critique' or 'tier_b'.",
        )

    return QuotaCheckResponse(allowed=allowed, remaining=remaining, reason=reason)


@router.post("/checkout", response_model=CheckoutSessionResponse)
def create_checkout_session(
    req: CheckoutSessionRequest,
    current_user: AuthClaims = Depends(get_current_user),
) -> CheckoutSessionResponse:
    """Creates a Stripe Checkout Session for subscription or credit pack purchase."""
    if not req.tier and not req.pack_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must specify either 'tier' (PRO or STUDIO) or 'pack_type'.",
        )

    # Subscription tier checkout
    if req.tier:
        if req.tier == TierLevel.FREE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot checkout for FREE tier.",
            )
        cfg = TIER_CONFIGS[req.tier]
        session_id = f"cs_sub_{req.tier.value.lower()}_{current_user.user_id[:8]}"
        checkout_url = f"https://checkout.stripe.com/pay/{session_id}"
        return CheckoutSessionResponse(
            session_id=session_id,
            checkout_url=checkout_url,
            mode="subscription",
            price_cents=cfg.price_cents,
        )

    # One-off credit pack checkout
    assert req.pack_type is not None
    pack_cfg = PACK_CONFIGS[req.pack_type]
    session_id = f"cs_pack_{req.pack_type.value.lower()}_{current_user.user_id[:8]}"
    checkout_url = f"https://checkout.stripe.com/pay/{session_id}"

    return CheckoutSessionResponse(
        session_id=session_id,
        checkout_url=checkout_url,
        mode="payment",
        price_cents=pack_cfg["price_cents"],
    )


@router.post("/portal")
def create_customer_portal(
    current_user: AuthClaims = Depends(get_current_user),
) -> Dict[str, str]:
    """Generates Stripe Customer Portal session URL for managing cards and subscriptions."""
    user = quota_manager.get_or_create_user(current_user.user_id)
    portal_id = f"bps_{user.stripe_customer_id or 'guest'}"
    return {
        "portal_url": f"https://billing.stripe.com/p/session/{portal_id}",
    }


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
) -> Dict[str, Any]:
    """Ingests raw Stripe webhooks, validates HMAC SHA-256 signature, and updates user state."""
    if not stripe_signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe-Signature header.",
        )

    raw_body = await request.body()

    try:
        result = billing_service.process_webhook_event(raw_body, stripe_signature)
        return result
    except ValueError as exc:
        logger.warning("Stripe webhook processing rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
