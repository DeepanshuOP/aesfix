"""Edit executor (Tier-A deterministic + Tier-B generative).

Why Tier-A and XMP are exported at top level:
Enables immediate single-line access for batch scripts, the Gradio demo,
and the AG@k evaluation harness.
"""
from aesfix.executor.guard import (
    GuardVerdict,
    IdentityGuard,
    compute_ssim_tensor,
)
from aesfix.executor.tier_a import (
    TIER_A_OPERATIONS,
    TierAExecutor,
    TierAResult,
    XMPWriter,
)

__all__ = [
    "TierAExecutor",
    "TierAResult",
    "XMPWriter",
    "TIER_A_OPERATIONS",
    "IdentityGuard",
    "GuardVerdict",
    "compute_ssim_tensor",
]
