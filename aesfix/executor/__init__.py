"""Edit executor (Tier-A deterministic + Tier-B generative).

Why Tier-A and XMP are exported at top level:
Enables immediate single-line access for batch scripts, the Gradio demo,
and the AG@k evaluation harness.
"""
from aesfix.executor.guard import (
    DINOv2Backend,
    DriftBackend,
    GuardVerdict,
    IdentityGuard,
    LPIPSBackend,
    RGBLayoutBackend,
    compute_ssim_tensor,
    rgb_layout_stat,
)
from aesfix.executor.region import (
    MaskGenerator,
    MaskedTierAExecutor,
    RegionMask,
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
    "DriftBackend",
    "RGBLayoutBackend",
    "LPIPSBackend",
    "DINOv2Backend",
    "rgb_layout_stat",
    "RegionMask",
    "MaskGenerator",
    "MaskedTierAExecutor",
]
