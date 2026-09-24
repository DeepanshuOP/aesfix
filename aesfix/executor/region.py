"""Mask-Conditioned Region-Targeted Tier-A+ Executor (AF-2.4 & Roadmap §18).

Why Tier-A+ region editing is essential for professional photography:
1. Localized Retouching Without Generative Distortion: Photographers rarely edit photos
   globally. A classic scenario is an underexposed portrait subject against a correctly
   exposed golden-hour sunset. Global exposure blows out the sky; localized Tier-A
   selectively lifts subject exposure while preserving the background.
2. Feathered Edge Blending: To eliminate artificial halos and harsh segmentation borders,
   all binary or bounding masks undergo analytical Gaussian feathering:
   I_final = M_feathered * I_edited + (1 - M_feathered) * I_original.
3. Decoupled Mask Generation: Supports SAM2 segmentation masks, bounding box coordinates,
   and semantic keywords ('subject', 'foreground', 'background', 'sky') with deterministic
   low-latency CPU fallbacks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aesfix.critic.critic import CriticFix
from aesfix.executor.tier_a import TierAExecutor, _pil_to_tensor, _tensor_to_pil


@dataclass
class RegionMask:
    """A spatial alpha mask defining the target region for a localized operation.

    Why this dataclass exists:
    Encapsulates the 2D floating-point weight grid in [0, 1] alongside metadata
    and feathering radius to ensure seamless compositing.
    """
    mask: torch.Tensor  # shape [1, 1, H, W] in [0.0, 1.0]
    region_type: str    # e.g. 'subject', 'background', 'bbox', 'custom'
    feather_radius: float = 5.0


class MaskGenerator:
    """Generates feathered spatial masks from region hints, bounding boxes, or heuristics."""

    @staticmethod
    def feather_mask(mask: torch.Tensor, radius: float = 7.0) -> torch.Tensor:
        """Applies Gaussian smoothing to a binary or soft mask to prevent border artifacts.

        Why Gaussian feathering:
        In photo editing, hard mask borders create visible edge halos and clipping.
        A smoothed transition zone ensures imperceptible blending.
        """
        if radius <= 0.5:
            return mask

        kernel_size = int(max(3, round(radius * 3) // 2 * 2 + 1))
        sigma = radius / 2.0

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=mask.device) - (kernel_size - 1) / 2.0
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()

        # 2D separable Gaussian filter
        g2d = g1d.unsqueeze(0) * g1d.unsqueeze(1)
        kernel = g2d.unsqueeze(0).unsqueeze(0)

        padding = kernel_size // 2
        padded = F.pad(mask, (padding, padding, padding, padding), mode="replicate")
        feathered = F.conv2d(padded, kernel)
        return torch.clamp(feathered, 0.0, 1.0)

    @classmethod
    def from_bbox(
        cls,
        shape: Tuple[int, int],
        bbox: Union[List[float], Tuple[float, float, float, float]],
        feather_radius: float = 9.0,
        device: Optional[torch.device] = None,
    ) -> RegionMask:
        """Constructs an elliptical feathered mask from normalized coordinates [ymin, xmin, ymax, xmax].

        Why elliptical soft gradient:
        Natural subjects (faces, products) are rounded. An inscribed super-ellipse
        provides much higher aesthetic blending fidelity than a box.
        """
        h, w = shape
        ymin, xmin, ymax, xmax = [float(v) for v in bbox]

        # Convert normalized [0, 1] to pixel space
        y0, x0 = int(ymin * h), int(xmin * w)
        y1, x1 = int(ymax * h), int(xmax * w)
        cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
        ry, rx = max(1.0, (y1 - y0) / 2.0), max(1.0, (x1 - x0) / 2.0)

        # Coordinate grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        )

        # Normalized elliptical distance
        dist_sq = ((grid_y - cy) / ry) ** 2 + ((grid_x - cx) / rx) ** 2
        raw_mask = torch.clamp(1.0 - torch.sqrt(dist_sq + 1e-6), 0.0, 1.0)
        raw_mask = raw_mask.unsqueeze(0).unsqueeze(0)

        feathered = cls.feather_mask(raw_mask, radius=feather_radius)
        return RegionMask(mask=feathered, region_type="bbox", feather_radius=feather_radius)

    @classmethod
    def from_semantic_hint(
        cls,
        image_tensor: torch.Tensor,
        hint: str,
        feather_radius: float = 11.0,
    ) -> RegionMask:
        """Derives a region mask from a semantic keyword hint ('subject', 'background', 'sky').

        Why fast heuristic fallback:
        Enables instantaneous zero-cost CPU processing without waiting for heavy multi-GB
        SAM2 weights during live preview sessions.
        """
        _, _, h, w = image_tensor.shape
        clean_hint = hint.strip().lower()

        if clean_hint in ("subject", "foreground", "person"):
            # Saliency-informed center-weighted subject mask: center 60% with soft falloff
            mask = cls.from_bbox((h, w), [0.15, 0.20, 0.85, 0.80], feather_radius=feather_radius, device=image_tensor.device)
            mask.region_type = "subject"
            return mask

        elif clean_hint in ("background", "env", "environment"):
            # Inverted subject mask: targets peripheral and backdrop elements
            subj_mask = cls.from_bbox((h, w), [0.15, 0.20, 0.85, 0.80], feather_radius=feather_radius, device=image_tensor.device)
            inverted = 1.0 - subj_mask.mask
            return RegionMask(mask=inverted, region_type="background", feather_radius=feather_radius)

        elif clean_hint in ("sky", "upper"):
            # Upper third luminance-favored gradient
            grid_y, _ = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=image_tensor.device),
                torch.arange(w, dtype=torch.float32, device=image_tensor.device),
                indexing="ij",
            )
            sky_grad = torch.clamp(1.0 - (grid_y / (h * 0.45)), 0.0, 1.0).unsqueeze(0).unsqueeze(0)
            feathered = cls.feather_mask(sky_grad, radius=feather_radius)
            return RegionMask(mask=feathered, region_type="sky", feather_radius=feather_radius)

        # Default fallback: broad center weighting
        return cls.from_bbox((h, w), [0.1, 0.1, 0.9, 0.9], feather_radius=feather_radius, device=image_tensor.device)


class MaskedTierAExecutor:
    """Applies Tier-A deterministic CV operations strictly within feathered regions (Tier-A+)."""

    def __init__(self, base_executor: Optional[TierAExecutor] = None) -> None:
        self.base_executor = base_executor or TierAExecutor()

    def execute_masked_op(
        self,
        image_input: Union[Image.Image, torch.Tensor, np.ndarray, str, Path],
        op: str,
        params: Dict[str, Any],
        mask: Union[RegionMask, torch.Tensor, np.ndarray, str],
        invert_mask: bool = False,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Applies a single photographic operation modulated by a region mask.

        Why alpha blending formula:
        I_out = M * I_edited + (1 - M) * I_original
        Guarantees that unmasked pixels are bitwise identical to the original image.
        """
        # Convert image input to tensor
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
            t_orig = _pil_to_tensor(pil_img)
        elif isinstance(image_input, Image.Image):
            t_orig = _pil_to_tensor(image_input)
        elif isinstance(image_input, np.ndarray):
            t_orig = _pil_to_tensor(Image.fromarray(image_input))
        elif isinstance(image_input, torch.Tensor):
            t_orig = image_input if image_input.ndim == 4 else image_input.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        _, _, h, w = t_orig.shape

        # Ingest and validate mask representation
        if isinstance(mask, RegionMask):
            m_tensor = mask.mask
        elif isinstance(mask, str):
            # Parse semantic hint (e.g. 'subject', 'sky')
            m_tensor = MaskGenerator.from_semantic_hint(t_orig, mask).mask
        elif isinstance(mask, torch.Tensor):
            m_tensor = mask if mask.ndim == 4 else mask.unsqueeze(0).unsqueeze(0)
        elif isinstance(mask, np.ndarray):
            arr = mask.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            m_tensor = torch.from_numpy(arr)
            while m_tensor.ndim < 4:
                m_tensor = m_tensor.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")

        # Resample mask to match image tensor shape if different
        if m_tensor.shape[2:] != (h, w):
            m_tensor = F.interpolate(m_tensor, size=(h, w), mode="bilinear", align_corners=False)

        if invert_mask:
            m_tensor = 1.0 - m_tensor

        m_tensor = torch.clamp(m_tensor, 0.0, 1.0).to(t_orig.device)

        # 1. Execute the base Tier-A operation globally on a clone
        t_edited = self.base_executor.execute_op(t_orig.clone(), op, params)

        # 2. Feathered alpha composite
        t_out = m_tensor * t_edited + (1.0 - m_tensor) * t_orig
        t_out = torch.clamp(t_out, 0.0, 1.0)

        out_pil = _tensor_to_pil(t_out)
        return out_pil, t_out

    def execute_masked_fix(
        self,
        image_input: Union[Image.Image, torch.Tensor, np.ndarray, str, Path],
        fix: CriticFix,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """Convenience method executing a CriticFix with its declared region_hint.

        Why: Provides direct interoperability with AspectCritic recommendations.
        """
        hint = fix.region_hint or "subject"
        return self.execute_masked_op(
            image_input=image_input,
            op=fix.op,
            params=fix.params,
            mask=hint,
        )
