"""Identity Guard and Anti-Reward-Hacking Drift Gate (AF-2.2 & Roadmap v6 R1.6).

Why the Identity Guard is the foundational gate of the AesFix closed loop:
1. Reward Hacking Prevention: When visual advice or generative edits (Tier-B) are evaluated
   purely on aesthetic gain, a model could hallucinate a completely new image.
2. Perceptual and Structural Invariance: The guard computes feature drift via a pluggable
   backend (RGB layout statistics, LPIPS, or DINOv2) alongside Structural Similarity (SSIM).
3. Hard Gate: If composite perceptual drift exceeds the threshold, the edit is rejected
   (verdict.is_valid = False) and zero aesthetic gain is credited.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class GuardVerdict:
    """Evaluation verdict emitted by the Identity Guard for an edit pair."""
    is_valid: bool
    composite_drift: float
    feature_drift: float
    structural_drift: float
    threshold: float
    backend_name: str = "rgb_layout_stat"
    rejection_reason: Optional[str] = None
    guard_backend: str = "SSIM + rgb_layout_stat"


def _to_rgb_tensor(image_input: Union[Image.Image, torch.Tensor, np.ndarray, str, Path]) -> torch.Tensor:
    """Converts diverse image inputs into a normalized float32 tensor [1, 3, H, W] in [0, 1]."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    elif isinstance(image_input, np.ndarray):
        arr = image_input.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3:
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return torch.from_numpy(arr)
    elif isinstance(image_input, torch.Tensor):
        t = image_input.float()
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.max() > 1.0:
            t = t / 255.0
        return t
    raise TypeError(f"Unsupported image type for Identity Guard: {type(image_input)}")


def compute_ssim_tensor(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> float:
    """Calculates mean Structural Similarity Index (SSIM) between two image tensors."""
    if x.shape[2:] != y.shape[2:]:
        y = F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False)

    lum_x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    lum_y = 0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2

    mu_x = F.avg_pool2d(lum_x, kernel_size=window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(lum_y, kernel_size=window_size, stride=1, padding=window_size // 2)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(lum_x * lum_x, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(lum_y * lum_y, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(lum_x * lum_y, kernel_size=window_size, stride=1, padding=window_size // 2) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / (denominator + 1e-7)
    return float(torch.clamp(ssim_map.mean(), 0.0, 1.0).item())


# ==============================================================================
# Pluggable Drift Backends (Roadmap v6 Part IV.7 & R1.6)
# ==============================================================================

class DriftBackend(ABC):
    """Abstract interface for pluggable perceptual and feature drift backends."""

    @abstractmethod
    def compute_drift(self, t_orig: torch.Tensor, t_edit: torch.Tensor) -> float:
        """Computes drift distance in [0, 1] between original and edited tensors [1, 3, H, W]."""
        raise NotImplementedError


def rgb_layout_stat(tensor: torch.Tensor) -> torch.Tensor:
    """Computes spatial layout color statistics across a multi-scale pooling grid (1x1, 2x2, 4x4).

    Note (Roadmap v6 R1.6): This is a spatial RGB layout statistic, not a semantic embedding.
    """
    p1 = F.adaptive_avg_pool2d(tensor, (1, 1)).flatten(1)
    p2 = F.adaptive_avg_pool2d(tensor, (2, 2)).flatten(1)
    p4 = F.adaptive_avg_pool2d(tensor, (4, 4)).flatten(1)

    stat = torch.cat([p1, p2, p4], dim=1)
    return F.normalize(stat, p=2, dim=1)


class RGBLayoutBackend(DriftBackend):
    """Default fast drift backend using multi-scale RGB layout statistics."""

    def compute_drift(self, t_orig: torch.Tensor, t_edit: torch.Tensor) -> float:
        stat_orig = rgb_layout_stat(t_orig)
        stat_edit = rgb_layout_stat(t_edit)
        cos_sim = float(torch.sum(stat_orig * stat_edit, dim=1).item())
        return float(np.clip(1.0 - cos_sim, 0.0, 1.0))


class LPIPSBackend(DriftBackend):
    """Lazy-loaded Learned Perceptual Image Patch Similarity (LPIPS) backend."""

    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        self.net = net
        self.device = device
        self._model: Optional[torch.nn.Module] = None

    def _get_model(self) -> torch.nn.Module:
        if self._model is None:
            try:
                import lpips
                self._model = lpips.LPIPS(net=self.net).to(self.device).eval()
            except ImportError as err:
                raise ImportError(
                    f"LPIPSBackend requires the 'lpips' package. Install with: pip install lpips ({err})"
                )
        return self._model

    def compute_drift(self, t_orig: torch.Tensor, t_edit: torch.Tensor) -> float:
        model = self._get_model()
        # LPIPS expects input tensors scaled to [-1, 1]
        x0 = (t_orig.to(self.device) * 2.0) - 1.0
        x1 = (t_edit.to(self.device) * 2.0) - 1.0
        if x0.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=x0.shape[2:], mode="bilinear", align_corners=False)
        with torch.no_grad():
            dist = model(x0, x1)
        return float(torch.clamp(dist.mean(), 0.0, 1.0).item())


class DINOv2Backend(DriftBackend):
    """Lazy-loaded DINOv2 ViT-S/14 CLS representation cosine drift backend."""

    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Optional[torch.nn.Module] = None

    def _get_model(self) -> torch.nn.Module:
        if self._model is None:
            try:
                from transformers import AutoModel
                self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            except Exception as err:
                raise RuntimeError(
                    f"DINOv2Backend failed to load model '{self.model_name}': {err}"
                )
        return self._model

    def compute_drift(self, t_orig: torch.Tensor, t_edit: torch.Tensor) -> float:
        model = self._get_model()
        with torch.no_grad():
            # Interpolate to standard ViT resolution (224x224)
            x0 = F.interpolate(t_orig.to(self.device), size=(224, 224), mode="bilinear", align_corners=False)
            x1 = F.interpolate(t_edit.to(self.device), size=(224, 224), mode="bilinear", align_corners=False)
            out0 = model(pixel_values=x0)
            out1 = model(pixel_values=x1)
            cls0 = F.normalize(out0.last_hidden_state[:, 0], p=2, dim=-1)
            cls1 = F.normalize(out1.last_hidden_state[:, 0], p=2, dim=-1)
            cos_sim = float(torch.sum(cls0 * cls1, dim=-1).item())
            return float(np.clip(1.0 - cos_sim, 0.0, 1.0))


# ==============================================================================
# Identity Guard Gate
# ==============================================================================

class IdentityGuard:
    """Anti-reward-hacking perceptual and structural drift gate."""

    def __init__(
        self,
        drift_threshold: float = 0.35,
        feature_weight: float = 0.60,
        structural_weight: float = 0.40,
        backend: Union[str, DriftBackend] = "rgb_layout_stat",
    ) -> None:
        self.drift_threshold = float(drift_threshold)
        self.feature_weight = float(feature_weight)
        self.structural_weight = float(structural_weight)

        if isinstance(backend, str):
            b = backend.lower()
            if b in {"rgb_layout_stat", "layout", "rgb"}:
                self.backend: DriftBackend = RGBLayoutBackend()
                self.backend_name = "rgb_layout_stat"
            elif b == "lpips":
                self.backend = LPIPSBackend()
                self.backend_name = "lpips"
            elif b in {"dinov2", "dino"}:
                self.backend = DINOv2Backend()
                self.backend_name = "dinov2"
            else:
                raise ValueError(f"Unknown guard backend: {backend!r}")
        elif isinstance(backend, DriftBackend):
            self.backend = backend
            self.backend_name = backend.__class__.__name__
        else:
            raise TypeError(f"Backend must be str or DriftBackend, got {type(backend)}")

    def _extract_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Backward-compatibility helper returning rgb_layout_stat."""
        return rgb_layout_stat(tensor)

    def evaluate(
        self,
        original: Union[Image.Image, torch.Tensor, np.ndarray, str, Path],
        edited: Union[Image.Image, torch.Tensor, np.ndarray, str, Path],
    ) -> GuardVerdict:
        """Evaluates whether the edited image maintains identity fidelity to the original."""
        t_orig = _to_rgb_tensor(original)
        t_edit = _to_rgb_tensor(edited)

        # 1. Structural similarity: 1.0 - SSIM in [0, 1]
        ssim_val = compute_ssim_tensor(t_orig, t_edit)
        structural_drift = float(np.clip(1.0 - ssim_val, 0.0, 1.0))

        # 2. Feature / perceptual drift from pluggable backend
        feature_drift = self.backend.compute_drift(t_orig, t_edit)

        # 3. Composite drift calculation
        composite_drift = (
            self.feature_weight * feature_drift + self.structural_weight * structural_drift
        )

        is_valid = composite_drift <= self.drift_threshold
        rejection_reason = None
        if not is_valid:
            rejection_reason = (
                f"Perceptual drift {composite_drift:.3f} exceeds tolerance {self.drift_threshold:.3f} "
                f"({self.backend_name}={feature_drift:.3f}, structural_drift={structural_drift:.3f}). "
                "Edit rejected: subject identity altered."
            )

        # Explicitly report the composite guard backends (SSIM structural + active feature backend)
        # to ensure honesty about what models actually run in the pipeline.
        full_guard_backend = f"SSIM + {self.backend_name}"
        return GuardVerdict(
            is_valid=is_valid,
            composite_drift=round(composite_drift, 4),
            feature_drift=round(feature_drift, 4),
            structural_drift=round(structural_drift, 4),
            threshold=self.drift_threshold,
            backend_name=self.backend_name,
            rejection_reason=rejection_reason,
            guard_backend=full_guard_backend,
        )

    def verify_gain(
        self,
        original_score: float,
        edited_score: float,
        original: Union[Image.Image, torch.Tensor, np.ndarray, str, Path],
        edited: Union[Image.Image, torch.Tensor, np.ndarray, str, Path],
    ) -> Tuple[float, GuardVerdict]:
        """Calculates verified aesthetic gain gated by the Identity Guard."""
        raw_gain = max(0.0, float(edited_score - original_score))
        verdict = self.evaluate(original, edited)

        if not verdict.is_valid:
            return 0.0, verdict

        return round(raw_gain, 4), verdict
