"""Tier-A Deterministic Image Executor and Adobe XMP Sidecar Engine (AF-2.1).

Why Tier-A exists alongside generative models:
1. Determinism and zero hallucination: Adjusting exposure, contrast, white balance,
   or horizon tilt mathematically alters existing pixels without generating synthetic artifacts.
2. Latency and COGS: Pure PyTorch/Kornia tensor transformations run in single-digit
   milliseconds on CPU, costing $0.00 in cloud inference fees.
3. Industry integration (The XMP principle): Professional photographers and agencies
   rely on Adobe Lightroom and Camera Raw. By exporting every verified fix into a
   standards-compliant .xmp sidecar, AesFix fits directly into existing editing pipelines
   instead of trapping edits in a proprietary format.
"""
from __future__ import annotations

import math
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:
    import kornia
    # Why dynamic resolution: Pylance and Pyright frequently flag 'kornia.enhance/filters/geometry'
    # as unresolvable submodule imports depending on the user's active Python environment or stub state.
    # Dynamically binding top-level attributes resolves cleanly without language-server squiggles.
    ke: Any = getattr(kornia, "enhance", None)
    kf: Any = getattr(kornia, "filters", None)
    _kgeom: Any = getattr(kornia, "geometry", None)
    kg: Any = getattr(_kgeom, "transform", None) if _kgeom is not None else None
    _HAS_KORNIA = bool(ke is not None and kf is not None and kg is not None)
except (ImportError, ModuleNotFoundError):
    ke = None
    kf = None
    kg = None
    _HAS_KORNIA = False

from aesfix.critic.critic import CriticFix

# The 12 Canonical Tier-A operations supported for round-trip execution and XMP export
TIER_A_OPERATIONS: Tuple[str, ...] = (
    "adjust_exposure",
    "adjust_contrast",
    "adjust_highlights",
    "adjust_shadows",
    "adjust_white_balance",
    "adjust_saturation",
    "adjust_vibrance",
    "adjust_clarity",
    "unsharp_mask",
    "denoise",
    "smart_crop",
    "correct_horizon",
)

# XML Namespaces for Adobe Lightroom Camera Raw settings
XMP_NS_X = "adobe:ns:meta/"
XMP_NS_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XMP_NS_CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"


@dataclass
class TierAResult:
    """Output artifact of a Tier-A deterministic execution pass.

    Why this dataclass exists:
    Maintains both PIL and Tensor representations alongside performance telemetry
    and the compiled XMP sidecar string for zero-friction export.
    """
    image: Image.Image
    tensor: torch.Tensor
    applied_ops: List[str]
    xmp_sidecar: str
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL Image to a float32 torch.Tensor [1, 3, H, W] in [0, 1].

    Why: Kornia and PyTorch operations require batch-first float32 channel representations.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0
    # [H, W, C] -> [1, C, H, W]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a [1, 3, H, W] or [3, H, W] tensor back into a uint8 PIL Image.

    Why: Clipping and type-casting ensure pixel values remain strictly valid 8-bit RGB.
    """
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t.squeeze(0)
    t = torch.clamp(t, 0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class TierAExecutor:
    """Differentiable and CPU-accelerated Tier-A image operations suite."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        # Default to CPU for deterministic, low-COGS execution
        self.device = device or torch.device("cpu")

    def execute_op(self, tensor: torch.Tensor, op: str, params: Dict[str, Any]) -> torch.Tensor:
        """Applies a single photographic operation to an image tensor.

        Why dispatch by op string: Allows flexible, decoupled replay of JSON instructions
        emitted by the Aspect Critic or loaded from an XMP preset.
        """
        tensor = tensor.to(self.device)

        if op == "adjust_exposure":
            # EV delta: scale intensity in linear domain then re-encode
            ev = float(params.get("ev_delta", params.get("ev", params.get("value", 0.0))))
            if abs(ev) > 1e-4:
                # Perceptually calibrated linear multiplier: I_out = I * 2^EV
                gain = math.pow(2.0, ev)
                tensor = torch.clamp(tensor * gain, 0.0, 1.0)

        elif op == "adjust_contrast":
            # Contrast: pivot around midtone 0.5
            if "contrast_factor" in params:
                warnings.warn(
                    "contrast_factor is deprecated; use canonical 'factor' instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
            factor = params.get("factor", params.get("contrast_factor"))
            if factor is None:
                contrast_val = float(params.get("contrast", params.get("value", 0.0)))
                factor = 1.0 + (contrast_val / 100.0)
            factor = max(0.1, min(float(factor), 3.0))
            if _HAS_KORNIA and ke is not None:
                tensor = ke.adjust_contrast(tensor, factor)
            else:
                tensor = torch.clamp(0.5 + factor * (tensor - 0.5), 0.0, 1.0)

        elif op == "adjust_highlights":
            # Smoothstep mask isolating luminance > 0.5 to lift/recover blown bright areas
            hl = float(params.get("highlights", params.get("value", 0.0))) / 100.0
            if abs(hl) > 1e-4:
                lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
                mask = torch.clamp((lum - 0.5) / 0.5, 0.0, 1.0)
                mask = mask * mask * (3.0 - 2.0 * mask)  # Cubic smoothstep for seamless blend
                tensor = torch.clamp(tensor + hl * mask * tensor, 0.0, 1.0)

        elif op == "adjust_shadows":
            # Smoothstep mask isolating luminance < 0.5 to lift or deepen dark details
            sh = float(params.get("shadows", params.get("value", 0.0))) / 100.0
            if abs(sh) > 1e-4:
                lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
                mask = torch.clamp((0.5 - lum) / 0.5, 0.0, 1.0)
                mask = mask * mask * (3.0 - 2.0 * mask)
                tensor = torch.clamp(tensor + sh * mask * (1.0 - tensor), 0.0, 1.0)

        elif op == "adjust_white_balance":
            # Color temperature (amber/blue) and tint (magenta/green)
            if "temperature_shift" in params:
                warnings.warn(
                    "temperature_shift is deprecated; use canonical 'temperature' instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if "temp" in params and "temperature" not in params:
                warnings.warn(
                    "temp is deprecated; use canonical 'temperature' instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
            temp = float(params.get("temperature", params.get("temp", params.get("temperature_shift", 0.0)))) / 100.0
            tint = float(params.get("tint", 0.0)) / 100.0
            # Red boosted / Blue lowered for warmth
            r = tensor[:, 0:1] * (1.0 + 0.25 * temp + 0.15 * tint)
            g = tensor[:, 1:2] * (1.0 - 0.20 * tint)
            b = tensor[:, 2:3] * (1.0 - 0.25 * temp + 0.15 * tint)
            tensor = torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)

        elif op == "adjust_saturation":
            factor = params.get("factor")
            if factor is None:
                sat_val = float(params.get("saturation", params.get("value", 0.0)))
                factor = 1.0 + (sat_val / 100.0)
            factor = max(0.0, min(float(factor), 3.0))
            if _HAS_KORNIA and ke is not None:
                tensor = ke.adjust_saturation(tensor, factor)
            else:
                lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
                tensor = torch.clamp(lum + factor * (tensor - lum), 0.0, 1.0)

        elif op == "adjust_vibrance":
            # Perceptual saturation: selectively boosts muted pixels more than saturated pixels
            vib = float(params.get("vibrance", params.get("value", 0.0))) / 100.0
            if abs(vib) > 1e-4:
                max_c, _ = torch.max(tensor, dim=1, keepdim=True)
                min_c, _ = torch.min(tensor, dim=1, keepdim=True)
                sat = max_c - min_c
                # Weight inversely proportional to existing saturation
                weight = 1.0 - sat
                lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
                tensor = torch.clamp(tensor + vib * weight * (tensor - lum), 0.0, 1.0)

        elif op == "adjust_clarity":
            # Midtone local contrast: unsharp mask with large radius on midtone frequencies
            clarity = float(params.get("clarity", params.get("value", 0.0))) / 100.0
            if abs(clarity) > 1e-4:
                if _HAS_KORNIA and kf is not None:
                    blurred = kf.gaussian_blur2d(tensor, (9, 9), (2.5, 2.5))
                else:
                    # Fallback average pool as box blur
                    blurred = torch.nn.functional.avg_pool2d(
                        torch.nn.functional.pad(tensor, (4, 4, 4, 4), mode="reflect"),
                        kernel_size=9, stride=1,
                    )
                lum = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
                # Midtone mask: peak at 0.5, zero at 0.0 and 1.0
                midtone_mask = 1.0 - torch.pow(2.0 * lum - 1.0, 2)
                tensor = torch.clamp(tensor + clarity * 0.5 * midtone_mask * (tensor - blurred), 0.0, 1.0)

        elif op == "unsharp_mask":
            amount = float(params.get("amount", 1.0))
            if amount > 10.0:  # Scaled as Lightroom 0..150
                amount = amount / 50.0
            radius = float(params.get("radius", 1.0))
            kernel_size = max(3, int(radius * 3) * 2 + 1)
            sigma = max(0.5, radius)
            if _HAS_KORNIA and kf is not None:
                blurred = kf.gaussian_blur2d(tensor, (kernel_size, kernel_size), (sigma, sigma))
            else:
                blurred = torch.nn.functional.avg_pool2d(
                    torch.nn.functional.pad(tensor, (1, 1, 1, 1), mode="reflect"),
                    kernel_size=3, stride=1,
                )
            high_pass = tensor - blurred
            tensor = torch.clamp(tensor + amount * high_pass, 0.0, 1.0)

        elif op == "denoise":
            # Denoise strength has exactly one unit: 0..100 (Lightroom-style)
            if "luminance_smoothing" in params and "strength" not in params:
                warnings.warn(
                    "luminance_smoothing is deprecated; use canonical 'strength' in [0, 100] instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                raw_strength = params["luminance_smoothing"]
            else:
                raw_strength = params.get("strength", 20.0)

            strength = float(raw_strength)
            if strength < 0.0 or strength > 100.0:
                raise ValueError(f"Denoise strength must be in [0, 100], got {strength}")

            norm_strength = strength / 100.0
            if norm_strength > 1e-4:
                if _HAS_KORNIA and kf is not None:
                    blurred = kf.gaussian_blur2d(tensor, (5, 5), (1.0 + norm_strength * 1.5, 1.0 + norm_strength * 1.5))
                else:
                    blurred = torch.nn.functional.avg_pool2d(
                        torch.nn.functional.pad(tensor, (1, 1, 1, 1), mode="reflect"),
                        kernel_size=3, stride=1,
                    )
                tensor = torch.clamp((1.0 - norm_strength * 0.6) * tensor + (norm_strength * 0.6) * blurred, 0.0, 1.0)

        elif op == "smart_crop":
            # Normalized box [top, left, bottom, right] or params dictionary
            _, _, h, w = tensor.shape
            top = float(params.get("top", 0.0))
            left = float(params.get("left", 0.0))
            bottom = float(params.get("bottom", 1.0))
            right = float(params.get("right", 1.0))

            if "crop_box" in params and isinstance(params["crop_box"], (list, tuple)) and len(params["crop_box"]) == 4:
                top, left, bottom, right = [float(v) for v in params["crop_box"]]
            elif ("rule" in params or "tightness" in params) and "top" not in params:
                warnings.warn(
                    "rule/tightness is deprecated for smart_crop; use canonical 'top', 'left', 'bottom', 'right' instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                tightness = float(params.get("tightness", 0.85))
                pad = max(0.0, (1.0 - tightness) / 2.0)
                top, left, bottom, right = pad, pad, 1.0 - pad, 1.0 - pad

            # Bound inside [0, 1]
            y1 = max(0, min(int(top * h), h - 10))
            y2 = max(y1 + 10, min(int(bottom * h), h))
            x1 = max(0, min(int(left * w), w - 10))
            x2 = max(x1 + 10, min(int(right * w), w))
            tensor = tensor[:, :, y1:y2, x1:x2]

        elif op == "correct_horizon":
            # Horizon straightening angle in degrees with black-border prevention crop
            angle = float(params.get("angle", params.get("crop_angle", 0.0)))
            if abs(angle) > 1e-3:
                angle_rad = abs(math.radians(angle))
                if _HAS_KORNIA and kg is not None:
                    angle_t = torch.tensor([angle], dtype=torch.float32, device=tensor.device)
                    tensor = kg.rotate(tensor, angle_t)
                else:
                    # Fallback PIL rotation if Kornia missing
                    pil_img = _tensor_to_pil(tensor)
                    pil_img = pil_img.rotate(angle, resample=Image.Resampling.BILINEAR)
                    tensor = _pil_to_tensor(pil_img).to(self.device)

                # Center crop to discard black rotated wedges
                _, _, h, w = tensor.shape
                pad_h = int(h * math.sin(angle_rad) * 0.75)
                pad_w = int(w * math.sin(angle_rad) * 0.75)
                if pad_h > 0 and pad_w > 0 and (h - 2 * pad_h) > 10 and (w - 2 * pad_w) > 10:
                    tensor = tensor[:, :, pad_h:h - pad_h, pad_w:w - pad_w]

        return tensor

    def execute_fixes(
        self,
        image_input: Union[Image.Image, torch.Tensor, np.ndarray, str, Path],
        fixes: List[Union[CriticFix, Dict[str, Any]]],
    ) -> TierAResult:
        """Executes a list of fixes sequentially and generates the matching XMP sidecar.

        Why: Ensures end-to-end execution of critic recommendations with atomic timing
        and verifiable output artifacts.
        """
        start_time = time.perf_counter()

        # Ingest input representation
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
            tensor = _pil_to_tensor(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            tensor = _pil_to_tensor(pil_img)
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
            tensor = _pil_to_tensor(pil_img)
        elif isinstance(image_input, torch.Tensor):
            tensor = image_input
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            pil_img = _tensor_to_pil(tensor)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        applied_ops: List[str] = []
        normalized_fixes: List[CriticFix] = []

        for item in fixes:
            if isinstance(item, dict):
                fix = CriticFix(
                    op=item.get("op", ""),
                    params=item.get("params", {}),
                    expected_aspect=item.get("expected_aspect", "composition"),
                    nl_instruction=item.get("nl_instruction", ""),
                    region_hint=item.get("region_hint"),
                )
            else:
                fix = item

            normalized_fixes.append(fix)
            tensor = self.execute_op(tensor, fix.op, fix.params)
            applied_ops.append(fix.op)

        out_pil = _tensor_to_pil(tensor)
        xmp_str = XMPWriter.generate_xmp(normalized_fixes)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return TierAResult(
            image=out_pil,
            tensor=tensor,
            applied_ops=applied_ops,
            xmp_sidecar=xmp_str,
            duration_ms=duration_ms,
        )


class XMPWriter:
    """Lightroom / Adobe Camera Raw compliant XMP sidecar serialization and deserialization.

    Why this class is critical:
    Allows AesFix to export presets and sidecars that natively open in Adobe Lightroom,
    Photoshop ACR, and Darktable, and allows parsing XMP presets back into CriticFix pipelines.
    """

    @classmethod
    def generate_xmp(
        cls,
        fixes: List[CriticFix],
        image_filename: Optional[str] = None,
    ) -> str:
        """Converts a sequence of CriticFix operations into an Adobe XMP XML document.

        Why formatted attributes: Lightroom parses camera-raw-settings attributes
        strictly using the 'crs:' namespace with signed decimal notation.
        """
        # Default neutral Lightroom attributes
        crs_attrs: Dict[str, str] = {
            "Version": "16.0",
            "ProcessVersion": "15.4",
            "WhiteBalance": "Custom",
        }

        # Accumulate parameters across fixes
        for fix in fixes:
            op = fix.op
            p = fix.params

            if op == "adjust_exposure":
                ev = float(p.get("ev_delta", p.get("ev", p.get("value", 0.0))))
                crs_attrs["Exposure2012"] = f"{ev:+.2f}"

            elif op == "adjust_contrast":
                if "contrast_factor" in p:
                    warnings.warn(
                        "contrast_factor is deprecated; use canonical 'factor' instead",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                contrast = float(p.get("contrast", p.get("value", 0.0)))
                factor_val = p.get("factor", p.get("contrast_factor"))
                if factor_val is not None:
                    contrast = (float(factor_val) - 1.0) * 100.0
                crs_attrs["Contrast2012"] = f"{int(round(contrast)):+d}"

            elif op == "adjust_highlights":
                hl = float(p.get("highlights", p.get("value", 0.0)))
                crs_attrs["Highlights2012"] = f"{int(round(hl)):+d}"

            elif op == "adjust_shadows":
                sh = float(p.get("shadows", p.get("value", 0.0)))
                crs_attrs["Shadows2012"] = f"{int(round(sh)):+d}"

            elif op == "adjust_white_balance":
                if "temperature_shift" in p:
                    warnings.warn(
                        "temperature_shift is deprecated; use canonical 'temperature' instead",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                if "temp" in p and "temperature" not in p:
                    warnings.warn(
                        "temp is deprecated; use canonical 'temperature' instead",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                temp = float(p.get("temperature", p.get("temp", p.get("temperature_shift", 0.0))))
                tint = float(p.get("tint", 0.0))
                crs_attrs["Temperature"] = f"{int(round(temp)):+d}"
                crs_attrs["Tint"] = f"{int(round(tint)):+d}"

            elif op == "adjust_saturation":
                sat = float(p.get("saturation", p.get("value", 0.0)))
                if "factor" in p:
                    sat = (float(p["factor"]) - 1.0) * 100.0
                crs_attrs["Saturation"] = f"{int(round(sat)):+d}"

            elif op == "adjust_vibrance":
                vib = float(p.get("vibrance", p.get("value", 0.0)))
                crs_attrs["Vibrance"] = f"{int(round(vib)):+d}"

            elif op == "adjust_clarity":
                clarity = float(p.get("clarity", p.get("value", 0.0)))
                crs_attrs["Clarity2012"] = f"{int(round(clarity)):+d}"

            elif op == "unsharp_mask":
                sharp = float(p.get("amount", 40.0))
                radius = float(p.get("radius", 1.0))
                crs_attrs["Sharpness"] = f"{int(round(sharp))}"
                crs_attrs["SharpenRadius"] = f"{radius:+.1f}"

            elif op == "denoise":
                if "luminance_smoothing" in p and "strength" not in p:
                    warnings.warn(
                        "luminance_smoothing is deprecated; use canonical 'strength' in [0, 100] instead",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    raw_noise = p["luminance_smoothing"]
                else:
                    raw_noise = p.get("strength", 25.0)
                noise = float(raw_noise)
                if noise < 0.0 or noise > 100.0:
                    raise ValueError(f"Denoise strength must be in [0, 100], got {noise}")
                crs_attrs["LuminanceSmoothing"] = f"{int(round(noise))}"

            elif op == "smart_crop":
                top = float(p.get("top", 0.0))
                left = float(p.get("left", 0.0))
                bottom = float(p.get("bottom", 1.0))
                right = float(p.get("right", 1.0))
                if "crop_box" in p and isinstance(p["crop_box"], (list, tuple)) and len(p["crop_box"]) == 4:
                    top, left, bottom, right = [float(v) for v in p["crop_box"]]
                elif ("rule" in p or "tightness" in p) and "top" not in p:
                    warnings.warn(
                        "rule/tightness is deprecated for smart_crop; use canonical 'top', 'left', 'bottom', 'right' instead",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    tightness = float(p.get("tightness", 0.85))
                    pad = max(0.0, (1.0 - tightness) / 2.0)
                    top, left, bottom, right = pad, pad, 1.0 - pad, 1.0 - pad
                crs_attrs["HasCrop"] = "True"
                crs_attrs["CropTop"] = f"{top:.4f}"
                crs_attrs["CropLeft"] = f"{left:.4f}"
                crs_attrs["CropBottom"] = f"{bottom:.4f}"
                crs_attrs["CropRight"] = f"{right:.4f}"

            elif op == "correct_horizon":
                angle = float(p.get("angle", p.get("crop_angle", 0.0)))
                crs_attrs["CropAngle"] = f"{angle:+.4f}"

        # Construct XML tree
        xmpmeta = ET.Element(
            f"{{{XMP_NS_X}}}xmpmeta",
            {f"{{{XMP_NS_X}}}xmptk": "AesFix Engine v0.2.0 (Lightroom Compatible)"},
        )
        rdf = ET.SubElement(xmpmeta, f"{{{XMP_NS_RDF}}}RDF")
        desc_attrs: Dict[str, str] = {f"{{{XMP_NS_RDF}}}about": image_filename or ""}

        for k, v in crs_attrs.items():
            desc_attrs[f"{{{XMP_NS_CRS}}}{k}"] = v

        ET.SubElement(rdf, f"{{{XMP_NS_RDF}}}Description", desc_attrs)

        # Register standard prefixes for clean serialization
        ET.register_namespace("x", XMP_NS_X)
        ET.register_namespace("rdf", XMP_NS_RDF)
        ET.register_namespace("crs", XMP_NS_CRS)

        xml_bytes = ET.tostring(xmpmeta, encoding="utf-8", xml_declaration=True)
        return xml_bytes.decode("utf-8")

    @classmethod
    def export_xmp(
        cls,
        fixes: List[CriticFix],
        output_path: Union[str, Path],
        image_filename: Optional[str] = None,
    ) -> Path:
        """Writes the generated XMP XML document to disk.

        Why: Provides the physical file saved alongside JPEG/RAW images in client shoots.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xml_content = cls.generate_xmp(fixes, image_filename=image_filename or path.stem)
        path.write_text(xml_content, encoding="utf-8")
        return path

    @classmethod
    def parse_xmp(cls, xmp_str: str) -> List[CriticFix]:
        """Parses an Adobe XMP XML string back into executable CriticFix objects.

        Why: Enables the DoD round-trip verification (XMP -> CriticFix -> Executor)
        and allows AesFix to ingest existing Lightroom user presets.
        """
        root = ET.fromstring(xmp_str)
        desc = root.find(f".//{{{XMP_NS_RDF}}}Description")
        if desc is None:
            return []

        fixes: List[CriticFix] = []

        def get_attr(key: str) -> Optional[str]:
            return desc.attrib.get(f"{{{XMP_NS_CRS}}}{key}")

        # 1. Exposure
        exp_val = get_attr("Exposure2012")
        if exp_val is not None:
            fixes.append(CriticFix(
                op="adjust_exposure",
                params={"ev_delta": float(exp_val)},
                expected_aspect="lighting",
                nl_instruction=f"Adjust exposure by {exp_val} EV stops in Camera Raw.",
            ))

        # 2. Contrast
        cont_val = get_attr("Contrast2012")
        if cont_val is not None:
            fixes.append(CriticFix(
                op="adjust_contrast",
                params={"contrast": float(cont_val)},
                expected_aspect="lighting",
                nl_instruction=f"Adjust contrast by {cont_val} points.",
            ))

        # 3. Highlights
        hl_val = get_attr("Highlights2012")
        if hl_val is not None:
            fixes.append(CriticFix(
                op="adjust_highlights",
                params={"highlights": float(hl_val)},
                expected_aspect="lighting",
                nl_instruction=f"Adjust highlights by {hl_val} points.",
            ))

        # 4. Shadows
        sh_val = get_attr("Shadows2012")
        if sh_val is not None:
            fixes.append(CriticFix(
                op="adjust_shadows",
                params={"shadows": float(sh_val)},
                expected_aspect="lighting",
                nl_instruction=f"Adjust shadows by {sh_val} points.",
            ))

        # 5. White balance
        temp_val = get_attr("Temperature")
        tint_val = get_attr("Tint")
        if temp_val is not None or tint_val is not None:
            fixes.append(CriticFix(
                op="adjust_white_balance",
                params={
                    "temperature": float(temp_val) if temp_val is not None else 0.0,
                    "tint": float(tint_val) if tint_val is not None else 0.0,
                },
                expected_aspect="color",
                nl_instruction="Adjust color temperature and tint balance.",
            ))

        # 6. Saturation
        sat_val = get_attr("Saturation")
        if sat_val is not None:
            fixes.append(CriticFix(
                op="adjust_saturation",
                params={"saturation": float(sat_val)},
                expected_aspect="color",
                nl_instruction=f"Adjust color saturation by {sat_val}%.",
            ))

        # 7. Vibrance
        vib_val = get_attr("Vibrance")
        if vib_val is not None:
            fixes.append(CriticFix(
                op="adjust_vibrance",
                params={"vibrance": float(vib_val)},
                expected_aspect="color",
                nl_instruction=f"Adjust vibrance by {vib_val}%.",
            ))

        # 8. Clarity
        clarity_val = get_attr("Clarity2012")
        if clarity_val is not None:
            fixes.append(CriticFix(
                op="adjust_clarity",
                params={"clarity": float(clarity_val)},
                expected_aspect="technical",
                nl_instruction=f"Enhance midtone local clarity by {clarity_val} points.",
            ))

        # 9. Unsharp Mask (Sharpness)
        sharp_val = get_attr("Sharpness")
        if sharp_val is not None:
            radius_val = get_attr("SharpenRadius")
            fixes.append(CriticFix(
                op="unsharp_mask",
                params={
                    "amount": float(sharp_val),
                    "radius": float(radius_val) if radius_val is not None else 1.0,
                },
                expected_aspect="technical",
                nl_instruction=f"Apply unsharp mask sharpening with amount {sharp_val}.",
            ))

        # 10. Denoise (LuminanceSmoothing)
        denoise_val = get_attr("LuminanceSmoothing")
        if denoise_val is not None:
            fixes.append(CriticFix(
                op="denoise",
                params={"strength": float(denoise_val)},
                expected_aspect="technical",
                nl_instruction=f"Apply luminance noise reduction smoothing of {denoise_val}.",
            ))

        # 11. Smart Crop
        if get_attr("HasCrop") == "True" or get_attr("CropTop") is not None:
            fixes.append(CriticFix(
                op="smart_crop",
                params={
                    "top": float(get_attr("CropTop") or 0.0),
                    "left": float(get_attr("CropLeft") or 0.0),
                    "bottom": float(get_attr("CropBottom") or 1.0),
                    "right": float(get_attr("CropRight") or 1.0),
                },
                expected_aspect="composition",
                nl_instruction="Recrop image to optimal compositional bounding box.",
            ))

        # 12. Correct Horizon (CropAngle)
        angle_val = get_attr("CropAngle")
        if angle_val is not None:
            fixes.append(CriticFix(
                op="correct_horizon",
                params={"angle": float(angle_val)},
                expected_aspect="composition",
                nl_instruction=f"Level tilted horizon by {angle_val} degrees.",
            ))

        return fixes
