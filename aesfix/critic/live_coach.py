"""Live Shooting Coach & Real-Time Viewfinder Guidance Engine (AF-6.5 & Roadmap §25 F4.5).

Why this module exists:
Post-capture editing cannot fix extreme camera shake, irreversibly clipped sensor highlights,
or subjects cropped out of the frame. This module implements low-latency (<30ms) capture-time
guidance over live camera viewfinders, computing horizon tilt, exposure clipping, compositional
power point alignment, and acutance before the shutter release.
"""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple, Union

import numpy as np

logger = logging.getLogger("aesfix.live_coach")

# Standard Rule-of-Thirds power point coordinates in normalized [0, 1] viewport space
RULE_OF_THIRDS_INTERSECTIONS: Tuple[Tuple[float, float], ...] = (
    (1.0 / 3.0, 1.0 / 3.0),
    (2.0 / 3.0, 1.0 / 3.0),
    (1.0 / 3.0, 2.0 / 3.0),
    (2.0 / 3.0, 2.0 / 3.0),
)


class NudgeSeverity(str, Enum):
    """Urgency level of a capture-time viewfinder nudge."""
    CRITICAL = "critical"
    WARNING = "warning"
    ADVISORY = "advisory"


class ShutterRecommendation(str, Enum):
    """Categorical shutter release readiness advice."""
    CAPTURE_READY = "capture_ready"
    CAUTION = "caution"
    HOLD_AND_ADJUST = "hold_and_adjust"


@dataclass
class ViewfinderNudge:
    """Actionable capture-time coaching guidance delivered to the viewfinder overlay."""
    aspect: str
    message: str
    severity: NudgeSeverity
    numerical_metric: float
    correction_action: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass
class ViewfinderFrameTelemetry:
    """Real-time diagnostics and guidance metrics extracted from a single viewfinder frame."""
    tilt_degrees: float
    highlight_clip_pct: float
    shadow_clip_pct: float
    subject_centroid: Tuple[float, float]
    thirds_distance: float
    sharpness_score: float
    aspect_estimates: Dict[str, float]
    capture_readiness_score: float
    shutter_recommendation: ShutterRecommendation
    nudges: List[ViewfinderNudge] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tilt_degrees": round(self.tilt_degrees, 2),
            "highlight_clip_pct": round(self.highlight_clip_pct, 2),
            "shadow_clip_pct": round(self.shadow_clip_pct, 2),
            "subject_centroid": [round(c, 3) for c in self.subject_centroid],
            "thirds_distance": round(self.thirds_distance, 3),
            "sharpness_score": round(self.sharpness_score, 2),
            "aspect_estimates": {k: round(v, 3) for k, v in self.aspect_estimates.items()},
            "capture_readiness_score": round(self.capture_readiness_score, 3),
            "shutter_recommendation": self.shutter_recommendation.value,
            "nudges": [n.to_dict() for n in self.nudges],
            "latency_ms": round(self.latency_ms, 2),
        }


class LiveShootingCoach:
    """Low-latency capture-time guidance engine operating over raw frame arrays.

    Why lightweight algorithmic probes are used:
    In mobile camera pipelines (30-60 FPS), deep VLM inference is too slow (500ms-2s).
    Fast NumPy/OpenCV spatial frequency, luminance histogram, and gradient orientation
    probes provide sub-30ms capture feedback directly on the mobile edge.
    """

    def __init__(
        self,
        tilt_warning_threshold: float = 1.5,
        highlight_clip_warning: float = 8.0,
        shadow_clip_warning: float = 12.0,
        sharpness_minimum: float = 40.0,
    ) -> None:
        self.tilt_warning_threshold = tilt_warning_threshold
        self.highlight_clip_warning = highlight_clip_warning
        self.shadow_clip_warning = shadow_clip_warning
        self.sharpness_minimum = sharpness_minimum

    def _ensure_grayscale_uint8(self, frame: Union[np.ndarray, Any]) -> np.ndarray:
        """Converts incoming frame (RGB/BGR/Grayscale float or uint8) into standard uint8 2D array."""
        if hasattr(frame, "convert"):
            # PIL Image support
            frame = np.array(frame.convert("RGB"))

        arr = np.asarray(frame)

        # Handle float normalized [0.0, 1.0]
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        # Convert 3D RGB to 2D Luminance using Rec. 601 coefficients
        if arr.ndim == 3:
            if arr.shape[2] >= 3:
                r = arr[:, :, 0].astype(np.float32)
                g = arr[:, :, 1].astype(np.float32)
                b = arr[:, :, 2].astype(np.float32)
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                return gray.astype(np.uint8)
            return arr[:, :, 0]
        return arr

    def detect_horizon_tilt(self, gray: np.ndarray) -> float:
        """Estimates camera tilt relative to the horizon via gradient orientation analysis.

        Why gradient phase distribution detects tilt:
        Natural and architectural scenes exhibit strong horizontal structural axes.
        The deviation of the dominant gradient orientation from 0° reveals camera roll.
        """
        # Downsample large frames for sub-5ms latency
        h, w = gray.shape
        if h > 240:
            step = max(1, int(round(h / 240.0)))
            gray_small = gray[::step, ::step]
        else:
            gray_small = gray

        # Compute Sobel-like gradients using simple finite differences
        gx = np.zeros_like(gray_small, dtype=np.float32)
        gy = np.zeros_like(gray_small, dtype=np.float32)

        gx[:, 1:-1] = gray_small[:, 2:].astype(np.float32) - gray_small[:, :-2].astype(np.float32)
        gy[1:-1, :] = gray_small[2:, :].astype(np.float32) - gray_small[:-2, :].astype(np.float32)

        mag = np.hypot(gx, gy)
        # Focus on prominent structural edges
        threshold = np.percentile(mag, 85)
        strong_mask = mag > max(10.0, threshold)

        if not np.any(strong_mask):
            return 0.0

        # Calculate angles in degrees [-90, +90]
        angles = np.arctan2(gy[strong_mask], gx[strong_mask]) * (180.0 / math.pi)

        # Map angles relative to the horizontal axis
        # Vertical edges have angles near ±90°; horizontal edges have angles near 0° or 180°
        norm_angles = (angles + 90.0) % 180.0 - 90.0
        # Filter to near-horizontal components within ±25°
        near_horizontal = norm_angles[np.abs(norm_angles) <= 25.0]

        if len(near_horizontal) < 20:
            return 0.0

        median_tilt = float(np.median(near_horizontal))
        return round(median_tilt, 2)

    def analyze_exposure_clipping(self, gray: np.ndarray) -> Tuple[float, float]:
        """Calculates percentage of sensor pixels lost to highlight burnout or crushed shadows."""
        total_pixels = gray.size
        if total_pixels == 0:
            return 0.0, 0.0

        highlight_blown = np.count_nonzero(gray >= 252)
        shadow_crushed = np.count_nonzero(gray <= 4)

        highlight_pct = (highlight_blown / total_pixels) * 100.0
        shadow_pct = (shadow_crushed / total_pixels) * 100.0

        return round(highlight_pct, 2), round(shadow_pct, 2)

    def locate_subject_centroid(self, gray: np.ndarray) -> Tuple[Tuple[float, float], float]:
        """Estimates primary subject centroid and distance to nearest Rule-of-Thirds power point.

        Why saliency + center-weighted gradient energy works:
        Viewers and autofocus systems look for high-contrast, structured subjects.
        Computing center of mass of the edge energy locates the primary compositional anchor.
        """
        h, w = gray.shape
        # Downsample for rapid centroid estimation
        stride = max(1, h // 180)
        sub = gray[::stride, ::stride].astype(np.float32)
        sh, sw = sub.shape

        # Gradient energy
        diff_y = np.abs(sub[1:, :] - sub[:-1, :])
        diff_x = np.abs(sub[:, 1:] - sub[:, :-1])

        energy = np.zeros((sh, sw), dtype=np.float32)
        energy[:-1, :] += diff_y
        energy[:, :-1] += diff_x

        total_energy = float(energy.sum())
        if total_energy < 1e-4:
            centroid = (0.5, 0.5)
        else:
            y_indices, x_indices = np.indices((sh, sw))
            cx = float((x_indices * energy).sum() / (total_energy * max(1, sw - 1)))
            cy = float((y_indices * energy).sum() / (total_energy * max(1, sh - 1)))
            centroid = (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)))

        # Compute Euclidean distance to nearest Rule-of-Thirds intersection
        min_dist = min(
            math.hypot(centroid[0] - px, centroid[1] - py)
            for px, py in RULE_OF_THIRDS_INTERSECTIONS
        )

        return centroid, round(min_dist, 3)

    def calculate_sharpness(self, gray: np.ndarray) -> float:
        """Measures high-frequency acutance via discrete Laplacian operator variance.

        Why variance of Laplacian measures focus:
        Sharp, in-focus edges produce high second-order spatial derivatives,
        whereas defocus or motion blur attenuates high frequencies into uniform gradients.
        """
        h, w = gray.shape
        stride = max(1, h // 240)
        sub = gray[::stride, ::stride].astype(np.float32)

        # 3x3 Laplacian kernel convolution via slice indexing
        lap = (
            -4.0 * sub[1:-1, 1:-1]
            + sub[:-2, 1:-1]
            + sub[2:, 1:-1]
            + sub[1:-1, :-2]
            + sub[1:-1, 2:]
        )
        variance = float(np.var(lap))
        return round(variance, 2)

    def evaluate_frame(
        self,
        frame: Union[np.ndarray, Any],
    ) -> ViewfinderFrameTelemetry:
        """Evaluates a raw video frame and synthesizes real-time capture nudges."""
        import time

        start_t = time.perf_counter()
        gray = self._ensure_grayscale_uint8(frame)

        tilt = self.detect_horizon_tilt(gray)
        hi_clip, sh_clip = self.analyze_exposure_clipping(gray)
        centroid, thirds_dist = self.locate_subject_centroid(gray)
        sharpness = self.calculate_sharpness(gray)

        nudges: List[ViewfinderNudge] = []

        # 1. Horizon Tilt Nudge
        if abs(tilt) >= self.tilt_warning_threshold:
            direction = "clockwise" if tilt > 0 else "counter-clockwise"
            deg = abs(tilt)
            severity = NudgeSeverity.WARNING if deg < 4.0 else NudgeSeverity.CRITICAL
            nudges.append(
                ViewfinderNudge(
                    aspect="composition",
                    message=f"Horizon tilted {deg:.1f}°: rotate {direction} to level frame.",
                    severity=severity,
                    numerical_metric=tilt,
                    correction_action=f"Roll camera {direction} by {deg:.1f}°",
                )
            )

        # 2. Highlight Clipping Nudge
        if hi_clip >= self.highlight_clip_warning:
            severity = NudgeSeverity.CRITICAL if hi_clip > 15.0 else NudgeSeverity.WARNING
            nudges.append(
                ViewfinderNudge(
                    aspect="lighting",
                    message=f"Highlight burnout risk: {hi_clip:.1f}% frame clipped to pure white.",
                    severity=severity,
                    numerical_metric=hi_clip,
                    correction_action="Lower exposure compensation (-0.7 EV) or tap to meter on bright area",
                )
            )

        # 3. Shadow Crushing Nudge
        if sh_clip >= self.shadow_clip_warning:
            nudges.append(
                ViewfinderNudge(
                    aspect="lighting",
                    message=f"Deep shadow crushing: {sh_clip:.1f}% frame in pure black.",
                    severity=NudgeSeverity.WARNING,
                    numerical_metric=sh_clip,
                    correction_action="Lift fill lighting or bracket exposure for shadow recovery",
                )
            )

        # 4. Acutance / Motion Blur Warning
        if sharpness < self.sharpness_minimum:
            severity = NudgeSeverity.CRITICAL if sharpness < (self.sharpness_minimum * 0.5) else NudgeSeverity.WARNING
            nudges.append(
                ViewfinderNudge(
                    aspect="technical",
                    message=f"Low acutance ({sharpness:.1f}): possible motion blur or missed autofocus.",
                    severity=severity,
                    numerical_metric=sharpness,
                    correction_action="Hold device steady and tap subject to re-acquire autofocus lock",
                )
            )

        # 5. Rule-of-Thirds Composition Nudge
        # If subject is near center (dist > 0.18 from thirds, but < 0.15 from dead center)
        dist_to_center = math.hypot(centroid[0] - 0.5, centroid[1] - 0.5)
        if thirds_dist > 0.18 and dist_to_center < 0.12:
            nudges.append(
                ViewfinderNudge(
                    aspect="composition",
                    message="Bullseye framing: subject dead-centered in frame.",
                    severity=NudgeSeverity.ADVISORY,
                    numerical_metric=thirds_dist,
                    correction_action="Pan slightly to place subject near upper or lower rule-of-thirds gridlines",
                )
            )

        # Compute aspect estimates from live telemetry
        comp_score = max(0.3, 1.0 - (min(abs(tilt), 10.0) / 15.0) - (min(thirds_dist, 0.3) * 0.8))
        light_score = max(0.2, 1.0 - (hi_clip / 40.0) - (sh_clip / 50.0))
        tech_score = min(1.0, max(0.2, sharpness / 120.0))
        aspects = {
            "composition": round(comp_score, 3),
            "lighting": round(light_score, 3),
            "color": 0.70,  # Standard baseline for real-time capture
            "subject_clarity": round(max(0.3, min(1.0, 0.5 + (0.5 * (1.0 - thirds_dist)))), 3),
            "technical": round(tech_score, 3),
        }

        readiness = float(sum(aspects.values()) / 5.0)

        # Determine shutter recommendation
        has_critical = any(n.severity == NudgeSeverity.CRITICAL for n in nudges)
        if has_critical or readiness < 0.52:
            rec = ShutterRecommendation.HOLD_AND_ADJUST
        elif len(nudges) > 0 or readiness < 0.72:
            rec = ShutterRecommendation.CAUTION
        else:
            rec = ShutterRecommendation.CAPTURE_READY

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return ViewfinderFrameTelemetry(
            tilt_degrees=tilt,
            highlight_clip_pct=hi_clip,
            shadow_clip_pct=sh_clip,
            subject_centroid=centroid,
            thirds_distance=thirds_dist,
            sharpness_score=sharpness,
            aspect_estimates=aspects,
            capture_readiness_score=round(readiness, 3),
            shutter_recommendation=rec,
            nudges=nudges,
            latency_ms=elapsed_ms,
        )

    def get_hud_overlay_data(self, telemetry: ViewfinderFrameTelemetry) -> Dict[str, Any]:
        """Compiles renderable HUD overlay markers for mobile viewfinder UI rendering."""
        status_colors = {
            ShutterRecommendation.CAPTURE_READY: "#10B981",  # Green
            ShutterRecommendation.CAUTION: "#F59E0B",        # Amber
            ShutterRecommendation.HOLD_AND_ADJUST: "#EF4444", # Red
        }

        return {
            "readiness_score": int(telemetry.capture_readiness_score * 100),
            "shutter_recommendation": telemetry.shutter_recommendation.value,
            "badge_color": status_colors.get(telemetry.shutter_recommendation, "#F59E0B"),
            "tilt_label": f"{telemetry.tilt_degrees:+.1f}°",
            "is_level": abs(telemetry.tilt_degrees) < self.tilt_warning_threshold,
            "focal_reticle": {
                "x_pct": round(telemetry.subject_centroid[0] * 100, 1),
                "y_pct": round(telemetry.subject_centroid[1] * 100, 1),
            },
            "top_nudge": telemetry.nudges[0].message if telemetry.nudges else "Framing & lighting balanced. Ready to capture.",
            "nudge_count": len(telemetry.nudges),
        }
