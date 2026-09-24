"""Synthetic Counterfactual Engine for Edit-Verified Ground Truth (AF-2.3 & Paper §13).

Why the Synthetic Counterfactual Engine is a scientific unfair advantage:
1. Zero Annotation Cost: By applying parameterized, invertible degradations to high-scoring
   reference photographs, we automatically generate provably correct (image, advice, target)
   triplets without paying human annotators.
2. Metric Sanity & AG@k Calibration: Provides an empirical upper bound / oracle ceiling.
   If the aesthetic gain metric cannot verify that the mathematical inverse of a known
   degradation improves the photo, the evaluation harness is flawed.
3. SFT Warm-Start: Teaches critic models exact numerical calibration (e.g. how much EV
   boost is needed to correct a -0.6 EV underexposure) prior to reinforcement learning.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from aesfix.critic.critic import CriticFix
    from aesfix.executor.tier_a import TierAExecutor

# Canonical invertible degradation operation families
SUPPORTED_DEGRADATIONS: Tuple[str, ...] = (
    "adjust_exposure",
    "adjust_contrast",
    "adjust_saturation",
    "adjust_white_balance",
    "correct_horizon",
)


@dataclass
class SyntheticTriplet:
    """A counterfactual triplet consisting of a degraded input, ground-truth fix, and target.

    Why this dataclass exists:
    Decouples data generation from downstream training pipelines, providing ready-to-train
    JSONL representations for instruction tuning and AG@k benchmarking.
    """
    degraded_image: Image.Image
    oracle_fix: CriticFix
    target_image: Image.Image
    degradation_op: str
    degradation_params: Dict[str, Any]
    recovery_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the triplet's semantic attributes to a dictionary."""
        oracle_data = (
            self.oracle_fix.model_dump()
            if hasattr(self.oracle_fix, "model_dump")
            else dict(self.oracle_fix)
        )
        return {
            "degradation_op": self.degradation_op,
            "degradation_params": self.degradation_params,
            "oracle_fix": oracle_data,
            "recovery_ratio": self.recovery_ratio,
            "metadata": self.metadata,
        }


class SyntheticCounterfactualEngine:
    """Config-driven engine synthesizing counterfactual edit-verification triplets."""

    def __init__(
        self,
        executor: Optional[TierAExecutor] = None,
        seed: Optional[int] = 42,
    ) -> None:
        """Initializes the engine with a Tier-A executor and deterministic seed."""
        if executor is None:
            from aesfix.executor.tier_a import TierAExecutor
            self.executor = TierAExecutor()
        else:
            self.executor = executor

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _sample_degradation(
        self,
        op: Optional[str] = None,
        severity: float = 1.0,
    ) -> Tuple[Any, Any]:
        """Generates a forward degradation fix and its exact analytical oracle inverse.

        Why analytical inversion:
        Guarantees that applying the oracle fix directly restores the degraded signal
        towards the ground-truth target without empirical search.
        """
        from aesfix.critic.critic import CriticFix

        chosen_op = op or random.choice(SUPPORTED_DEGRADATIONS)
        scale = max(0.2, min(float(severity), 2.0))

        if chosen_op == "adjust_exposure":
            # Underexposure/overexposure within non-clipping bounds
            direction = random.choice([-1.0, 1.0])
            ev_delta = round(direction * (0.25 + 0.35 * random.random()) * scale, 2)
            deg_fix = CriticFix(
                op="adjust_exposure",
                params={"ev_delta": ev_delta},
                expected_aspect="lighting",
                nl_instruction=f"Introduce exposure shift of {ev_delta:+.2f} EV.",
            )
            oracle_fix = CriticFix(
                op="adjust_exposure",
                params={"ev_delta": -ev_delta},
                expected_aspect="lighting",
                nl_instruction=f"Correct exposure by {-ev_delta:+.2f} EV stops.",
            )

        elif chosen_op == "adjust_contrast":
            # Contrast compression (factor < 1.0 preserves all dynamic range without clipping)
            factor = round(0.50 + 0.25 * (1.0 - scale * 0.4), 2)
            deg_fix = CriticFix(
                op="adjust_contrast",
                params={"factor": factor},
                expected_aspect="lighting",
                nl_instruction=f"Distort contrast by scale factor {factor:.2f}.",
            )
            oracle_fix = CriticFix(
                op="adjust_contrast",
                params={"factor": round(1.0 / factor, 3)},
                expected_aspect="lighting",
                nl_instruction=f"Restore balanced contrast curve with factor {1.0 / factor:.2f}.",
            )

        elif chosen_op == "adjust_saturation":
            # Desaturation (factor < 1.0 compresses chroma without clipping at 255)
            factor = round(0.35 + 0.30 * (1.0 - scale * 0.4), 2)
            deg_fix = CriticFix(
                op="adjust_saturation",
                params={"factor": factor},
                expected_aspect="color",
                nl_instruction=f"Shift color saturation to factor {factor:.2f}.",
            )
            oracle_fix = CriticFix(
                op="adjust_saturation",
                params={"factor": round(1.0 / factor, 3)},
                expected_aspect="color",
                nl_instruction=f"Correct color vividness back to normal with factor {1.0 / factor:.2f}.",
            )

        elif chosen_op == "adjust_white_balance":
            # Temperature and tint shift -> opposite direction oracle
            temp_shift = round(random.choice([-1.0, 1.0]) * (15.0 + 15.0 * random.random()) * scale, 1)
            tint_shift = round(random.choice([-1.0, 1.0]) * (8.0 + 8.0 * random.random()) * scale, 1)
            deg_fix = CriticFix(
                op="adjust_white_balance",
                params={"temperature": temp_shift, "tint": tint_shift},
                expected_aspect="color",
                nl_instruction=f"Impose color cast (temp={temp_shift:+.1f}, tint={tint_shift:+.1f}).",
            )
            oracle_fix = CriticFix(
                op="adjust_white_balance",
                params={"temperature": -temp_shift, "tint": -tint_shift},
                expected_aspect="color",
                nl_instruction=f"Neutralize color cast by applying temp={-temp_shift:+.1f}, tint={-tint_shift:+.1f}.",
            )

        elif chosen_op == "adjust_highlights":
            hl_shift = round(random.choice([-1.0, 1.0]) * (25.0 + 15.0 * random.random()) * scale, 1)
            deg_fix = CriticFix(
                op="adjust_highlights",
                params={"highlights": hl_shift},
                expected_aspect="lighting",
                nl_instruction=f"Alter highlights by {hl_shift:+.1f} points.",
            )
            oracle_fix = CriticFix(
                op="adjust_highlights",
                params={"highlights": -hl_shift},
                expected_aspect="lighting",
                nl_instruction=f"Restore highlights by {-hl_shift:+.1f} points.",
            )

        elif chosen_op == "adjust_shadows":
            sh_shift = round(random.choice([-1.0, 1.0]) * (25.0 + 15.0 * random.random()) * scale, 1)
            deg_fix = CriticFix(
                op="adjust_shadows",
                params={"shadows": sh_shift},
                expected_aspect="lighting",
                nl_instruction=f"Alter shadows by {sh_shift:+.1f} points.",
            )
            oracle_fix = CriticFix(
                op="adjust_shadows",
                params={"shadows": -sh_shift},
                expected_aspect="lighting",
                nl_instruction=f"Restore shadows by {-sh_shift:+.1f} points.",
            )

        elif chosen_op == "correct_horizon":
            # Horizon tilt: e.g. +3.0 degrees tilt -> -3.0 degrees straightening
            tilt_angle = round(random.choice([-1.0, 1.0]) * (1.5 + 2.0 * random.random()) * scale, 1)
            deg_fix = CriticFix(
                op="correct_horizon",
                params={"angle": tilt_angle},
                expected_aspect="composition",
                nl_instruction=f"Tilt horizon by {tilt_angle:+.1f} degrees.",
            )
            oracle_fix = CriticFix(
                op="correct_horizon",
                params={"angle": -tilt_angle},
                expected_aspect="composition",
                nl_instruction=f"Level tilted horizon by rotating {-tilt_angle:+.1f} degrees.",
            )

        else:
            raise ValueError(f"Unsupported degradation operation: {chosen_op}")

        return deg_fix, oracle_fix

    def generate_triplet(
        self,
        target_image: Image.Image,
        op: Optional[str] = None,
        severity: float = 1.0,
    ) -> SyntheticTriplet:
        """Synthesizes a counterfactual triplet from a clean target image.

        Why: Creates the complete (degraded, oracle_fix, target) datum.
        """
        deg_fix, oracle_fix = self._sample_degradation(op=op, severity=severity)

        # Apply degradation to target
        deg_result = self.executor.execute_fixes(target_image, [deg_fix])
        degraded_img = deg_result.image

        triplet = SyntheticTriplet(
            degraded_image=degraded_img,
            oracle_fix=oracle_fix,
            target_image=target_image,
            degradation_op=deg_fix.op,
            degradation_params=deg_fix.params,
        )

        # Compute recovery ratio to verify metric sanity
        recovery = self.evaluate_oracle_recovery(triplet)
        triplet.recovery_ratio = recovery

        return triplet

    def evaluate_oracle_recovery(self, triplet: SyntheticTriplet) -> float:
        """Evaluates signal recovery percentage achieved by executing the oracle fix.

        Why MSE recovery ratio:
        Measures the fraction of degradation distortion eliminated by the oracle advice:
        Recovery = 1.0 - (MSE(restored, target) / MSE(degraded, target))
        Roadmap DoD requirement: Must achieve >= 0.90 (90%) recovery.
        """
        # Execute oracle fix on degraded image
        restored_res = self.executor.execute_fixes(triplet.degraded_image, [triplet.oracle_fix])
        restored_img = restored_res.image

        # Align spatial dimensions via center crop if rotation/crop caused boundary trim
        t_w, t_h = triplet.target_image.size
        d_w, d_h = triplet.degraded_image.size
        r_w, r_h = restored_img.size
        min_w, min_h = min(t_w, d_w, r_w), min(t_h, d_h, r_h)

        def get_center_crop(im: Image.Image) -> np.ndarray:
            w, h = im.size
            x0 = (w - min_w) // 2
            y0 = (h - min_h) // 2
            cropped = im.crop((x0, y0, x0 + min_w, y0 + min_h))
            return np.array(cropped, dtype=np.float32)

        arr_target = get_center_crop(triplet.target_image)
        arr_degraded = get_center_crop(triplet.degraded_image)
        arr_restored = get_center_crop(restored_img)

        mse_degraded = float(np.mean((arr_degraded - arr_target) ** 2))
        mse_restored = float(np.mean((arr_restored - arr_target) ** 2))

        if mse_degraded < 1e-4:
            return 1.0  # Trivial case: no meaningful degradation

        recovery = float(np.clip(1.0 - (mse_restored / mse_degraded), 0.0, 1.0))
        return round(recovery, 4)

    def generate_batch(
        self,
        target_images: List[Image.Image],
        count: int,
    ) -> List[SyntheticTriplet]:
        """Synthesizes a batch of counterfactual triplets sampled across target images.

        Why: Scales up zero-cost dataset creation for SFT warm-start and ablation runs.
        """
        triplets: List[SyntheticTriplet] = []
        if not target_images:
            return triplets

        for i in range(count):
            img = random.choice(target_images)
            triplet = self.generate_triplet(img)
            triplets.append(triplet)

        return triplets

    def export_dataset(
        self,
        triplets: List[SyntheticTriplet],
        output_dir: Union[str, Path],
        prefix: str = "counterfactual",
    ) -> Path:
        """Exports synthesized triplets to disk with JSONL metadata and JPEGs.

        Why: Produces the exact dataset artifact consumed by LoRA SFT training scripts.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        img_dir = out_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = out_path / f"{prefix}_triplets.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for idx, trip in enumerate(triplets):
                deg_filename = f"{prefix}_{idx:05d}_degraded.jpg"
                tgt_filename = f"{prefix}_{idx:05d}_target.jpg"

                trip.degraded_image.save(img_dir / deg_filename, quality=95)
                trip.target_image.save(img_dir / tgt_filename, quality=95)

                entry = trip.to_dict()
                entry["degraded_image_path"] = str(img_dir / deg_filename)
                entry["target_image_path"] = str(img_dir / tgt_filename)
                f.write(json.dumps(entry) + "\n")

        return jsonl_path
