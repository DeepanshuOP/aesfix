"""Edit Proposers for AesFix Closed-Loop Agent (S5).

Implements both heuristic and symmetric grid-based best-of-k proposers.
Per SHIP (arXiv 2608.21748), every decision explicitly logs k (the number of evaluated
candidates) so the downstream certifier can be calibrated on the true multi-candidate policy.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from aesfix.critic import CriticFix
from aesfix.executor import TierAExecutor

logger = logging.getLogger("aesfix.proposer")


@dataclass
class CandidateEvaluation:
    """Audit record for a single candidate considered during proposer selection."""
    op: str
    params: Dict[str, Any]
    score_cand: float
    delta: float
    guards_passed: bool
    guard_reasons: List[str] = field(default_factory=list)


@dataclass
class ProposerDecision:
    """Best-of-k decision emitted by an edit proposer."""
    selected_proposal: Optional[CriticFix]
    k: int
    evaluated_candidates: List[CandidateEvaluation]
    best_gain: float
    stopping_reason: str = ""


class BaseProposer(ABC):
    """Abstract base contract for edit proposers."""

    @abstractmethod
    def propose(
        self,
        current_image: Image.Image,
        original_image: Image.Image,
        current_score: float,
        noise_floor: float,
        primary_judge: Any,
        executor: TierAExecutor,
        guard: Optional[Any] = None,
        overprocessing_cfg: Optional[Dict[str, float]] = None,
        style_intent: Optional[Dict[str, Any]] = None,
        secondary_judges: Optional[List[Any]] = None,
    ) -> ProposerDecision:
        """Evaluates candidates and selects the best admissible edit."""
        pass


class GridProposer(BaseProposer):
    """Symmetric candidate set proposer with Best-of-k selection and guard filtering.

    Candidates (k=15):
    - Exposure: ±0.3, ±0.7, ±1.2, ±1.6 EV (8 candidates)
    - Contrast: 0.8, 1.25, 1.6 factor (3 candidates)
    - Signed WB: ±10, ±25 temperature shift (4 candidates)
    """

    def __init__(self) -> None:
        self.candidate_specs = [
            ("adjust_exposure", {"ev_delta": -1.6}, "lighting"),
            ("adjust_exposure", {"ev_delta": -1.2}, "lighting"),
            ("adjust_exposure", {"ev_delta": -0.7}, "lighting"),
            ("adjust_exposure", {"ev_delta": -0.3}, "lighting"),
            ("adjust_exposure", {"ev_delta": 0.3}, "lighting"),
            ("adjust_exposure", {"ev_delta": 0.7}, "lighting"),
            ("adjust_exposure", {"ev_delta": 1.2}, "lighting"),
            ("adjust_exposure", {"ev_delta": 1.6}, "lighting"),
            ("adjust_contrast", {"factor": 0.8}, "lighting"),
            ("adjust_contrast", {"factor": 1.25}, "lighting"),
            ("adjust_contrast", {"factor": 1.6}, "lighting"),
            ("adjust_white_balance", {"temperature": -25.0, "tint": 0.0}, "color"),
            ("adjust_white_balance", {"temperature": -10.0, "tint": 0.0}, "color"),
            ("adjust_white_balance", {"temperature": 10.0, "tint": 0.0}, "color"),
            ("adjust_white_balance", {"temperature": 25.0, "tint": 0.0}, "color"),
        ]

    def _check_clipping_guard(
        self,
        cand_img: Image.Image,
        orig_img: Image.Image,
        cfg: Dict[str, float],
        is_low_key: bool = False,
    ) -> Tuple[bool, List[str]]:
        reasons = []
        try:
            arr_cand = np.asarray(cand_img.convert("RGB"), dtype=np.float32) / 255.0
            arr_orig = np.asarray(orig_img.convert("RGB"), dtype=np.float32) / 255.0

            luma_cand = 0.299 * arr_cand[:, :, 0] + 0.587 * arr_cand[:, :, 1] + 0.114 * arr_cand[:, :, 2]
            luma_orig = 0.299 * arr_orig[:, :, 0] + 0.587 * arr_orig[:, :, 1] + 0.114 * arr_orig[:, :, 2]

            hl_cand = float(np.mean(np.max(arr_cand, axis=-1) >= 0.99))
            hl_orig = float(np.mean(np.max(arr_orig, axis=-1) >= 0.99))
            hl_growth = hl_cand - hl_orig

            max_hl = float(cfg.get("max_highlight_clipping_growth", 0.02))
            if hl_growth > max_hl:
                reasons.append(f"Highlight clipping grew by {hl_growth:+.4f} > {max_hl:.4f}")

            if is_low_key:
                p5_cand = float(np.percentile(luma_cand, 5))
                p5_orig = float(np.percentile(luma_orig, 5))
                black_lift = p5_cand - p5_orig
                max_lift = float(cfg.get("max_black_level_lift_low_key", 0.03))
                if black_lift > max_lift:
                    reasons.append(f"Black level lifted by {black_lift:+.4f} > {max_lift:.4f} on low-key image")

            max_c = np.max(arr_cand, axis=-1)
            min_c = np.min(arr_cand, axis=-1)
            sat = np.where(max_c > 1e-4, (max_c - min_c) / np.maximum(max_c, 1e-4), 0.0)
            sat_cand = float(np.mean(sat >= 0.99))
            max_sat = float(cfg.get("max_saturation_clipping_growth", 0.05))
            if sat_cand > max_sat:
                reasons.append(f"Saturation clipping reached {sat_cand:.4f} > {max_sat:.4f}")

        except Exception as e:
            logger.debug("Clipping check error: %s", e)

        return len(reasons) == 0, reasons

    def propose(
        self,
        current_image: Image.Image,
        original_image: Image.Image,
        current_score: float,
        noise_floor: float,
        primary_judge: Any,
        executor: TierAExecutor,
        guard: Optional[Any] = None,
        overprocessing_cfg: Optional[Dict[str, float]] = None,
        style_intent: Optional[Dict[str, Any]] = None,
        secondary_judges: Optional[List[Any]] = None,
    ) -> ProposerDecision:
        cfg = overprocessing_cfg or {
            "max_highlight_clipping_growth": 0.02,
            "max_black_level_lift_low_key": 0.03,
            "max_saturation_clipping_growth": 0.05,
        }
        is_low_key = bool(
            style_intent
            and (
                style_intent.get("style_type") in ("low-key", "night", "silhouette")
                or style_intent.get("is_low_key", False)
                or style_intent.get("style") in ("low-key", "low_key", "night")
            )
        )

        evaluated: List[CandidateEvaluation] = []
        admissible_candidates: List[Tuple[CriticFix, float, float]] = []

        # 1. Generate all candidate images in memory
        cand_items: List[Tuple[CriticFix, Image.Image]] = []
        for op, params, aspect in self.candidate_specs:
            fix = CriticFix(
                op=op,
                params=params,
                expected_aspect=aspect,
                direction=1,
                nl_instruction=f"Grid candidate {op} {params}",
            )
            try:
                res = executor.execute_fixes(image_input=current_image, fixes=[fix])
                cand_items.append((fix, res.image))
            except Exception as e:
                evaluated.append(CandidateEvaluation(
                    op=op, params=params, score_cand=0.0, delta=0.0, guards_passed=False, guard_reasons=[str(e)]
                ))

        if not cand_items:
            return ProposerDecision(
                selected_proposal=None,
                k=len(self.candidate_specs),
                evaluated_candidates=evaluated,
                best_gain=0.0,
                stopping_reason="All candidate executions raised exceptions.",
            )

        # 2. Score candidates (batched if supported)
        cand_pils = [item[1] for item in cand_items]
        cand_scores: List[float] = []
        if hasattr(primary_judge, "evaluate_batch_pil"):
            cand_scores = primary_judge.evaluate_batch_pil(cand_pils)
        else:
            for c_img in cand_pils:
                if hasattr(primary_judge, "evaluate_pil"):
                    cand_scores.append(float(primary_judge.evaluate_pil(c_img)))
                else:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                        t_path = tf.name
                    c_img.save(t_path, quality=95)
                    try:
                        s_c = float(primary_judge.evaluate(t_path).raw_score)
                    except Exception:
                        s_c = float(primary_judge.evaluate(t_path))
                    cand_scores.append(s_c)
                    try:
                        os.unlink(t_path)
                    except Exception:
                        pass

        # 3. Check guards for each candidate
        import tempfile
        tmp_dir = None
        orig_p = None
        if guard is not None or secondary_judges:
            tmp_dir = tempfile.mkdtemp()
            orig_p = f"{tmp_dir}/orig.jpg"
            original_image.save(orig_p, quality=95)

        try:
            for (fix, c_img), s_cand in zip(cand_items, cand_scores):
                op = fix.op
                params = fix.params
                delta = s_cand - current_score
                guards_passed = True
                reasons = []

                # Clipping Guard
                clip_ok, clip_reasons = self._check_clipping_guard(c_img, original_image, cfg, is_low_key=is_low_key)
                if not clip_ok:
                    guards_passed = False
                    reasons.extend(clip_reasons)

                # Style Intent Guard: Disallow positive exposure lifts on low-key images
                if is_low_key and op == "adjust_exposure" and params.get("ev_delta", 0.0) > 0:
                    guards_passed = False
                    reasons.append("Positive exposure lift disallowed on low-key style")

                # Primary Judge Non-inferiority / improvement requirement
                if delta <= noise_floor:
                    guards_passed = False
                    reasons.append(f"Delta {delta:+.4f} <= noise floor {noise_floor:.4f}")

                # Identity Guard & Secondary Judges
                if tmp_dir and (guard is not None or secondary_judges):
                    cand_p = f"{tmp_dir}/cand.jpg"
                    c_img.save(cand_p, quality=95)

                    if guard is not None:
                        verdict = guard.evaluate(original=orig_p, edited=cand_p)
                        if not verdict.is_valid:
                            guards_passed = False
                            reasons.append(f"Identity drift {verdict.composite_drift:.3f} > {verdict.threshold:.3f}")

                    if secondary_judges:
                        for sec_j in secondary_judges:
                            try:
                                s_sec_curr = float(sec_j.evaluate(orig_p).raw_score)
                                s_sec_cand = float(sec_j.evaluate(cand_p).raw_score)
                                sec_eps = getattr(sec_j, "noise_floor", 0.01)
                                if (s_sec_cand - s_sec_curr) < -sec_eps:
                                    guards_passed = False
                                    reasons.append(f"Secondary judge {sec_j.name} degraded beyond -eps")
                                    break
                            except Exception:
                                pass

                evaluated.append(CandidateEvaluation(
                    op=op,
                    params=params,
                    score_cand=round(s_cand, 4),
                    delta=round(delta, 4),
                    guards_passed=guards_passed,
                    guard_reasons=reasons,
                ))

                if guards_passed:
                    admissible_candidates.append((fix, s_cand, delta))
        finally:
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        k_count = len(self.candidate_specs)
        if not admissible_candidates:
            return ProposerDecision(
                selected_proposal=None,
                k=k_count,
                evaluated_candidates=evaluated,
                best_gain=0.0,
                stopping_reason="No candidate passed all guards and improved beyond epsilon.",
            )

        # Best-of-k selection: maximize score gain delta
        admissible_candidates.sort(key=lambda x: x[2], reverse=True)
        best_fix, best_s, best_d = admissible_candidates[0]

        return ProposerDecision(
            selected_proposal=best_fix,
            k=k_count,
            evaluated_candidates=evaluated,
            best_gain=round(best_d, 4),
            stopping_reason=f"Selected best candidate {best_fix.op} ({best_fix.params}) with Delta={best_d:+.4f}",
        )


class HeuristicProposer(BaseProposer):
    """Heuristic aspect critic proposer (k=number of heuristic candidate fixes, typically 1).

    Wraps HeuristicAspectCritic to emit candidate fixes based on analytical image histograms,
    then verifies them against clipping, identity, and non-inferiority guards.
    """

    def __init__(self) -> None:
        from aesfix.critic.critic import HeuristicAspectCritic
        self.critic = HeuristicAspectCritic()

    def propose(
        self,
        current_image: Image.Image,
        original_image: Image.Image,
        current_score: float,
        noise_floor: float,
        primary_judge: Any,
        executor: TierAExecutor,
        guard: Optional[Any] = None,
        overprocessing_cfg: Optional[Dict[str, float]] = None,
        style_intent: Optional[Dict[str, Any]] = None,
        secondary_judges: Optional[List[Any]] = None,
    ) -> ProposerDecision:
        cfg = overprocessing_cfg or {
            "max_highlight_clipping_growth": 0.02,
            "max_black_level_lift_low_key": 0.03,
            "max_saturation_clipping_growth": 0.05,
        }
        is_low_key = bool(style_intent and style_intent.get("style_type") in ("low-key", "night", "silhouette"))

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            curr_p = f"{tmp_dir}/curr.jpg"
            orig_p = f"{tmp_dir}/orig.jpg"
            current_image.save(curr_p, quality=95)
            original_image.save(orig_p, quality=95)

            critique = self.critic.critique(curr_p)
            if not critique.fixes:
                return ProposerDecision(
                    selected_proposal=None,
                    k=1,
                    evaluated_candidates=[],
                    best_gain=0.0,
                    stopping_reason="Critic proposed no further fixes (histogram within normal bounds)",
                )

            evaluated: List[CandidateEvaluation] = []
            admissible_candidates: List[Tuple[CriticFix, float, float]] = []
            k_count = len(critique.fixes)

            for fix in critique.fixes:
                try:
                    res = executor.execute_fixes(image_input=current_image, fixes=[fix])
                except Exception as e:
                    evaluated.append(CandidateEvaluation(
                        op=fix.op,
                        params=fix.params,
                        score_cand=0.0,
                        delta=0.0,
                        guards_passed=False,
                        guard_reasons=[str(e)],
                    ))
                    continue

                cand_p = f"{tmp_dir}/cand_{fix.op}.jpg"
                res.image.save(cand_p, quality=95)

                guards_passed = True
                reasons = []

                # Identity guard
                if guard is not None:
                    verdict = guard.evaluate(original=orig_p, edited=cand_p)
                    if not verdict.is_valid:
                        guards_passed = False
                        reasons.append(f"Identity drift {verdict.composite_drift:.3f} > {verdict.threshold:.3f}")

                # Clipping guard
                grid_helper = GridProposer()
                clip_ok, clip_reasons = grid_helper._check_clipping_guard(res.image, original_image, cfg, is_low_key=is_low_key)
                if not clip_ok:
                    guards_passed = False
                    reasons.extend(clip_reasons)

                # Primary judge score
                try:
                    s_cand = float(primary_judge.evaluate(cand_p).raw_score)
                except Exception:
                    s_cand = float(primary_judge.evaluate(cand_p))
                delta = s_cand - current_score

                if delta <= noise_floor:
                    guards_passed = False
                    reasons.append(f"Delta {delta:+.4f} <= noise floor {noise_floor:.4f}")

                # Secondary judges
                if secondary_judges:
                    for sec_j in secondary_judges:
                        try:
                            s_sec_curr = float(sec_j.evaluate(orig_p).raw_score)
                            s_sec_cand = float(sec_j.evaluate(cand_p).raw_score)
                            sec_eps = getattr(sec_j, "noise_floor", 0.01)
                            if (s_sec_cand - s_sec_curr) < -sec_eps:
                                guards_passed = False
                                reasons.append(f"Secondary judge {sec_j.name} degraded beyond -eps")
                                break
                        except Exception:
                            pass

                evaluated.append(CandidateEvaluation(
                    op=fix.op,
                    params=fix.params,
                    score_cand=round(s_cand, 4),
                    delta=round(delta, 4),
                    guards_passed=guards_passed,
                    guard_reasons=reasons,
                ))

                if guards_passed:
                    admissible_candidates.append((fix, s_cand, delta))

            if not admissible_candidates:
                return ProposerDecision(
                    selected_proposal=None,
                    k=k_count,
                    evaluated_candidates=evaluated,
                    best_gain=0.0,
                    stopping_reason="Proposed heuristic fix failed guards or noise-floor check.",
                )

            admissible_candidates.sort(key=lambda x: x[2], reverse=True)
            best_fix, best_s, best_d = admissible_candidates[0]

            return ProposerDecision(
                selected_proposal=best_fix,
                k=k_count,
                evaluated_candidates=evaluated,
                best_gain=round(best_d, 4),
                stopping_reason=f"Selected heuristic candidate {best_fix.op} ({best_fix.params}) with Delta={best_d:+.4f}",
            )

