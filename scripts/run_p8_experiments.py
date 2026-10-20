#!/usr/bin/env python3
"""Run P8 Core Experiments (T1–T4): Proposer Benchmark, Overoptimization Curve, Policy Certifier.

Author: Deepanshu <Deepanshuop@gmail.com>

T1: Proposer comparison on 500 synthetic cases (Wilson CIs, mean ΔE, degradation breakdown, dark.jpg visuals).
    Configurations:
    - heuristic + v1
    - grid + v1
    - grid + v2
    - grid + v2 + TopIQ-NR guard
    - grid + v2 + guard + clipping/intent guards
T2: Overoptimization curve (k in {1, 2, 3, 5, 8, 12, 15}, 5 seeds per k, single step best-of-k).
    Standardized judge gain vs ground truth change vs harm rate for v2, v1, rsinema.
    Plot: docs/figures/overoptimization_k.png
T3: Policy-level certifier on synthetic data (100 scenes, 50/50 split, 20 seeds).
    Compares candidate-level LTT (k=1), policy-level LTT (best-of-15), naive s > epsilon.
    Alpha in {0.05, 0.10, 0.20}, delta = 0.1.
T4: Handled by scripts/run_t4_caveat_check.py (completed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.agent.proposer import GridProposer, HeuristicProposer
from aesfix.critic import CriticFix
from aesfix.executor import TierAExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("p8_experiments")

CASES_CSV = REPO_ROOT / "results" / "synth_v2_cases.csv"
VISUALS_DIR = REPO_ROOT / "results" / "visuals"
FIGURES_DIR = REPO_ROOT / "docs" / "figures"
CONFIG_EPSILON = REPO_ROOT / "config" / "judge_epsilon.json"


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculates Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half_width = z * math.sqrt((p * (1.0 - p) + (z**2) / (4 * n)) / n) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return p, lower, upper


def compute_lab_delta_e(img1: Image.Image, img2: Image.Image) -> float:
    """Computes CIEDE2000 color difference between two images."""
    if img1.size != img2.size:
        img1 = img1.resize(img2.size, Image.Resampling.BILINEAR)
    arr1 = np.asarray(img1.convert("RGB"), dtype=np.float32) / 255.0
    arr2 = np.asarray(img2.convert("RGB"), dtype=np.float32) / 255.0
    lab1 = rgb2lab(arr1)
    lab2 = rgb2lab(arr2)
    de = deltaE_ciede2000(lab1, lab2)
    return float(np.mean(de))


def compute_clipping_metrics(img: Image.Image) -> Dict[str, float]:
    """Computes highlight clipping, saturation clipping, and low-key black level."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hl = float(np.mean(np.max(arr, axis=-1) >= 0.99))
    max_c = np.max(arr, axis=-1)
    min_c = np.min(arr, axis=-1)
    sat = np.where(max_c > 1e-4, (max_c - min_c) / np.maximum(max_c, 1e-4), 0.0)
    sat_clip = float(np.mean(sat >= 0.99))
    p5_luma = float(np.percentile(luma, 5))
    return {"hl_clipping": hl, "sat_clipping": sat_clip, "p5_luma": p5_luma}


def check_candidate_guards(
    cand_metrics: Dict[str, float],
    orig_metrics: Dict[str, float],
    op: str,
    params: Dict[str, Any],
    is_low_key: bool = False,
    delta_topiq: Optional[float] = None,
    eps_topiq: float = 0.005,
    use_topiq_guard: bool = False,
    use_clipping_guards: bool = False,
) -> Tuple[bool, List[str]]:
    """Evaluates admissibility of candidate edit."""
    reasons = []
    if use_topiq_guard and delta_topiq is not None:
        if delta_topiq < -eps_topiq:
            reasons.append(f"TopIQ degraded: {delta_topiq:+.4f} < -{eps_topiq:.4f}")

    if use_clipping_guards:
        hl_growth = cand_metrics["hl_clipping"] - orig_metrics["hl_clipping"]
        if hl_growth > 0.02:
            reasons.append(f"Highlight clipping grew by {hl_growth:+.4f} > 0.02")
        if cand_metrics["sat_clipping"] > 0.05:
            reasons.append(f"Saturation clipping {cand_metrics['sat_clipping']:.4f} > 0.05")
        if is_low_key:
            b_lift = cand_metrics["p5_luma"] - orig_metrics["p5_luma"]
            if b_lift > 0.03:
                reasons.append(f"Black level lifted by {b_lift:+.4f} > 0.03 on low-key")
            if op == "adjust_exposure" and params.get("ev_delta", 0.0) > 0:
                reasons.append("Positive EV disallowed on low-key")

    return len(reasons) == 0, reasons


print("p8_experiments module loaded.")
