#!/usr/bin/env python3
"""Monitors Kaggle kernel deepanshuop12/aesfix-p8-experiments and downloads outputs upon completion."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

KERNEL_SLUG = "deepanshuop12/aesfix-p8-experiments"
OUT_DIR = REPO_ROOT / "results" / "kaggle_p8_outputs"


def main():
    print(f"=== Monitoring Kaggle Kernel: {KERNEL_SLUG} ===")
    api = KaggleApi()
    api.authenticate()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    prev_status = None
    consecutive_errors = 0

    while True:
        try:
            res = api.kernels_status(KERNEL_SLUG)
            raw_status = getattr(res, "status", None)
            status_name = getattr(raw_status, "name", str(raw_status))
            fail_msg = getattr(res, "failure_message", None)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"Warning: transient network error ({consecutive_errors}): {e}", flush=True)
            time.sleep(20)
            continue

        elapsed_min = (time.time() - start_time) / 60.0
        if status_name != prev_status:
            print(f"[{elapsed_min:.1f}m] Kernel status: {status_name} (fail msg: {fail_msg})", flush=True)
            prev_status = status_name
        else:
            print(f"[{elapsed_min:.1f}m] Kernel status: {status_name}", flush=True)

        if "COMPLETE" in status_name:
            print(f"\nKernel completed successfully after {elapsed_min:.1f} minutes!")
            print(f"Downloading outputs to {OUT_DIR}...")
            api.kernels_output(KERNEL_SLUG, path=str(OUT_DIR))
            print(f"Successfully downloaded kernel outputs to {OUT_DIR}")
            sys.exit(0)
        elif "ERROR" in status_name or "FAILED" in status_name:
            print(f"\nKernel failed with error: {fail_msg}")
            sys.exit(1)

        time.sleep(30)


if __name__ == "__main__":
    main()
