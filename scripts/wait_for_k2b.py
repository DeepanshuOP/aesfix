#!/usr/bin/env python3
"""Poll Kaggle kernel status until completion with robust retry."""
import time
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    api = KaggleApi()
    api.authenticate()
    kernel_slug = "deepanshuop12/aesfix-k2-dear-judges"
    print(f"Monitoring {kernel_slug}...")
    prev_status = None
    start_time = time.time()
    consecutive_errors = 0

    while True:
        try:
            res = api.kernels_status(kernel_slug)
            raw_status = getattr(res, "status", None)
            status_name = getattr(raw_status, "name", str(raw_status))
            fail_msg = getattr(res, "failure_message", None)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"Warning: transient network error querying status ({consecutive_errors}): {e}", flush=True)
            time.sleep(20)
            continue

        elapsed_min = (time.time() - start_time) / 60.0
        if status_name != prev_status:
            print(f"[{elapsed_min:.1f}m] Status changed to: {status_name} (failure: {fail_msg})", flush=True)
            prev_status = status_name
        else:
            print(f"[{elapsed_min:.1f}m] Status: {status_name}", flush=True)

        if "COMPLETE" in status_name:
            print(f"Kernel completed successfully after {elapsed_min:.1f} minutes!")
            sys.exit(0)
        elif "ERROR" in status_name or "FAILED" in status_name:
            print(f"Kernel failed with error: {fail_msg}")
            sys.exit(1)

        time.sleep(30)

if __name__ == "__main__":
    main()
