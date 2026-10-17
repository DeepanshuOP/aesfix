"""Playwright Automated End-to-End Visual Verification on 3 Benchmark Samples (Task D8).

Why this script exists:
Automates visual inspection across mobile (375px), tablet (768px), and desktop (1440px)
viewports against the live Next.js studio and FastAPI pipeline backend.
Saves all verified screenshots to %TEMP%\\aesfix-shots\\ per Task D8 specifications.
"""
from __future__ import annotations

import os
from pathlib import Path
from playwright.sync_api import sync_playwright

VIEWPORTS = [
    {"name": "375", "width": 375, "height": 812},
    {"name": "768", "width": 768, "height": 1024},
    {"name": "1440", "width": 1440, "height": 900},
]

SAMPLES = [
    {"id": "test", "label": "Reference", "file": "test.jpg"},
    {"id": "dark", "label": "Underexposed", "file": "dark.jpg"},
    {"id": "blur", "label": "Motion Blur", "file": "blur.jpg"},
]


def run_demo_visual_tests() -> None:
    shots_dir = Path(os.environ.get("TEMP", r"C:\Users\SHREE\AppData\Local\Temp")) / "aesfix-shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Target screenshot directory: {shots_dir}")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)

        for sample in SAMPLES:
            print(f"\n--- Testing Sample: {sample['id']} ({sample['label']}) ---")

            for vp in VIEWPORTS:
                context = browser.new_context(
                    viewport={"width": vp["width"], "height": vp["height"]},
                    device_scale_factor=1,
                )
                page = context.new_page()

                # Navigate to Studio
                page.goto("http://localhost:3000", wait_until="networkidle")
                page.wait_for_timeout(1000)

                # Click sample button
                sample_btn = page.get_by_role("button", name=sample["label"], exact=True)
                sample_btn.click()
                page.wait_for_timeout(1500)

                # Trigger diagnosis & edit verification
                run_btn = page.get_by_role("button", name="Run Grounded Diagnosis & Fix")
                run_btn.click()

                # Wait for pipeline completion (aspect scores section renders)
                print(f"    Waiting for pipeline completion on viewport {vp['name']}...")
                page.locator("text=2. Measured Aspect Scores").wait_for(state="visible", timeout=60000)
                page.wait_for_timeout(2000)

                shot_path = shots_dir / f"sample_{sample['id']}_{vp['name']}.png"
                page.screenshot(path=str(shot_path), full_page=True)
                print(f"    [+] Saved: {shot_path.name} ({shot_path.stat().st_size} bytes)")

                context.close()

        browser.close()

    print("\n[OK] All 9 visual screenshots generated successfully in:")
    print(f"    {shots_dir}")


if __name__ == "__main__":
    run_demo_visual_tests()
