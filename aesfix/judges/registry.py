"""Judge Environment Registry & Dependency Diagnostics (Roadmap v6 Part VI & R1.2).

Registers external IQA and Aesthetic judges from Roadmap Part VI with exact
install commands and version pins. Provides a dry-run diagnostic that reports
which judges import successfully on the current environment without mock fallbacks.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any, Dict, List, Optional


JUDGE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pyiqa": {
        "name": "pyiqa (IQA-PyTorch)",
        "module": "pyiqa",
        "install_command": "pip install pyiqa>=0.1.13",
        "version_pin": ">=0.1.13",
        "description": "Comprehensive IQA/IAA library (MUSIQ, TOPIQ, LIQE, CLIP-IQA, NIMA).",
        "role": "Expands judge pool for Goodhart Map and reference deltas.",
    },
    "imscore": {
        "name": "imscore",
        "module": "imscore",
        "install_command": "pip install imscore>=0.2.0",
        "version_pin": ">=0.2.0",
        "description": "Aesthetic quality scoring engine.",
        "role": "Frozen reference judge.",
    },
    "hpsv2_hpsv3": {
        "name": "HPSv2 / HPSv3",
        "module": "hpsv2",
        "install_command": "pip install hpsv2>=1.1.0",
        "version_pin": ">=1.1.0",
        "description": "Human Preference Score model trained on pairwise comparisons.",
        "role": "Preference-aligned judge in frozen ensemble.",
    },
    "qalign": {
        "name": "Q-Align / OneAlign",
        "module": "transformers",
        "install_command": "pip install transformers>=4.40.0 accelerate",
        "version_pin": ">=4.40.0",
        "description": "MLLM-based aesthetic and technical quality assessor.",
        "role": "Instruction-tuned aesthetic judge.",
    },
    "aesthetic_predictor_v2_5": {
        "name": "aesthetic-predictor-v2-5 (APv2.5)",
        "module": "aesthetic_predictor_v2_5",
        "install_command": "pip install git+https://github.com/discus0434/aesthetic-predictor-v2-5.git",
        "version_pin": "v2.5",
        "description": "Modern SigLIP/CLIP-based aesthetic predictor.",
        "role": "Modern visual aesthetic anchor.",
    },
    "mear_dear": {
        "name": "MEAR (DEAR Pairwise Model)",
        "module": "dear",
        "install_command": "pip install git+https://github.com/DEAR-Bench/DEAR.git",
        "version_pin": "cvpr2025",
        "description": "Pairwise aesthetic preference judge on rendering differences.",
        "role": "Rendering-difference baseline judge.",
    },
}


def check_judge_import(judge_key: str) -> Dict[str, Any]:
    """Tests if a judge's python module can be imported, strictly without mock fallbacks.

    Returns:
        Dict with import status, version (if available), or exception details.
    """
    info = JUDGE_REGISTRY[judge_key]
    module_name = info["module"]

    status: Dict[str, Any] = {
        "key": judge_key,
        "name": info["name"],
        "module": module_name,
        "installed": False,
        "version": None,
        "error": None,
        "install_command": info["install_command"],
        "role": info["role"],
    }

    try:
        mod = importlib.import_module(module_name)
        status["installed"] = True
        status["version"] = getattr(mod, "__version__", "unknown")
    except ImportError as err:
        status["error"] = str(err)
    except Exception as err:
        status["error"] = f"Error during import: {err}"

    return status


def run_diagnostics() -> List[Dict[str, Any]]:
    """Runs dry-run import diagnostics across all registered judges."""
    return [check_judge_import(k) for k in JUDGE_REGISTRY]


def print_diagnostic_report(results: List[Dict[str, Any]]) -> None:
    """Prints a structured console report of judge import availability."""
    print("=" * 80)
    print("JUDGE ENVIRONMENT DIAGNOSTICS (Roadmap v6 Part VI & R1.2)")
    print("=" * 80)

    installed = [r for r in results if r["installed"]]
    missing = [r for r in results if not r["installed"]]

    print(f"\nAvailable Judges ({len(installed)}/{len(results)}):")
    if installed:
        for r in installed:
            print(f"  [x] {r['name']:<30} (v{r['version']})")
    else:
        print("  (None installed in current environment)")

    print(f"\nMissing / Uninstalled Judges ({len(missing)}/{len(results)}):")
    for r in missing:
        print(f"  [ ] {r['name']:<30}")
        print(f"      Module:  {r['module']}")
        print(f"      Reason:  {r['error']}")
        print(f"      Install: {r['install_command']}")
        print(f"      Role:    {r['role']}")

    print("\n" + "=" * 80)
    print("INTEGRITY RULE: AesFix never uses mocks or heuristics as fallbacks")
    print("for uninstalled judges. Experiments requiring uninstalled judges")
    print("must halt and report blockers until dependencies are provisioned.")
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Judge registry and import diagnostics.")
    parser.parse_args()

    results = run_diagnostics()
    print_diagnostic_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
