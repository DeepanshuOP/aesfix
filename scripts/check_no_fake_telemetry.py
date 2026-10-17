"""CI check verifying no fake providers or models in evidence ledger or telemetry runs (F1).

Fails with exit code 1 if any entry in results/agent_runs/runs.jsonl or results/ledger.jsonl
has a provider or model starting with 'fake'.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_PATH = REPO_ROOT / "results" / "agent_runs" / "runs.jsonl"
LEDGER_PATH = REPO_ROOT / "results" / "ledger.jsonl"


def check_no_fake_telemetry() -> None:
    violations: list[str] = []

    # Check runs.jsonl
    if RUNS_PATH.exists():
        with open(RUNS_PATH, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    p = str(data.get("provider", "")).lower()
                    m = str(data.get("model_id", "")).lower()
                    if p.startswith("fake") or m.startswith("fake"):
                        violations.append(f"{RUNS_PATH.name}:{idx} has fake provider='{p}' or model_id='{m}'")
                except json.JSONDecodeError as err:
                    violations.append(f"{RUNS_PATH.name}:{idx} corrupt json: {err}")

    # Check ledger.jsonl
    if LEDGER_PATH.exists():
        with open(LEDGER_PATH, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    cfg = data.get("config", {})
                    p = str(cfg.get("provider", "")).lower()
                    m = str(cfg.get("model_id", "")).lower()
                    if p.startswith("fake") or m.startswith("fake"):
                        violations.append(f"{LEDGER_PATH.name}:{idx} has fake provider='{p}' or model_id='{m}'")
                except json.JSONDecodeError as err:
                    violations.append(f"{LEDGER_PATH.name}:{idx} corrupt json: {err}")

    if violations:
        print("Integrity Violation: Found fake test entries in evidence files:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        sys.exit(1)

    print("Integrity Check Passed: Zero fake providers or models in evidence ledger or telemetry runs.")


if __name__ == "__main__":
    check_no_fake_telemetry()
