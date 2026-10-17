"""One-off cleanup script for F1 telemetry pollution.

Moves entries with provider or model starting with 'fake' from results/agent_runs/runs.jsonl
to archive/test_pollution/runs_fake.jsonl without silently deleting anything.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_PATH = REPO_ROOT / "results" / "agent_runs" / "runs.jsonl"
ARCHIVE_PATH = REPO_ROOT / "archive" / "test_pollution" / "runs_fake.jsonl"


def cleanup_telemetry() -> tuple[int, int]:
    if not RUNS_PATH.exists():
        print(f"File not found: {RUNS_PATH}")
        return 0, 0

    all_entries = []
    with open(RUNS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            all_entries.append(json.loads(line_str))

    fake_entries = []
    real_entries = []

    for entry in all_entries:
        p = str(entry.get("provider", ""))
        m = str(entry.get("model_id", ""))
        if p.startswith("fake") or m.startswith("fake"):
            fake_entries.append(entry)
        else:
            real_entries.append(entry)

    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVE_PATH, "w", encoding="utf-8") as f:
        for entry in fake_entries:
            f.write(json.dumps(entry) + "\n")

    with open(RUNS_PATH, "w", encoding="utf-8") as f:
        for entry in real_entries:
            f.write(json.dumps(entry) + "\n")

    print(
        f"Cleanup complete: Moved {len(fake_entries)} fake entries to {ARCHIVE_PATH.relative_to(REPO_ROOT)}. "
        f"Kept {len(real_entries)} real entries in {RUNS_PATH.relative_to(REPO_ROOT)}."
    )
    return len(fake_entries), len(real_entries)


if __name__ == "__main__":
    cleanup_telemetry()
