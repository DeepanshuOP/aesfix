#!/usr/bin/env python3
"""Check that results/ledger.jsonl is strictly append-only and schema-compliant.

Integrity Rules (AGF4):
1. Every line present in the base commit (e.g. merge-base with origin/main) must exist
   identically, in the same order, as a prefix of current results/ledger.jsonl.
   No existing line may be modified or deleted.
2. Every line in the current ledger must be valid JSON and contain required ledger
   metadata (run_id, commit, timestamp, metrics). Dummy entries (e.g. {'test': 1})
   are strictly rejected.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LEDGER_PATH = REPO_ROOT / "results" / "ledger.jsonl"
REQUIRED_FIELDS = ("run_id", "commit", "timestamp", "metrics")


def resolve_base_commit(requested_base: Optional[str] = None) -> Optional[str]:
    """Finds the git commit to compare against for append-only validation."""
    candidates: List[str] = []
    if requested_base:
        candidates.append(requested_base)
    if "GITHUB_BASE_REF" in os.environ:
        candidates.append(f"origin/{os.environ['GITHUB_BASE_REF']}")
    candidates.extend(["origin/main", "main", "HEAD~1"])

    for ref in candidates:
        try:
            res = subprocess.run(
                ["git", "merge-base", "HEAD", ref],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            continue

    # Fallback: check if the ref itself can be resolved directly
    for ref in candidates:
        try:
            res = subprocess.run(
                ["git", "rev-parse", ref],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            continue

    return None


def get_base_ledger_lines(base_commit: str, ledger_rel_path: str = "results/ledger.jsonl") -> Optional[List[str]]:
    """Retrieves lines of the ledger file as of base_commit."""
    try:
        res = subprocess.run(
            ["git", "show", f"{base_commit}:{ledger_rel_path}"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return [line.strip() for line in res.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return None


def validate_ledger_entries(lines: List[str]) -> List[str]:
    """Verifies that all ledger lines parse as JSON and contain required fields."""
    errors: List[str] = []
    for idx, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
        except Exception as exc:
            errors.append(f"Line {idx}: Invalid JSON syntax ({exc})")
            continue

        if not isinstance(entry, dict):
            errors.append(f"Line {idx}: Entry must be a JSON object, got {type(entry).__name__}")
            continue

        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            errors.append(
                f"Line {idx}: Missing required ledger field(s) {missing}. "
                f"Invalid dummy entry: {line[:80]}"
            )
    return errors


def verify_append_only(
    base_lines: List[str], current_lines: List[str]
) -> List[str]:
    """Ensures current_lines starts with base_lines without modifications or deletions."""
    errors: List[str] = []
    if len(current_lines) < len(base_lines):
        errors.append(
            f"Ledger truncation or deletion detected: base commit had {len(base_lines)} lines, "
            f"but current ledger has only {len(current_lines)} lines."
        )
        return errors

    for idx, base_line in enumerate(base_lines, 1):
        curr_line = current_lines[idx - 1]
        if curr_line != base_line:
            errors.append(
                f"Ledger line {idx} was modified!\n"
                f"  Base:    {base_line[:120]}\n"
                f"  Current: {curr_line[:120]}"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that results/ledger.jsonl is append-only.")
    parser.add_argument("--base", default=None, help="Base git ref to compare against.")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help="Path to results/ledger.jsonl file.",
    )
    args = parser.parse_args()

    ledger_path: Path = args.ledger
    if not ledger_path.exists():
        print(f"Ledger file not found at '{ledger_path}'. Nothing to verify.")
        return 0

    current_lines = [line.strip() for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Validate structure of all entries
    schema_errors = validate_ledger_entries(current_lines)
    if schema_errors:
        print(f"FAILED: Found {len(schema_errors)} invalid ledger entry(ies):")
        for err in schema_errors:
            print(f"  - {err}")
        return 1

    # Check git base
    base_commit = resolve_base_commit(args.base)
    if not base_commit:
        print("Note: No base commit identified. Skipping diff check; schema check passed.")
        return 0

    base_lines = get_base_ledger_lines(base_commit)
    if base_lines is None:
        print(f"Note: Ledger did not exist at base commit {base_commit[:8]}. Schema check passed.")
        return 0

    diff_errors = verify_append_only(base_lines, current_lines)
    if diff_errors:
        print(f"FAILED: Ledger append-only violation relative to base {base_commit[:8]}:")
        for err in diff_errors:
            print(f"  - {err}")
        return 1

    appended_count = len(current_lines) - len(base_lines)
    print(
        f"SUCCESS: Ledger integrity verified against base {base_commit[:8]}. "
        f"{len(base_lines)} base lines preserved, {appended_count} legitimate run(s) appended."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
