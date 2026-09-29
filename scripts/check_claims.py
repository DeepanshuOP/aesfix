#!/usr/bin/env python3
"""Check documentation and paper claims against the AesFix evidence ledger.

Rule (AF-Roadmap v6 & Part XI):
Any number followed by a metric keyword (SRCC, PLCC, AG, HWR, FCR, accuracy, kappa, alpha)
inside docs/ or paper/ must be tagged [run:<id>] with an id present in results/ledger.jsonl.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.ledger import DEFAULT_LEDGER_PATH, load_ledger  # noqa: E402

CLAIM_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9_.-])[-+]?\d+(?:\.\d+)?%?\s*(?:SRCC|PLCC|AG|HWR|FCR|accuracy|kappa|alpha)\b",
    re.IGNORECASE,
)
RUN_TAG_PATTERN = re.compile(r"\[run:([a-zA-Z0-9_\-]+)\]")


def get_known_run_ids(ledger_path: Path) -> Set[str]:
    """Retrieves all valid run IDs from the ledger."""
    if not ledger_path.exists():
        return set()
    runs = load_ledger(ledger_path)
    return {str(r.get("run_id")) for r in runs if "run_id" in r}


def scan_file_for_claims(
    file_path: Path, known_run_ids: Set[str]
) -> List[Tuple[int, str, str]]:
    """Scans a file for metric claims and validates their ledger provenance.

    Returns:
        List of (line_num, claim_text, error_message).
    """
    violations: List[Tuple[int, str, str]] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                matches = list(CLAIM_PATTERN.finditer(line))
                if not matches:
                    continue

                tag_match = RUN_TAG_PATTERN.search(line)
                if not tag_match:
                    for m in matches:
                        violations.append((
                            line_num,
                            m.group(0),
                            f"Untagged metric claim '{m.group(0)}' without [run:<id>] tag.",
                        ))
                else:
                    run_id = tag_match.group(1)
                    if run_id not in known_run_ids:
                        for m in matches:
                            violations.append((
                                line_num,
                                m.group(0),
                                f"Claim references unknown run_id '{run_id}' not found in ledger.",
                            ))
    except Exception as err:
        violations.append((0, "", f"Error reading {file_path}: {err}"))
    return violations


def check_directories(
    dirs: List[Path], ledger_path: Path
) -> Tuple[int, List[str]]:
    """Scans directories for unverified metric claims."""
    known_run_ids = get_known_run_ids(ledger_path)
    all_violations: List[str] = []
    total_files_scanned = 0

    for directory in dirs:
        if not directory.exists():
            continue
        for root, dirs_list, files in os.walk(directory):
            # Exclude archive directories from scan
            dirs_list[:] = [d for d in dirs_list if d.lower() != "archive"]
            for file_name in files:
                if file_name.lower().endswith((".md", ".txt", ".tex", ".rst")):
                    file_path = Path(root) / file_name
                    total_files_scanned += 1
                    file_violations = scan_file_for_claims(file_path, known_run_ids)
                    for line_num, claim_text, msg in file_violations:
                        all_violations.append(
                            f"{file_path}:{line_num}: [CLAIM VIOLATION] {msg}"
                        )

    return total_files_scanned, all_violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify metric claims in docs/ and paper/ against evidence ledger."
    )
    parser.add_argument(
        "--ledger",
        type=str,
        default=str(DEFAULT_LEDGER_PATH),
        help="Path to ledger.jsonl",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["docs", "paper"],
        help="Directories to scan for claims",
    )

    args = parser.parse_args()
    ledger_path = Path(args.ledger)
    scan_dirs = [Path(d) for d in args.dirs]

    print(f"Scanning directories {scan_dirs} against ledger '{ledger_path}'...")
    total_scanned, violations = check_directories(scan_dirs, ledger_path)

    print(f"Scanned {total_scanned} documentation and paper files.")
    if violations:
        sys.stderr.write(
            f"\nFAILED: Found {len(violations)} metric claim integrity violation(s):\n"
        )
        for v in violations:
            sys.stderr.write(f"  - {v}\n")
        return 1

    print("SUCCESS: All metric claims are verified against ledger runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
