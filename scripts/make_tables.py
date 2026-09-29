#!/usr/bin/env python3
"""Render Markdown summary and comparison tables from AesFix ledger runs.

Usage:
    python scripts/make_tables.py [--run-id RUN_ID] [--ledger PATH] [--output FILE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.ledger import DEFAULT_LEDGER_PATH, load_ledger, get_run_by_id


def format_run_table(run: Dict[str, Any]) -> str:
    """Renders a detailed Markdown table for an individual run."""
    run_id = run.get("run_id", "unknown")
    timestamp = run.get("timestamp", "unknown")
    commit = run.get("commit", "unknown")
    hardware = run.get("hardware", "unknown")
    cost = run.get("cost_usd")
    cost_str = f"${cost:.4f}" if cost is not None else "N/A"
    config_hash = run.get("config_hash", "unknown")
    manifest_hash = run.get("data_manifest_hash", "unknown")

    lines: List[str] = [
        f"### Run `{run_id}`",
        "",
        "| Attribute | Value |",
        "| :--- | :--- |",
        f"| **Run ID** | `{run_id}` |",
        f"| **Timestamp (UTC)** | `{timestamp}` |",
        f"| **Git Commit** | `{commit}` |",
        f"| **Hardware** | {hardware} |",
        f"| **Cost (USD)** | {cost_str} |",
        f"| **Config SHA-256** | `{config_hash[:16]}...` |",
        f"| **Data Manifest SHA-256** | `{manifest_hash[:16]}...` |",
        "",
        "#### Metrics",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
    ]

    metrics = run.get("metrics", {})
    if not metrics:
        lines.append("| *(none)* | *(no metrics recorded)* |")
    else:
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                lines.append(f"| `{k}` | {v:.4f} |")
            else:
                lines.append(f"| `{k}` | {v} |")

    lines.append("")
    return "\n".join(lines)


def format_summary_table(runs: List[Dict[str, Any]]) -> str:
    """Renders a comparison table across multiple runs."""
    if not runs:
        return "*(Evidence ledger is empty — no runs recorded)*\n"

    # Collect all unique metric keys
    metric_keys = sorted(
        {k for r in runs for k in r.get("metrics", {}).keys()}
    )

    headers = ["Run ID", "Commit", "Timestamp", "Hardware"] + metric_keys
    col_align = [":---", ":---", ":---", ":---"] + [":---:" for _ in metric_keys]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(col_align) + " |",
    ]

    for r in runs:
        rid = r.get("run_id", "unknown")[:8]
        commit = r.get("commit", "unknown")[:8]
        ts = r.get("timestamp", "")[:19]
        hw = r.get("hardware", "unknown")[:25]
        row = [f"`{rid}`", f"`{commit}`", ts, hw]
        metrics = r.get("metrics", {})
        for k in metric_keys:
            val = metrics.get(k)
            if val is None:
                row.append("-")
            elif isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Markdown tables from AesFix ledger.")
    parser.add_argument("--run-id", type=str, help="Render detailed table for specific run ID.")
    parser.add_argument("--ledger", type=str, default=str(DEFAULT_LEDGER_PATH), help="Path to ledger.jsonl.")
    parser.add_argument("--output", type=str, help="Optional output markdown file path.")

    args = parser.parse_args()
    ledger_path = Path(args.ledger)

    if not ledger_path.exists():
        text = f"Ledger file not found at `{ledger_path}`.\n"
    elif args.run_id:
        run = get_run_by_id(args.run_id, ledger_path=ledger_path)
        if not run:
            sys.stderr.write(f"Error: Run ID '{args.run_id}' not found in ledger '{ledger_path}'.\n")
            return 1
        text = format_run_table(run)
    else:
        runs = load_ledger(ledger_path=ledger_path)
        text = format_summary_table(runs)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Rendered table written to {args.output}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
