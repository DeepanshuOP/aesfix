#!/usr/bin/env python3
"""Render Markdown summary and comparison tables from AesFix ledger runs.

Usage:
    python scripts/make_tables.py [--run-id RUN_ID] [--ledger PATH] [--output FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aesfix.ledger import DEFAULT_LEDGER_PATH, get_run_by_id, load_ledger  # noqa: E402



def compute_sha256(file_path: Path) -> str:
    """Computes hexadecimal SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def provenance_footer(file_path: Path) -> str:
    """Returns standardized Markdown provenance footer with relative path and SHA-256."""
    try:
        rel_path = file_path.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        rel_path = file_path.as_posix()
    file_hash = compute_sha256(file_path)
    return f"\n*Source: `{rel_path}` | SHA-256: `{file_hash}`*\n"


def format_koniq_table(koniq_path: Path) -> str:
    """Renders Markdown table from KonIQ benchmark results file."""
    with open(koniq_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = data.get("run_manifest", {})
    metrics = data.get("metrics_summary", {})

    lines: List[str] = [
        "### KonIQ-10k Benchmark (K1 Evaluation)",
        "",
        f"**Hardware**: {manifest.get('gpu_device', 'N/A')} | **Runtime**: {manifest.get('runtime_seconds', 'N/A')}s ({manifest.get('gpu_hours', 'N/A')} GPU-h) | **Precision**: {manifest.get('precision', 'N/A')}",
        "",
        "| Judge Model | Measured SRCC | Measured PLCC | Status |",
        "| :--- | :---: | :---: | :--- |",
    ]

    for model, m in sorted(metrics.items()):
        srcc = m.get("measured_srcc", "N/A")
        plcc = m.get("measured_plcc", "N/A")
        srcc_str = f"{srcc:.4f}" if isinstance(srcc, float) else str(srcc)
        plcc_str = f"{plcc:.4f}" if isinstance(plcc, float) else str(plcc)
        lines.append(f"| `{model}` | {srcc_str} | {plcc_str} | measured |")

    lines.append(provenance_footer(koniq_path))
    return "\n".join(lines)


def format_dear_pilot_table(dear_path: Path) -> str:
    """Renders Markdown table from DEAR benchmark results file."""
    with open(dear_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = data.get("run_manifest", {})
    pairwise = data.get("test_pairwise_metrics", {}) or data.get("metrics", {}).get("pairwise_accuracy_all", {})

    lines: List[str] = [
        "### DEAR Benchmark (K2 Pilot Evaluation)",
        "",
        f"**Hardware**: {manifest.get('gpu_device', 'N/A')} | **Runtime**: {manifest.get('runtime_seconds', 'N/A')}s ({manifest.get('gpu_hours', 'N/A')} GPU-h)",
        "",
        "| Judge Model | Pairwise Accuracy | Spearman Margin | Status |",
        "| :--- | :---: | :---: | :--- |",
    ]

    for model, m in sorted(pairwise.items()):
        acc = m.get("pairwise_accuracy", "N/A")
        sp = m.get("spearman_margin", "N/A")
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        sp_str = f"{sp:+.4f}" if isinstance(sp, float) else str(sp)
        lines.append(f"| `{model}` | {acc_str} | {sp_str} | measured |")

    lines.append(provenance_footer(dear_path))
    return "\n".join(lines)


def format_run_table(run: Dict[str, Any], ledger_path: Optional[Path] = None) -> str:
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

    if ledger_path is not None and ledger_path.exists():
        lines.append(provenance_footer(ledger_path))
    else:
        lines.append("")
    return "\n".join(lines)


def format_summary_table(runs: List[Dict[str, Any]], ledger_path: Optional[Path] = None) -> str:
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

    if ledger_path is not None and ledger_path.exists():
        lines.append(provenance_footer(ledger_path))
    else:
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Markdown tables from AesFix results and ledger.")
    parser.add_argument("--run-id", type=str, help="Render detailed table for specific run ID.")
    parser.add_argument("--ledger", type=str, default=str(DEFAULT_LEDGER_PATH), help="Path to ledger.jsonl.")
    parser.add_argument("--koniq", type=str, help="Render table from koniq_results.json.")
    parser.add_argument("--dear", type=str, help="Render table from dear_results.json.")
    parser.add_argument("--output", type=str, help="Optional output markdown file path.")

    args = parser.parse_args()
    ledger_path = Path(args.ledger)

    if args.koniq:
        kp = Path(args.koniq)
        if not kp.exists():
            sys.stderr.write(f"Error: KonIQ file not found at '{kp}'\n")
            return 1
        text = format_koniq_table(kp)
    elif args.dear:
        dp = Path(args.dear)
        if not dp.exists():
            sys.stderr.write(f"Error: DEAR file not found at '{dp}'\n")
            return 1
        text = format_dear_pilot_table(dp)
    elif args.run_id:
        if not ledger_path.exists():
            sys.stderr.write(f"Error: Ledger file not found at '{ledger_path}'.\n")
            return 1
        run = get_run_by_id(args.run_id, ledger_path=ledger_path)
        if not run:
            sys.stderr.write(f"Error: Run ID '{args.run_id}' not found in ledger '{ledger_path}'.\n")
            return 1
        text = format_run_table(run, ledger_path=ledger_path)
    else:
        if not ledger_path.exists():
            text = f"Ledger file not found at `{ledger_path}`.\n"
        else:
            runs = load_ledger(ledger_path=ledger_path)
            text = format_summary_table(runs, ledger_path=ledger_path)

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

