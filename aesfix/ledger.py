"""AesFix Evidence Ledger (AF-4.4 & Roadmap v6).

Every metric value reported in documentation, papers, or leaderboards must originate
from an executed run logged to results/ledger.jsonl with cryptographic provenance.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("aesfix.ledger")

DEFAULT_LEDGER_PATH = Path("results/ledger.jsonl")


def get_ledger_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Returns effective ledger path respecting explicit arg, AESFIX_LEDGER_PATH env var, or DEFAULT_LEDGER_PATH."""
    if path is not None:
        return Path(path)
    env_path = os.environ.get("AESFIX_LEDGER_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_LEDGER_PATH


def compute_dict_hash(d: Dict[str, Any]) -> str:
    """Computes deterministic SHA-256 hash of a dictionary."""
    canonical_json = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_git_commit() -> str:
    """Retrieves the current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted"


def get_hardware_string() -> str:
    """Constructs a descriptive string of host CPU and GPU hardware."""
    parts: List[str] = []
    proc = platform.processor() or platform.machine() or "unknown_cpu"
    parts.append(proc)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            parts.append(f"GPU: {device_name}")
        else:
            parts.append("CPU-only")
    except Exception:
        parts.append("CPU-only")
    return "; ".join(parts)


def append_run(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    data_manifest: Dict[str, Any],
    cost_usd: Optional[float] = None,
    ledger_path: Optional[Union[str, Path]] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Appends an immutable record of an executed benchmark/experiment run to the ledger.

    Args:
        metrics: Dictionary of evaluated metrics (e.g. ag_at_1, hwr, fcr).
        config: Experiment configuration parameters.
        data_manifest: Dataset splits, identifiers, and checksums.
        cost_usd: Monetary cost incurred during inference/API execution.
        ledger_path: Target path for the JSONL ledger (falls back to AESFIX_LEDGER_PATH or DEFAULT_LEDGER_PATH).
        run_id: Optional UUID4 string; auto-generated if omitted.

    Returns:
        The complete run entry dictionary written to disk.
    """
    path = get_ledger_path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rid = run_id or str(uuid.uuid4())
    utc_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    commit_hash = get_git_commit()
    config_hash = compute_dict_hash(config)
    manifest_hash = compute_dict_hash(data_manifest)
    hardware = get_hardware_string()

    record: Dict[str, Any] = {
        "run_id": rid,
        "timestamp": utc_now,
        "commit": commit_hash,
        "config_hash": config_hash,
        "data_manifest_hash": manifest_hash,
        "hardware": hardware,
        "cost_usd": cost_usd,
        "config": config,
        "data_manifest": data_manifest,
        "metrics": metrics,
    }

    line = json.dumps(record, sort_keys=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    logger.info("Recorded ledger run %s to %s", rid, path)
    return record


def load_ledger(ledger_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Loads all run records from the ledger JSONL file."""
    path = get_ledger_path(ledger_path)
    if not path.exists():
        return []

    runs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError as err:
                logger.warning("Corrupt ledger line %d in %s: %s", line_num, path, err)
    return runs


def get_run_by_id(run_id: str, ledger_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Fetches a specific run record by UUID from the ledger."""
    for run in load_ledger(ledger_path):
        if run.get("run_id") == run_id:
            return run
    return None
