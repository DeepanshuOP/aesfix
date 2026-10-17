"""Global pytest fixtures for AesFix test suite."""
from __future__ import annotations

from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def isolate_telemetry_and_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirects telemetry runs.jsonl and evidence ledger.jsonl to temporary directories for every test.

    Why (F1 Telemetry Isolation):
    Prevents tests with mock/fake providers from writing to results/agent_runs/runs.jsonl
    or results/ledger.jsonl and polluting production/benchmark evidence files.
    """
    temp_runs = tmp_path / "runs.jsonl"
    temp_ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("AESFIX_RUNS_PATH", str(temp_runs))
    monkeypatch.setenv("AESFIX_LEDGER_PATH", str(temp_ledger))
