"""KonIQ-10k Dataset Parser & Quality Annotation Loader (AQ8).

Parses metadata and MOS ratings from D:\\aesfix-data\\koniq10k for non-CLIP aesthetic alignment.
Provides statistical summaries, record iteration, and manifest ledger logging.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("aesfix.data.koniq")

DEFAULT_KONIQ_DIR = Path(r"D:\aesfix-data\koniq10k")


@dataclass
class KonIQSummary:
    """Statistical summary of the KonIQ-10k dataset."""
    total_samples: int
    mean_mos: float
    std_mos: float
    min_mos: float
    max_mos: float
    columns: List[str]
    manifest_sha256: str
    images_dir_exists: bool
    distribution_bins: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes KonIQ summary for reporting and ledger telemetry."""
        return {
            "total_samples": self.total_samples,
            "mean_mos": self.mean_mos,
            "std_mos": self.std_mos,
            "min_mos": self.min_mos,
            "max_mos": self.max_mos,
            "columns": self.columns,
            "manifest_sha256": self.manifest_sha256,
            "images_dir_exists": self.images_dir_exists,
            "distribution_bins": self.distribution_bins,
        }


def _file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_koniq_manifest_hash(root_dir: Path) -> str:
    """Computes composite SHA-256 hash of KonIQ-10k metadata files."""
    csv_file = root_dir / "koniq10k_scores_and_distributions.csv"
    if not csv_file.exists():
        return ""
    return _file_sha256(csv_file)


def parse_koniq_dataset(koniq_dir: Path | str = DEFAULT_KONIQ_DIR) -> KonIQSummary:
    """Parses KonIQ-10k dataset annotations and calculates statistical distributions."""
    root = Path(koniq_dir)
    csv_path = root / "koniq10k_scores_and_distributions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"KonIQ-10k scores CSV not found at: {csv_path}")

    images_dir = root / "images"
    images_dir_exists = images_dir.exists() and images_dir.is_dir()

    mos_values: List[float] = []
    columns: List[str] = []
    c1_sum = c2_sum = c3_sum = c4_sum = c5_sum = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            mos = float(row["MOS"])
            mos_values.append(mos)
            c1_sum += int(row.get("c1", 0))
            c2_sum += int(row.get("c2", 0))
            c3_sum += int(row.get("c3", 0))
            c4_sum += int(row.get("c4", 0))
            c5_sum += int(row.get("c5", 0))

    n = len(mos_values)
    if n == 0:
        raise ValueError(f"KonIQ-10k scores CSV at {csv_path} is empty")

    mean_mos = float(sum(mos_values) / n)
    variance = sum((x - mean_mos) ** 2 for x in mos_values) / n
    std_mos = float(variance ** 0.5)
    min_mos = float(min(mos_values))
    max_mos = float(max(mos_values))

    manifest_sha = _file_sha256(csv_path)

    return KonIQSummary(
        total_samples=n,
        mean_mos=round(mean_mos, 4),
        std_mos=round(std_mos, 4),
        min_mos=round(min_mos, 4),
        max_mos=round(max_mos, 4),
        columns=columns,
        manifest_sha256=manifest_sha,
        images_dir_exists=images_dir_exists,
        distribution_bins={
            "bad_c1": c1_sum,
            "poor_c2": c2_sum,
            "fair_c3": c3_sum,
            "good_c4": c4_sum,
            "excellent_c5": c5_sum,
        },
    )


def iter_koniq_samples(
    koniq_dir: Path | str = DEFAULT_KONIQ_DIR,
    max_records: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yields parsed samples from KonIQ-10k dataset."""
    root = Path(koniq_dir)
    csv_path = root / "koniq10k_scores_and_distributions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"KonIQ-10k CSV not found at: {csv_path}")

    images_dir = root / "images"
    count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["image_name"]
            img_path = images_dir / img_name if images_dir.exists() else None
            yield {
                "image_name": img_name,
                "image_path": str(img_path) if img_path and img_path.exists() else None,
                "mos": float(row["MOS"]),
                "sd": float(row.get("SD", 0.0)),
                "mos_zscore": float(row.get("MOS_zscore", 0.0)),
                "total_votes": int(row.get("c_total", 0)),
            }
            count += 1
            if max_records and count >= max_records:
                break


def log_koniq_dataset_manifest(
    koniq_dir: Path | str = DEFAULT_KONIQ_DIR,
    ledger_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Logs KonIQ-10k dataset manifest to ledger with manifest SHA-256 hash (AQ8)."""
    from aesfix.ledger import DEFAULT_LEDGER_PATH, append_run

    summary = parse_koniq_dataset(koniq_dir)
    lp = Path(ledger_path) if ledger_path else DEFAULT_LEDGER_PATH

    metrics = {
        "dataset_total_samples": summary.total_samples,
        "dataset_mean_mos": summary.mean_mos,
        "dataset_std_mos": summary.std_mos,
        "dataset_min_mos": summary.min_mos,
        "dataset_max_mos": summary.max_mos,
    }
    config = {
        "dataset": "KonIQ-10k",
        "csv_columns": summary.columns,
    }
    manifest = {
        "dataset": "KonIQ-10k",
        "manifest_sha256": summary.manifest_sha256,
        "root_path": str(koniq_dir),
        "total_images": summary.total_samples,
    }

    return append_run(
        metrics=metrics,
        config=config,
        data_manifest=manifest,
        cost_usd=0.0,
        ledger_path=lp,
    )
