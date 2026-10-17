"""DEAR (Dataset for Evaluating the Aesthetics of Rendering / REPID) Parser (AQ8).

Parses root markup metadata from D:\\aesfix-data\\dear without downloading the 476 GB image repo.
Provides scene analysis, train/test split disjointness verification, pairwise rendering
comparison stats, and manifest ledger logging.
"""
from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("aesfix.data.dear")

DEFAULT_DEAR_DIR = Path(r"D:\aesfix-data\dear")


@dataclass
class DEARSummary:
    """Summary metrics and manifest characteristics of the DEAR dataset."""
    total_scenes: int
    train_scenes: int
    test_scenes: int
    is_scene_disjoint: bool
    raw_equals_subsample: bool
    raw_markup_sha256: str
    unique_style_pairs_count: int
    unique_style_pairs: List[Tuple[str, str]]
    median_votes_per_pair: float
    mean_votes_per_pair: float
    total_annotations_subsample: int
    manifest_sha256: str
    split_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes dataset summary for logging and ledger entries."""
        return {
            "total_scenes": self.total_scenes,
            "train_scenes": self.train_scenes,
            "test_scenes": self.test_scenes,
            "is_scene_disjoint": self.is_scene_disjoint,
            "raw_equals_subsample": self.raw_equals_subsample,
            "raw_markup_sha256": self.raw_markup_sha256,
            "unique_style_pairs_count": self.unique_style_pairs_count,
            "unique_style_pairs": [f"{left}->{right}" for left, right in self.unique_style_pairs],
            "median_votes_per_pair": self.median_votes_per_pair,
            "mean_votes_per_pair": self.mean_votes_per_pair,
            "total_annotations_subsample": self.total_annotations_subsample,
            "manifest_sha256": self.manifest_sha256,
            "split_counts": self.split_counts,
        }


def _file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dear_manifest_hash(root_dir: Path) -> str:
    """Computes composite SHA-256 hash of the markup files."""
    filenames = [
        "processed_markup.csv",
        "raw_markup.tsv",
        "raw_subsample_markup.tsv",
        "test_processed_markup.csv",
        "train_processed_markup.csv",
    ]
    h = hashlib.sha256()
    for fname in sorted(filenames):
        p = root_dir / fname
        if p.is_file():
            h.update(p.name.encode("utf-8"))
            with open(p, "rb") as fh:
                while chunk := fh.read(65536):
                    h.update(chunk)
    return h.hexdigest()


def parse_dear_dataset(dear_dir: Path | str = DEFAULT_DEAR_DIR) -> DEARSummary:
    """Parses root DEAR markup files and verifies split integrity."""
    root = Path(dear_dir)
    if not root.exists():
        raise FileNotFoundError(f"DEAR directory not found at: {root}")

    # 1. Compare raw_markup.tsv and raw_subsample_markup.tsv
    raw_path = root / "raw_markup.tsv"
    sub_path = root / "raw_subsample_markup.tsv"

    raw_sha = _file_sha256(raw_path) if raw_path.exists() else ""
    sub_sha = _file_sha256(sub_path) if sub_path.exists() else ""
    raw_equals_subsample = bool(raw_sha and sub_sha and raw_sha == sub_sha)

    # 2. Inspect raw votes per pair in raw_markup.tsv
    votes_per_pair: Dict[Tuple[str, str], int] = {}
    total_votes = 0
    if raw_path.exists():
        with open(raw_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                img_l = row.get("INPUT:image_left", "")
                img_r = row.get("INPUT:image_right", "")
                pair_key = (img_l, img_r)
                votes_per_pair[pair_key] = votes_per_pair.get(pair_key, 0) + 1
                total_votes += 1

    counts = list(votes_per_pair.values())
    if counts:
        counts.sort()
        n = len(counts)
        median_votes = float(counts[n // 2]) if n % 2 != 0 else float((counts[n // 2 - 1] + counts[n // 2]) / 2)
        mean_votes = round(float(sum(counts) / n), 2)
    else:
        median_votes, mean_votes = 0.0, 0.0

    # 3. Parse processed splits for scene disjointness and style pairs
    train_path = root / "train_processed_markup.csv"
    test_path = root / "test_processed_markup.csv"
    full_path = root / "processed_markup.csv"

    train_scenes: set[str] = set()
    train_rows = 0
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_scenes.add(row["name"])
                train_rows += 1

    test_scenes: set[str] = set()
    test_rows = 0
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                test_scenes.add(row["name"])
                test_rows += 1

    full_scenes: set[str] = set()
    style_pairs: set[Tuple[str, str]] = set()
    full_rows = 0
    if full_path.exists():
        with open(full_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_scenes.add(row["name"])
                style_pairs.add((row["left"], row["right"]))
                full_rows += 1

    is_scene_disjoint = bool(len(train_scenes.intersection(test_scenes)) == 0)
    manifest_sha = compute_dear_manifest_hash(root)

    return DEARSummary(
        total_scenes=len(full_scenes),
        train_scenes=len(train_scenes),
        test_scenes=len(test_scenes),
        is_scene_disjoint=is_scene_disjoint,
        raw_equals_subsample=raw_equals_subsample,
        raw_markup_sha256=raw_sha,
        unique_style_pairs_count=len(style_pairs),
        unique_style_pairs=sorted(list(style_pairs)),
        median_votes_per_pair=median_votes,
        mean_votes_per_pair=mean_votes,
        total_annotations_subsample=total_votes,
        manifest_sha256=manifest_sha,
        split_counts={"train": train_rows, "test": test_rows, "full": full_rows},
    )


def iter_dear_records(
    split: str = "train",
    dear_dir: Path | str = DEFAULT_DEAR_DIR,
    max_records: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yields parsed pairwise rendering comparison records from DEAR processed markup."""
    root = Path(dear_dir)
    file_map = {
        "train": root / "train_processed_markup.csv",
        "test": root / "test_processed_markup.csv",
        "full": root / "processed_markup.csv",
    }
    target_file = file_map.get(split)
    if not target_file or not target_file.exists():
        raise FileNotFoundError(f"DEAR split file not found for split '{split}': {target_file}")

    count = 0
    with open(target_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "scene": row["name"],
                "left_style": row["left"],
                "right_style": row["right"],
                "mos": float(row["mos"]),
                "mse": float(row.get("mse", 0.0)),
            }
            count += 1
            if max_records and count >= max_records:
                break


def log_dear_dataset_manifest(
    dear_dir: Path | str = DEFAULT_DEAR_DIR,
    ledger_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Logs DEAR dataset manifest to ledger with manifest SHA-256 hash (AQ8)."""
    from aesfix.ledger import DEFAULT_LEDGER_PATH, append_run

    summary = parse_dear_dataset(dear_dir)
    lp = Path(ledger_path) if ledger_path else DEFAULT_LEDGER_PATH

    metrics = {
        "dataset_total_scenes": summary.total_scenes,
        "dataset_train_scenes": summary.train_scenes,
        "dataset_test_scenes": summary.test_scenes,
        "dataset_is_scene_disjoint": summary.is_scene_disjoint,
        "dataset_raw_equals_subsample": summary.raw_equals_subsample,
        "dataset_unique_style_pairs": summary.unique_style_pairs_count,
        "dataset_median_votes_per_pair": summary.median_votes_per_pair,
        "dataset_mean_votes_per_pair": summary.mean_votes_per_pair,
        "dataset_total_annotations": summary.total_annotations_subsample,
    }
    config = {
        "dataset": "DEAR_REPID",
        "split_disjoint": summary.is_scene_disjoint,
        "raw_markup_sha256": summary.raw_markup_sha256,
    }
    manifest = {
        "dataset": "DEAR",
        "manifest_sha256": summary.manifest_sha256,
        "root_path": str(dear_dir),
        "total_scenes": summary.total_scenes,
    }

    return append_run(
        metrics=metrics,
        config=config,
        data_manifest=manifest,
        cost_usd=0.0,
        ledger_path=lp,
    )
