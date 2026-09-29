"""Data Pool Manifest Builder & Canonical Dataset Audit (AF-Roadmap v6 R1.1).

Scans local dataset directories, computes cryptographic SHA-256 checksums,
and verifies the presence and licensing state of canonical photographic datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aesfix.data.manifest")

# Canonical reference datasets required for AesFix calibration & benchmarking (Roadmap v6 Part VI)
CANONICAL_DATASETS: Dict[str, Dict[str, Any]] = {
    "FiveK": {
        "full_name": "MIT-Adobe FiveK Dataset",
        "expected_subdirs": ["fivek", "mit_adobe_fivek", "FiveK"],
        "source": "MIT / Adobe Research (Bychkovsky et al., CVPR 2011)",
        "license_name": "CC BY-NC-SA 3.0",
        "license_note": "Non-commercial research only. User must accept Adobe/MIT terms before downloading.",
        "requires_license_acceptance": True,
        "role": "Expert photographic retouching pairs (Retouchers A-E) for Tier-A calibration.",
    },
    "AADB": {
        "full_name": "Aesthetics and Attributes Database",
        "expected_subdirs": ["aadb", "AADB", "aesthetics_attributes"],
        "source": "Adobe Research (Kong et al., ECCV 2016)",
        "license_name": "Adobe Research Academic License",
        "license_note": "Academic use only. Requires accepting Adobe Research license agreement.",
        "requires_license_acceptance": True,
        "role": "Ground-truth attribute ratings for SigLIP aspect probe training.",
    },
    "SPAQ": {
        "full_name": "Smartphone Photography Attribute and Quality Database",
        "expected_subdirs": ["spaq", "SPAQ"],
        "source": "Fang et al., CVPR 2020",
        "license_name": "PolyU Research License",
        "license_note": "Non-commercial research use only. Requires agreement with authors.",
        "requires_license_acceptance": True,
        "role": "11k smartphone photos with brightness/contrast/colorfulness attribute labels.",
    },
    "KonIQ-10k": {
        "full_name": "KonIQ-10k Ecological Image Quality Assessment",
        "expected_subdirs": ["koniq-10k", "koniq10k", "KonIQ-10k"],
        "source": "University of Konstanz (Hosu et al., TIP 2020)",
        "license_name": "CC BY-NC-SA 4.0",
        "license_note": "Attribution-NonCommercial-ShareAlike 4.0 International.",
        "requires_license_acceptance": True,
        "role": "In-the-wild aesthetic distribution for reference pool normalization.",
    },
    "PPR10K": {
        "full_name": "Portrait Photo Retouching 10K",
        "expected_subdirs": ["ppr10k", "PPR10K"],
        "source": "Liang et al., CVPR 2021",
        "license_name": "Non-commercial Research Agreement",
        "license_note": "Portrait retouching pairs; requires non-commercial academic research agreement.",
        "requires_license_acceptance": True,
        "role": "Portrait-specific retouched raw/target pairs.",
    },
    "DEAR": {
        "full_name": "Diverse Aesthetic Assessment and Retouching",
        "expected_subdirs": ["dear", "DEAR"],
        "source": "CVPR 2025 / DEAR Authors",
        "license_name": "Research-Only Data Agreement",
        "license_note": "100-pair subset public; full pairwise dataset requires author approval.",
        "requires_license_acceptance": True,
        "role": "Human pairwise preference on rendering variations; calibration set D_cal.",
    },
}


def compute_file_sha256(file_path: Path, block_size: int = 1048576) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_canonical_datasets(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Checks presence of canonical datasets in data_dir without downloading."""
    audit_results: Dict[str, Dict[str, Any]] = {}
    existing_subdirs = {
        p.name.lower(): p
        for p in data_dir.iterdir()
        if p.is_dir()
    } if data_dir.exists() else {}

    for key, info in CANONICAL_DATASETS.items():
        found = False
        found_path: Optional[Path] = None
        for alias in info["expected_subdirs"]:
            if alias.lower() in existing_subdirs:
                found = True
                found_path = existing_subdirs[alias.lower()]
                break

        audit_results[key] = {
            "name": info["full_name"],
            "present": found,
            "path": str(found_path) if found_path else None,
            "license": info["license_name"],
            "license_note": info["license_note"],
            "requires_license_acceptance": info["requires_license_acceptance"],
            "role": info["role"],
        }

    return audit_results


def build_manifest(data_dir: Path = Path("data")) -> Dict[str, Any]:
    """Builds a cryptographic pool manifest for all files present under data_dir."""
    files_manifest: List[Dict[str, Any]] = []

    if data_dir.exists():
        for root, _, files in os.walk(data_dir):
            for fname in sorted(files):
                fpath = Path(root) / fname
                rel_path = fpath.relative_to(data_dir).as_posix()
                try:
                    sha256 = compute_file_sha256(fpath)
                    files_manifest.append({
                        "path": rel_path,
                        "sha256": sha256,
                        "size_bytes": fpath.stat().st_size,
                    })
                except Exception as err:
                    logger.warning("Could not hash %s: %s", fpath, err)

    canonical_audit = audit_canonical_datasets(data_dir)

    manifest: Dict[str, Any] = {
        "manifest_version": "1.0",
        "data_root": str(data_dir),
        "total_files": len(files_manifest),
        "files": files_manifest,
        "canonical_audit": canonical_audit,
    }

    return manifest


def print_audit_report(canonical_audit: Dict[str, Dict[str, Any]]) -> None:
    """Prints a clear report of present vs missing canonical datasets."""
    print("=" * 80)
    print("CANONICAL DATASET AUDIT (Roadmap v6 Part VI)")
    print("=" * 80)

    present = [k for k, v in canonical_audit.items() if v["present"]]
    missing = [k for k, v in canonical_audit.items() if not v["present"]]

    print(f"\nPresent datasets ({len(present)}):")
    if present:
        for k in present:
            info = canonical_audit[k]
            print(f"  [x] {k:<10} - {info['name']} (Path: {info['path']})")
    else:
        print("  (None found under data/)")

    print(f"\nMissing datasets ({len(missing)}):")
    for k in missing:
        info = canonical_audit[k]
        print(f"  [ ] {k:<10} - {info['name']}")
        print(f"      License: {info['license']}")
        print(f"      Note:    {info['license_note']}")
        print(f"      Role:    {info['role']}")

    print("\n" + "=" * 80)
    print("NOTICE: No datasets were downloaded. Datasets marked with restrictive")
    print("licenses must be accepted and acquired by the user before training/eval.")
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data pool manifest and audit datasets.")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output", type=str, default="data/manifest.json", help="Output path for manifest JSON")

    args = parser.parse_args()
    data_path = Path(args.data_dir)

    manifest = build_manifest(data_path)
    print_audit_report(manifest["canonical_audit"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest with {manifest['total_files']} file(s) saved to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
