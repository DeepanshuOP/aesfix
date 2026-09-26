"""Unit tests for Open-Source Release Manifest & Compliance Auditor (AF-4.3).

Why these tests are critical:
1. Prevents copyright and license infringements by verifying that raw pixel binaries
   cannot be registered in release manifests without triggering an audit violation.
2. Confirms that sensitive files (.env, secrets, private keys) are strictly blocked from release packages.
3. Validates that Hugging Face ModelCard and DatasetCard generators produce valid YAML frontmatter.
4. Verifies cryptographic SHA-256 checksum generation for artifact integrity.
"""
from __future__ import annotations

import json
from pathlib import Path

from aesfix.data.release import (
    DatasetCardBuilder,
    LicenseComplianceAuditor,
    ModelCardBuilder,
    ReleaseManifestBuilder,
    compute_sha256,
)


def test_license_auditor_blocks_raw_images() -> None:
    """Verifies that distributing raw copyrighted image binaries is strictly rejected."""
    is_valid_jpg, reason_jpg = LicenseComplianceAuditor.audit_manifest_entry("data/test.jpg")
    assert is_valid_jpg is False
    assert "Redistribution violation" in str(reason_jpg)

    is_valid_png, _ = LicenseComplianceAuditor.audit_manifest_entry("data/test.png")
    assert is_valid_png is False

    # Allowed when explicitly specified
    is_valid_override, _ = LicenseComplianceAuditor.audit_manifest_entry(
        "data/test.jpg", allow_raw_images=True
    )
    assert is_valid_override is True


def test_license_auditor_blocks_sensitive_files() -> None:
    """Confirms that credential and secret files are caught by the security gate."""
    is_valid_env, reason_env = LicenseComplianceAuditor.audit_manifest_entry(".env")
    assert is_valid_env is False
    assert "Security violation" in str(reason_env)

    is_valid_key, _ = LicenseComplianceAuditor.audit_manifest_entry("secrets/prod_api.key")
    assert is_valid_key is False


def test_license_auditor_permits_manifest_files() -> None:
    """Verifies that allowed open-source formats pass audit inspection."""
    valid_paths = [
        "checkpoints/model.safetensors",
        "manifests/synthetic_triplets.jsonl",
        "configs/sft_config.json",
        "docs/README.md",
    ]
    for path in valid_paths:
        is_valid, reason = LicenseComplianceAuditor.audit_manifest_entry(path)
        assert is_valid is True, f"Failed on valid path {path}: {reason}"


def test_license_auditor_directory_scan(tmp_path: Path) -> None:
    """Tests recursive directory auditing on simulated release directory."""
    # Create valid files
    (tmp_path / "model.safetensors").write_bytes(b"dummy_weights")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")

    # Initial scan should be clean
    violations = LicenseComplianceAuditor.audit_directory(tmp_path)
    assert len(violations) == 0

    # Add an offending file
    (tmp_path / "unauthorized_photo.jpeg").write_bytes(b"image_bytes")
    violations_with_image = LicenseComplianceAuditor.audit_directory(tmp_path)
    assert len(violations_with_image) == 1


def test_model_card_builder_yaml_structure() -> None:
    """Verifies standard Hugging Face Model Card formatting."""
    card = ModelCardBuilder.build_model_card(
        model_name="AesFix-Qwen2.5-VL-7B-LoRA",
        ag1_score=0.0824,
        concordance_rate=0.768,
    )

    # Check YAML frontmatter boundaries
    assert card.startswith("---")
    assert "license: apache-2.0" in card
    assert "base_model: Qwen/Qwen2.5-VL-7B-Instruct" in card
    assert "+0.0824" in card
    assert "76.8%" in card
    assert "@article{aesfix2026" in card


def test_dataset_card_builder_yaml_structure() -> None:
    """Verifies standard Hugging Face Dataset Card formatting."""
    card = DatasetCardBuilder.build_dataset_card(
        dataset_name="AesFix-Synthetic-Counterfactual-50K",
        num_triplets=50000,
        signal_recovery_rate=0.914,
    )

    assert card.startswith("---")
    assert "license: cc-by-nc-4.0" in card
    assert "50,000" in card
    assert "91.4%" in card


def test_release_manifest_builder(tmp_path: Path) -> None:
    """Tests artifact registration, SHA-256 calculation, and manifest export."""
    builder = ReleaseManifestBuilder()
    content = b"sample_safetensors_weights_for_testing"

    expected_sha = compute_sha256(content)

    artifact = builder.add_artifact(
        artifact_id="test_weights_v1",
        name="Test Weights",
        artifact_type="model_weights",
        license_type="Apache-2.0",
        file_path="checkpoints/test_weights.safetensors",
        content_bytes=content,
        metadata={"precision": "bf16"},
    )

    assert artifact.sha256_hash == expected_sha
    assert artifact.size_bytes == len(content)

    # Export to JSON
    manifest_path = tmp_path / "release_manifest.json"
    builder.export_json(manifest_path)

    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["version"] == "1.0.0"
    assert data["total_artifacts"] == 1
    assert data["artifacts"][0]["artifact_id"] == "test_weights_v1"
