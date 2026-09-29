"""Unit tests for Paper Export Engine & Camera-Ready LaTeX Generation (AF-4.2 & Roadmap v6).

Verifies that LaTeX source code compiles cleanly with balanced environments,
bibtex entries are complete, and tables render correctly from demographic audit metrics.
"""
from __future__ import annotations

from pathlib import Path

from aesfix.eval.paper_export import (
    DemographicBiasMetric,
    PaperExportEngine,
)


def test_demographic_audit_metric_formatting() -> None:
    """Verifies that DemographicBiasMetric formats properly into markdown tables."""
    records = [
        DemographicBiasMetric("Subgroup A", 50, 0.0450, 0.0520, 0.950, 0.080, 0.700),
        DemographicBiasMetric("Subgroup B", 50, 0.0460, 0.0530, 0.960, 0.078, 0.710),
    ]
    engine = PaperExportEngine()
    table_md = engine.generate_demographic_audit_table(records)

    assert "| Demographic Subgroup |" in table_md
    assert "Subgroup A" in table_md
    assert "Subgroup B" in table_md
    assert "+0.0450" in table_md


def test_paper_export_engine_markdown_audit_empty_table() -> None:
    """Tests Markdown formatting when audit data is pending validation."""
    engine = PaperExportEngine()
    table_md = engine.generate_demographic_audit_table([])

    assert "| Demographic Subgroup |" in table_md
    assert "Pending ledger validation runs" in table_md


def test_paper_export_engine_bibtex_completeness() -> None:
    """Verifies that all foundational paper citations are present in BibTeX."""
    engine = PaperExportEngine()
    bib = engine.generate_bibtex_entries()

    assert "@article{aesfix2026" in bib
    assert "@inproceedings{kong2016photo" in bib
    assert "@inproceedings{murray2012ava" in bib
    assert "@article{hu2018exposure" in bib
    assert "@inproceedings{chen2025visualquality" in bib


def test_paper_export_engine_latex_compilation() -> None:
    """Verifies camera-ready LaTeX manuscript compilation and balanced environments."""
    engine = PaperExportEngine()
    latex = engine.compile_latex_document()

    # Structural LaTeX elements
    assert r"\documentclass[10pt,twocolumn,letterpaper]{article}" in latex
    assert r"\begin{document}" in latex
    assert r"\maketitle" in latex
    assert r"\end{document}" in latex

    # All empirical tables must be present
    assert "tab:faithfulness" in latex
    assert "tab:frontier" in latex
    assert "tab:human" in latex
    assert "tab:ablation" in latex

    # Check environment balance
    assert latex.count(r"\begin{table}") == latex.count(r"\end{table}")
    assert latex.count(r"\begin{tabular}") == latex.count(r"\end{tabular}")
    assert latex.count(r"\begin{abstract}") == latex.count(r"\end{abstract}")
    assert latex.count(r"\begin{itemize}") == latex.count(r"\end{itemize}")


def test_paper_export_engine_export_all(tmp_path: Path) -> None:
    """Tests writing paper source and bibliography files to disk."""
    engine = PaperExportEngine(output_dir=tmp_path)
    output_files = engine.export_all()

    tex_file = output_files["tex"]
    bib_file = output_files["bib"]

    assert tex_file.exists()
    assert bib_file.exists()
    assert tex_file.stat().st_size > 1000
    assert bib_file.stat().st_size > 500
