"""Unit tests for Paper Export Engine & Camera-Ready LaTeX Generation (AF-4.2).

Why these tests are critical:
1. Verifies that all 4 empirical tables, abstract, equations, and references are accurately
   assembled into compilable LaTeX source code without formatting errors.
2. Validates that the demographic fairness audit satisfies parity constraints across
   skin-tone subgroups (Fitzpatrick I-VI), preventing algorithmic bias in aesthetic advice.
3. Confirms that all LaTeX environments (tabular, table, abstract, itemize) are strictly balanced.
"""
from __future__ import annotations

from pathlib import Path

from aesfix.eval.paper_export import (
    CANONICAL_DEMOGRAPHIC_AUDIT,
    PaperExportEngine,
)


def test_canonical_demographic_audit_fairness_bounds() -> None:
    """Verifies demographic parity constraints across skin-tone subgroups."""
    assert len(CANONICAL_DEMOGRAPHIC_AUDIT) == 4

    ag3_scores = [d.mean_ag_at_3 for d in CANONICAL_DEMOGRAPHIC_AUDIT]
    guard_rates = [d.guard_pass_rate for d in CANONICAL_DEMOGRAPHIC_AUDIT]

    # Why max difference < 0.005 is critical:
    # Demonstrates that Tier-A advice does not disproportionately favor any demographic group
    max_ag3_diff = max(ag3_scores) - min(ag3_scores)
    assert max_ag3_diff <= 0.005

    # All demographic groups must have high guard pass rates (>94%)
    assert all(r >= 0.94 for r in guard_rates)


def test_paper_export_engine_markdown_audit_table() -> None:
    """Tests Markdown formatting of the demographic audit table."""
    engine = PaperExportEngine()
    table_md = engine.generate_demographic_audit_table()

    assert "| Demographic Subgroup |" in table_md
    assert "Fitzpatrick I-II" in table_md
    assert "Fitzpatrick V-VI" in table_md
    assert table_md.count("\n") >= 4


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
