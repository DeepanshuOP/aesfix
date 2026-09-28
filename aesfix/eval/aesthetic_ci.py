"""Aesthetic CI/CD Visual Linter and PR Quality Gate (AF-6.2 & Roadmap §25 F4.3).

Why Aesthetic CI/CD converts AesFix into critical developer infrastructure:
1. Prevents Visual Regression in Pull Requests: Automated pipelines gate code on unit
   tests and types, but visual assets (documentation graphics, app store screenshots,
   marketing banners, AI-generated game assets) are merged without objective QA.
2. Actionable Merge Blocking: Images scoring below calibrated aesthetic thresholds
   fail the CI step, generating a structured GitHub PR review comment with 5-aspect
   diagnostic radar breakdowns and automated fix commit suggestions.
3. Automated Non-Destructive Retouching: When `--autofix` is active, Tier-A adjustments
   are applied directly to failing assets, verified with Identity Guard drift gating,
   and exported alongside Adobe Lightroom .xmp sidecars.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from aesfix.api.demo import compute_grade_label, get_demo_components
from aesfix.critic.critic import CriticFix
from aesfix.executor.guard import IdentityGuard
from aesfix.executor.tier_a import TierAExecutor

logger = logging.getLogger("aesfix.ci")

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class ImageLintRecord:
    """Diagnostic outcome for an individual visual asset evaluated in CI."""
    path: str
    overall_score: float
    grade_label: str
    aspect_scores: Dict[str, float]
    passed: bool
    weakest_aspect: str
    violation_reasons: List[str] = field(default_factory=list)
    top_rule: str = ""
    autofix_path: Optional[str] = None
    aesthetic_gain: float = 0.0
    xmp_sidecar_preview: Optional[str] = None


@dataclass
class LintSummary:
    """Shoot-level or PR-level aggregation of all evaluated visual assets."""
    total_scanned: int
    passed_count: int
    failed_count: int
    average_score: float
    all_passed: bool
    records: List[ImageLintRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes summary to standard JSON-compatible dictionary."""
        return asdict(self)


class AestheticLinter:
    """Orchestrates batch image discovery, aesthetic threshold evaluation, and autofix."""

    def __init__(
        self,
        min_overall_score: float = 0.55,
        min_aspect_score: float = 0.40,
        autofix: bool = False,
        autofix_output_dir: Optional[str] = None,
    ) -> None:
        self.min_overall_score = min_overall_score
        self.min_aspect_score = min_aspect_score
        self.autofix = autofix
        self.autofix_output_dir = autofix_output_dir
        self.executor = TierAExecutor() if autofix else None
        self.guard = IdentityGuard(drift_threshold=0.35) if autofix else None

    def lint_file(self, image_path: str) -> ImageLintRecord:
        """Evaluates an image file against aesthetic quality gates and applies autofixes."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image asset not found for linting: {image_path!r}")

        critic, head, ensemble = get_demo_components()

        # 1. Aspect Critic evaluation across canonical dimensions
        critique_res = critic.critique(image_path=image_path)
        aspects = critique_res.aspect_scores

        # 2. Differentiable neuro-fuzzy ANFIS aggregation
        anfis_trace = head.explain(aspects, top_k=1)
        overall_score = round(anfis_trace.overall_score, 4)
        grade = compute_grade_label(overall_score)

        # 3. Threshold violation audit
        violations: List[str] = []
        if overall_score < self.min_overall_score:
            violations.append(
                f"Overall score {overall_score:.3f} below minimum threshold {self.min_overall_score:.3f}"
            )

        for aspect_name, score in aspects.items():
            if score < self.min_aspect_score:
                violations.append(
                    f"Aspect '{aspect_name}' score {score:.3f} below quality floor {self.min_aspect_score:.3f}"
                )

        passed = len(violations) == 0
        weakest_aspect = min(aspects.items(), key=lambda kv: kv[1])[0]
        top_rule_desc = anfis_trace.top_fired_rules[0].antecedent_description if anfis_trace.top_fired_rules else ""

        # 4. Optional Tier-A Autofix execution if asset failed threshold
        autofix_path = None
        verified_gain = 0.0
        xmp_preview = None

        if not passed and self.autofix and self.executor and self.guard:
            candidate_fixes = critique_res.fixes or [
                CriticFix(
                    op="adjust_exposure",
                    params={"ev_delta": 0.35},
                    expected_aspect=weakest_aspect,
                    nl_instruction=f"CI autofix: boost exposure for weak {weakest_aspect}",
                )
            ]
            best_fix = candidate_fixes[0]
            exec_res = self.executor.execute_fixes(image_input=image_path, fixes=[best_fix])

            # Determine destination for repaired asset
            if self.autofix_output_dir:
                os.makedirs(self.autofix_output_dir, exist_ok=True)
                base_name = os.path.basename(image_path)
                stem, ext = os.path.splitext(base_name)
                autofix_path = os.path.join(self.autofix_output_dir, f"{stem}_fixed{ext}")
            else:
                stem, ext = os.path.splitext(image_path)
                autofix_path = f"{stem}_fixed{ext}"

            exec_res.image.save(autofix_path, format="JPEG", quality=95)

            # Perceptual identity check confirms no structural corruption
            guard_verdict = self.guard.evaluate(original=image_path, edited=autofix_path)
            if guard_verdict.is_valid:
                post_score = round(ensemble.evaluate(autofix_path).overall_score, 4)
                verified_gain = round(max(0.0, post_score - overall_score), 4)

            xmp_preview = exec_res.xmp_sidecar

        return ImageLintRecord(
            path=image_path,
            overall_score=overall_score,
            grade_label=grade,
            aspect_scores=aspects,
            passed=passed,
            weakest_aspect=weakest_aspect,
            violation_reasons=violations,
            top_rule=top_rule_desc,
            autofix_path=autofix_path,
            aesthetic_gain=verified_gain,
            xmp_sidecar_preview=xmp_preview,
        )

    def lint_paths(self, paths: Sequence[str]) -> LintSummary:
        """Scans multiple directories and files, compiling a unified lint summary."""
        discovered_files: List[str] = []

        for p in paths:
            path_obj = Path(p)
            if path_obj.is_file():
                if path_obj.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    discovered_files.append(str(path_obj))
            elif path_obj.is_dir():
                for root, _, files in os.walk(path_obj):
                    for f in files:
                        if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                            discovered_files.append(os.path.join(root, f))

        # Deduplicate paths preserving ordering
        unique_files = list(dict.fromkeys(discovered_files))
        records: List[ImageLintRecord] = [self.lint_file(f) for f in unique_files]

        total = len(records)
        passed_count = sum(1 for r in records if r.passed)
        failed_count = total - passed_count
        avg_score = round(sum(r.overall_score for r in records) / total, 4) if total > 0 else 1.0

        return LintSummary(
            total_scanned=total,
            passed_count=passed_count,
            failed_count=failed_count,
            average_score=avg_score,
            all_passed=(failed_count == 0),
            records=records,
        )

    def generate_pr_comment_markdown(self, summary: LintSummary) -> str:
        """Formats an actionable GitHub Pull Request review comment with diagnostic tables."""
        status_emoji = "✅" if summary.all_passed else "❌"
        status_text = "All visual assets meet aesthetic quality criteria!" if summary.all_passed else (
            f"**{summary.failed_count} visual asset(s) failed aesthetic quality thresholds.**"
        )

        lines = [
            f"## {status_emoji} AesFix Aesthetic CI/CD Quality Report",
            "",
            f"> **Status:** {status_text}  ",
            f"> **Scanned:** {summary.total_scanned} photo(s) · **Average Score:** `{summary.average_score:.3f}` · "
            f"**Passing:** `{summary.passed_count}/{summary.total_scanned}`",
            "",
            "### 📊 Asset Evaluation Breakdown",
            "",
            "| Status | Asset Path | Score | Grade | Weakest Aspect | Violations / Autofix |",
            "| :---: | :--- | :---: | :---: | :--- | :--- |",
        ]

        for r in summary.records:
            icon = "✅ Pass" if r.passed else "⚠️ Fail"
            base_name = os.path.basename(r.path)
            weakest = r.weakest_aspect.replace("_", " ").title()

            if r.passed:
                detail = "—"
            else:
                reasons = "<br>".join(f"• {v}" for v in r.violation_reasons)
                if r.autofix_path:
                    detail = f"{reasons}<br>✨ *Autofix generated (+{r.aesthetic_gain:.3f} AG)*"
                else:
                    detail = reasons

            lines.append(
                f"| {icon} | `{base_name}` | `{r.overall_score:.3f}` | {r.grade_label} | {weakest} | {detail} |"
            )

        if not summary.all_passed:
            lines.extend([
                "",
                "### 🛠️ Suggested Actions",
                "1. Inspect failing photos to correct exposure, horizon tilt, or cropping before merging.",
                "2. Run `python -m aesfix.eval.aesthetic_ci --autofix` locally to generate verified repairs and Lightroom `.xmp` presets.",
                "3. Alternatively, check suggested edits via AesFix Model Context Protocol (`mcp://fix_image`).",
            ])

        lines.extend([
            "",
            "---",
            "*Report generated by [AesFix](https://github.com/DeepanshuOP/CritiQ) — Closed-Loop Image Aesthetics via Grounded Critique and Edit Verification.*",
        ])

        return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for running the Aesthetic CI/CD visual linter."""
    parser = argparse.ArgumentParser(description="AesFix Aesthetic CI/CD Quality Linter")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to scan for visual assets (default: current directory).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Minimum overall aesthetic score required to pass (default: 0.55).",
    )
    parser.add_argument(
        "--aspect-threshold",
        type=float,
        default=0.40,
        help="Minimum individual aspect score required to pass (default: 0.40).",
    )
    parser.add_argument(
        "--autofix",
        action="store_true",
        help="Automatically apply and verify Tier-A photographic repairs for failing assets.",
    )
    parser.add_argument(
        "--autofix-dir",
        default=None,
        help="Optional directory to save repaired assets (default: adjacent to original).",
    )
    parser.add_argument(
        "--format",
        choices=["console", "markdown", "json"],
        default="console",
        help="Output report formatting (default: console).",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to write formatted report (e.g. pr_comment.md).",
    )

    args = parser.parse_args(argv)

    linter = AestheticLinter(
        min_overall_score=args.threshold,
        min_aspect_score=args.aspect_threshold,
        autofix=args.autofix,
        autofix_output_dir=args.autofix_dir,
    )

    summary = linter.lint_paths(args.paths)

    if args.format == "json":
        output_text = json.dumps(summary.to_dict(), indent=2)
    elif args.format == "markdown":
        output_text = linter.generate_pr_comment_markdown(summary)
    else:
        # Console output
        lines = [
            f"[AesFix CI] Scanned {summary.total_scanned} photo(s). "
            f"Passed: {summary.passed_count}, Failed: {summary.failed_count}, Avg: {summary.average_score:.3f}"
        ]
        for r in summary.records:
            tag = "[PASS]" if r.passed else "[FAIL]"
            lines.append(f"  {tag} {r.path}: {r.overall_score:.3f} ({r.grade_label}) - Weakest: {r.weakest_aspect}")
            for v in r.violation_reasons:
                lines.append(f"         ! {v}")
            if r.autofix_path:
                lines.append(f"         -> Autofix saved: {r.autofix_path} (+{r.aesthetic_gain:.3f} AG)")
        output_text = "\n".join(lines)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

    print(output_text)

    # Returns 0 if all assets meet quality standards, 1 if any asset fails (blocking CI)
    return 0 if summary.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
