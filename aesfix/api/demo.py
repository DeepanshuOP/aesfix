"""Gradio v2 Internal Demo & End-to-End Pipeline (AF-1.4).

Why this demo satisfies AF-1.4 and Core Features F1.1, F1.2, F1.6:
1. Replaces the legacy monolithic app.py with a modular pipeline combining:
   - Aspect Critic (AF-1.1): 5-dimensional ratings, qualitative rationales, and falsifiable fix plans.
   - Differentiable ANFIS Head (AF-1.2): overall aesthetic score + fired IF-THEN rules trace.
   - Frozen Judge Ensemble (AF-0.4): cross-judge disagreement variance and uncertainty honesty badges.
2. Visualizes the 5-aspect profile via an automated radar chart.
3. DoD: Operates end-to-end in <8s per image on standard CPU dev boxes.
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to ensure thread-safe headless rendering
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from aesfix.critic import CriticFix, CriticOutput, HeuristicAspectCritic
from aesfix.data.loaders import ASPECT_DIMENSIONS
from aesfix.head import ANFISHead, RulebookTrace
from aesfix.judges import JudgeEnsemble, JudgeResult, create_default_ensemble

logger = logging.getLogger("aesfix.demo")


@dataclass
class PipelineResult:
    """Comprehensive diagnostic result produced by the AesFix core pipeline."""
    image_path: str
    overall_score: float
    grade_label: str
    aspect_scores: Dict[str, float]
    anfis_trace: RulebookTrace
    critic_output: CriticOutput
    judge_result: JudgeResult
    candidate_fixes: List[CriticFix]
    radar_image: Optional[Image.Image]
    latency_seconds: float
    uncertainty_badge: str


def compute_grade_label(score: float) -> str:
    """Convert continuous [0.0, 1.0] aesthetic rating to intuitive letter grade."""
    if score >= 0.75:
        return "EXCELLENT (A)"
    elif score >= 0.60:
        return "GOOD (B)"
    elif score >= 0.45:
        return "MEDIOCRE (C)"
    else:
        return "NEEDS FIX (D)"


def generate_radar_chart(aspect_scores: Dict[str, float]) -> Image.Image:
    """Generate a 5-spoke photographic aspect radar plot.

    Why radar visualization matters:
    Enables photographers to instantly recognize the visual profile of their image
    (e.g., strong composition paired with severe lighting underexposure).
    """
    labels = [dim.replace("_", " ").title() for dim in ASPECT_DIMENSIONS]
    values = [aspect_scores.get(dim, 0.5) for dim in ASPECT_DIMENSIONS]

    num_vars = len(labels)
    # Compute angle for each spoke, closing the polygon by repeating the first angle
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True), dpi=120)
    fig.patch.set_facecolor("#18181b")  # Dark theme background
    ax.set_facecolor("#27272a")

    ax.plot(angles_closed, values_closed, color="#38bdf8", linewidth=2.5, linestyle="solid")
    ax.fill(angles_closed, values_closed, color="#38bdf8", alpha=0.35)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#e4e4e7", fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], color="#a1a1aa", fontsize=8)
    ax.grid(color="#52525b", linestyle="--", alpha=0.6)
    ax.spines["polar"].set_color("#52525b")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


# Module-level singletons for fast re-use across demo requests
_GLOBAL_CRITIC: Optional[HeuristicAspectCritic] = None
_GLOBAL_HEAD: Optional[ANFISHead] = None
_GLOBAL_ENSEMBLE: Optional[JudgeEnsemble] = None


def get_demo_components() -> tuple[HeuristicAspectCritic, ANFISHead, JudgeEnsemble]:
    """Lazy initialize and cache pipeline components to avoid re-instantiation latency."""
    global _GLOBAL_CRITIC, _GLOBAL_HEAD, _GLOBAL_ENSEMBLE
    if _GLOBAL_CRITIC is None:
        _GLOBAL_CRITIC = HeuristicAspectCritic()
    if _GLOBAL_HEAD is None:
        _GLOBAL_HEAD = ANFISHead(num_inputs=5, num_rules=20, monotonic=True)
        _GLOBAL_HEAD.eval()
    if _GLOBAL_ENSEMBLE is None:
        _GLOBAL_ENSEMBLE = create_default_ensemble(include_heavy_models=False)
    return _GLOBAL_CRITIC, _GLOBAL_HEAD, _GLOBAL_ENSEMBLE


def run_aesfix_pipeline(
    image_input: Union[str, Image.Image],
    user_critique: Optional[str] = None,
    genre: Optional[str] = None,
) -> PipelineResult:
    """Execute the complete AesFix diagnostic pipeline on an image.

    DoD constraint: Must complete in <8s on developer machine.
    """
    start_time = time.time()
    temp_file_to_clean = None

    if isinstance(image_input, str):
        image_path = image_input
    else:
        # Materialize PIL Image to temporary file for components expecting disk paths
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            image_input.convert("RGB").save(tf.name, quality=95)
            image_path = tf.name
            temp_file_to_clean = tf.name

    try:
        critic, head, ensemble = get_demo_components()

        # 1. Critic stage (aspect ratings, diagnostic rationales, candidate fixes)
        critic_out = critic.critique(image_path, user_critique=user_critique, genre=genre)

        # 2. ANFIS Interpretable Head stage (neuro-fuzzy rule aggregation and trace)
        anfis_trace = head.explain(critic_out.aspect_scores, top_k=3)
        overall_score = anfis_trace.overall_score
        grade_label = compute_grade_label(overall_score)

        # 3. Frozen Judge Ensemble stage (objective validation + disagreement variance)
        judge_res = ensemble.evaluate(image_path)

        # Disagreement badge signals honesty when judges diverge on subjective taste
        if judge_res.is_uncertain:
            uncertainty_badge = f"⚠️ High Aesthetic Disagreement (Var: {judge_res.disagreement:.3f}) — Taste-Dependent"
        else:
            uncertainty_badge = f"✅ Judge Consensus (Var: {judge_res.disagreement:.3f}) — High Confidence"

        # 4. Visualization stage
        radar_img = generate_radar_chart(critic_out.aspect_scores)

        elapsed = round(time.time() - start_time, 2)
        logger.info("AesFix pipeline completed in %s seconds for %s", elapsed, image_path)

        return PipelineResult(
            image_path=image_path,
            overall_score=overall_score,
            grade_label=grade_label,
            aspect_scores=critic_out.aspect_scores,
            anfis_trace=anfis_trace,
            critic_output=critic_out,
            judge_result=judge_res,
            candidate_fixes=critic_out.fixes,
            radar_image=radar_img,
            latency_seconds=elapsed,
            uncertainty_badge=uncertainty_badge,
        )
    finally:
        if temp_file_to_clean and os.path.exists(temp_file_to_clean):
            try:
                os.remove(temp_file_to_clean)
            except OSError:
                pass


def build_demo_app():
    """Build the Gradio Blocks UI for interactive inspection."""
    import gradio as gr

    with gr.Blocks(title="AesFix — Grounded Photo Critic") as demo:
        gr.Markdown(
            "# 📸 AesFix — Grounded Photo Critic (v2 Demo)\n"
            "*The critic that proves its advice works — edit-verified, interpretable, objective.*"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Photograph")
                genre_dropdown = gr.Dropdown(
                    choices=["general", "portrait", "landscape", "street", "listing", "dating"],
                    value="general",
                    label="Goal / Genre Context",
                )
                user_critique_text = gr.Textbox(
                    placeholder="Optional: Enter a critique or observation (e.g. 'lighting feels flat')",
                    label="User Observation (Multimodal Mode)",
                    lines=2,
                )
                analyze_btn = gr.Button("🔍 Diagnose & Recommend Fixes", variant="primary")

            with gr.Column(scale=1):
                verdict_md = gr.Markdown("### Diagnosis Pending")
                radar_plot = gr.Image(type="pil", label="5-Aspect Radar Profile")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🧠 Neuro-Fuzzy Rulebook Trace (ANFIS Head C3)")
                rules_trace_md = gr.Markdown("No analysis yet.")

            with gr.Column():
                gr.Markdown("### 🛠️ Proposed Falsifiable Fixes (Executor Vocabulary)")
                fixes_md = gr.Markdown("No fixes proposed yet.")

        def _ui_predict(img, genre, critique):
            if img is None:
                return "### Please upload an image first.", None, "", ""
            res = run_aesfix_pipeline(img, user_critique=critique, genre=genre)

            verdict_text = f"""
### Overall Grade: **{res.grade_label}** (Score: `{res.overall_score:.2f}/1.00`)
**Latency:** `{res.latency_seconds}s` | **Ensemble Score:** `{res.judge_result.overall_score:.2f}`

> {res.uncertainty_badge}

**Summary:** {res.critic_output.overall_summary}
"""
            # Format candidate fixes
            if res.candidate_fixes:
                fixes_lines = []
                for idx, fix in enumerate(res.candidate_fixes, 1):
                    fixes_lines.append(
                        f"**Fix {idx}: `{fix.op}`** (Targets: `{fix.expected_aspect}`)\n"
                        f"- *Instruction:* {fix.nl_instruction}\n"
                        f"- *Parameters:* `{fix.params}`"
                    )
                fixes_text = "\n\n".join(fixes_lines)
            else:
                fixes_text = "✅ No critical fixes required — image meets quality thresholds across all aspects."

            rules_text = f"```\n{res.anfis_trace.rulebook_summary}\n```"

            return verdict_text, res.radar_image, rules_text, fixes_text

        analyze_btn.click(
            fn=_ui_predict,
            inputs=[input_image, genre_dropdown, user_critique_text],
            outputs=[verdict_md, radar_plot, rules_trace_md, fixes_md],
        )

    return demo


if __name__ == "__main__":
    app = build_demo_app()
    app.launch(server_name="127.0.0.1", server_port=7860)
