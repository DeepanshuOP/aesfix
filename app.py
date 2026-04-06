import gradio as gr
import os

# Import your system
from our_system.image_model import extract_image_features
from our_system.llm_extractor import extract_aesthetic_features
from our_system.fuzzy_system import build_fuzzy_system, evaluate_aesthetics
from our_system.aggregator import identify_aspect_driver
from our_system.explainer import generate_explanation

# Load fuzzy system once
fuzzy_sim = build_fuzzy_system()

# DARK THEME CSS mimicking a professional dashboard
CSS = """
body, .gradio-container {
    background-color: #0F172A !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* Force dark mode */
.dark, .gradio-container.dark {
    background-color: #0F172A !important;
}

/* Override Gradio elements container to fit dark mode organically */
.gradio-container *, .prose * {
    border-color: #334155 !important;
}

.header {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    margin-bottom: 2rem;
    border-bottom: 2px solid #334155;
}
.header h1 {
    color: #E2E8F0 !important;
    font-weight: 900;
    font-size: 3rem;
    margin: 0;
    letter-spacing: -0.5px;
}
.header p {
    color: #94A3B8;
    font-size: 1.15rem;
    margin-top: 0.5rem;
    font-weight: 500;
}
.panel {
    background: #1E293B !important;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.5), 0 4px 6px -2px rgba(0,0,0,0.3);
    border: 1px solid #334155 !important;
    color: #E2E8F0 !important;
}
.panel h3 {
    color: #E2E8F0 !important;
    margin-top: 0;
    border-bottom: 1px solid #334155;
    padding-bottom: 12px;
    margin-bottom: 18px;
    font-weight: 700;
    font-size: 1.25rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.score-card {
    background: #1E293B;
    border: 1px solid #334155;
    color: #E2E8F0;
    border-radius: 12px;
    padding: 35px 20px;
    text-align: center;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}
.score-card h2 {
    font-size: 1.15rem;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #94A3B8 !important;
    font-weight: 700;
}
.score-card .score {
    font-size: 5.5rem;
    font-weight: 900;
    margin: 15px 0;
    line-height: 1;
    color: #22D3EE;
    text-shadow: 0 0 25px rgba(34, 211, 238, 0.25);
}
.score-card .label-badge {
    font-size: 1.15rem;
    font-weight: 800;
    background: rgba(34, 197, 94, 0.15);
    color: #22C55E;
    padding: 8px 24px;
    border-radius: 30px;
    display: inline-block;
    border: 1px solid rgba(34, 197, 94, 0.4);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.feature-row {
    margin-bottom: 18px;
}
.feature-name {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    margin-bottom: 8px;
    color: #E2E8F0;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.feature-bar-bg {
    background: #0F172A;
    height: 14px;
    border-radius: 7px;
    overflow: hidden;
    border: 1px solid #334155;
}
.feature-bar-fill {
    background: #22D3EE;
    height: 100%;
    border-radius: 7px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 10px rgba(34, 211, 238, 0.4);
}
.explanation-panel {
    background: #0F172A;
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #22D3EE;
    color: #E2E8F0;
    font-size: 1.05rem;
    line-height: 1.7;
    margin-top: 15px;
    border-top: 1px solid #334155;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
}
.analyze-btn {
    background: #22C55E !important;
    color: #0F172A !important;
    font-weight: 900 !important;
    font-size: 1.15rem !important;
    border: none !important;
    padding: 16px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.25) !important;
    transition: all 0.2s ease !important;
    margin-top: 10px !important;
    width: 100% !important;
    letter-spacing: 1px !important;
}
.analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(34, 197, 94, 0.4) !important;
    background: #16a34a !important;
}
.error-msg {
    color: #fca5a5;
    font-weight: bolder;
    text-align: center;
    padding: 20px;
    background: rgba(239, 68, 68, 0.1);
    border-radius: 8px;
    border: 1px solid #ef4444;
    margin-bottom: 20px;
}

/* Ensure textareas and inputs look beautiful in dark mode */
textarea, input, .box {
    background-color: #0F172A !important;
    color: #E2E8F0 !important;
}
"""

def generate_score_html(label, score):
    return f"""
    <div class="score-card">
        <h2>Aesthetic Score</h2>
        <div class="score">{score:.2f}</div>
        <div class="label-badge">{label}</div>
    </div>
    """

def generate_features_html(features):
    html = '<div class="panel"><h3>Feature Breakdown</h3>'
    for k, v in features.items():
        name = k.replace('_', ' ')
        percentage = v * 100 # scaling to 0-100%
        percentage = max(0, min(100, percentage))
        html += f"""
        <div class="feature-row">
            <div class="feature-name">
                <span style="color: #94A3B8;">{name}</span>
                <span style="color: #E2E8F0;">{v:.2f}</span>
            </div>
            <div class="feature-bar-bg">
                <div class="feature-bar-fill" style="width: {percentage}%"></div>
            </div>
        </div>
        """
    html += '</div>'
    return html

def generate_explanation_html(explanation, driver):
    return f"""
    <div class="panel">
        <h3>Analysis Insight</h3>
        <p style="font-size: 1.1rem; color: #94A3B8; margin-bottom: 0;">
            <strong>Primary Driver:</strong> 
            <span style="color: #22D3EE; font-weight: 800; text-transform: uppercase; margin-left: 8px;">
                {driver.replace('_', ' ')}
            </span>
        </p>
        <div class="explanation-panel">
            {explanation}
        </div>
    </div>
    """

def analyze(image, critique):
    if image is None:
        return (
            '<div class="error-msg">Please upload an image first.</div>',
            "",
            ""
        )
    
    try:
        # Save uploaded image temporarily
        image_path = "temp.jpg"
        image.save(image_path)

        # ---- IMAGE FEATURES ----
        img_features = extract_image_features(image_path)

        # ---- TEXT FEATURES ----
        has_critique = bool(critique and critique.strip())
        if has_critique:
            text_features = extract_aesthetic_features(critique)
        else:
            text_features = None

        # ---- FUSION ----
        agg = {}
        if has_critique and text_features:
            # Multimodal: 70% text + 30% image
            for key in ["composition", "lighting", "color", "subject_clarity", "technical"]:
                agg[key] = round(
                    0.70 * text_features.get(key, 0.5) + 0.30 * img_features.get(key, 0.5), 4
                )
        else:
            # Image-only: 100% CLIP zero-shot features
            for key in ["composition", "lighting", "color", "subject_clarity", "technical"]:
                agg[key] = round(img_features.get(key, 0.5), 4)

        # ---- FUZZY ----
        label, score = evaluate_aesthetics(
            fuzzy_sim,
            agg["composition"],
            agg["lighting"],
            agg["color"],
            agg["subject_clarity"],
            agg["technical"]
        )

        # ---- DRIVER ----
        driver = identify_aspect_driver(agg, label)

        # ---- EXPLANATION ----
        explanation = generate_explanation(label, score, agg, None)

        score_html = generate_score_html(label, score)
        features_html = generate_features_html(agg)
        explanation_html = generate_explanation_html(explanation, driver)

        return score_html, features_html, explanation_html

    except Exception as e:
        error_html = f'<div class="error-msg">Error during analysis: {str(e)}</div>'
        return error_html, "", ""

# ---- UI DEFINITION ----
# Setup custom Gradio theme
theme = gr.themes.Default(
    primary_hue="cyan",
    secondary_hue="cyan",
    neutral_hue="slate"
).set(
    body_background_fill="#0F172A",
    body_text_color="#E2E8F0",
    color_accent_soft="#1E293B",
    block_background_fill="#1E293B",
    block_border_color="#334155",
    block_label_text_color="#94A3B8",
    input_background_fill="#0F172A",
    button_primary_background_fill="#22C55E",
    button_primary_text_color="#0F172A"
)

with gr.Blocks(title="CritiQ - AI Analytics") as interface:
    # Inject CSS
    gr.HTML(f"<style>{CSS}</style>")
    
    # Header
    gr.HTML("""
        <div class="header">
            <h1>CritiQ Dashboard</h1>
            <p>Diagnostic Aesthetic Evaluation Model</p>
        </div>
    """)
    
    with gr.Row(equal_height=False):
        # Left Column: Inputs
        with gr.Column(scale=4):
            gr.HTML('<div><h2 style="color: #94A3B8; font-weight: 800; font-size: 1.15rem; text-transform: uppercase;">1. Input Source</h2></div>')
            img_input = gr.Image(type="pil", label="Upload Image")
            text_input = gr.Textbox(
                label="Critique Context (Optional)", 
                placeholder="e.g., 'The lighting is very dramatic but composition feels unbalanced...'",
                lines=4
            )
            analyze_btn = gr.Button("RUN ANALYSIS ➔", elem_classes=["analyze-btn"], variant="primary")

        # Right Column: Outputs
        with gr.Column(scale=6):
            gr.HTML('<div><h2 style="color: #94A3B8; font-weight: 800; font-size: 1.15rem; text-transform: uppercase;">2. Diagnostic Results</h2></div>')
            score_output = gr.HTML(
                value='<div class="panel" style="height: auto; text-align: center; color: #94A3B8; padding: 60px 20px; font-size: 1.1rem;">Awaiting image source...<br><br>Upload an image and click <b>RUN ANALYSIS</b> to generate diagnostics.</div>'
            )
            features_output = gr.HTML()
            explanation_output = gr.HTML()

    # Event binding with queue
    analyze_btn.click(
        fn=analyze,
        inputs=[img_input, text_input],
        outputs=[score_output, features_output, explanation_output],
        api_name="evaluate"
    )

if __name__ == "__main__":
    # Pass theme and css to launch() to resolve Gradio 6.0 UserWarning
    interface.queue().launch(theme=theme, css=CSS)