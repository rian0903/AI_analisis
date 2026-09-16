"""
streamlit_app.py
================
Streamlit web interface for the Multi-Feature Content Moderation System.
"""

import io
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CLASS_NAMES, MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = "models/final_model.keras"

CLASS_CONFIG = {
    "Safe": {
        "color": "#2ecc71",
        "bg": "#1a3d2b",
        "emoji": "✅",
        "badge_text": "SAFE CONTENT",
    },
    "Violence": {
        "color": "#e67e22",
        "bg": "#3d2b1a",
        "emoji": "⚠️",
        "badge_text": "VIOLENT CONTENT",
    },
    "NSFW": {
        "color": "#e74c3c",
        "bg": "#3d1a1a",
        "emoji": "🔞",
        "badge_text": "NSFW CONTENT",
    },
}

st.set_page_config(
    page_title="Multi-Feature Content Moderation AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 1.5rem 0 1rem;
}

.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea, #764ba2, #f64f59);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.main-subtitle {
    color: #8892b0;
    font-size: 1rem;
}

.glass-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.2rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.8rem 2rem;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    border: 2px solid;
    margin: 1rem 0;
}

.section-title {
    color: #ccd6f6;
    font-size: 1.1rem;
    font-weight: 600;
    border-left: 3px solid #667eea;
    padding-left: 0.8rem;
    margin: 1rem 0 0.8rem;
}

.pipeline-tag {
    background: rgba(102, 126, 234, 0.2);
    border: 1px solid #667eea;
    color: #829bf8;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Multi-Feature AI Model…")
def load_predictor(model_path: str):
    try:
        from src.predict import ImagePredictor
        return ImagePredictor(model_path=model_path)
    except Exception as exc:
        logger.warning("Predictor loading error: %s", exc)
        return None


@st.cache_data(show_spinner="Computing Grad-CAM…")
def compute_gradcam(_model, input_tensors: dict, class_index: int):
    from src.utils import GradCAM
    try:
        grad_cam = GradCAM(_model)
        heatmap = grad_cam.compute_heatmap(input_tensors, class_index=class_index)
        orig_uint8 = np.uint8(np.squeeze(input_tensors["input_global"]) * 255)
        colorised, overlay = grad_cam.overlay_heatmap(heatmap, orig_uint8)
        return orig_uint8, colorised, overlay
    except Exception as exc:
        logger.warning("Grad-CAM failed: %s", exc)
        return None, None, None


def build_probability_chart(probabilities: dict, predicted_class: str) -> go.Figure:
    colors_map = {"Safe": "#2ecc71", "Violence": "#e67e22", "NSFW": "#e74c3c"}
    classes = list(probabilities.keys())
    values = [probabilities[c] * 100 for c in classes]
    bar_colors = [colors_map.get(c, "#667eea") for c in classes]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=classes,
            orientation="h",
            marker=dict(color=bar_colors),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color="white", size=13),
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 110], gridcolor="rgba(255,255,255,0.07)", tickfont=dict(color="#8892b0")),
        yaxis=dict(tickfont=dict(color="#ccd6f6", size=14)),
        margin=dict(l=10, r=60, t=10, b=10),
        height=180,
    )
    return fig


def render_sidebar(predictor) -> None:
    with st.sidebar:
        st.markdown("<h2 style='color:#ccd6f6;font-size:1.2rem;'>🛡️ Content Moderation AI</h2>", unsafe_allow_html=True)
        st.divider()

        if predictor is not None:
            st.success("✅ Multi-Feature Model Loaded", icon="✅")
            st.markdown(
                """
                <div class="glass-card" style="font-size:0.85rem; color:#ccd6f6;">
                    <b>Backbone:</b> MobileNetV2<br>
                    <b>Branches:</b> Global + Human Crop + Pose<br>
                    <b>Classes:</b> Safe · Violence · NSFW
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.error("❌ Model not found at models/final_model.keras", icon="❌")
            st.info("Train the model first using:\n`python src/train.py`", icon="ℹ️")

        st.divider()
        st.markdown("<p style='color:#8892b0;font-size:0.8rem;'>MobileNetV2 Multi-Feature Fusion Pipeline</p>", unsafe_allow_html=True)


def render_result(result: dict, predictor) -> None:
    predicted_class = result["prediction"]
    confidence = result["confidence"] * 100.0
    probabilities = result["probabilities"]
    human = result["human_analysis"]
    pipeline = result["pipeline"]
    cfg = CLASS_CONFIG[predicted_class]

    st.markdown(
        f"""
        <div style='text-align:center;'>
            <div class="result-badge" style='color:{cfg["color"]}; border-color:{cfg["color"]}; background:{cfg["bg"]};'>
                {cfg["emoji"]} {cfg["badge_text"]}
            </div>
            <br>
            <span class="pipeline-tag">Pipeline: {pipeline}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_conf, col_chart = st.columns([1, 1.8])

    with col_conf:
        st.markdown(
            f"""
            <div class="glass-card" style='text-align:center;'>
                <div style='color:#8892b0;font-size:0.85rem;'>CONFIDENCE SCORE</div>
                <div style='font-size:2.4rem;font-weight:700;color:{cfg["color"]};'>{confidence:.1f}%</div>
                <div style='color:#ccd6f6;font-weight:600;'>{predicted_class}</div>
                <div style='color:#8892b0;font-size:0.8rem;margin-top:0.4rem;'>
                    Inference: {result["inference_time_ms"]} ms
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="glass-card" style="font-size:0.85rem; color:#ccd6f6;">
                <b>Human Detection:</b> {"Yes (" + str(human["person_count"]) + " person)" if human["person_detected"] else "No"}<br>
                <b>Pose Landmark:</b> {"Detected" if human["pose_detected"] else "None / Fallback"}<br>
                <b>Low Confidence Flag:</b> {"⚠️ Yes" if result.get("low_confidence") else "No"}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_chart:
        st.markdown("<div class='section-title'>Class Probability Distribution</div>", unsafe_allow_html=True)
        fig = build_probability_chart(probabilities, predicted_class)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.divider()

    st.markdown("<div class='section-title'>💡 AI Explanation & Semantic Analysis</div>", unsafe_allow_html=True)
    if "explanation_summary" in result:
        st.info(result["explanation_summary"], icon="ℹ️")

    st.divider()

    st.markdown("<div class='section-title'>🔬 Grad-CAM Explainability</div>", unsafe_allow_html=True)
    if "input_tensors" in result:
        class_idx = CLASS_NAMES.index(predicted_class)
        orig_uint8, colorised, overlay = compute_gradcam(predictor.model, result["input_tensors"], class_idx)

        if orig_uint8 is not None:
            tabs = st.tabs(["📷 Original", "🌡️ Heatmap", "🎨 Overlay"])
            with tabs[0]:
                st.image(orig_uint8, caption="Original Image", use_column_width=True)
            with tabs[1]:
                st.image(colorised, caption="Grad-CAM Heatmap", use_column_width=True)
            with tabs[2]:
                st.image(overlay, caption="Heatmap Overlay", use_column_width=True)


def main() -> None:
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">🛡️ Content Moderation AI</div>
            <div class="main-subtitle">MobileNetV2 Multi-Feature Classification Architecture</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor = load_predictor(MODEL_PATH)
    render_sidebar(predictor)

    col_upload, col_result = st.columns([1, 1.6], gap="large")

    with col_upload:
        st.markdown("<div class='section-title'>📁 Upload Image</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(label="Browse image", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")

        if uploaded_file is not None:
            pil_image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
            st.image(pil_image, caption=f"📄 {uploaded_file.name}", use_column_width=True)
            img_np = np.array(pil_image, dtype=np.float32) / 255.0

    with col_result:
        st.markdown("<div class='section-title'>🔍 Analysis Results</div>", unsafe_allow_html=True)

        if uploaded_file is None:
            st.markdown(
                """
                <div class="glass-card" style='text-align:center;padding:3rem 1.5rem;'>
                    <div style='font-size:3rem'>🖼️</div>
                    <div style='color:#8892b0;margin-top:0.8rem;'>Upload an image to start analysis</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif predictor is None:
            st.error("Model not available. Please train the model first.")
        else:
            with st.spinner("Analyzing image features…"):
                try:
                    result = predictor.predict_array(img_np, source_name=uploaded_file.name)
                    render_result(result, predictor)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()
