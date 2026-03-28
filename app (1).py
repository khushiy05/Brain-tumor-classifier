"""
Brain Tumor Classifier — Streamlit Web App
Run with:  streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown

from utils.predict import load_model, predict

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.title-block { text-align: center; padding: 2rem 0 1rem; }
.title-block h1 { font-family: 'Space Mono', monospace; font-size: 2.4rem; color: #58a6ff; letter-spacing: -1px; }
.title-block p { color: #8b949e; font-size: 1rem; margin-top: 0.3rem; }
.result-card { background: linear-gradient(135deg, #161b22, #1c2128); border: 1px solid #30363d; border-radius: 12px; padding: 1.5rem 2rem; margin-top: 1.5rem; text-align: center; }
.result-card h2 { font-family: 'Space Mono', monospace; color: #58a6ff; }
.result-card .confidence { font-size: 2.5rem; font-weight: 600; color: #3fb950; }
.warning-card { background: #2d1b1b; border: 1px solid #f85149; border-radius: 8px; padding: 1rem 1.5rem; margin-top: 1rem; color: #ff7b72; font-size: 0.88rem; }
.stButton > button { background: #238636; color: white; border: none; border-radius: 8px; padding: 0.6rem 2.5rem; font-family: 'Space Mono', monospace; font-size: 0.9rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
    <h1>🧠 Brain Tumor Classifier</h1>
    <p>Upload an MRI scan — CNN identifies tumor type in seconds</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
This app uses a **Convolutional Neural Network (CNN)** trained on ~3,000 brain MRI images to detect and classify tumors into four categories.

**Classes:**
- 🔴 Glioma
- 🟠 Meningioma
- 🟡 Pituitary
- 🟢 No Tumor

**Model:** Custom CNN · 3 Conv blocks · ~89% test accuracy

**Stack:** TensorFlow · Keras · Streamlit
""")
    st.caption("⚠️ For educational purposes only.")

MODEL_PATH = "model/brain_tumor_cnn.h5"
GDRIVE_ID  = "1FWqljdnC9IU2yl6S4X4KsfdRGSa_72g7"

@st.cache_resource(show_spinner="Loading model...")
def get_model():
    os.makedirs("model", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model (first time only)..."):
            gdown.download(id=GDRIVE_ID, output=MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = get_model()

st.markdown("#### Upload MRI Image")
uploaded = st.file_uploader("Supported formats: JPG, PNG, JPEG", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

CLASSES     = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
CLASS_EMOJI = {"Glioma": "🔴", "Meningioma": "🟠", "No Tumor": "🟢", "Pituitary": "🟡"}
CLASS_INFO  = {
    "Glioma":     "A tumor originating in glial cells of the brain or spine.",
    "Meningioma": "Usually benign tumor arising from the meninges.",
    "Pituitary":  "Tumor in the pituitary gland; often treatable.",
    "No Tumor":   "No tumor detected in the MRI scan.",
}

if uploaded:
    image = Image.open(uploaded)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(image, caption="Uploaded MRI", width=300)
    with col2:
        if st.button("🔍 Classify Tumor"):
            with st.spinner("Analysing..."):
                label, confidence, probs = predict(model, image)
            emoji = CLASS_EMOJI[label]
            st.markdown(f"""
            <div class="result-card">
                <h2>{emoji} {label}</h2>
                <div class="confidence">{confidence:.1f}%</div>
                <p style="color:#8b949e; margin-top:0.5rem;">confidence</p>
                <p style="margin-top:1rem; font-size:0.9rem;">{CLASS_INFO[label]}</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("#### Class Probabilities")
            fig, ax = plt.subplots(figsize=(5, 3), facecolor='#161b22')
            colors = ['#58a6ff' if c == label else '#30363d' for c in CLASSES]
            bars   = ax.barh(CLASSES, [p * 100 for p in probs], color=colors, height=0.5)
            ax.set_facecolor('#161b22')
            ax.tick_params(colors='#8b949e')
            ax.spines[:].set_color('#30363d')
            ax.set_xlabel('Probability (%)', color='#8b949e')
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{prob*100:.1f}%', va='center', color='#e6edf3', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("""
            <div class="warning-card">
                ⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. Always consult a licensed medical professional for diagnosis.
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#8b949e; font-size:0.82rem;'>Built with TensorFlow + Streamlit · <a href='https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri' style='color:#58a6ff;'>Dataset: Kaggle</a></p>", unsafe_allow_html=True)
