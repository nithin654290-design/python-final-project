"""
Streamlit front end for the GA-CNN skin lesion classifier.

    streamlit run app.py
"""

from __future__ import annotations

import io
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from model import build_model
from preprocess import (
    CLASS_LABELS,
    CLASS_NAMES,
    CLASS_RISK,
    clean_image,
    eval_transforms,
)

CHECKPOINT_PATH = os.environ.get("SKIN_MODEL_PATH", "best_skin_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RISK_COLORS = {"Malignant": "#d62828", "Pre-cancerous": "#f77f00", "Benign": "#2a9d8f"}

st.set_page_config(page_title="Skin Lesion Classifier (GA-CNN)", page_icon="🔬", layout="centered")


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> Tuple[torch.nn.Module, List[str], Dict]:
    """Load the trained checkpoint once per session."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE).eval()
    return model, checkpoint.get("class_names", CLASS_NAMES), config


def predict(model: torch.nn.Module, image: Image.Image) -> np.ndarray:
    """Run the DullRazor + CLAHE pipeline and return class probabilities."""
    import cv2

    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    cleaned = clean_image(bgr)
    pil = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    tensor = eval_transforms()(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)[0]
    return probabilities.cpu().numpy()


st.title("🔬 Skin Lesion Classification")
st.caption("Novel Skin Disease Classification Model Using a GA-CNN Hybrid Architecture")

try:
    model, class_names, config = load_model(CHECKPOINT_PATH)
except FileNotFoundError:
    st.error(
        f"Model checkpoint `{CHECKPOINT_PATH}` not found. "
        "Run `python ga_optimizer.py` then `python train.py` first."
    )
    st.stop()

with st.sidebar:
    st.subheader("Model configuration")
    st.json({key: value for key, value in config.items() if key != "ga_settings"})
    st.caption(f"Device: {DEVICE}")

uploaded = st.file_uploader("Upload a dermoscopic lesion image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a JPG or PNG image of a skin lesion to get a prediction.")
    st.stop()

image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
st.image(image, caption="Uploaded image", width=320)

with st.spinner("Analyzing lesion..."):
    probabilities = predict(model, image)

order = np.argsort(probabilities)[::-1]
top_index = int(order[0])
top_key = class_names[top_index]
confidence = float(probabilities[top_index]) * 100
risk = CLASS_RISK.get(top_key, "Benign")
color = RISK_COLORS[risk]

st.markdown(
    f"""
    <div style="background:{color};padding:22px;border-radius:12px;text-align:center;color:white;">
        <div style="font-size:14px;letter-spacing:2px;opacity:.85;">PREDICTED DIAGNOSIS</div>
        <div style="font-size:30px;font-weight:700;margin:6px 0;">{CLASS_LABELS[top_key]}</div>
        <div style="font-size:18px;">Confidence: {confidence:.1f}% &nbsp;|&nbsp; Risk: {risk}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Top-3 probabilities")
top3 = pd.DataFrame(
    {
        "Class": [CLASS_LABELS[class_names[i]] for i in order[:3]],
        "Probability": [float(probabilities[i]) for i in order[:3]],
    }
).set_index("Class")
st.bar_chart(top3)

with st.expander("All class probabilities"):
    st.dataframe(
        pd.DataFrame(
            {
                "Class": [CLASS_LABELS[name] for name in class_names],
                "Risk": [CLASS_RISK[name] for name in class_names],
                "Probability": [f"{float(p) * 100:.2f}%" for p in probabilities],
            }
        ),
        hide_index=True,
        width="stretch",
    )

st.warning(
    "Research and educational use only. This tool is not a medical device and does not "
    "replace examination by a qualified dermatologist."
)
