"""
image_processing_view.py — Standalone Streamlit app showing the full
9-stage preprocessing pipeline AND the Grad-CAM visual explanation.

Run with:
    streamlit run image_processing_view.py

This complements app.py. It is optimised for dual-screen or demo use:
the entire page is the visual pipeline + Grad-CAM, with no prediction UI.
"""

import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from explainability import run_gradcam
from preprocessing import (
    convert_to_rgb,
    expand_batch_dim,
    image_to_array,
    is_likely_leaf,
    normalize_array,
    resize_image,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriScan · Processing Pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#18332a; --muted:#6f8177; --line:#e5eee8; --cream:#f7faf5; --green:#16724d; --purple:#7c3aed; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
[data-testid="stAppViewContainer"] { background:var(--cream); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#fbfdfb; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding:2rem 1.15rem; }
.block-container { max-width:1320px; padding:2.8rem 3.5rem 3rem; }
h1,h2,h3,h4 { font-family:'Space Grotesk', sans-serif; letter-spacing:-.04em; color:var(--ink); }
.stage-header { background:white; border:1px solid var(--line); border-radius:18px;
  padding:1rem 1.2rem; margin-bottom:.6rem;
  box-shadow:0 6px 20px rgba(25,68,46,.05); }
.stage-header .kicker { color:var(--green); font-size:.72rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.14em; }
.stage-header .title { font:600 1.05rem 'Space Grotesk'; margin:.2rem 0 .1rem; }
.stage-header .desc { color:var(--muted); font-size:.85rem; line-height:1.55; }
.gradcam-wrap { background:#0b1a13; border-radius:22px; padding:1.6rem 1.8rem;
  border:1px solid #1d3527; margin-top:1rem; }
.gradcam-wrap h3 { color:#a8dbb8; font-family:'Space Grotesk'; }
.gradcam-wrap p { color:#6a9b7e; font-size:.88rem; }
.stButton > button { border-radius:12px; border:0; background:var(--green); color:white;
  font-weight:700; min-height:2.8rem; transition:all 160ms; box-shadow:0 6px 14px rgba(22,114,77,.18); }
.stButton > button:hover { background:#105d3e; transform:translateY(-1px); }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "agriscan_model.keras")
    indices_path = os.path.join(base_dir, "class_indices.json")
    model = tf.keras.models.load_model(model_path) if os.path.exists(model_path) else None
    class_names = {}
    if os.path.exists(indices_path):
        with open(indices_path, "r", encoding="utf-8") as fh:
            class_names = {v: k for k, v in json.load(fh).items()}
    return model, class_names


model, class_names = get_model()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔬 Pipeline Explorer")
    st.caption("Upload a leaf image to walk through every preprocessing stage and see the Grad-CAM explanation.")
    st.divider()

    uploaded_file = st.file_uploader(
        "Leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Grad-CAM settings**")
    gradcam_alpha = st.slider("Heatmap intensity", 0.2, 0.8, 0.4, 0.05,
                              help="Alpha blend of heatmap over the original image")
    show_raw_heatmap = st.checkbox("Show raw heatmap alongside overlay", value=True)

    st.divider()
    st.markdown("**Display**")
    show_histograms = st.checkbox("Show pixel histograms", value=True)
    show_augmentation = st.checkbox("Show augmentation preview", value=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.markdown("## 🔬 AgriScan · Image Processing Pipeline")
st.caption(
    "Every transformation your leaf image goes through before reaching the AI model "
    "— visualised side by side."
)

if not uploaded_file:
    st.info("Upload a leaf image in the sidebar to begin.")
    st.stop()

if model is None or not class_names:
    st.warning("Model file not found. Grad-CAM will be skipped; preprocessing stages will still display.")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
image = Image.open(uploaded_file)
rgb_image, original_mode, was_converted = convert_to_rgb(image)
is_leaf, green_ratio, green_mask = is_likely_leaf(rgb_image)
resized_image = resize_image(rgb_image)
img_array = image_to_array(resized_image)
normed_array = normalize_array(img_array)
batched_array = expand_batch_dim(normed_array)

if not is_leaf:
    st.warning(
        f"Low leaf signal ({green_ratio*100:.0f}%). "
        "This image may not contain a clear leaf — Grad-CAM may be less meaningful."
    )

w, h = image.size

# ---------------------------------------------------------------------------
# Helper: section header HTML
# ---------------------------------------------------------------------------

def section_header(num: str, title: str, desc: str):
    st.markdown(
        f'<div class="stage-header">'
        f'<div class="kicker">Stage {num}</div>'
        f'<div class="title">{title}</div>'
        f'<div class="desc">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Stage 1 & 2: Original + RGB conversion ─────────────────────────────────
st.divider()
col1, col2 = st.columns(2, gap="large")
with col1:
    section_header("1", "Original upload",
                   f"Source image as received. Size: {w} × {h} px · Mode: {image.mode}")
    st.image(image, use_container_width=True)
with col2:
    section_header("2", "RGB conversion",
                   f"Mode: {original_mode} → RGB" if was_converted else
                   "Already in RGB — no conversion needed.")
    st.image(rgb_image, use_container_width=True)

# ── Stage 3: Leaf detection mask ────────────────────────────────────────────
st.divider()
col3, col4 = st.columns(2, gap="large")
with col3:
    section_header("3", "Leaf detection mask",
                   f"Green-dominant + yellow-brown pixels highlighted. "
                   f"Signal: {green_ratio*100:.1f}% — {'✓ PASS (≥10%)' if is_leaf else '✗ FAIL (<10%)'}")
    st.image(green_mask.astype(np.uint8) * 255, use_container_width=True,
             caption="White = detected leaf pixels")
with col4:
    section_header("4", "Resize → 224 × 224",
                   f"Bicubic-downsampled from {w} × {h} to model input resolution 224 × 224 px.")
    st.image(resized_image, use_container_width=True)

# ── Stage 5: Normalisation ──────────────────────────────────────────────────
st.divider()
section_header("5", "Normalisation [0, 1]",
               "Pixel values divided by 255 → float32 range [0.0, 1.0]. "
               "Appearance is identical; only tensor values change.")
norm_display = (normed_array * 255).astype(np.uint8)
col5a, col5b = st.columns(2, gap="large")
with col5a:
    st.image(norm_display, use_container_width=True, caption="Normalised image (display-scaled back to uint8)")
with col5b:
    if show_histograms:
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
        axes[0].hist(img_array.ravel(), bins=48, color="#6e9fd0")
        axes[0].set_title("Raw pixels [0–255]", fontsize=9)
        axes[1].hist(normed_array.ravel(), bins=48, color="#6ab47b")
        axes[1].set_title("Normalised [0–1]", fontsize=9)
        for ax in axes:
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            ax.tick_params(labelsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Enable 'Show pixel histograms' in the sidebar to display histograms.")

# ── Stage 6/7/8: RGB channel splits ─────────────────────────────────────────
st.divider()
section_header("6–8", "RGB channel splits",
               "Each colour channel isolated. Healthy tissue dominates the green channel. "
               "Disease often appears as elevated red or suppressed green.")
r_only = np.zeros_like(img_array); r_only[:, :, 0] = img_array[:, :, 0]
g_only = np.zeros_like(img_array); g_only[:, :, 1] = img_array[:, :, 1]
b_only = np.zeros_like(img_array); b_only[:, :, 2] = img_array[:, :, 2]
ch_col1, ch_col2, ch_col3 = st.columns(3, gap="medium")
with ch_col1: st.image(r_only, caption="🔴 Red channel", use_container_width=True)
with ch_col2: st.image(g_only, caption="🟢 Green channel", use_container_width=True)
with ch_col3: st.image(b_only, caption="🔵 Blue channel", use_container_width=True)

# ── Stage 9: Batch expansion summary ────────────────────────────────────────
st.divider()
section_header("9", "Batch expansion",
               f"Array expanded from {img_array.shape} → {batched_array.shape}. "
               f"Model input ready · range [{batched_array.min():.3f}, {batched_array.max():.3f}]")
st.success(f"✓ Model input tensor ready — shape `{batched_array.shape}` · dtype `float32`")

# ── Augmentation preview (optional) ─────────────────────────────────────────
if show_augmentation:
    st.divider()
    section_header("★", "Augmentation preview (training-only)",
                   "Illustrative augmentations applied during training. Not used at inference time.")
    aug_gen = ImageDataGenerator(rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    aug_iter = aug_gen.flow(np.expand_dims(img_array.astype(np.float32), 0), batch_size=1)
    aug_cols = st.columns(4)
    for i, acol in enumerate(aug_cols):
        with acol:
            st.image(next(aug_iter)[0].astype(np.uint8), caption=f"Aug {i+1}", use_container_width=True)

# ---------------------------------------------------------------------------
# Grad-CAM section
# ---------------------------------------------------------------------------
st.divider()
st.markdown("## 🧠 Grad-CAM Visual Explanation")
st.caption(
    "Gradient-weighted Class Activation Mapping: shows **which leaf regions** "
    "most influenced the model's prediction. Red/yellow = high activation · Blue = low."
)

if model is None:
    st.warning("Model not loaded — skipping Grad-CAM.")
else:
    with st.spinner("Running Grad-CAM…"):
        try:
            # Run prediction to get class index
            raw_pred = model.predict(batched_array, verbose=0)[0]
            pred_idx  = int(np.argmax(raw_pred))
            pred_cls  = class_names.get(pred_idx, "Unknown")
            pred_conf = float(np.max(raw_pred)) * 100

            # Run Grad-CAM
            gradcam_result = run_gradcam(
                model=model,
                original_image_rgb=rgb_image,
                preprocessed_array=batched_array,
                predicted_class_index=pred_idx,
                alpha=gradcam_alpha,
            )

            st.markdown(
                f'<div class="gradcam-wrap">'
                f'<h3>Prediction: {pred_cls.replace("___"," · ").replace("_"," ")} '
                f'({pred_conf:.1f}% confidence)</h3>'
                f'<p>Conv layer used: <code style="background:transparent;color:#a8dbb8">'
                f'{gradcam_result["layer_used"]}</code> · Heatmap blend: {gradcam_alpha:.0%}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if show_raw_heatmap:
                gc_col1, gc_col2 = st.columns(2, gap="large")
                with gc_col1:
                    st.image(
                        gradcam_result["overlay_image"],
                        caption="Heatmap overlay — regions that drove the prediction",
                        use_container_width=True,
                    )
                with gc_col2:
                    raw_u8  = (gradcam_result["heatmap"] * 255).astype(np.uint8)
                    raw_jet = cv2.applyColorMap(raw_u8, cv2.COLORMAP_JET)
                    raw_rgb = cv2.cvtColor(raw_jet, cv2.COLOR_BGR2RGB)
                    st.image(
                        raw_rgb,
                        caption="Raw Grad-CAM heatmap (false-colour, no original)",
                        use_container_width=True,
                    )
            else:
                st.image(
                    gradcam_result["overlay_image"],
                    caption="Heatmap overlay — regions that drove the prediction",
                    use_container_width=True,
                )

            # Legend
            st.markdown(
                '<div style="display:flex;align-items:center;gap:12px;margin-top:.6rem">'
                '<span style="font-size:.78rem;color:#6f8177">Low activation</span>'
                '<div style="flex:1;height:10px;border-radius:99px;'
                'background:linear-gradient(90deg,#00f,#0ff,#0f0,#ff0,#f00)"></div>'
                '<span style="font-size:.78rem;color:#6f8177">High activation</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"Grad-CAM failed: {exc}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "AgriScan · Standalone pipeline & Grad-CAM viewer · "
    "Transfer-learning plant disease detection. "
    "For decision support, not a substitute for an agronomist."
)
