"""
AgriScan — professional plant disease detection interface.
Run with: streamlit run app.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from disease_info import disease_info
from preprocessing import (
    convert_to_rgb,
    expand_batch_dim,
    image_to_array,
    is_likely_leaf,
    normalize_array,
    preprocess_for_prediction,
    resize_image,
)

DEBUG_MODE = os.environ.get("AGRISCAN_DEBUG") == "1"

st.set_page_config(
    page_title="AgriScan | Plant health intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#18332a; --muted:#6f8177; --line:#e5eee8; --cream:#f7faf5; --green:#16724d; --mint:#e9f5ed; --gold:#e9b949; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
[data-testid="stAppViewContainer"] { background:var(--cream); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#fbfdfb; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding:2rem 1.15rem; }
.block-container { max-width:1320px; padding:2.8rem 3.5rem 3rem; }
h1,h2,h3,h4 { font-family:'Space Grotesk', sans-serif; letter-spacing:-.04em; color:var(--ink); }
h1 { font-size:clamp(2.3rem,4vw,4rem); line-height:1.02; margin-bottom:.8rem; }
h2 { font-size:2rem; }
.agri-brand { display:flex; align-items:center; gap:.7rem; margin-bottom:2.6rem; }
.agri-mark { width:38px;height:38px;border-radius:13px;background:#dff2e5;display:grid;place-items:center;font-size:1.35rem; }
.agri-brand strong { font:700 1.1rem 'Space Grotesk'; }
.agri-brand span { display:block; font-size:.68rem; color:var(--muted); letter-spacing:.11em; text-transform:uppercase; margin-top:2px; }
.eyebrow { color:var(--green); text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:700; margin-bottom:1rem; }
.hero-copy { max-width:700px; }
.hero-copy p { font-size:1.07rem; line-height:1.7; color:var(--muted); max-width:610px; }
.hero-badge { display:inline-flex; gap:.5rem; align-items:center; padding:.45rem .7rem; background:#eaf5ed; color:var(--green); border-radius:999px; font-size:.76rem; font-weight:700; margin-bottom:1.35rem; }
.hero-badge i { display:block;width:7px;height:7px;border-radius:50%;background:#42a66d; box-shadow:0 0 0 4px #d2edda; }
.section-card { background:white; border:1px solid var(--line); border-radius:24px; padding:1.55rem; box-shadow:0 12px 36px rgba(25,68,46,.055); }
.upload-card { margin-top:2rem; }
.card-kicker { color:var(--muted); font-size:.73rem; text-transform:uppercase; letter-spacing:.14em; font-weight:700; }
.card-title { font:600 1.35rem 'Space Grotesk'; margin:.35rem 0 .25rem; }
.card-subtitle { color:var(--muted); font-size:.88rem; margin-bottom:1.2rem; }
[data-testid="stFileUploader"] { border:1.5px dashed #b9d6c3; border-radius:18px; background:#f8fcf8; padding:.55rem; }
[data-testid="stFileUploader"] section { border:0; }
.stButton > button { border-radius:12px; border:0; background:var(--green); color:white; font-weight:700; min-height:3rem; transition:all 160ms cubic-bezier(.23,1,.32,1); box-shadow:0 6px 14px rgba(22,114,77,.18); }
.stButton > button:hover { background:#105d3e; transform:translateY(-1px); box-shadow:0 10px 18px rgba(22,114,77,.22); }
.stButton > button:active { transform:scale(.97); }
[data-baseweb="select"] > div { border-color:var(--line); border-radius:12px; min-height:3rem; }
.agri-stat { padding:1rem 1.1rem; border-radius:16px; background:#f6faf6; border:1px solid #edf4ee; }
.agri-stat .label { color:var(--muted); text-transform:uppercase; letter-spacing:.1em; font-size:.65rem; font-weight:700; }
.agri-stat .value { font:700 1.35rem 'Space Grotesk'; margin-top:.35rem; }
.result-shell { margin-top:2rem; animation:rise 360ms cubic-bezier(.23,1,.32,1); }
.result-head { display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; }
.result-pill { padding:.45rem .7rem; border-radius:999px; background:#eaf5ed; color:var(--green); font-size:.73rem; font-weight:700; }
.diagnosis { font:700 clamp(1.5rem,2.4vw,2.4rem) 'Space Grotesk'; margin:.4rem 0 .8rem; }
.diagnosis.bad { color:#af4f35; }
.diagnosis.good { color:#16724d; }
.conf-track { height:10px; background:#edf2ed; border-radius:99px; overflow:hidden; margin:.8rem 0 .35rem; }
.conf-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#e7b94a,#65ad67,#16724d); width:var(--fill); animation:grow 600ms ease-out; }
.conf-label { color:var(--muted); font-size:.8rem; }
.advice-box { background:#f7fbf6; border-left:4px solid #6ab47b; border-radius:0 15px 15px 0; padding:1rem 1.1rem; height:100%; }
.advice-box strong { display:block; font:600 1rem 'Space Grotesk'; margin-bottom:.45rem; }
.advice-box p { color:#557064; font-size:.9rem; line-height:1.6; margin:0; }
.tip-box { background:#fff9e9; border:1px solid #f4e5ba; padding:.85rem 1rem; border-radius:14px; color:#765e22; font-size:.85rem; }
.footer { color:#94a59b; font-size:.75rem; text-align:center; padding:2.2rem 0 0; }
[data-testid="stImage"] { animation:rise 300ms ease-out; }
@keyframes rise { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
@keyframes grow { from{width:0} to{width:var(--fill)} }
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def get_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "agriscan_model.keras")
    indices_path = os.path.join(base_dir, "class_indices.json")
    model = tf.keras.models.load_model(model_path) if os.path.exists(model_path) else None
    class_names = {}
    if os.path.exists(indices_path):
        with open(indices_path, "r", encoding="utf-8") as handle:
            class_names = {v: k for k, v in json.load(handle).items()}
    return model, class_names

def render_confidence(confidence: float):
    st.markdown(
        f'<div class="conf-track"><div class="conf-fill" style="--fill:{min(confidence,100):.1f}%"></div></div>'
        f'<div class="conf-label">Model confidence · <strong>{confidence:.1f}%</strong></div>',
        unsafe_allow_html=True,
    )

model, class_names = get_model()

with st.sidebar:
    st.markdown('<div class="agri-brand"><div class="agri-mark">🌿</div><div><strong>AgriScan</strong><span>Plant health intelligence</span></div></div>', unsafe_allow_html=True)
    st.markdown("### Scan workspace")
    st.caption("Use a clear, well-lit leaf image for the most reliable diagnosis.")
    st.divider()
    st.markdown("**Crop focus**")
    crop_options = ["Auto-Detect (All Crops)", "Wheat", "Rice", "Potato", "Cotton", "Sugarcane"]
    selected_crop = st.selectbox("Crop focus", crop_options, label_visibility="collapsed")
    st.divider()
    st.markdown("**About the model**")
    st.caption("AgriScan uses MobileNetV2 transfer learning across 22 crop and disease classes.")
    if DEBUG_MODE:
        show_pipeline = st.checkbox("Show processing pipeline", value=False)
    else:
        show_pipeline = False
        st.caption("Developer pipeline is available in debug mode.")

st.markdown('<div class="eyebrow">Field-ready crop intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy"><div class="hero-badge"><i></i> AI-assisted plant health scan</div><h1>See what your plants are telling you.</h1><p>Upload a leaf photo to identify potential disease symptoms and get practical next steps in seconds.</p></div>', unsafe_allow_html=True)

if model is None or not class_names:
    st.error("The trained model or class index file is missing. Run the training workflow before starting a scan.")
    st.stop()

st.markdown('<div class="section-card upload-card"><div class="card-kicker">01 · Add a leaf image</div><div class="card-title">Start a new scan</div><div class="card-subtitle">JPG, JPEG, or PNG · one leaf per image works best</div>', unsafe_allow_html=True)
upload_col, camera_col = st.columns(2, gap="medium")
with upload_col:
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
with camera_col:
    camera_file = st.camera_input("Take a photo", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

source_file = camera_file if camera_file is not None else uploaded_file
if source_file:
    image = Image.open(source_file)
    rgb_image, original_mode, was_converted = convert_to_rgb(image)
    is_leaf, green_ratio, green_mask = is_likely_leaf(rgb_image)
    resized_image = resize_image(rgb_image)
    img_array = image_to_array(resized_image)
    normed_array = normalize_array(img_array)
    batched_array = expand_batch_dim(normed_array)
    if not is_leaf:
        st.warning("This image may not contain a clear leaf. Try a closer, brighter photo before scanning.")

    preview_col, details_col = st.columns([1.15, .85], gap="large")
    with preview_col:
        st.markdown('<div class="section-card"><div class="card-kicker">Image preview</div>', unsafe_allow_html=True)
        st.image(rgb_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with details_col:
        st.markdown('<div class="section-card"><div class="card-kicker">Scan readiness</div><div class="card-title">One image, three checks</div>', unsafe_allow_html=True)
        w, h = image.size
        a, b = st.columns(2)
        with a: st.markdown(f'<div class="agri-stat"><div class="label">Dimensions</div><div class="value">{w} × {h}</div></div>', unsafe_allow_html=True)
        with b: st.markdown(f'<div class="agri-stat"><div class="label">Leaf signal</div><div class="value">{green_ratio*100:.0f}%</div></div>', unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="agri-stat"><div class="label">Selected scope</div><div class="value" style="font-size:1.05rem">'+selected_crop+'</div></div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Run health scan  →", use_container_width=True, disabled=not is_leaf):
            with st.spinner("Analyzing leaf patterns..."):
                st.session_state["raw_prediction"] = model.predict(batched_array, verbose=0)[0]
        st.markdown('</div>', unsafe_allow_html=True)

    if show_pipeline:
        with st.expander("Developer view · nine-stage processing pipeline", expanded=False):
            st.caption("Diagnostics for model development. The exact tensor below is sent to the trained classifier.")
            st.write(f"1. Original upload: `{w} × {h}` · mode `{image.mode}`")
            st.write(f"2. RGB conversion: `{original_mode}` → `RGB`" if was_converted else "2. RGB conversion: already RGB")
            st.write(f"3. Leaf validity: `{green_ratio*100:.1f}%` green-dominant pixels · {'PASS' if is_leaf else 'FAIL'}")
            st.image(green_mask, caption="Green-dominant pixel mask", use_container_width=True)
            st.write(f"4. Resize: `{w} × {h}` → `224 × 224`")
            fig, axes = plt.subplots(1, 2, figsize=(8, 2.5))
            axes[0].hist(img_array.ravel(), bins=48, color="#6e9fd0"); axes[0].set_title("Raw pixels")
            axes[1].hist(normed_array.ravel(), bins=48, color="#6ab47b"); axes[1].set_title("Normalized")
            st.pyplot(fig); plt.close(fig)
            st.write(f"5–7. Array normalization and batch expansion: `{img_array.shape}` → `{batched_array.shape}`")
            aug_gen = ImageDataGenerator(rotation_range=20, zoom_range=.2, horizontal_flip=True)
            aug_iter = aug_gen.flow(np.expand_dims(img_array.astype(np.float32), 0), batch_size=1)
            st.caption("8. Augmentation preview (training-only illustration)")
            aug_cols = st.columns(4)
            for i, col in enumerate(aug_cols):
                with col: st.image(next(aug_iter)[0].astype(np.uint8), caption=f"Aug {i+1}", use_container_width=True)
            st.success(f"9. Model input ready · shape `{batched_array.shape}` · range [{batched_array.min():.1f}, {batched_array.max():.1f}]")

if "raw_prediction" in st.session_state and source_file:
    raw_prediction = st.session_state["raw_prediction"]
    crop_results = {}
    for crop_name in ["Wheat", "Rice", "Potato", "Cotton", "Sugarcane"]:
        indices = [idx for idx, name in class_names.items() if name.lower().startswith(crop_name.lower())]
        if indices:
            probs = raw_prediction[indices]
            local = int(np.argmax(probs)); idx = indices[local]
            crop_results[crop_name] = (class_names[idx], float(probs[local] / np.sum(probs) * 100) if np.sum(probs) else 0, float(raw_prediction[idx]))
    overall_index = int(np.argmax(raw_prediction))
    overall_class = class_names.get(overall_index, "Unknown")
    overall_conf = float(np.max(raw_prediction)) * 100
    if selected_crop != "Auto-Detect (All Crops)" and selected_crop in crop_results:
        disease_class, confidence, _ = crop_results[selected_crop]
    else:
        disease_class, confidence = overall_class, overall_conf
    info = disease_info.get(disease_class, {})
    healthy = "healthy" in disease_class.lower()
    st.markdown('<div class="result-shell"><div class="section-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="result-head"><div><div class="card-kicker">02 · Scan result</div><div class="diagnosis {"good" if healthy else "bad"}">{"Healthy leaf detected" if healthy else "Potential issue detected"}</div></div><div class="result-pill">{selected_crop}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#6f8177;font-size:.95rem;margin-bottom:.8rem">Model label: <strong style="color:#18332a">{disease_class.replace("___", " · ").replace("_", " ")}</strong></div>', unsafe_allow_html=True)
    render_confidence(confidence)
    if confidence < 70: st.markdown('<div class="tip-box" style="margin-top:1rem">Confidence is below 70%. For a stronger result, retake the photo in natural light with the full leaf in frame.</div>', unsafe_allow_html=True)
    st.divider()
    advice_a, advice_b = st.columns(2, gap="large")
    with advice_a:
        st.markdown(f'<div class="advice-box"><strong>What we found</strong><p>{info.get("description", "No description available.")}</p></div>', unsafe_allow_html=True)
    with advice_b:
        st.markdown(f'<div class="advice-box"><strong>Recommended next step</strong><p>{info.get("treatment", "No treatment available.")}</p></div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">AgriScan · Transfer-learning plant disease detection · For decision support, not a substitute for an agronomist.</div>', unsafe_allow_html=True)
