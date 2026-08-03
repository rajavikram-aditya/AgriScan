"""
app.py - AgriScan: AI Disease Prediction + Expandable Processing Pipeline

Single Streamlit app with the prediction flow as the main view and
the full 9-stage image processing pipeline in a collapsible sidebar.
Run with:  streamlit run app.py
"""

import os
import json
import streamlit as st
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from disease_info import disease_info
from preprocessing import (
    convert_to_rgb,
    is_likely_leaf,
    resize_image,
    image_to_array,
    normalize_array,
    expand_batch_dim,
    preprocess_for_prediction,
)

# Page configuration
st.set_page_config(
    page_title="AgriScan - Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
)


# Load model and class indices
@st.cache_resource
def load_model_and_classes():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "agriscan_model.keras")
    class_indices_path = os.path.join(base_dir, "class_indices.json")

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = None

    class_names = {}
    if os.path.exists(class_indices_path):
        with open(class_indices_path, "r") as f:
            class_indices = json.load(f)
            class_names = {v: k for k, v in class_indices.items()}

    return model, class_names


model, class_names = load_model_and_classes()

# ── Sidebar: Image Processing Pipeline (expandable) ─────────────────────────
with st.sidebar:
    st.header("🔬 Image Processing Pipeline")
    st.caption(
        "Expand each stage below after uploading an image to see "
        "every preprocessing step applied before prediction."
    )
    show_pipeline = st.checkbox("Enable Pipeline View", value=False)

# ── Main area ────────────────────────────────────────────────────────────────
st.title("🌱 AgriScan -- Plant Disease Detection")
st.write("Upload a plant leaf image and get disease prediction with treatment advice.")
st.divider()

crop_type = st.radio(
    "Select Crop Category",
    [
        "Fruit/Vegetable (Potato, Tomato, Cotton, Sugarcane)",
        "Grain Crops (Corn, Rice, Wheat)",
    ],
)
st.divider()

if model is None or not class_names:
    st.error("Model or class indices not found! Please run the training script first.")
else:
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Uploaded Leaf Image")
        st.image(image, width=350)

        # ── Run shared preprocessing (used by both pipeline view & prediction) ──
        rgb_image, original_mode, was_converted = convert_to_rgb(image)
        is_leaf, green_ratio, green_mask = is_likely_leaf(rgb_image)
        resized_image = resize_image(rgb_image)
        img_array = image_to_array(resized_image)
        normed_array = normalize_array(img_array)
        batched_array = expand_batch_dim(normed_array)

        # ── Sidebar pipeline stages (only if enabled) ───────────────────────
        if show_pipeline:
            with st.sidebar:
                st.divider()

                # Stage 1
                with st.expander("Stage 1: Original Upload", expanded=False):
                    st.image(image, use_container_width=True)
                    w, h = image.size
                    st.caption(f"**Dimensions:** {w} x {h} px  |  **Mode:** {image.mode}")

                # Stage 2
                with st.expander("Stage 2: Color Space Conversion", expanded=False):
                    st.image(rgb_image, use_container_width=True)
                    if was_converted:
                        st.caption(f"Converted from **{original_mode}** to **RGB**.")
                    else:
                        st.caption("No conversion needed -- already **RGB**.")

                # Stage 3
                with st.expander("Stage 3: Leaf Validity Check", expanded=False):
                    st.metric("Green-Dominant Pixel Ratio", f"{green_ratio * 100:.1f}%")
                    if is_leaf:
                        st.success("PASS -- accepted as a leaf (>= 15%).")
                    else:
                        st.error("FAIL -- rejected (< 15%). Not a leaf.")

                    # Green mask overlay
                    rgb_arr = np.array(rgb_image)
                    overlay = rgb_arr.copy()
                    overlay[green_mask] = [0, 255, 0]
                    overlay[~green_mask] = (overlay[~green_mask] * 0.3).astype(np.uint8)
                    st.image(overlay, caption="Green-Dominant Pixel Mask", use_container_width=True)

                # Stage 4
                with st.expander("Stage 4: Resize to 224x224", expanded=False):
                    col_a, col_b = st.columns(2)
                    w, h = image.size
                    with col_a:
                        st.image(rgb_image, caption=f"Before: {w}x{h}", use_container_width=True)
                    with col_b:
                        st.image(resized_image, caption="After: 224x224", use_container_width=True)

                # Stage 5 & 6
                with st.expander("Stage 5-6: Pixel Distribution & Normalization", expanded=False):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
                    ax1.hist(img_array.ravel(), bins=64, color="#4A90D9", edgecolor="black", linewidth=0.3)
                    ax1.set_title("Raw (0-255)", fontsize=10)
                    ax1.set_xlabel("Value")
                    ax2.hist(normed_array.ravel(), bins=64, color="#7ED321", edgecolor="black", linewidth=0.3)
                    ax2.set_title("Normalized (0-1)", fontsize=10)
                    ax2.set_xlabel("Value")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        f"Before: [{img_array.min()}, {img_array.max()}]  |  "
                        f"After: [{normed_array.min():.1f}, {normed_array.max():.1f}]"
                    )

                # Stage 7
                with st.expander("Stage 7: Batch Dimension Expansion", expanded=False):
                    st.markdown(
                        f"`{normed_array.shape}`  ->  `{batched_array.shape}`"
                    )
                    st.caption("Leading batch dimension added for model.predict().")

                # Stage 8
                with st.expander("Stage 8: Augmentation Preview (Training Only)", expanded=False):
                    st.caption(
                        "These are illustrative examples generated live. "
                        "Augmentation only happens during training, not inference."
                    )
                    aug_gen = ImageDataGenerator(
                        rotation_range=20, zoom_range=0.2, horizontal_flip=True
                    )
                    aug_input = np.expand_dims(img_array.astype(np.float32), axis=0)
                    aug_iter = aug_gen.flow(aug_input, batch_size=1)
                    cols = st.columns(2)
                    for i in range(4):
                        augmented = next(aug_iter)[0].astype(np.uint8)
                        with cols[i % 2]:
                            st.image(augmented, caption=f"Aug #{i+1}", use_container_width=True)

                # Stage 9
                with st.expander("Stage 9: Model Input Confirmation", expanded=False):
                    st.success(
                        f"Shape: **{batched_array.shape}**  |  "
                        f"dtype: **{batched_array.dtype}**  |  "
                        f"Range: [{batched_array.min():.1f}, {batched_array.max():.1f}]"
                    )
                    st.info("This exact tensor is fed to agriscan_model.keras.")

        # ── Main prediction flow ────────────────────────────────────────────
        if st.button("🔍 Predict Disease"):
            if not is_leaf:
                st.warning(
                    "⚠️ This doesn't look like a leaf image. "
                    "Please upload a clear photo of a plant leaf."
                )
            else:
                with st.spinner("Analyzing image..."):
                    prediction = model.predict(batched_array)

                index = int(np.argmax(prediction))
                confidence = float(np.max(prediction)) * 100
                disease_class = class_names.get(index, "Unknown")

                st.divider()
                st.subheader("Prediction Result")

                if confidence < 70.0:
                    st.warning("⚠️ Low confidence in prediction. Please try a clearer photo!")
                    st.info(f"Top guess: {disease_class} (Confidence: {confidence:.2f}%)")
                else:
                    if "healthy" in disease_class.lower():
                        st.success(f"🌿 {disease_class}")
                    else:
                        st.error(f"⚠️ {disease_class}")

                    st.info(f"Confidence: {confidence:.2f}%")

                # Display info and treatment
                info = disease_info.get(disease_class, {})
                description = info.get("description", "No description available.")
                treatment = info.get("treatment", "No treatment available.")

                st.markdown("### Description")
                st.write(description)

                st.markdown("### Recommended Treatment")
                st.write(treatment)

st.divider()
st.caption("AgriScan - Transfer Learning based Plant Disease Detection System")
