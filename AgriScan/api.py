"""
AgriScan REST API — Flask server that exposes the trained model over HTTP.

Run with:
    python api.py

Endpoints:
    GET  /health   — liveness check; confirms model loaded correctly
    POST /predict  — accepts a multipart image file, returns JSON prediction
    POST /process  — returns base64-encoded images of each pipeline step
    POST /gradcam  — returns Grad-CAM heatmap overlay as a base64 PNG
"""

import base64
import io
import json
import os
import sys

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# ---------------------------------------------------------------------------
# Bootstrap: make sure we can import the project's own modules regardless of
# where the script is launched from.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from disease_info import disease_info  # noqa: E402
from explainability import run_gradcam  # noqa: E402
from preprocessing import (  # noqa: E402
    convert_to_rgb,
    expand_batch_dim,
    image_to_array,
    is_likely_leaf,
    normalize_array,
    resize_image,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # Allow requests from the Vite dev server (port 3000)

# ---------------------------------------------------------------------------
# Model loading — done once at startup
# ---------------------------------------------------------------------------
_model = None
_class_names: dict[int, str] = {}


def _load_model():
    """Load the Keras model and class index map into module-level globals."""
    global _model, _class_names

    model_path = os.path.join(BASE_DIR, "agriscan_model.keras")
    indices_path = os.path.join(BASE_DIR, "class_indices.json")

    if not os.path.exists(model_path):
        print(f"[AgriScan API] WARNING: model file not found at {model_path}")
        return

    if not os.path.exists(indices_path):
        print(f"[AgriScan API] WARNING: class index file not found at {indices_path}")
        return

    # Import TF lazily to give a faster startup on machines without GPU
    import tensorflow as tf  # noqa: PLC0415

    print("[AgriScan API] Loading model…", flush=True)
    _model = tf.keras.models.load_model(model_path)

    with open(indices_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    _class_names = {v: k for k, v in raw.items()}  # index → class name
    print(f"[AgriScan API] Model ready. {len(_class_names)} classes.", flush=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
CROP_PREFIXES = ["Wheat", "Rice", "Potato", "Cotton", "Sugarcane", "Tomato"]


def _predict(image: Image.Image, selected_crop: str = "Auto-detect") -> dict:
    """Run the full prediction pipeline on a PIL image and return a dict."""
    if _model is None or not _class_names:
        raise RuntimeError("Model is not loaded.")

    rgb_image, _, _ = convert_to_rgb(image)
    is_leaf, green_ratio, _ = is_likely_leaf(rgb_image)

    resized = resize_image(rgb_image)
    arr = image_to_array(resized)
    normed = normalize_array(arr)
    batched = expand_batch_dim(normed)

    raw_prediction: np.ndarray = _model.predict(batched, verbose=0)[0]

    # Build per-crop results
    crop_results: dict[str, tuple[str, float, float]] = {}
    for crop in CROP_PREFIXES:
        indices = [
            idx
            for idx, name in _class_names.items()
            if name.lower().startswith(crop.lower())
        ]
        if not indices:
            continue
        probs = raw_prediction[indices]
        local_best = int(np.argmax(probs))
        best_idx = indices[local_best]
        crop_conf = (
            float(probs[local_best] / np.sum(probs) * 100) if np.sum(probs) > 0 else 0
        )
        crop_results[crop] = (
            _class_names[best_idx],
            crop_conf,
            float(raw_prediction[best_idx]),
        )

    # Choose result
    overall_idx = int(np.argmax(raw_prediction))
    overall_class = _class_names.get(overall_idx, "Unknown")
    overall_conf = float(np.max(raw_prediction)) * 100

    if selected_crop not in ("Auto-detect", "") and selected_crop in crop_results:
        disease_class, confidence, _ = crop_results[selected_crop]
    else:
        disease_class, confidence = overall_class, overall_conf

    info = disease_info.get(disease_class, {})
    is_healthy = "healthy" in disease_class.lower()

    # Human-readable label: "Wheat___Leaf_Rust" → "Wheat · Leaf Rust"
    label = disease_class.replace("___", " · ").replace("_", " ")

    return {
        "disease_class": disease_class,
        "label": label,
        "confidence": round(confidence, 1),
        "is_healthy": is_healthy,
        "is_leaf": bool(is_leaf),
        "green_ratio": round(float(green_ratio) * 100, 1),
        "description": info.get(
            "description", "No description available for this class."
        ),
        "treatment": info.get(
            "treatment", "No treatment information available for this class."
        ),
        "selected_crop": selected_crop,
        "top_crops": {
            crop: {
                "label": res[0].replace("___", " · ").replace("_", " "),
                "confidence": round(res[1], 1),
            }
            for crop, res in crop_results.items()
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness check — lets the frontend know whether the model is ready."""
    if _model is None or not _class_names:
        return jsonify({"status": "model_not_loaded"}), 503
    return jsonify({"status": "ok", "classes": len(_class_names)})


@app.post("/predict")
def predict():
    """
    Accept a multipart/form-data POST with:
        file   — the image file (required)
        crop   — crop filter string, e.g. "Wheat" or "Auto-detect" (optional)

    Returns JSON prediction or a JSON error with an appropriate HTTP status.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided. Send the image as 'file' field."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    selected_crop = request.form.get("crop", "Auto-detect").strip()

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not open image: {exc}"}), 422

    if _model is None:
        return jsonify({"error": "Model is not loaded. Start the server after training."}), 503

    try:
        result = _predict(image, selected_crop)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)



# ---------------------------------------------------------------------------
# Helper — PIL image → base64 PNG data URI
# ---------------------------------------------------------------------------
def _img_to_b64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# /process — return visual processing pipeline steps as base64 images
# ---------------------------------------------------------------------------
@app.post("/process")
def process_pipeline():
    """
    Accept a multipart image upload and return base64-encoded PNG images
    for each processing step so the frontend can display the full pipeline.

    Steps returned:
        1. original   — decoded original upload
        2. rgb        — after RGB conversion (same if already RGB)
        3. resized    — 224×224 thumbnail
        4. green_mask — boolean leaf-detection mask (green overlay)
        5. normalized — pixel values mapped back to 0-255 for display
        6. channels   — R, G, B channel splits displayed as grayscale
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        original = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not open image: {exc}"}), 422

    # ── Step 1: original (re-encoded cleanly) ──────────────────────────────
    original_clean = original.copy()
    # Flatten alpha if needed for display
    if original_clean.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", original_clean.size, (255, 255, 255))
        bg.paste(original_clean, mask=original_clean.split()[-1])
        original_clean = bg

    # ── Step 2: RGB ────────────────────────────────────────────────────────
    rgb_img = original.convert("RGB")

    # ── Step 3: Resized 224×224 ────────────────────────────────────────────
    resized = rgb_img.resize((224, 224), Image.LANCZOS)

    # ── Step 4: Green leaf mask ────────────────────────────────────────────
    arr = np.array(rgb_img).astype(int)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    green_pixels  = (g > r) & (g > b) & (g > 30)
    yellow_brown  = (r > 60) & (g > 40) & (abs(r - g) < 45) & (b < g - 15)
    mask = (green_pixels | yellow_brown).astype(np.uint8)
    green_ratio   = float(mask.sum()) / (arr.shape[0] * arr.shape[1])
    # Overlay: highlight detected pixels in vivid green, darken the rest
    overlay = np.array(rgb_img).copy()
    overlay[mask == 1] = [34, 197, 94]       # bright green
    overlay[mask == 0] = (overlay[mask == 0] * 0.35).astype(np.uint8)
    mask_img = Image.fromarray(overlay.astype(np.uint8))

    # ── Step 5: Normalised (scaled back to [0,255] for display) ───────────
    norm_arr = (np.array(resized).astype(np.float32) / 255.0)  # [0,1]
    norm_display = (norm_arr * 255).astype(np.uint8)
    norm_img = Image.fromarray(norm_display)

    # ── Step 6: Individual RGB channels ───────────────────────────────────
    r_arr = np.zeros_like(np.array(resized))
    g_arr = np.zeros_like(np.array(resized))
    b_arr = np.zeros_like(np.array(resized))
    resized_arr = np.array(resized)
    r_arr[:, :, 0] = resized_arr[:, :, 0]
    g_arr[:, :, 1] = resized_arr[:, :, 1]
    b_arr[:, :, 2] = resized_arr[:, :, 2]

    return jsonify({
        "steps": [
            {
                "id": "original",
                "label": "1 · Original upload",
                "description": f"Source image as received ({original.width}×{original.height} px, mode: {original.mode})",
                "image": _img_to_b64(original_clean),
            },
            {
                "id": "rgb",
                "label": "2 · RGB conversion",
                "description": "Converted to 3-channel RGB. Alpha channels stripped, palette images expanded.",
                "image": _img_to_b64(rgb_img),
            },
            {
                "id": "resized",
                "label": "3 · Resize → 224×224",
                "description": "Bicubic-downsampled to the model's fixed input resolution (224×224 px).",
                "image": _img_to_b64(resized),
            },
            {
                "id": "green_mask",
                "label": "4 · Leaf detection mask",
                "description": f"Vegetation pixels highlighted in green ({green_ratio*100:.1f}% leaf signal). Combines green-dominant and yellow-brown disease pixels.",
                "image": _img_to_b64(mask_img),
                "meta": {"green_ratio": round(green_ratio * 100, 1), "is_leaf": green_ratio >= 0.10},
            },
            {
                "id": "normalized",
                "label": "5 · Normalised [0, 1]",
                "description": "Pixel values divided by 255 to produce float32 values in [0, 1]. Appearance is identical — only the tensor values change.",
                "image": _img_to_b64(norm_img),
            },
            {
                "id": "channel_r",
                "label": "6a · Red channel",
                "description": "Isolated red channel. Bright areas indicate high red intensity.",
                "image": _img_to_b64(Image.fromarray(r_arr)),
            },
            {
                "id": "channel_g",
                "label": "6b · Green channel",
                "description": "Isolated green channel. Healthy leaf tissue dominates here.",
                "image": _img_to_b64(Image.fromarray(g_arr)),
            },
            {
                "id": "channel_b",
                "label": "6c · Blue channel",
                "description": "Isolated blue channel. Typically lowest in healthy leaves.",
                "image": _img_to_b64(Image.fromarray(b_arr)),
            },
        ],
        "summary": {
            "original_size": f"{original.width}×{original.height}",
            "original_mode": original.mode,
            "model_input":   "224×224",
            "green_ratio":   round(green_ratio * 100, 1),
            "is_leaf":       green_ratio >= 0.10,
        }
    })


# ---------------------------------------------------------------------------
# /gradcam — Grad-CAM visual explanation
# ---------------------------------------------------------------------------
@app.post("/gradcam")
def gradcam():
    """
    Accept a multipart/form-data POST with:
        file   — image file (required)
        crop   — crop filter string, same as /predict (optional)
        alpha  — heatmap blend strength 0.0-1.0 (optional, default 0.4)

    Returns JSON with:
        overlay   — base64 PNG of the heatmap blended onto the original image
        heatmap   — base64 PNG of the raw false-colour heatmap (no original)
        layer_used — name of the conv layer that produced the activations
        class_index — predicted class index
        class_name  — predicted class string
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        original = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Could not open image: {exc}"}), 422

    if _model is None:
        return jsonify({"error": "Model is not loaded."}), 503

    alpha = float(request.form.get("alpha", 0.4))
    alpha = max(0.0, min(1.0, alpha))

    try:
        # ── Preprocess (reusing existing pipeline) ─────────────────────────
        rgb_img = original.convert("RGB")
        resized = rgb_img.resize((224, 224))
        arr     = np.array(resized).astype(np.float32) / 255.0
        batched = np.expand_dims(arr, axis=0)          # (1, 224, 224, 3)

        # ── Predict to get class index ─────────────────────────────────────
        raw_pred   = _model.predict(batched, verbose=0)[0]
        class_idx  = int(np.argmax(raw_pred))
        class_name = _class_names.get(class_idx, "Unknown")

        # ── Grad-CAM ───────────────────────────────────────────────────────
        result = run_gradcam(
            model=_model,
            original_image_rgb=rgb_img,
            preprocessed_array=batched,
            predicted_class_index=class_idx,
            alpha=alpha,
        )

        # ── Encode overlay image ───────────────────────────────────────────
        overlay_b64 = _img_to_b64(result["overlay_image"])

        # ── Encode raw heatmap as false-colour PNG ─────────────────────────
        import cv2 as _cv2
        raw_u8  = (result["heatmap"] * 255).astype(np.uint8)
        raw_jet = _cv2.applyColorMap(raw_u8, _cv2.COLORMAP_JET)
        raw_rgb = _cv2.cvtColor(raw_jet, _cv2.COLOR_BGR2RGB)
        heatmap_b64 = _img_to_b64(Image.fromarray(raw_rgb))

    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Grad-CAM failed: {exc}"}), 500

    return jsonify({
        "overlay":     overlay_b64,
        "heatmap":     heatmap_b64,
        "layer_used":  result["layer_used"],
        "class_index": class_idx,
        "class_name":  class_name,
        "alpha":       alpha,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _load_model()
    port = int(os.environ.get("AGRISCAN_API_PORT", 5000))
    print(f"[AgriScan API] Starting on http://localhost:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False)
