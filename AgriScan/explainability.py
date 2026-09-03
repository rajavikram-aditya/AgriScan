"""
explainability.py — Grad-CAM visual explainability for AgriScan.

Exposes four public functions:
    find_last_conv_layer(model)          → str
    generate_gradcam(model, array, cls)  → np.ndarray  (224×224, values in [0,1])
    overlay_heatmap(image_rgb, heatmap)  → PIL.Image
    run_gradcam(model, image_rgb, array, class_index) → dict

Dependencies: tensorflow, numpy, cv2, Pillow
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Layer detection
# ---------------------------------------------------------------------------

def find_last_conv_layer(model) -> str:
    """
    Programmatically find the last convolutional layer in *model* by iterating
    model.layers in reverse and returning the first layer whose output tensor
    has exactly 4 dimensions (batch, height, width, channels).

    Three strategies are tried in order, to handle differences across
    TF/Keras versions (Keras 2 vs Keras 3 / TF 2.16+):

    1. ``layer.output_shape``  — works in Keras 2 / TF ≤ 2.15
    2. ``layer.output.shape``  — works in Keras 3 for connected functional layers
    3. ``isinstance`` type check — final fallback; finds the last Conv2D by
       class type (picks Conv_1 for MobileNetV2, the correct Grad-CAM target)

    Raises RuntimeError if no suitable layer is found.
    """
    import tensorflow as tf  # lazy import

    # ── Strategy 1: output_shape property (Keras 2 / older TF) ──────────────
    for layer in reversed(model.layers):
        try:
            out_shape = layer.output_shape
        except Exception:
            out_shape = None

        if out_shape is not None:
            shapes = out_shape if isinstance(out_shape, list) else [out_shape]
            for shape in shapes:
                if isinstance(shape, (list, tuple)) and len(shape) == 4:
                    print(f"[Grad-CAM] Using conv layer (output_shape): "
                          f"{layer.name!r}  shape: {shape}")
                    return layer.name

    # ── Strategy 2: layer.output.shape (Keras 3 / TF 2.16+) ─────────────────
    for layer in reversed(model.layers):
        try:
            shape = tuple(layer.output.shape)
        except Exception:
            continue
        if len(shape) == 4:
            print(f"[Grad-CAM] Using conv layer (output.shape): "
                  f"{layer.name!r}  shape: {shape}")
            return layer.name

    # ── Strategy 3: type-based fallback ──────────────────────────────────────
    # Prefer plain Conv2D (last one before pooling, e.g. 'Conv_1' in MobileNetV2)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"[Grad-CAM] Using conv layer (Conv2D type): {layer.name!r}")
            return layer.name

    # Accept DepthwiseConv2D as last resort
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D)):
            print(f"[Grad-CAM] Using conv layer (DepthwiseConv2D type): "
                  f"{layer.name!r}")
            return layer.name

    raise RuntimeError(
        "Could not find a convolutional layer with 4-D output in the model. "
        "Make sure the model includes a convolutional backbone."
    )


# ---------------------------------------------------------------------------
# 2. Core Grad-CAM computation
# ---------------------------------------------------------------------------

def generate_gradcam(
    model,
    preprocessed_image_array: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for *class_index*.

    Parameters
    ----------
    model :
        The loaded Keras model.
    preprocessed_image_array : np.ndarray
        Already resized, normalised, and batched tensor with shape
        (1, 224, 224, 3).  Do **not** reprocess here; reuse the output of
        preprocessing.py's existing pipeline.
    class_index : int
        The predicted (or target) class index.

    Returns
    -------
    np.ndarray
        A 2-D float32 array of shape (224, 224) with values in [0, 1].
    """
    import tensorflow as tf  # lazy import — keeps module lightweight

    conv_layer_name = find_last_conv_layer(model)

    # Build a sub-model that outputs (conv activations, final predictions)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
            model.output,
        ],
    )

    # Forward pass + gradient tape
    inp = tf.cast(preprocessed_image_array, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        conv_outputs, predictions = grad_model(inp, training=False)
        # Score for the target class (scalar)
        class_score = predictions[:, class_index]

    # Gradients of the class score w.r.t. the conv layer output
    grads = tape.gradient(class_score, conv_outputs)  # shape: (1, h, w, C)

    # Pool gradients spatially → one importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # Weight each channel of the conv output, then sum across channels
    conv_outputs = conv_outputs[0]  # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    heatmap = tf.squeeze(heatmap)  # (h, w)

    # ReLU — keep only positive contributions
    heatmap = tf.maximum(heatmap, 0).numpy()

    # Normalise to [0, 1] — guard against a zero heatmap
    h_max = heatmap.max()
    if h_max == 0:
        return np.zeros((224, 224), dtype=np.float32)
    heatmap = heatmap / h_max

    # Upsample from conv-output size (e.g. 7×7) to model input size (224×224)
    heatmap_resized = cv2.resize(
        heatmap.astype(np.float32),
        (224, 224),
        interpolation=cv2.INTER_LINEAR,
    )

    return heatmap_resized.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Heatmap overlay
# ---------------------------------------------------------------------------

def overlay_heatmap(
    original_image_rgb,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Blend a Grad-CAM heatmap on top of the original image.

    Parameters
    ----------
    original_image_rgb :
        A PIL Image or numpy uint8 array (H, W, 3), any resolution.
    heatmap : np.ndarray
        Float32 array with values in [0, 1], typically (224, 224).
    alpha : float
        Opacity of the heatmap overlay (0 = image only, 1 = heatmap only).

    Returns
    -------
    PIL.Image
        Blended RGB image ready for display.
    """
    # Convert PIL → numpy if needed
    if isinstance(original_image_rgb, Image.Image):
        orig_arr = np.array(original_image_rgb.convert("RGB"))
    else:
        orig_arr = np.array(original_image_rgb, dtype=np.uint8)

    h, w = orig_arr.shape[:2]

    # Resize heatmap to match the display image resolution
    heatmap_sized = cv2.resize(
        heatmap,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    # Scale to uint8 and apply a colour map (JET: blue→green→yellow→red)
    heatmap_u8 = (heatmap_sized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    # cv2 uses BGR; convert to RGB for PIL
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Alpha-blend: result = (1-α)·original + α·heatmap
    orig_bgr  = cv2.cvtColor(orig_arr,    cv2.COLOR_RGB2BGR)
    heat_bgr  = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    blended   = cv2.addWeighted(orig_bgr, 1 - alpha, heat_bgr, alpha, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return Image.fromarray(blended_rgb)


# ---------------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------------

def run_gradcam(
    model,
    original_image_rgb,
    preprocessed_array: np.ndarray,
    predicted_class_index: int,
    alpha: float = 0.4,
) -> dict:
    """
    Full Grad-CAM pipeline: compute heatmap and blend with the original image.

    Parameters
    ----------
    model :
        The loaded Keras model.
    original_image_rgb :
        PIL Image or numpy array at display resolution (any size, RGB).
    preprocessed_array : np.ndarray
        Already preprocessed batch tensor, shape (1, 224, 224, 3).
    predicted_class_index : int
        Class index to explain (typically the argmax of the prediction).
    alpha : float
        Heatmap blend strength passed to overlay_heatmap().

    Returns
    -------
    dict with keys:
        "heatmap"       — raw 2-D float32 numpy array (224×224, values [0,1])
        "overlay_image" — PIL Image with heatmap blended onto the original
        "layer_used"    — name of the conv layer that was used
    """
    layer_name = find_last_conv_layer(model)
    heatmap    = generate_gradcam(model, preprocessed_array, predicted_class_index)
    overlay    = overlay_heatmap(original_image_rgb, heatmap, alpha=alpha)

    return {
        "heatmap":       heatmap,
        "overlay_image": overlay,
        "layer_used":    layer_name,
    }
