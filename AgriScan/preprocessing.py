"""
preprocessing.py — Shared image preprocessing module for AgriScan.

Both app.py (AI prediction view) and image_processing_view.py (pipeline view)
import from this single module to guarantee identical preprocessing logic.
"""

import numpy as np
from PIL import Image


def convert_to_rgb(img: Image.Image) -> tuple[Image.Image, str, bool]:
    """
    Convert a PIL image to RGB mode.

    Returns:
        (converted_image, original_mode, was_converted)
    """
    original_mode = img.mode
    if original_mode != "RGB":
        return img.convert("RGB"), original_mode, True
    return img, original_mode, False


def is_likely_leaf(img: Image.Image, min_green_ratio: float = 0.15) -> tuple[bool, float, np.ndarray]:
    """
    Check whether an image is likely a leaf by measuring the fraction
    of green-dominant pixels.

    Returns:
        (is_leaf, green_ratio, green_mask)
        - is_leaf: True if green_ratio >= min_green_ratio
        - green_ratio: float in [0, 1]
        - green_mask: boolean ndarray, True where pixel is green-dominant
    """
    rgb_img = img.convert("RGB") if img.mode != "RGB" else img
    arr = np.array(rgb_img).astype(int)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Green-dominant: G > R, G > B, and bright enough to exclude shadows
    green_mask = (g > r) & (g > b) & (g > 30)
    green_ratio = float(np.sum(green_mask)) / (arr.shape[0] * arr.shape[1])

    return green_ratio >= min_green_ratio, green_ratio, green_mask


def resize_image(img: Image.Image, target_size: tuple[int, int] = (224, 224)) -> Image.Image:
    """Resize a PIL image to the target dimensions."""
    return img.resize(target_size)


def image_to_array(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a uint8 numpy array.
    Strips the alpha channel if present.
    """
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] to [0.0, 1.0]."""
    return arr.astype(np.float32) / 255.0


def expand_batch_dim(arr: np.ndarray) -> np.ndarray:
    """Add a leading batch dimension: (H, W, C) -> (1, H, W, C)."""
    return np.expand_dims(arr, axis=0)


def preprocess_for_prediction(img: Image.Image) -> np.ndarray:
    """
    Full preprocessing pipeline: RGB -> resize -> array -> normalize -> batch.
    Returns a (1, 224, 224, 3) float32 array ready for model.predict().
    """
    rgb, _, _ = convert_to_rgb(img)
    resized = resize_image(rgb)
    arr = image_to_array(resized)
    normed = normalize_array(arr)
    batched = expand_batch_dim(normed)
    return batched
