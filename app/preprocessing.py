"""Preprocessing: parse uploads and prepare model input."""

from io import BytesIO

import numpy as np
from PIL import Image

from app.config import WAFER_RGB

# Reference RGB colors for reverse-mapping images to pixel states
_REF_COLORS = np.array(list(WAFER_RGB.values()), dtype=np.float32)  # (3, 3)


def parse_npz_upload(file_bytes: bytes) -> np.ndarray:
    """Parse an uploaded .npz file into raw 52x52 integer arrays.

    Returns: np.ndarray of shape (N, 52, 52) with dtype int.
    """
    data = np.load(BytesIO(file_bytes))

    # Try 'arr_0' first (matches dataset format), else first key
    keys = list(data.keys())
    if not keys:
        raise ValueError("NPZ file is empty — no arrays found.")
    key = "arr_0" if "arr_0" in keys else keys[0]
    arr = data[key]

    # Validate and reshape
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]  # (52, 52) → (1, 52, 52)
    if arr.ndim != 3 or arr.shape[1:] != (52, 52):
        raise ValueError(f"Expected shape (N, 52, 52), got {arr.shape}")

    return arr.astype(int)


def parse_image_upload(file_bytes: bytes) -> np.ndarray:
    """Parse an uploaded PNG/JPG wafer map image into a 52x52 integer array.

    Reverse-maps RGB pixel colors to discrete states {0, 1, 2}
    using nearest Euclidean distance.

    Returns: np.ndarray of shape (1, 52, 52) with dtype int.
    """
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize((52, 52), Image.NEAREST)
    pixels = np.array(img, dtype=np.float32)  # (52, 52, 3)

    # Compute distance to each reference color and pick nearest
    # pixels: (52, 52, 3), _REF_COLORS: (3, 3)
    diff = pixels[:, :, np.newaxis, :] - _REF_COLORS[np.newaxis, np.newaxis, :, :]
    distances = np.sum(diff**2, axis=-1)  # (52, 52, 3)
    mapped = np.argmin(distances, axis=-1)  # (52, 52)

    return mapped[np.newaxis, ...].astype(int)


def parse_upload(filename: str, file_bytes: bytes) -> np.ndarray:
    """Auto-detect format and parse an uploaded file.

    Returns: np.ndarray of shape (N, 52, 52) with dtype int.
    """
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "npz":
        return parse_npz_upload(file_bytes)
    if ext in ("png", "jpg", "jpeg"):
        return parse_image_upload(file_bytes)
    raise ValueError(f"Unsupported file format: .{ext}")


def prepare_for_model(raw_images: np.ndarray) -> np.ndarray:
    """Convert raw 52x52 int arrays to one-hot encoded model input.

    Matches the training pipeline:
    1. Clip pixel value 3 → 0 (undocumented state)
    2. One-hot encode → (N, 52, 52, 3) float32

    Input:  (N, 52, 52) int, values in {0, 1, 2, 3}
    Output: (N, 52, 52, 3) float32
    """
    images = raw_images.copy()
    images[images == 3] = 0
    images = np.clip(images, 0, 2)
    return np.eye(3, dtype=np.float32)[images]
