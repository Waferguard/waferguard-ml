"""Inference utilities: artifact loading and model prediction.

Kept separate from financial.py so that financial.py remains
Streamlit-compatible (no TensorFlow import at module load time).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def load_artifacts(artifact_dir: str) -> dict:
    """Load saved training artifacts from {artifact_dir}/artifacts.pkl."""
    path = Path(artifact_dir) / "artifacts.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def run_model(model_path: str, x_batch: np.ndarray) -> np.ndarray:
    """Load a single model and run batch inference.

    Args:
        model_path: path to the .keras model file selected by the user.
        x_batch: input array of shape (N, 52, 52, 3).

    Returns probability array of shape (N, num_classes).

    """
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    return model.predict(x_batch, verbose=0)
