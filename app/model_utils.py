"""Model loading and prediction utilities."""

import numpy as np
import streamlit as st
import tensorflow as tf

from app.config import MODEL_REGISTRY
from app.labels import ID_TO_PATTERN


@st.cache_resource
def load_model(model_key: str) -> tf.keras.Model:
    """Load and cache a Keras model. Cached across reruns and sessions."""
    info = MODEL_REGISTRY[model_key]
    model = tf.keras.models.load_model(str(info["path"]))
    return model


def predict_single(model: tf.keras.Model, prepared_input: np.ndarray) -> dict:
    """Run prediction on a single prepared wafer map.

    Args:
        prepared_input: shape (1, 52, 52, 3) float32

    Returns:
        dict with class_id, pattern_name, confidence, probabilities

    """
    probs = model.predict(prepared_input, verbose=0)[0]
    class_id = int(np.argmax(probs))
    return {
        "class_id": class_id,
        "pattern_name": ID_TO_PATTERN[class_id],
        "confidence": float(probs[class_id]),
        "probabilities": probs,
    }


def predict_batch(model: tf.keras.Model, prepared_inputs: np.ndarray) -> list[dict]:
    """Run prediction on a batch of prepared wafer maps.

    Args:
        prepared_inputs: shape (N, 52, 52, 3) float32

    Returns:
        list of result dicts, one per wafer

    """
    all_probs = model.predict(prepared_inputs, verbose=0)
    results = []
    for i, probs in enumerate(all_probs):
        class_id = int(np.argmax(probs))
        results.append({
            "index": i,
            "class_id": class_id,
            "pattern_name": ID_TO_PATTERN[class_id],
            "confidence": float(probs[class_id]),
            "probabilities": probs,
        })
    return results
