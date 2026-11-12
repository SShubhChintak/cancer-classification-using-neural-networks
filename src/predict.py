"""
Shared inference helpers for the saved ANN cancer-classification model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cancer_model.keras"
SCALER_PATH = ROOT / "models" / "training_scaler.joblib"
FEATURES_PATH = ROOT / "metadata" / "feature_names.json"
CLASS_PATH = ROOT / "metadata" / "class_mapping.json"


def load_artifacts():
    """Load the trained Keras model, fitted scaler, feature order and labels."""
    required = (MODEL_PATH, SCALER_PATH, FEATURES_PATH, CLASS_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required model artefacts are missing: "
            + ", ".join(missing)
            + ". Run python src/train_model.py first or restore the saved artefacts."
        )

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    class_mapping = json.loads(CLASS_PATH.read_text(encoding="utf-8"))

    if model.input_shape[-1] != len(feature_names):
        raise ValueError("Model input dimension does not match saved feature list.")
    if scaler.n_features_in_ != len(feature_names):
        raise ValueError("Scaler feature count does not match saved feature list.")

    return model, scaler, feature_names, class_mapping


def predict_from_values(values: Iterable[float], artifacts=None):
    """
    Predict a class from 30 measurements.

    Returns a dictionary containing label, class id and class probabilities.
    This is an educational demonstration, not a medical diagnosis.
    """
    model, scaler, feature_names, class_mapping = artifacts or load_artifacts()
    values = list(values)

    if len(values) != len(feature_names):
        raise ValueError(f"Exactly {len(feature_names)} numerical measurements are required.")

    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("All measurements must be valid numerical values.") from exc

    if not np.isfinite(numeric).all():
        raise ValueError("Measurements must be finite numbers.")
    if (numeric < 0).any():
        raise ValueError("Tumor measurements must be non-negative.")

    frame = pd.DataFrame([numeric], columns=feature_names)
    scaled = scaler.transform(frame)
    probabilities = model.predict(scaled, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))

    return {
        "class_id": predicted_class,
        "label": class_mapping[str(predicted_class)],
        "malignant_probability": float(probabilities[0]),
        "benign_probability": float(probabilities[1]),
        "model_probability": float(probabilities[predicted_class]),
    }
