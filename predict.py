"""
Prediction helpers for the saved Cancer Classification ANN.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from tensorflow import keras


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "cancer_model.keras"
SCALER_PATH = ROOT / "training_scaler.joblib"


def load_artifacts():
    """Load the saved ANN and the exact fitted scaler."""
    missing = [str(p) for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Model artefacts not found: "
            + ", ".join(missing)
            + ". Run python cancer_classification.py first."
        )

    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    dataset = load_breast_cancer()
    feature_names = list(dataset.feature_names)
    class_mapping = {"0": "Malignant", "1": "Benign"}

    if model.input_shape[-1] != len(feature_names):
        raise ValueError("Model input size does not match the dataset features.")
    if scaler.n_features_in_ != len(feature_names):
        raise ValueError("Scaler feature count does not match the dataset features.")

    return model, scaler, feature_names, class_mapping


def predict_from_values(values, artifacts=None):
    """Predict from 30 numerical measurements; this is not a medical diagnosis."""
    model, scaler, feature_names, class_mapping = artifacts or load_artifacts()
    values = list(values)

    if len(values) != len(feature_names):
        raise ValueError(f"Exactly {len(feature_names)} numerical values are required.")

    try:
        numeric_values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("All inputs must be valid numerical values.") from exc

    if not np.isfinite(numeric_values).all():
        raise ValueError("Inputs must be finite numbers.")
    if (numeric_values < 0).any():
        raise ValueError("Tumor measurements must be non-negative.")

    input_frame = pd.DataFrame([numeric_values], columns=feature_names)
    scaled_input = scaler.transform(input_frame)
    probabilities = model.predict(scaled_input, verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))

    return {
        "class_id": predicted_class,
        "label": class_mapping[str(predicted_class)],
        "malignant_probability": float(probabilities[0]),
        "benign_probability": float(probabilities[1]),
        "model_probability": float(probabilities[predicted_class]),
    }
