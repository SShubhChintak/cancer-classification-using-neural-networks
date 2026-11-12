"""
Gradio interface for the saved ANN cancer-classification demonstration.

Run from the repository root:
    python src/app.py

The app expects model and preprocessing artefacts created by train_model.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd
from sklearn.datasets import load_breast_cancer

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import load_artifacts, predict_from_values  # noqa: E402


model, scaler, feature_names, class_mapping = load_artifacts()

# Training-dataset medians are convenient demonstration defaults,
# not clinical patient measurements.
dataset = load_breast_cancer()
training_frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
defaults = {
    feature: float(training_frame[feature].median())
    for feature in feature_names
}

mean_features = [f for f in feature_names if f.startswith("mean ")]
error_features = [f for f in feature_names if f.endswith("error")]
worst_features = [f for f in feature_names if f.startswith("worst ")]

assert len(mean_features) == len(error_features) == len(worst_features) == 10
assert mean_features + error_features + worst_features == feature_names


def run_prediction(*values):
    """Validate form values and return classification, probabilities and disclaimer."""
    try:
        result = predict_from_values(
            values,
            artifacts=(model, scaler, feature_names, class_mapping),
        )
    except ValueError as exc:
        return "Invalid input", "", str(exc)

    output = (
        f"Predicted Classification: {result['label']}\n"
        f"Predicted Class: {result['class_id']}\n"
        f"Model Probability: {result['model_probability'] * 100:.2f}%"
    )
    probabilities = (
        f"Malignant: {result['malignant_probability'] * 100:.2f}%\n"
        f"Benign: {result['benign_probability'] * 100:.2f}%"
    )
    disclaimer = (
        "Educational AI prediction only. This is not a medical diagnosis "
        "or a substitute for professional medical evaluation."
    )
    return output, probabilities, disclaimer


def add_feature_group(title, features, open_by_default=False):
    components = []
    with gr.Accordion(title, open=open_by_default):
        for start in range(0, len(features), 2):
            with gr.Row():
                for feature in features[start : start + 2]:
                    components.append(
                        gr.Number(
                            label=feature.title(),
                            value=defaults[feature],
                            precision=6,
                        )
                    )
    return components


with gr.Blocks(title="Breast Cancer Classification System") as demo:
    gr.Markdown(
        """
        # Breast Cancer Classification System
        ### Artificial Neural Network-based educational prediction demonstration

        Enter the 30 numerical measurements in the three groups below.
        """
    )
    gr.Markdown("**Class mapping:** 0 = Malignant · 1 = Benign")
    inputs = []
    inputs.extend(add_feature_group("1. Mean Measurements", mean_features, True))
    inputs.extend(add_feature_group("2. Standard Error Measurements", error_features))
    inputs.extend(add_feature_group("3. Worst Measurements", worst_features))

    predict_button = gr.Button("Predict Tumor Classification", variant="primary")
    prediction_output = gr.Textbox(label="Predicted Classification", lines=3, interactive=False)
    probability_output = gr.Textbox(label="Class Probabilities", lines=2, interactive=False)
    information_output = gr.Textbox(label="Important Information", lines=3, interactive=False)

    predict_button.click(
        fn=run_prediction,
        inputs=inputs,
        outputs=[prediction_output, probability_output, information_output],
    )
    gr.Markdown(
        """
        ---
        **Disclaimer:** This application is developed for educational purposes.
        Its output must not be used for clinical diagnosis or decision-making.
        """
    )

if __name__ == "__main__":
    demo.launch()
