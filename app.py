"""
Gradio application for Cancer Classification and Prediction Using Neural Networks.

Run from the repository root:
    python app.py

The application requires cancer_model.keras and training_scaler.joblib.
"""
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from predict import load_artifacts, predict_from_values


model, scaler, feature_names, class_mapping = load_artifacts()

# Dataset medians provide demonstration defaults; they are not patient data.
dataset = load_breast_cancer()
dataset_frame = pd.DataFrame(dataset.data, columns=dataset.feature_names)
default_values = {
    feature: float(dataset_frame[feature].median())
    for feature in feature_names
}

mean_features = [f for f in feature_names if f.startswith("mean ")]
error_features = [f for f in feature_names if f.endswith("error")]
worst_features = [f for f in feature_names if f.startswith("worst ")]

assert len(mean_features) == len(error_features) == len(worst_features) == 10
assert mean_features + error_features + worst_features == feature_names


def run_prediction(*values):
    try:
        result = predict_from_values(
            values,
            artifacts=(model, scaler, feature_names, class_mapping),
        )
    except ValueError as exc:
        return "Invalid input", "", str(exc)

    prediction = (
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
    return prediction, probabilities, disclaimer


def add_measurement_group(title, features, open_by_default=False):
    components = []
    with gr.Accordion(title, open=open_by_default):
        for start in range(0, len(features), 2):
            with gr.Row():
                for feature in features[start:start + 2]:
                    components.append(
                        gr.Number(
                            label=feature.title(),
                            value=default_values[feature],
                            precision=6,
                        )
                    )
    return components


with gr.Blocks(title="Breast Cancer Classification System") as demo:
    gr.Markdown(
        """
        # Breast Cancer Classification System
        ### Artificial Neural Network-based educational prediction application

        Enter the 30 numerical tumor measurements below.
        """
    )
    gr.Markdown("**Class mapping:** 0 = Malignant · 1 = Benign")

    inputs = []
    inputs.extend(add_measurement_group("1. Mean Measurements", mean_features, True))
    inputs.extend(add_measurement_group("2. Standard Error Measurements", error_features))
    inputs.extend(add_measurement_group("3. Worst Measurements", worst_features))

    predict_button = gr.Button("Predict Tumor Classification", variant="primary")
    prediction_output = gr.Textbox(
        label="Predicted Classification", lines=3, interactive=False
    )
    probability_output = gr.Textbox(
        label="Class Probabilities", lines=2, interactive=False
    )
    information_output = gr.Textbox(
        label="Important Information", lines=3, interactive=False
    )

    predict_button.click(
        fn=run_prediction,
        inputs=inputs,
        outputs=[prediction_output, probability_output, information_output],
    )

    gr.Markdown(
        """
        ---
        **Disclaimer:** This application is for educational purposes only.
        Its output must not be used for clinical diagnosis or decision-making.
        """
    )


if __name__ == "__main__":
    demo.launch()
