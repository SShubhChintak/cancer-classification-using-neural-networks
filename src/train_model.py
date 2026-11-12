"""
Train the Breast Cancer Wisconsin ANN classifier.

Run from the repository root:
    python src/train_model.py

The split and preprocessing sequence follows the documented Colab workflow.
Training results can vary slightly because neural-network initialization and
backend execution are stochastic; the README/report metrics describe the
specific author-supplied Colab run.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
METADATA_DIR = ROOT / "metadata"
RESULTS_DIR = ROOT / "results"

for directory in (MODEL_DIR, METADATA_DIR, RESULTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")
feature_names = list(X.columns)
class_mapping = {"0": "Malignant", "1": "Benign"}

# 2. Stratified 80:20 development/test split
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 3. Reserve 10% of development data for validation
X_fit, X_val, y_fit, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.10,
    random_state=42,
    stratify=y_dev,
)

# 4. Fit the final scaler on training data only
scaler = StandardScaler()
X_fit_scaled = scaler.fit_transform(X_fit)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. Build the documented ANN (30 -> 20 -> 2; 662 parameters)
model = keras.Sequential(
    [
        keras.layers.Input(shape=(len(feature_names),), name="input_layer"),
        keras.layers.Dense(20, activation="relu", name="hidden_layer"),
        keras.layers.Dense(2, activation="softmax", name="output_layer"),
    ],
    name="Cancer_Classification_ANN",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 6. Train for the documented initial experiment
history = model.fit(
    X_fit_scaled,
    y_fit.to_numpy(),
    validation_data=(X_val_scaled, y_val.to_numpy()),
    epochs=10,
    batch_size=32,
    shuffle=True,
    verbose=1,
)

# 7. Evaluate once on the held-out test set
test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test.to_numpy(),
    verbose=0,
)
probabilities = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(probabilities, axis=1)
y_true = y_test.to_numpy()

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
# Rows: actual [Malignant, Benign]; columns: predicted [Malignant, Benign]
confusion_df = pd.DataFrame(
    cm,
    index=["Actual_Malignant", "Actual_Benign"],
    columns=["Predicted_Malignant", "Predicted_Benign"],
)
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Malignant", "Benign"],
    digits=4,
    zero_division=0,
)
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1], zero_division=0
)

# With labels [0, 1], malignant is class 0 and benign is class 1.
specificity = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0
evaluation = pd.DataFrame(
    [
        ("Testing Accuracy", accuracy_score(y_true, y_pred) * 100, "%"),
        ("Malignant Precision", precision[0] * 100, "%"),
        ("Malignant Recall / Sensitivity", recall[0] * 100, "%"),
        ("Malignant F1 Score", f1[0] * 100, "%"),
        ("Benign Precision", precision[1] * 100, "%"),
        ("Benign Recall", recall[1] * 100, "%"),
        ("Benign F1 Score", f1[1] * 100, "%"),
        ("Specificity", specificity * 100, "%"),
        ("Testing Loss", float(test_loss), "Loss"),
    ],
    columns=["Performance Metric", "Result", "Unit"],
)

# 8. Save model and all required inference metadata
model.save(MODEL_DIR / "cancer_model.keras")
joblib.dump(scaler, MODEL_DIR / "training_scaler.joblib")

(METADATA_DIR / "feature_names.json").write_text(
    json.dumps(feature_names, indent=4), encoding="utf-8"
)
(METADATA_DIR / "class_mapping.json").write_text(
    json.dumps(class_mapping, indent=4), encoding="utf-8"
)

training_history = pd.DataFrame(history.history)
training_history.insert(0, "Epoch", range(1, len(training_history) + 1))
training_history.to_csv(RESULTS_DIR / "training_history.csv", index=False)
evaluation.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
confusion_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")
(RESULTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

metadata = {
    "project_title": "Cancer Classification and Prediction Using Neural Networks",
    "author": "SANKET KISHOR SUTAR",
    "dataset": "Breast Cancer Wisconsin (Diagnostic), Scikit-learn",
    "samples": int(len(X)),
    "features": int(len(feature_names)),
    "class_mapping": class_mapping,
    "model_architecture": {
        "type": "Artificial Neural Network",
        "input_features": 30,
        "hidden_layers": 1,
        "hidden_neurons": 20,
        "hidden_activation": "ReLU",
        "output_neurons": 2,
        "output_activation": "Softmax",
        "total_parameters": int(model.count_params()),
    },
    "training_configuration": {
        "development_test_split": "80:20",
        "training_validation_split": "90:10 of development portion",
        "random_state": 42,
        "epochs": 10,
        "batch_size": 32,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "Sparse categorical cross-entropy",
    },
    "split_sizes": {
        "training": int(len(X_fit)),
        "validation": int(len(X_val)),
        "testing": int(len(X_test)),
    },
    "software_versions": {
        "python_note": "Use the Python version supported by the installed TensorFlow release.",
        "tensorflow": tf.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    },
    "this_run_test_accuracy_percent": round(float(test_accuracy) * 100, 4),
    "this_run_test_loss": round(float(test_loss), 6),
}
(METADATA_DIR / "model_metadata.json").write_text(
    json.dumps(metadata, indent=4), encoding="utf-8"
)

print("\nTraining and evaluation completed.")
print(f"Training samples: {len(X_fit)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Test loss: {test_loss:.4f}")
print(f"Artifacts saved under: {ROOT}")
