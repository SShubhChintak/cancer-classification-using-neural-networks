"""
Cancer Classification and Prediction Using Neural Networks
Training and evaluation source code.

Run from the repository root:
    python cancer_classification.py

This script trains the documented ANN workflow and saves:
    cancer_model.keras
    training_scaler.joblib

A newly trained run can differ slightly from the results recorded from the
specific Google Colab execution in the report, because neural-network training
can be stochastic.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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


ROOT = Path(__file__).resolve().parent

# Load the Breast Cancer Wisconsin (Diagnostic) dataset.
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")
feature_names = list(X.columns)

print("DATASET SHAPE:", X.shape)
print("CLASS MAPPING: 0 = Malignant, 1 = Benign")

# Create a stratified 80:20 development/test split.
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Reserve 10% of the development portion for validation.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_dev, y_dev, test_size=0.10, random_state=42, stratify=y_dev
)

# Fit StandardScaler only on the training partition.
training_scaler = StandardScaler()
X_fit_scaled = training_scaler.fit_transform(X_fit)
X_val_scaled = training_scaler.transform(X_val)
X_test_scaled = training_scaler.transform(X_test)

# Build the documented 30 -> 20 -> 2 Artificial Neural Network.
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

print("\nNEURAL NETWORK ARCHITECTURE")
model.summary()

history = model.fit(
    X_fit_scaled,
    y_fit.to_numpy(),
    validation_data=(X_val_scaled, y_val.to_numpy()),
    epochs=10,
    batch_size=32,
    shuffle=True,
    verbose=1,
)

# Evaluate once on the held-out test data.
test_loss, test_accuracy = model.evaluate(
    X_test_scaled, y_test.to_numpy(), verbose=0
)
probabilities = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(probabilities, axis=1)
y_true = y_test.to_numpy()

print("\nTEST ACCURACY:", f"{test_accuracy * 100:.2f}%")
print("TEST LOSS:", f"{test_loss:.4f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Malignant", "Benign"],
        digits=4,
        zero_division=0,
    )
)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
print("CONFUSION MATRIX (actual rows; predicted columns)")
print(cm)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1], zero_division=0
)
specificity = cm[1, 1] / cm[1].sum() if cm[1].sum() else 0.0

print("\nCLASS-WISE PERFORMANCE")
print(f"Malignant precision: {precision[0] * 100:.2f}%")
print(f"Malignant recall / sensitivity: {recall[0] * 100:.2f}%")
print(f"Malignant F1-score: {f1[0] * 100:.2f}%")
print(f"Benign precision: {precision[1] * 100:.2f}%")
print(f"Benign recall: {recall[1] * 100:.2f}%")
print(f"Benign F1-score: {f1[1] * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")

# Save the trained ANN and the exact scaler used for preprocessing.
model_path = ROOT / "cancer_model.keras"
scaler_path = ROOT / "training_scaler.joblib"

model.save(model_path)
joblib.dump(training_scaler, scaler_path)

print("\nMODEL SAVED:", model_path)
print("SCALER SAVED:", scaler_path)
print("Training and evaluation completed.")

# Retain history in memory for optional plotting during an interactive run.
