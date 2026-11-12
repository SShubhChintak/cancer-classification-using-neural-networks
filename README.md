# Cancer Classification and Prediction Using Neural Networks

**Author:** SANKET KISHOR SUTAR

An educational machine-learning project that uses an Artificial Neural Network (ANN) to classify samples in the Breast Cancer Wisconsin (Diagnostic) dataset as **Malignant (0)** or **Benign (1)**. The project was developed in Python using Google Colab and includes exploratory data analysis, preprocessing, model training, evaluation, and an interactive Gradio demonstration.

> **Important:** This is an educational demonstration, not a clinically validated diagnostic system. It must not be used for medical diagnosis, treatment, or other clinical decisions.

## Project overview

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), provided through Scikit-learn
- **Dataset size:** 569 samples, 30 numerical input features, 2 classes
- **Class mapping:** 0 = Malignant; 1 = Benign
- **Model:** Feed-forward Artificial Neural Network
- **Architecture:** 30 inputs → Dense(20, ReLU) → Dense(2, Softmax)
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss:** Sparse categorical cross-entropy
- **Training:** 10 epochs, batch size 32
- **Environment:** Python, Google Colab
- **Interface:** Gradio

## Dataset

The 30 features are grouped as:
1. Mean measurements
2. Standard-error measurements
3. Worst measurements

The target classes in the dataset are malignant and benign. The dataset contains 212 malignant and 357 benign samples.

## Methodology

1. Load the Breast Cancer Wisconsin (Diagnostic) dataset.
2. Explore dataset dimensions, columns, missing values, duplicates, descriptive statistics, class distribution, feature distributions, and correlations.
3. Separate input features and target labels.
4. Create a stratified 80:20 train/test split (455 development samples and 114 held-out test samples).
5. Split the 455 development samples into 409 training and 46 validation samples.
6. Fit the final StandardScaler on the 409 training samples only; transform validation and test data with the fitted scaler.
7. Build and train the ANN for 10 epochs.
8. Evaluate on the held-out test set using accuracy, precision, recall, F1-score, and a confusion matrix.
9. Provide an interactive Gradio form for 30 numerical measurements.
10. Save the Keras model, scaler, feature order, class mapping, metadata, and result files.

## Reported results

The supplied Google Colab run recorded:

| Metric | Result |
|---|---:|
| Training accuracy | 95.60% |
| Validation accuracy | 95.65% |
| Held-out testing accuracy | 93.86% |
| Testing loss | 0.1408 |
| Malignant precision | 92.68% |
| Malignant recall / sensitivity | 90.48% |
| Malignant F1-score | 91.57% |
| Benign precision | 94.52% |
| Benign recall | 95.83% |
| Benign F1-score | 95.17% |
| Specificity | 95.83% |

Confusion matrix (actual rows; predicted columns):

| | Predicted Malignant | Predicted Benign |
|---|---:|---:|
| Actual Malignant | 38 | 4 |
| Actual Benign | 3 | 69 |

These results describe one stratified held-out split and are not evidence of clinical performance.

## Interactive demonstration

The Gradio interface accepts 30 numeric values, arranged into Mean, Standard Error, and Worst measurement groups. It applies the saved training scaler and displays the predicted class and the model's class probabilities.

Example shown in the supplied demonstration screenshot:
- Predicted class: Benign (class 1)
- Benign probability: 86.14%
- Malignant probability: 13.86%

Model probabilities are not clinically calibrated diagnostic certainty.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── notebooks/          # Completed Google Colab notebook (.ipynb)
├── src/                # Python source files and Gradio application
├── models/             # Trained model and fitted scaler
├── metadata/           # Feature names, class mapping, model metadata
├── results/            # Training history and evaluation outputs
├── screenshots/        # Actual project output screenshots
├── documentation/      # Final report (DOCX and PDF)
└── presentation/       # PowerPoint presentation
```

Folders will be populated with the corresponding project files as they are uploaded.

## Requirements and execution

The exact dependency versions should be recorded in `requirements.txt` from the final working environment. The primary development workflow is the completed notebook in Google Colab.

Once the source files and saved model artefacts are uploaded, use the instructions in the respective source-code files to run training or launch the Gradio interface.

## Limitations

- The dataset is a fixed public benchmark with 569 samples.
- Evaluation uses one stratified train/test split.
- No independent external cohort or prospective clinical validation was conducted.
- The interface accepts numerical measurements, not mammograms or clinical records.
- A high score on this dataset does not establish suitability for clinical use.

## Acknowledgements and references

- Breast Cancer Wisconsin (Diagnostic) dataset, UCI Machine Learning Repository.
- Scikit-learn dataset utilities and documentation.
- TensorFlow/Keras documentation.
- Gradio documentation.

See the project report in `documentation/` for the detailed background, methodology, implementation, results, references, and evidence notes.

---

**SANKET KISHOR SUTAR**
