# Cancer Classification and Prediction Using Neural Networks

**Developed by: SANKET KISHOR SUTAR**

An educational machine-learning project that uses an Artificial Neural Network (ANN) to classify samples from the Breast Cancer Wisconsin (Diagnostic) dataset as **Malignant (0)** or **Benign (1)**.

> **Disclaimer:** This project is an educational demonstration, not a clinically validated diagnostic system. Its predictions must not be used for medical diagnosis or treatment decisions.

## Project overview

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), available through Scikit-learn
- **Samples:** 569
- **Input features:** 30 numerical measurements
- **Classes:** Malignant and Benign
- **Model:** Artificial Neural Network
- **Architecture:** 30 input features → Dense(20, ReLU) → Dense(2, Softmax)
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss function:** Sparse categorical cross-entropy
- **Training:** 10 epochs, batch size 32
- **Development environment:** Google Colab
- **Interactive interface:** Gradio

## Dataset

The 30 numerical features are grouped into:
1. Mean measurements
2. Standard-error measurements
3. Worst measurements

The dataset contains 212 malignant and 357 benign samples. The class mapping used by the model is 0 = Malignant and 1 = Benign.

## Project workflow

1. Load the Breast Cancer Wisconsin (Diagnostic) dataset.
2. Explore the dataset, missing values, duplicate records, descriptive statistics, distributions and feature correlations.
3. Split the data into development and held-out test portions using stratification.
4. Create a training and validation split from the development portion.
5. Fit StandardScaler on training data only and transform validation/test data with that fitted scaler.
6. Build and train the ANN.
7. Evaluate the held-out test data using accuracy, precision, recall, F1-score and a confusion matrix.
8. Use the saved model and scaler in the Gradio prediction application.

## Recorded results from the supplied Google Colab run

| Metric | Result |
|---|---:|
| Training accuracy | 95.60% |
| Validation accuracy | 95.65% |
| Testing accuracy | 93.86% |
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

These figures describe the specific supplied Colab run. A new training run may produce slightly different values.

## Interactive prediction application

The Gradio interface accepts 30 numerical measurements in three groups: Mean, Standard Error and Worst. It applies the saved training scaler, passes the values to the ANN, and displays the predicted class and both class probabilities.

The supplied application screenshot shows an example output of Benign (class 1), with 86.14% Benign and 13.86% Malignant model probability. These probabilities are not clinically calibrated diagnostic certainty.

## Repository files

```text
cancer-classification-using-neural-networks/
│
├── README.md
├── .gitignore
├── Cancer_Classification_Project_clean.ipynb
├── cancer_classification.py
├── predict.py
├── app.py
├── cancer_model.keras
├── training_scaler.joblib
│
├── documentation/
│   ├── Cancer_Classification_Project_Report.docx
│   └── Cancer_Classification_Project_Report.pdf
│
├── presentation/
│   └── Cancer_Classification_Project_Presentation.pptx
│
└── screenshots/
    ├── Dataset and EDA screenshots
    ├── Training and accuracy screenshots
    ├── Evaluation screenshots
    └── Final prediction screenshots
```

The notebook, trained model, scaler, report, presentation and screenshots are added as the corresponding project files are uploaded.

## Running the project

Install the required libraries:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn tensorflow gradio joblib
```

Train the model and save the model/scaler files:

```bash
python cancer_classification.py
```

Launch the interactive application after the model files have been created:

```bash
python app.py
```

The completed Google Colab notebook is the primary record of the project workflow. The standalone Python files provide training, prediction and interface code.

## Limitations

- The dataset is a fixed public benchmark containing 569 samples.
- Evaluation is based on a held-out split from this dataset.
- No independent external cohort or prospective clinical validation was conducted.
- The application accepts numerical measurements and is not an image-based medical system.
- Strong benchmark performance does not establish clinical suitability.

## References

- Breast Cancer Wisconsin (Diagnostic) dataset, UCI Machine Learning Repository.
- Scikit-learn documentation.
- TensorFlow and Keras documentation.
- Gradio documentation.

---

**SANKET KISHOR SUTAR**
