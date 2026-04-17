# Student Dropout Risk Predictor

**COM 572 Machine Learning – Coursework Task 1**  
Predicting Student Dropout and Academic Success using Machine Learning

---

## Overview

This project builds and deploys a machine learning application that predicts whether a student
in higher education is at risk of dropping out. The system is built using Python and Streamlit,
trained on the UCI Machine Learning Repository *Predict Students' Dropout and Academic Success*
dataset (ID 697), and deployed as an interactive web application.

The prediction model is a **Random Forest** classifier trained directly within the application at
start-up. No pre-saved model files (such as `.pkl`) are used at any point, ensuring the
application is fully reproducible and transparent.

---

## Project Structure

```
.
├── app.py               # Streamlit web application (main entry point)
├── train.py             # Standalone training and evaluation script
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── data.csv             # (Optional) Bundled dataset for offline use
```

---

## Features

- Predicts student dropout risk based on eight simplified input features
- Displays dropout probability alongside a clear, plain-English interpretation
- Includes a model performance dashboard with accuracy, precision, recall, F1-score, and confusion matrix
- Trained entirely within the application – no external model files required
- Deployable on Streamlit Cloud with no additional configuration

---

## The Eight Input Features

The following features are used by the model. Their names have been simplified from the original
UCI dataset column names to improve accessibility for non-technical users.

| Simplified Name        | Description                                               |
|------------------------|-----------------------------------------------------------|
| Previous Grades        | First-semester average grade (0–20 scale)                 |
| Attendance             | Number of curricular units credited in semester 1         |
| Tuition Fees Status    | Whether tuition fees are paid up to date (Yes/No)         |
| Age                    | Student age at enrolment                                  |
| Gender                 | Gender recorded at enrolment (Male/Female)                |
| Course                 | Academic programme enrolled in                            |
| Study Time             | Daytime or evening attendance pattern                     |
| Scholarship Holder     | Whether the student holds a scholarship (Yes/No)          |

---

## Target Variable

The original dataset contains three target classes: `Dropout`, `Enrolled`, and `Graduate`.  
This project converts the target to a **binary classification** problem:

- **Dropout = 1** (positive class, the outcome we aim to detect)
- **Not Dropout = 0** (`Enrolled` and `Graduate` combined)

This simplification is justified because the primary operational goal is to identify students
at risk of leaving before completing their qualification.

---

## Machine Learning Models

Three models are implemented and compared:

| Model                | Role                              | Justification                                      |
|----------------------|-----------------------------------|----------------------------------------------------|
| Logistic Regression  | Linear baseline                   | Simple, interpretable, useful for benchmarking     |
| Decision Tree        | Non-linear single model           | Captures non-linear patterns but prone to overfitting |
| Random Forest        | Ensemble model (final selection)  | Robust, well-calibrated, resistant to overfitting  |

**Random Forest was selected as the final model** because it consistently outperformed both
alternatives on F1-score and cross-validation stability.

---

## Getting Started

### Prerequisites

- Python 3.10 or above
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/student-dropout-predictor.git
cd student-dropout-predictor

# Install dependencies
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser. The model will be trained automatically on first launch.

### Run the Training Script Only

```bash
python train.py
```

This will load the dataset, train all three models, evaluate them, and print a comparison table
and final evaluation of the tuned Random Forest to the terminal.

---

## Dataset

**Name:** Predict Students' Dropout and Academic Success  
**Source:** UCI Machine Learning Repository  
**URL:** https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success  
**Records:** 4,424 students  
**Features:** 36 original features (8 selected for this project)  
**Target:** Dropout / Enrolled / Graduate (binarised to Dropout vs Not Dropout)

The dataset is loaded automatically via the `ucimlrepo` package. A local `data.csv` copy
(semicolon-separated) can be placed in the project root as a fallback.

---

## Deployment on Streamlit Cloud

1. Push all project files to a public GitHub repository.
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud).
3. Click **New App** and connect your GitHub repository.
4. Set the **Main file path** to `app.py`.
5. Click **Deploy**. No environment variables or secrets are required.

The application will install dependencies from `requirements.txt` automatically.

---

## Evaluation Summary

Metrics are reported on a held-out test set (20% of the dataset, stratified by class):

| Metric     | Value (approx.) |
|------------|-----------------|
| Accuracy   | ~77–80%         |
| Precision  | ~72–76%         |
| Recall     | ~75–80%         |
| F1-Score   | ~74–78%         |

*Exact values are displayed in the **Model Performance** section of the Streamlit application.*

---

## Important Disclaimer

This application is a research and educational tool. Predictions must not be used as the sole
basis for any academic, administrative, or disciplinary decision. All outputs should be reviewed
by a qualified academic adviser. Student data must be handled in accordance with the
UK General Data Protection Regulation (UK GDPR) and relevant institutional data governance policies.

---

## References

Please refer to the accompanying 2,000-word technical report for a full IEEE-style reference list
and detailed academic justification of all design decisions.

---

## Module Information

**Module:** COM 572 – Machine Learning  
**Module Leader:** Sahan Perera  
**Task:** Coursework Task 1  
**Submission Deadline:** 21 April 2026
