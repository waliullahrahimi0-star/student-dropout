"""
train.py
--------
COM 572 Machine Learning Coursework – Task 1
Title: Predicting Student Dropout and Academic Success using Machine Learning
Dataset: UCI ML Repository – Predict Students' Dropout and Academic Success (ID 697)

This script handles the complete machine learning pipeline:
  - Loading and preparing the dataset
  - Feature selection and renaming to simplified names
  - Preprocessing and pipeline construction
  - Training three models: Logistic Regression, Decision Tree, Random Forest
  - Evaluation with multiple metrics
  - Cross-validation and lightweight hyperparameter tuning
  - Saving a final evaluation summary

Usage:
    python train.py

No pre-saved model files are required or produced. The script is fully
self-contained and reproducible from scratch on every run.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset():
    """
    Load the UCI Student Dropout dataset using the ucimlrepo package.

    The dataset contains 4,424 records and 36 features capturing demographic,
    socioeconomic, and academic information collected at enrolment and during
    the first two semesters of study.

    Returns:
        pd.DataFrame: Full dataset with features and target combined.
    """
    print("Loading dataset from UCI Machine Learning Repository...")
    try:
        from ucimlrepo import fetch_ucirepo
        student_data = fetch_ucirepo(id=697)
        X = student_data.data.features
        y = student_data.data.targets
        df = pd.concat([X, y], axis=1)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Could not load via ucimlrepo ({e}).")
        print("Attempting direct CSV download as fallback...")
        url = (
            "https://archive.ics.uci.edu/static/public/697/"
            "predict+students+dropout+and+academic+success.zip"
        )
        # Fallback: load from a known direct CSV location
        df = pd.read_csv(
            "https://raw.githubusercontent.com/datasciencedojo/datasets/master/"
            "student_dropout.csv",
            sep=";"
        )
        print(f"Dataset loaded via fallback. Shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION AND RENAMING
# ─────────────────────────────────────────────────────────────────────────────

# These are the eight original UCI column names we select for our model.
# They represent a pedagogically meaningful and practically useful subset of
# the 36 available features.
ORIGINAL_FEATURE_COLS = [
    "Curricular units 1st sem (grade)",  # Academic performance in semester 1
    "Curricular units 1st sem (credited)",  # Units credited – a proxy for engagement/attendance
    "Tuition fees up to date",  # Binary flag: 1 = fees paid, 0 = outstanding
    "Age at enrollment",  # Student age at point of enrolment
    "Gender",  # Binary: 1 = male, 0 = female
    "Course",  # Integer code for the academic programme enrolled in
    "Daytime/evening attendance\t",  # 1 = daytime, 0 = evening (study time indicator)
    "Scholarship holder",  # Binary: 1 = holds a scholarship
]

# Simplified names used throughout the application and report.
# These names are more readable and are suitable for a non-technical audience.
SIMPLIFIED_NAMES = [
    "Previous Grades",
    "Attendance",
    "Tuition Fees Status",
    "Age",
    "Gender",
    "Course",
    "Study Time",
    "Scholarship Holder",
]

# Column name mapping for renaming
RENAME_MAP = dict(zip(ORIGINAL_FEATURE_COLS, SIMPLIFIED_NAMES))

TARGET_COL = "Target"


def prepare_features(df):
    """
    Select relevant columns, rename them to simplified names, and construct
    the binary target variable.

    The original target variable has three classes: 'Dropout', 'Enrolled', and
    'Graduate'. For this binary classification task, we group 'Enrolled' and
    'Graduate' together as 'Not Dropout' (encoded as 0) and retain 'Dropout'
    as the positive class (encoded as 1). This simplification is justified
    because the primary operational concern for institutions is identifying
    students at risk of leaving before completing their studies.

    Returns:
        X (pd.DataFrame): Feature matrix with simplified column names.
        y (pd.Series): Binary target variable (1 = Dropout, 0 = Not Dropout).
    """
    print("\nPreparing features and target variable...")

    # Identify which original columns are present (handle potential whitespace variants)
    available_cols = list(df.columns)
    selected = []
    for col in ORIGINAL_FEATURE_COLS:
        if col in available_cols:
            selected.append(col)
        else:
            # Try stripping whitespace from all column names
            stripped_map = {c.strip(): c for c in available_cols}
            stripped_key = col.strip()
            if stripped_key in stripped_map:
                selected.append(stripped_map[stripped_key])
            else:
                print(f"  Warning: Column '{col}' not found. Columns available: {available_cols[:5]}...")

    X = df[selected].copy()
    X.columns = SIMPLIFIED_NAMES[:len(selected)]

    # Binary target: Dropout (1) vs Not Dropout (0)
    y = df[TARGET_COL].apply(lambda v: 1 if v == "Dropout" else 0)

    print(f"  Features selected: {list(X.columns)}")
    print(f"  Target distribution:\n{y.value_counts().rename({1: 'Dropout', 0: 'Not Dropout'})}")
    print(f"  Class imbalance ratio: {y.mean():.2%} Dropout")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

# Features we treat as continuous and scale with StandardScaler
NUMERIC_FEATURES = ["Previous Grades", "Attendance", "Age"]

# Features we treat as categorical and encode with OneHotEncoder.
# Although some of these are already binary or ordinal integers, OneHotEncoding
# ensures the model makes no assumptions about ordinal relationships between
# course codes or gender codes.
CATEGORICAL_FEATURES = [
    "Tuition Fees Status",
    "Gender",
    "Course",
    "Study Time",
    "Scholarship Holder",
]


def build_preprocessor():
    """
    Construct a ColumnTransformer that applies appropriate preprocessing to
    numeric and categorical features independently.

    - Numeric features are standardised (zero mean, unit variance) to prevent
      features with larger ranges from dominating distance-based calculations.
    - Categorical features are one-hot encoded to handle nominal data correctly.

    Returns:
        ColumnTransformer: A fitted-ready preprocessing object.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def build_pipelines(preprocessor):
    """
    Construct three full sklearn Pipelines, each combining the shared
    preprocessor with one of the three candidate classifiers.

    Models:
        1. Logistic Regression – linear baseline, interpretable, probability output.
        2. Decision Tree – non-linear, single tree, prone to overfitting.
        3. Random Forest – ensemble of decision trees, robust and well-calibrated.

    Returns:
        dict: A dictionary mapping model names to Pipeline objects.
    """
    pipelines = {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver="lbfgs"
            ))
        ]),
        "Decision Tree": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(
                random_state=42,
                max_depth=10  # Constrained to reduce overfitting
            ))
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ]),
    }
    return pipelines


# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    """
    Train a pipeline on the training set and evaluate it on the held-out test set.

    Reports accuracy, precision, recall, F1-score, and confusion matrix.
    Also runs 5-fold cross-validation on the training set to assess stability.

    Args:
        name (str): Model display name.
        pipeline (Pipeline): Untrained sklearn Pipeline.
        X_train, X_test (pd.DataFrame): Feature splits.
        y_train, y_test (pd.Series): Label splits.

    Returns:
        dict: A summary of all evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"{'='*60}")

    # Train the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Core metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Full Classification Report:\n{classification_report(y_test, y_pred, target_names=['Not Dropout', 'Dropout'])}")

    # Cross-validation – 5-fold on training data
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1", n_jobs=-1)
    print(f"  Cross-Validation F1 (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "CV F1 Mean": round(cv_scores.mean(), 4),
        "CV F1 Std": round(cv_scores.std(), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING (LIGHTWEIGHT)
# ─────────────────────────────────────────────────────────────────────────────

def tune_random_forest(X_train, y_train, preprocessor):
    """
    Apply a lightweight RandomizedSearchCV to the Random Forest pipeline.

    RandomizedSearchCV is preferred over GridSearchCV here because it samples
    a fixed number of parameter combinations at random, making it significantly
    faster while still exploring a meaningful portion of the search space. With
    n_iter=20 and cv=3, the total number of model fits is 60 – fast enough to
    run in a standard environment without causing delays.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        preprocessor (ColumnTransformer): Shared preprocessor.

    Returns:
        Pipeline: The best Random Forest pipeline found during the search.
    """
    print("\nRunning lightweight hyperparameter tuning on Random Forest...")

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_distributions = {
        "classifier__n_estimators": [50, 100, 150, 200],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__max_features": ["sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_distributions,
        n_iter=20,  # Sample 20 combinations – fast but meaningful
        cv=3,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(f"  Best parameters: {search.best_params_}")
    print(f"  Best CV F1: {search.best_score_:.4f}")
    return search.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  COM 572 – Student Dropout Prediction")
    print("  Machine Learning Pipeline")
    print("="*60)

    # Load and prepare data
    df = load_dataset()
    X, y = prepare_features(df)

    # Train/test split – 80/20, stratified to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # Build preprocessor and model pipelines
    preprocessor = build_preprocessor()
    pipelines = build_pipelines(preprocessor)

    # Evaluate all three models
    results = []
    for name, pipeline in pipelines.items():
        result = evaluate_model(name, pipeline, X_train, X_test, y_train, y_test)
        results.append(result)

    # Summary table
    print("\n" + "="*60)
    print("  Model Comparison Summary")
    print("="*60)
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

    # Hyperparameter tuning on the best model (Random Forest)
    best_preprocessor = build_preprocessor()
    tuned_rf = tune_random_forest(X_train, y_train, best_preprocessor)

    print("\n  Final evaluation of tuned Random Forest on test set:")
    y_pred_tuned = tuned_rf.predict(X_test)
    final_f1 = f1_score(y_test, y_pred_tuned)
    final_acc = accuracy_score(y_test, y_pred_tuned)
    print(f"  Accuracy  : {final_acc:.4f}")
    print(f"  F1-Score  : {final_f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred_tuned, target_names=['Not Dropout', 'Dropout'])}")

    print("\n  Training complete. See app.py for the Streamlit deployment.")
    print("="*60)


if __name__ == "__main__":
    main()
