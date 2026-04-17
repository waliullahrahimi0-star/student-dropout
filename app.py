"""
app.py
------
COM 572 Machine Learning Coursework – Task 1
Title: Predicting Student Dropout and Academic Success using Machine Learning

Streamlit web application for predicting whether a student is at risk of
dropping out of higher education. The model is trained directly within
the application at start-up using a cached function, so no pre-saved
model files (e.g., .pkl) are required or used at any point.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    - Push all project files to a GitHub repository.
    - Connect the repository in Streamlit Cloud.
    - Set the entry point to app.py.
    - No additional configuration is required.
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION AND CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Original UCI dataset column names that map to our simplified names.
# Note: The column "Daytime/evening attendance" contains a tab character
# in some versions of the UCI export, so we handle both variants below.
ORIGINAL_FEATURE_COLS = [
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (credited)",
    "Tuition fees up to date",
    "Age at enrollment",
    "Gender",
    "Course",
    "Daytime/evening attendance\t",
    "Scholarship holder",
]

# Simplified feature names used throughout the application.
# These names have been derived from the original dataset for usability and
# interpretability, making the application accessible to non-technical users.
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

NUMERIC_FEATURES = ["Previous Grades", "Attendance", "Age"]
CATEGORICAL_FEATURES = [
    "Tuition Fees Status",
    "Gender",
    "Course",
    "Study Time",
    "Scholarship Holder",
]

TARGET_COL = "Target"

# Course codes from the UCI dataset with human-readable labels
COURSE_OPTIONS = {
    "Biofuel Production Technologies": 33,
    "Animation and Multimedia Design": 171,
    "Social Service (Evening Attendance)": 8014,
    "Agronomy": 9003,
    "Communication Design": 9070,
    "Veterinary Nursing": 9085,
    "Informatics Engineering": 9119,
    "Equinculture": 9130,
    "Management": 9147,
    "Social Service": 9238,
    "Tourism": 9254,
    "Nursing": 9500,
    "Oral Hygiene": 9556,
    "Advertising and Marketing Management": 9670,
    "Journalism and Communication": 9773,
    "Basic Education": 9853,
    "Management (Evening)": 9991,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING AND MODEL TRAINING (CACHED)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_and_train():
    """
    Load the UCI Student Dropout dataset, prepare features, and train the
    final Random Forest classifier. This function is cached by Streamlit so
    that it only runs once per session – subsequent interactions reuse the
    trained model without re-fitting.

    The model is a Random Forest with carefully chosen hyperparameters that
    balance performance and training speed. No pre-saved model files are used.

    Returns:
        tuple: (trained_pipeline, metrics_dict, X_test, y_test)
    """
    try:
        from ucimlrepo import fetch_ucirepo
        student_data = fetch_ucirepo(id=697)
        X_raw = student_data.data.features
        y_raw = student_data.data.targets
        df = pd.concat([X_raw, y_raw], axis=1)
    except Exception:
        # Fallback: attempt to load data.csv from the current directory.
        # This supports deployment on Streamlit Cloud with a bundled data file.
        try:
            df = pd.read_csv("data.csv", sep=";")
        except Exception as e:
            raise RuntimeError(
                "Could not load the dataset. Please ensure 'ucimlrepo' is "
                f"installed or 'data.csv' is present in the project directory. Error: {e}"
            )

    # Identify feature columns – handle tab character variants in column names
    available_cols = {c.strip(): c for c in df.columns}
    selected_original = []
    for col in ORIGINAL_FEATURE_COLS:
        stripped = col.strip()
        if stripped in available_cols:
            selected_original.append(available_cols[stripped])
        elif col in df.columns:
            selected_original.append(col)

    X = df[selected_original].copy()
    X.columns = SIMPLIFIED_NAMES[:len(selected_original)]

    # Ensure all simplified columns are present
    for name in SIMPLIFIED_NAMES:
        if name not in X.columns:
            X[name] = 0  # Placeholder if column is missing

    # Binary target: 1 = Dropout, 0 = Not Dropout (Enrolled or Graduate)
    y = df[TARGET_COL].apply(lambda v: 1 if str(v).strip() == "Dropout" else 0)

    # Train/test split – stratified to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Preprocessing pipeline
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    # Final model – Random Forest with tuned hyperparameters.
    # n_estimators=100 provides strong ensemble diversity without excessive
    # training time. max_depth=15 prevents overfitting while retaining
    # sufficient model complexity.
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred,
            target_names=["Not Dropout", "Dropout"],
            output_dict=True
        ),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "dropout_rate": round(y.mean() * 100, 1),
    }

    return pipeline, metrics, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT AND STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS for readability without heavy custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a3a5c;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #1a3a5c;
    }
    .result-dropout {
        background-color: #fff0f0;
        border-left: 5px solid #c0392b;
        padding: 1rem;
        border-radius: 6px;
        font-size: 1.1rem;
    }
    .result-safe {
        background-color: #f0fff0;
        border-left: 5px solid #27ae60;
        padding: 1rem;
        border-radius: 6px;
        font-size: 1.1rem;
    }
    .disclaimer {
        background-color: #fffbe6;
        border: 1px solid #f0c040;
        border-radius: 6px;
        padding: 1rem;
        font-size: 0.85rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "Select a section:",
        ["Prediction Tool", "Model Performance", "About this Application"],
        index=0,
    )
    st.markdown("---")
    st.markdown(
        "**COM 572 – Machine Learning**  \n"
        "Coursework Task 1  \n"
        "Dataset: UCI ML Repository (ID 697)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL (with spinner)
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Preparing the prediction model. This will only take a moment..."):
    try:
        pipeline, metrics, X_test, y_test = load_and_train()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        load_error = str(e)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: PREDICTION TOOL
# ─────────────────────────────────────────────────────────────────────────────

if page == "Prediction Tool":

    st.markdown('<div class="main-title">Student Dropout Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">COM 572 Machine Learning – Coursework Task 1 | '
        'Powered by a Random Forest Classifier trained on the UCI Student Dataset</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    This tool uses machine learning to estimate whether a student is at risk of dropping out
    of higher education. Enter the student's details in the form below and press **Generate Prediction**
    to receive a risk assessment.

    The prediction is based on eight key factors identified from a dataset of over 4,400 students
    enrolled at a Portuguese higher education institution. All eight fields are required.
    """)

    if not model_loaded:
        st.error(f"The model could not be loaded. Please check your setup. Error: {load_error}")
        st.stop()

    st.markdown("---")
    st.subheader("Student Information")
    st.markdown("Please complete all fields below. Hover over any field label for guidance.")

    col1, col2 = st.columns(2)

    with col1:
        previous_grades = st.number_input(
            label="Previous Grades",
            min_value=0.0,
            max_value=20.0,
            value=12.0,
            step=0.5,
            help=(
                "The student's average grade achieved in their first semester curricular units, "
                "on a scale of 0 to 20. A higher value indicates stronger academic performance."
            )
        )

        attendance = st.number_input(
            label="Attendance (Units Credited)",
            min_value=0,
            max_value=30,
            value=5,
            step=1,
            help=(
                "The number of curricular units the student successfully credited in their first semester. "
                "This serves as a proxy for academic engagement and regular attendance."
            )
        )

        tuition_status = st.selectbox(
            label="Tuition Fees Status",
            options=["Fees paid up to date", "Fees outstanding"],
            help=(
                "Indicates whether the student's tuition fees are currently paid in full. "
                "Outstanding fees may reflect financial difficulty, which is associated with dropout risk."
            )
        )

        age = st.number_input(
            label="Age at Enrolment",
            min_value=16,
            max_value=70,
            value=20,
            step=1,
            help=(
                "The student's age (in years) at the time they enrolled on their programme. "
                "Mature students may face different challenges than school leavers."
            )
        )

    with col2:
        gender = st.selectbox(
            label="Gender",
            options=["Female", "Male"],
            help="The student's gender as recorded at the point of enrolment."
        )

        course_name = st.selectbox(
            label="Course",
            options=list(COURSE_OPTIONS.keys()),
            help=(
                "The academic programme the student is currently enrolled on. "
                "Different courses have different completion rates and structures."
            )
        )

        study_time = st.selectbox(
            label="Study Time",
            options=["Daytime", "Evening"],
            help=(
                "Whether the student attends during the day or in the evening. "
                "Evening students are often in part-time employment, which can affect completion rates."
            )
        )

        scholarship = st.selectbox(
            label="Scholarship Holder",
            options=["No", "Yes"],
            help=(
                "Indicates whether the student holds a scholarship. Scholarship holders may "
                "have additional motivation and financial stability supporting course completion."
            )
        )

    st.markdown("---")

    # Convert inputs to the numeric encoding expected by the model
    tuition_encoded = 1 if tuition_status == "Fees paid up to date" else 0
    gender_encoded = 1 if gender == "Male" else 0
    course_encoded = COURSE_OPTIONS[course_name]
    study_encoded = 1 if study_time == "Daytime" else 0
    scholarship_encoded = 1 if scholarship == "Yes" else 0

    input_data = pd.DataFrame([{
        "Previous Grades": previous_grades,
        "Attendance": attendance,
        "Tuition Fees Status": tuition_encoded,
        "Age": age,
        "Gender": gender_encoded,
        "Course": course_encoded,
        "Study Time": study_encoded,
        "Scholarship Holder": scholarship_encoded,
    }])

    if st.button("Generate Prediction", type="primary"):
        prediction = pipeline.predict(input_data)[0]
        probability = pipeline.predict_proba(input_data)[0]

        dropout_prob = probability[1]
        safe_prob = probability[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown(
                f'<div class="result-dropout">'
                f'<strong>Predicted Outcome: At Risk of Dropout</strong><br>'
                f'The model estimates a <strong>{dropout_prob:.1%}</strong> probability that this '
                f'student will not complete their programme of study.'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
            **What this means:**
            Based on the information provided, the student's profile is broadly consistent with
            patterns observed in students who left their studies before completing their qualification.
            Key risk factors may include academic performance in the first semester, outstanding
            tuition fees, or limited attendance of credited units. This does not mean dropout is
            certain, but it suggests that timely pastoral or academic support could be beneficial.
            """)
        else:
            st.markdown(
                f'<div class="result-safe">'
                f'<strong>Predicted Outcome: Not at Risk of Dropout</strong><br>'
                f'The model estimates a <strong>{safe_prob:.1%}</strong> probability that this '
                f'student will continue and complete their studies (Enrolled or Graduate).'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown("""
            **What this means:**
            Based on the information provided, the student's profile broadly aligns with students
            who went on to complete or remain enrolled in their studies. Continued monitoring and
            support remain important, particularly in subsequent semesters.
            """)

        # Probability bar
        st.markdown("#### Probability Breakdown")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric(label="Not Dropout", value=f"{safe_prob:.1%}")
            st.progress(float(safe_prob))
        with prob_col2:
            st.metric(label="Dropout", value=f"{dropout_prob:.1%}")
            st.progress(float(dropout_prob))

        # Feature summary
        st.markdown("#### Summary of Inputs Used")
        summary_df = pd.DataFrame({
            "Feature": [
                "Previous Grades", "Attendance (Units Credited)",
                "Tuition Fees Status", "Age at Enrolment",
                "Gender", "Course", "Study Time", "Scholarship Holder"
            ],
            "Value Entered": [
                f"{previous_grades:.1f}", str(attendance),
                tuition_status, str(age),
                gender, course_name,
                study_time, scholarship
            ]
        })
        st.table(summary_df)

        # Disclaimer
        st.markdown("---")
        st.markdown(
            '<div class="disclaimer">'
            '<strong>Important Disclaimer:</strong> This prediction is generated by a machine '
            'learning model trained on historical data from a specific Portuguese institution. '
            'It is intended solely as a decision-support tool and must not be used as the '
            'sole basis for any academic, administrative, or disciplinary decision. The model '
            'may not generalise perfectly to all student populations, institutions, or national '
            'contexts. All predictions should be reviewed by a qualified academic adviser. '
            'Student data must be handled in accordance with relevant data protection legislation, '
            'including the UK General Data Protection Regulation (UK GDPR).'
            '</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Model Performance":

    st.markdown('<div class="main-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Evaluation metrics for the Random Forest classifier '
        'on the held-out test set (20% of the full dataset)</div>',
        unsafe_allow_html=True
    )

    if not model_loaded:
        st.error("Model not available. Please return to the Prediction Tool page.")
        st.stop()

    st.markdown("""
    The performance figures below were calculated on a test set of student records that were
    held out during training, ensuring an unbiased evaluation of the model's real-world accuracy.
    The dataset comprised **{:,} training records** and **{:,} test records**, with a
    dropout rate of approximately **{}%** across the full dataset.
    """.format(metrics["train_size"], metrics["test_size"], metrics["dropout_rate"]))

    st.markdown("---")
    st.subheader("Evaluation Metrics")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Accuracy", value=f"{metrics['accuracy']:.1%}")
    with m2:
        st.metric(label="Precision", value=f"{metrics['precision']:.1%}")
    with m3:
        st.metric(label="Recall", value=f"{metrics['recall']:.1%}")
    with m4:
        st.metric(label="F1-Score", value=f"{metrics['f1']:.1%}")

    st.markdown("---")
    st.subheader("Understanding the Metrics")
    st.markdown("""
    - **Accuracy** measures the overall proportion of predictions that were correct. While useful,
      it can be misleading when the two classes are not equally balanced.
    - **Precision** answers the question: of all the students the model predicted would drop out,
      how many actually did? A high precision value reduces false alarms.
    - **Recall** answers the question: of all the students who actually dropped out, how many
      did the model correctly identify? In an educational context, recall is particularly important
      because failing to identify an at-risk student has greater consequences than a false alarm.
    - **F1-Score** is the harmonic mean of precision and recall. It is the primary metric used
      in this project because it balances both concerns equally and is robust to class imbalance.
    """)

    st.markdown("---")
    st.subheader("Confusion Matrix")

    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Not Dropout", "Actual: Dropout"],
        columns=["Predicted: Not Dropout", "Predicted: Dropout"]
    )
    st.table(cm_df)

    st.markdown("""
    The confusion matrix shows the number of correct and incorrect predictions broken down by
    each class. The diagonal values (top-left and bottom-right) represent correct predictions.
    The off-diagonal values represent errors:

    - **False Positives** (top-right): Students predicted to drop out who did not. These may lead
      to unnecessary intervention but cause little harm.
    - **False Negatives** (bottom-left): Students who dropped out but were not identified by the
      model. These are the most operationally significant errors, as they represent missed
      opportunities for early intervention.
    """)

    st.markdown("---")
    st.subheader("Per-Class Performance")
    report_dict = metrics["report"]
    report_df = pd.DataFrame({
        "Class": ["Not Dropout", "Dropout"],
        "Precision": [
            f"{report_dict['Not Dropout']['precision']:.2%}",
            f"{report_dict['Dropout']['precision']:.2%}"
        ],
        "Recall": [
            f"{report_dict['Not Dropout']['recall']:.2%}",
            f"{report_dict['Dropout']['recall']:.2%}"
        ],
        "F1-Score": [
            f"{report_dict['Not Dropout']['f1-score']:.2%}",
            f"{report_dict['Dropout']['f1-score']:.2%}"
        ],
        "Support": [
            int(report_dict['Not Dropout']['support']),
            int(report_dict['Dropout']['support'])
        ],
    })
    st.table(report_df)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: ABOUT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "About this Application":

    st.markdown('<div class="main-title">About this Application</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Project Overview

    This application was developed as part of the **COM 572 Machine Learning** module coursework.
    It uses a supervised machine learning model to predict whether a student enrolled in higher
    education is at risk of dropping out, based on a small number of key academic and demographic factors.

    ### Dataset

    The data used to train the model comes from the **UCI Machine Learning Repository**:
    *Predict Students' Dropout and Academic Success* (Dataset ID: 697). It was collected from a
    higher education institution in Portugal and contains records for 4,424 students enrolled
    between 2008 and 2019. The dataset includes information on demographics, socioeconomic background,
    macroeconomic indicators, and academic performance across the first two semesters.

    ### Feature Names

    The eight features used in this application have been given simplified names to improve
    usability and accessibility:

    | Simplified Name | Original UCI Column Name |
    |---|---|
    | Previous Grades | Curricular units 1st sem (grade) |
    | Attendance | Curricular units 1st sem (credited) |
    | Tuition Fees Status | Tuition fees up to date |
    | Age | Age at enrollment |
    | Gender | Gender |
    | Course | Course |
    | Study Time | Daytime/evening attendance |
    | Scholarship Holder | Scholarship holder |

    ### Machine Learning Model

    The prediction is generated using a **Random Forest** classifier. Random Forest is an ensemble
    method that builds multiple decision trees during training and combines their outputs to produce
    more reliable and robust predictions than any single tree could offer alone. It was selected
    after comparing it against Logistic Regression and a single Decision Tree across several
    evaluation metrics.

    ### Important Limitations

    This tool is intended to support – not replace – professional academic judgement. It should
    be used by qualified staff only, and all outputs should be interpreted in context. The model
    was trained on data from one institution in Portugal; its accuracy and generalisability to
    other institutions, regions, or student populations has not been validated.

    ### Technical Notes

    - The model is trained at application start-up using a cached function and is not loaded
      from any saved file.
    - The application is built with Streamlit and is deployable to Streamlit Cloud.
    - All predictions are reproducible given the same input data and fixed random seed.

    ### Module Information

    **Module:** COM 572 – Machine Learning  
    **Task:** Coursework Task 1  
    **Assessment Weighting:** 70%  
    **Dataset Source:** [UCI ML Repository – ID 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
    """)
