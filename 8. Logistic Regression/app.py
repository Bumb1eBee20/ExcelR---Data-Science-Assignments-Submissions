# ============================================================
# DIABETES PREDICTION STREAMLIT APPLICATION
# ============================================================

import os
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# GET PROJECT DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model_path = os.path.join(
        BASE_DIR,
        "logistic_regression_model.pkl"
    )

    imputer_path = os.path.join(
        BASE_DIR,
        "imputer.pkl"
    )

    scaler_path = os.path.join(
        BASE_DIR,
        "scaler.pkl"
    )

    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    return model, imputer, scaler


model, imputer, scaler = load_artifacts()


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🩺 Diabetes Prediction System")

st.write(
    """
    Enter the patient's health information below to predict
    the likelihood of diabetes using a Logistic Regression model.
    """
)

st.divider()


# ============================================================
# USER INPUT SECTION
# ============================================================

st.subheader("Patient Health Information")


# Create two columns for cleaner UI

col1, col2 = st.columns(2)


with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=72.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )


with col2:

    insulin = st.number_input(
        "Insulin Level",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.50,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button("🔍 Predict Diabetes", use_container_width=True):


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    # IMPORTANT:
    # Feature order must exactly match the training dataset

    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],

        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )


    # ========================================================
    # HANDLE INVALID ZERO VALUES
    # ========================================================

    # These columns were treated as missing when they contained 0
    # during model training.

    cols_with_invalid_zeros = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]


    input_data[cols_with_invalid_zeros] = (
        input_data[cols_with_invalid_zeros].replace(0, float("nan"))
    )


    # ========================================================
    # APPLY MEDIAN IMPUTATION
    # ========================================================

    input_imputed = imputer.transform(input_data)


    # ========================================================
    # APPLY FEATURE SCALING
    # ========================================================

    input_scaled = scaler.transform(input_imputed)


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    prediction = model.predict(input_scaled)[0]

    prediction_probability = model.predict_proba(
        input_scaled
    )[0]


    diabetes_probability = prediction_probability[1] * 100
    no_diabetes_probability = prediction_probability[0] * 100


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.divider()

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error("⚠️ Prediction: Diabetes Detected")

    else:

        st.success("✅ Prediction: No Diabetes Detected")


    # Display probabilities

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Probability of No Diabetes",
            f"{no_diabetes_probability:.2f}%"
        )


    with result_col2:

        st.metric(
            "Probability of Diabetes",
            f"{diabetes_probability:.2f}%"
        )


    # ========================================================
    # OPTIONAL: DISPLAY INPUT DATA
    # ========================================================

    with st.expander("View Entered Patient Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("About This Model"):

    st.write(
        """
        This application uses a Logistic Regression machine learning
        model trained on diabetes-related health features.

        The preprocessing steps include:

        1. Handling invalid zero values as missing values.
        2. Median imputation for missing values.
        3. Standardization using StandardScaler.
        4. Prediction using Logistic Regression.
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    """
    ⚠️ Disclaimer: This application is created for educational purposes.
    The prediction should not be considered a medical diagnosis.
    Please consult a qualified healthcare professional for medical advice.
    """
)