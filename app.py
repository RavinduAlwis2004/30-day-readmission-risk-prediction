import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="30-Day Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

model = joblib.load("final_model.pkl")


# --------------------------------------------------
# Options
# --------------------------------------------------

age_options = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]

gender_options = [
    "Female",
    "Male",
    "Unknown/Invalid"
]

admission_type_labels = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    5: "Not Available",
    6: "Unknown",
    7: "Trauma Center",
    8: "Not Mapped"
}

admission_source_labels = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from a Hospital",
    5: "Transfer from a Skilled Nursing Facility",
    6: "Transfer from another Healthcare Facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    9: "Not Available",
    10: "Transfer from Critical Access Hospital",
    11: "Normal Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    17: "Unknown",
    20: "Not Mapped",
    22: "Transfer from Hospital / Same Facility",
    25: "Transfer from Ambulatory Surgery Center"
}


# Diagnosis options from the original dataset
diag_1_options = [
    str(x) for x in sorted(
        pd.read_csv("diabetic_data.csv")["diag_1"]
        .replace("?", pd.NA)
        .dropna()
        .unique()
    )
]

diag_2_options = [
    str(x) for x in sorted(
        pd.read_csv("diabetic_data.csv")["diag_2"]
        .replace("?", pd.NA)
        .dropna()
        .unique()
    )
]

diag_3_options = [
    str(x) for x in sorted(
        pd.read_csv("diabetic_data.csv")["diag_3"]
        .replace("?", pd.NA)
        .dropna()
        .unique()
    )
]


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🏥 30-Day Hospital Readmission Risk Predictor")

st.write(
    "Enter patient and hospital information to estimate "
    "the predicted probability of readmission within 30 days."
)

st.info(
    "This tool provides a machine-learning prediction for educational "
    "and decision-support purposes. It is not a medical diagnosis."
)


# --------------------------------------------------
# Patient Information
# --------------------------------------------------

st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:

    age = st.selectbox(
        "Age Group",
        age_options
    )

with col2:

    gender = st.selectbox(
        "Gender",
        gender_options
    )


# --------------------------------------------------
# Admission Information
# --------------------------------------------------

st.header("Admission Information")

col1, col2 = st.columns(2)

with col1:

    admission_type = st.selectbox(
        "Admission Type",
        options=list(admission_type_labels.keys()),
        format_func=lambda x: admission_type_labels[x]
    )

with col2:

    admission_source = st.selectbox(
        "Admission Source",
        options=list(admission_source_labels.keys()),
        format_func=lambda x: admission_source_labels[x]
    )


# --------------------------------------------------
# Hospital Utilization
# --------------------------------------------------

st.header("Hospital Utilization")

col1, col2, col3, col4 = st.columns(4)

with col1:

    time_in_hospital = st.number_input(
        "Time in Hospital (days)",
        min_value=1,
        max_value=14,
        value=4
    )

with col2:

    num_lab_procedures = st.number_input(
        "Number of Lab Procedures",
        min_value=0,
        max_value=132,
        value=40
    )

with col3:

    num_procedures = st.number_input(
        "Number of Procedures",
        min_value=0,
        max_value=6,
        value=1
    )

with col4:

    num_medications = st.number_input(
        "Number of Medications",
        min_value=0,
        max_value=81,
        value=10
    )


col1, col2, col3, col4 = st.columns(4)

with col1:

    number_outpatient = st.number_input(
        "Outpatient Visits",
        min_value=0,
        max_value=42,
        value=0
    )

with col2:

    number_emergency = st.number_input(
        "Emergency Visits",
        min_value=0,
        max_value=76,
        value=0
    )

with col3:

    number_inpatient = st.number_input(
        "Previous Inpatient Visits",
        min_value=0,
        max_value=21,
        value=1
    )

with col4:

    number_diagnoses = st.number_input(
        "Number of Diagnoses",
        min_value=1,
        max_value=16,
        value=5
    )


# --------------------------------------------------
# Diagnosis Information
# --------------------------------------------------

st.header("Diagnosis Information")

col1, col2, col3 = st.columns(3)

with col1:

    diag_1 = st.selectbox(
        "Primary Diagnosis",
        diag_1_options
    )

with col2:

    diag_2 = st.selectbox(
        "Secondary Diagnosis",
        diag_2_options
    )

with col3:

    diag_3 = st.selectbox(
        "Additional Diagnosis",
        diag_3_options
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Readmission Risk",
    type="primary",
    use_container_width=True
):

    # Create patient DataFrame
    patient = pd.DataFrame({
        "time_in_hospital": [time_in_hospital],
        "num_lab_procedures": [num_lab_procedures],
        "num_procedures": [num_procedures],
        "num_medications": [num_medications],
        "number_outpatient": [number_outpatient],
        "number_emergency": [number_emergency],
        "number_inpatient": [number_inpatient],
        "number_diagnoses": [number_diagnoses],
        "age": [age],
        "gender": [gender],
        "admission_type_id": [admission_type],
        "admission_source_id": [admission_source],
        "diag_1": [diag_1],
        "diag_2": [diag_2],
        "diag_3": [diag_3]
    })


    # Prediction
    prediction = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0][1]


    # --------------------------------------------------
    # Display Results
    # --------------------------------------------------

    st.header("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted 30-Day Readmission Probability",
            f"{probability:.2%}"
        )

    with col2:

        if prediction == 1:

            st.error(
                "Higher Predicted Readmission Risk"
            )

        else:

            st.success(
                "Lower Predicted Readmission Risk"
            )


    st.caption(
        "The prediction is generated by the trained Random Forest "
        "classification model."
    )