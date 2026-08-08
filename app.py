import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# Page configuration
st.set_page_config(
    page_title="AI-Enabled Personalized Cardiac Rehabilitation",
    layout="wide"
)

st.title("AI-Enabled Personalized Cardiac Rehabilitation")
st.subheader("Exercise Prescription Recommendation System")

# Sidebar
page = st.sidebar.radio(

    "Navigation",

    [
        "Prediction",
        "About"
    ]
)

# LOAD MODELS

risk_model = joblib.load("models/Risk_DecisionTree_RFE_LR.pkl")

frequency_model = joblib.load("models/Frequency_CatBoost_ANOVA.pkl")

intensity_model = joblib.load("models/Intensity_LightGBM_RF.pkl")

time_model = joblib.load("models/Time_CatBoost_RFE_SVM.pkl")

type_model = tf.keras.models.load_model(
    "models/Type_BPNN_Mutual_Info.keras"
)

# LOAD FEATURE ENCODERS

risk_encoder = joblib.load(
    "encoders/risk_feature_encoders.pkl"
)

frequency_encoder = joblib.load(
    "encoders/frequency_feature_encoders.pkl"
)

intensity_encoder = joblib.load(
    "encoders/intensity_feature_encoders.pkl"
)

time_encoder = joblib.load(
    "encoders/time_feature_encoders.pkl"
)

type_encoder = joblib.load(
    "encoders/type_feature_encoders.pkl"
)

# TARGET ENCODERS

frequency_target = joblib.load(
    "encoders/frequency_target_encoder.pkl"
)

intensity_target = joblib.load(
    "encoders/intensity_target_encoder.pkl"
)

time_target = joblib.load(
    "encoders/time_target_encoder.pkl"
)

type_target = joblib.load(
    "encoders/type_target_encoder.pkl"
)

# SCALER

type_scaler = joblib.load(
    "scalers/type_scaler.pkl"
)


# FEATURE LISTS

risk_features = [
'RISK  - Risk Type', 
'Exercise Habit - Frequency', 
'Exercise Habit - Mode', 
'Walking', 
'Diagnosis', 
'Gait', 
'Risk Factor - DM', 
'Total_Muscle_Power', 
'Smoking', 
'Functional Activity', 
'Posture', 
'ROM', 
'Test Today - METS', 
'Marital Status', 
'Risk Factor - ECHO - EF'
]

frequency_features = [
'Weekly_Exercise_Duration', 
'Risk Factor - Exercise', 
'Cooling Down', 
'Balance in Sitting and Standing', 
'Walking', 
'Risk Factor - Smoking', 
'Exercise Habit - Duration', 
'Occupation', 
'Predicted Risk Level Encoded', 
'Gait', 
'Risk Factor - ECHO - EF', 
'Test Today - peak HR', 
'Age', 
'Family History',
'Risk Factor - HPL'
]

intensity_features = [
'Age', 
'Test Today - peak HR', 
'Exercise Habit - Duration', 
'Weekly_Exercise_Duration', 
'Test Today - Termination Cause', 
'Risk Factor - ECHO - EF', 
'Exercise Habit - Frequency', 
'Risk Factor - BMI', 
'RISK  - Risk Type', 
'ECG Resting', 
'Predicted Risk Level Encoded', 
'Occupation', 
'Diagnosis', 
'Exercise Habit - Mode', 
'Risk Factor - Exercise'
]

time_features = [
'Weekly_Exercise_Duration', 
'Exercise Habit - Frequency', 
'Posture', 
'Exercise Habit - Mode', 
'Test Today - Termination Cause', 
'Risk Factor - HPL', 
'Total_Muscle_Power', 
'Gender', 
'Lives With', 
'Cooling Down', 
'Marital Status', 
'Target HR (bpm)', 
'Predicted Risk Level Encoded', 
'Test Today - peak HR', 
'Risk Factor - Stress'
]

type_features = [
'Total_Muscle_Power',
'Walking',
'Gait',
'Living Environment',
'ROM',
'Diagnosis',
'Balance in Sitting and Standing',
'Family History',
'Test Today - peak HR',
'RISK  - Risk Type',
'ECG Resting',
'Risk Factor - ECHO - EF',
'Cooling Down',
'Risk Factor - Stress',
'Risk Factor - Smoking'
]

# Prediction page
import pandas as pd

def encode_dataframe(df, encoders, features):

    df = df.copy()

    for col in features:

        if col not in df.columns:
            continue

        # If this feature has a saved LabelEncoder
        if col in encoders:

            encoder = encoders[col]

            mapping = {
                str(cls).strip().lower(): i
                for i, cls in enumerate(encoder.classes_)
            }

            value = str(df.at[0, col]).strip().lower()

            # Unknown category -> -1
            # Convert the column to object first so it can hold integers
            df[col] = df[col].astype(object)
            df.at[0, col] = mapping.get(value, -1)
            
        else:
            # No encoder exists for this column
            # Convert any remaining object/category column to integer codes
            if df[col].dtype == object or str(df[col].dtype) == "category":

                df[col] = (
                    pd.Categorical(
                        df[col].astype(str).str.strip().str.lower()
                    ).codes
                )

    # Final safety: LightGBM only accepts numeric columns
    for col in features:

        if col in df.columns:

            if df[col].dtype == object or str(df[col].dtype) == "category":

                df[col] = (
                    pd.to_numeric(df[col], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                )

    return df


if page == "Prediction":

    st.header("Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["M", "F"],
            key="gender"

        )

        age = st.number_input(
            "Age",
            min_value=21,
            max_value=87,
            value=60
        )

        marital_status = st.selectbox(
            "Marital Status",
            [
                "married",
                "single",
                "divorced",
                "widow"
            ],
            key="marital_status"
        )

        lives_with = st.selectbox(
            "Lives With",
            [
                "family",
                "friends",
                "alone"
            ],
            key="lives_with"
        )

        living_environment = st.selectbox(
            "Living Environment",
            [
                "FOS",
                "landed"
            ],
            key="living_environment"
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "employed",
                "self employed",
                "not working",
                "retired"
            ],
            key="occupation"
        )

        smoking = st.selectbox(
            "Smoking",
            [
                "yes",
                "no",
                "ex smoker"
            ],
            key="smoking"
        )


        family_history = st.selectbox(
            "Family History",
            [
                "yes",
                "no"
            ],
            key="family_history"
        )

        exercise_frequency = st.number_input(
            "Exercise Habit - Frequency",
            0,
            14,
            3
        )

        exercise_duration = st.number_input(
            "Exercise Habit - Duration",
            0,
            480,
            30
        )

        exercise_mode = st.selectbox(
            "Exercise Habit - Mode",
            [
                "walking",
                "jogging",
                "cycling",
                "others"
            ],
            key="exercise_mode"
        )

        rom = st.selectbox(
            "ROM",
            [
                "normal",
                "abnormal"
            ],
            key="rom"
        )

    with col2:

        ul_r = st.slider(
            "Muscle Power - UL - Right",
            0,
            5,
            5
        )

        ul_l = st.slider(
            "Muscle Power - UL - Left",
            0,
            5,
            5
        )

        ll_r = st.slider(
            "Muscle Power - LL - Right",
            0,
            5,
            5
        )

        ll_l = st.slider(
            "Muscle Power - LL - Left",
            0,
            5,
            5
        )

        balance = st.selectbox(
            "Balance in Sitting and Standing",
            [
                "yes",
                "no"
            ],
            key="balance"
        )

        functional = st.selectbox(
            "Functional Activity",
            [
                "independent",
                "assisted"
            ],
            key="functional"
        )

        walking = st.selectbox(
            "Walking",
            [
                "independent",
                "dependent"
            ],
            key="walking"
        )

        gait = st.selectbox(
            "Gait",
            [
                "normal",
                "abnormal"
            ],
            key="gait"
        )

        posture = st.selectbox(
            "Posture",
            [
                "normal",
                "abnormal"
            ],
            key="posture"
        )

    st.divider()
            
    st.subheader("Clinical Assessment")

    col3, col4 = st.columns(2)

    with col3:

        risk_dm = st.selectbox(
            "Risk Factor - DM",
            ["yes", "no"],
            key="risk_dm"
        )

        risk_hpl = st.selectbox(
            "Risk Factor - HPL",
            ["yes", "no"],
            key="risk_hpl"
        )

        risk_exercise = st.selectbox(
            "Risk Factor - Exercise",
            ["active", "moderate", "inactive"],
            key="risk_exercise"
        )

        risk_stress = st.selectbox(
            "Risk Factor - Stress",
            ["yes", "no"],
            key="risk_stress"
        )

        risk_smoking = st.selectbox(
            "Risk Factor - Smoking",
            ["yes", "no", "ex smoker"],
            key="risk_smoking"
        )

        risk_bmi = st.selectbox(
            "Risk Factor - BMI",
            [
                "underweight",
                "healthy",
                "overweight",
                "obese"
            ],
            key="risk_bmi"
        )

        risk_echo = st.selectbox(
            "Risk Factor - ECHO - EF",
            [
                "normal",
                "borderline",
                "reduced"
            ],
            key="risk_echo"
        )

    with col4:

        termination = st.selectbox(
            "Test Today - Termination Cause",
            [
                "Complete Test",
                "Fatigue",
                "Medical Condition",
                "Physical Discomfort",
                "Medical Condition+Physical Discomfort",
                "Fatigue+Medical Condition",
                "Fatigue+Physical Discomfort"
            ],
            key="termination"
        )

        peak_hr = st.selectbox(
            "Test Today - Peak HR",
            [
                "low intensity",
                "moderate intensity",
                "high intensity",
                "maximum intensity",
                "above maximum intensity"
            ],
            key="peak_hr"
        )

        mets = st.selectbox(
            "Test Today - METS",
            [
                "low intensity",
                "moderate intensity (low)",
                "moderate intensity (high)",
                "high intensity"
            ],
            key="mets"
        )

        ecg = st.selectbox(
            "ECG Resting",
            [
                "normal",
                "Q wave",
                "sinus rhythm",
                "sinus rhythm+Q wave",
                "sinus rhythm+Ectopics",
                "sinus rhythm+st depression",
                "sinus rhythm+T wave inversion",
                "T wave inversion",
                "T wave inversion+Q wave",
                "T wave inversion+st depression",
                "st depression",
                "st depression+Q wave",
                "Q wave ectopics"
            ],
            key="ecg"
        )

        risk_type = st.selectbox(
            "RISK - Risk Type",
            [
                "low",
                "moderate",
                "low to moderate",
                "moderate to high",
                "high"
            ],
            key="risk_type"
        )

        diagnosis = st.selectbox(
            "Diagnosis",
            [
                "PCI",
                "conservative",
                "CABG",
                "surgical",
                "PCI+conservative",
                "PCI+CABG",
                "surgical+conservative",
                "conservative + CABG"
            ],
            key="diagnosis"
        )

        target_bpm = st.number_input(
            "Target HR (bpm)",
            min_value=58,
            max_value=156,
            value=100
        )

        cooling = st.selectbox(
                "Cooling Down",
                [
                    "Visual",
                    "Extended"
                ],
                key="cooling"
            )
        

    patient_data = {

        "Gender": gender,
        "Age": age,
        "Marital Status": marital_status,
        "Lives With": lives_with,
        "Living Environment": living_environment,
        "Occupation": occupation,
        "Smoking": smoking,
        "Family History": family_history,
        "Exercise Habit - Frequency": exercise_frequency,
        "Exercise Habit - Duration": exercise_duration,
        "Exercise Habit - Mode": exercise_mode,
        "ROM": rom,
        "Walking": walking,
        "Gait": gait,
        "Posture": posture,
        "Functional Activity": functional,
        "Balance in Sitting and Standing": balance,
        "Risk Factor - DM": risk_dm,
        "Risk Factor - HPL": risk_hpl,
        "Risk Factor - Exercise": risk_exercise,
        "Risk Factor - Stress": risk_stress,
        "Risk Factor - Smoking": risk_smoking,
        "Risk Factor - BMI": risk_bmi,
        "Risk Factor - ECHO - EF": risk_echo,
        "Diagnosis": diagnosis,
        "Cooling Down": cooling,
        "Target HR (bpm)": target_bpm,
        "Test Today - Termination Cause": termination,
        "Test Today - peak HR": peak_hr,
        "Test Today - METS": mets,
        "ECG Resting": ecg,
        "RISK  - Risk Type": risk_type,        
        "Muscle Power - UL - Right": ul_r,
        "Muscle Power - UL - Left": ul_l,
        "Muscle Power - LL - Right": ll_r,
        "Muscle Power - LL - Left": ll_l,
    }

    st.divider()

    if st.button("Generate Exercise Prescription"):

        # Create DataFrame

        patient_df = pd.DataFrame([patient_data])

        # Derived Features

        patient_df["Weekly_Exercise_Duration"] = (
            patient_df["Exercise Habit - Frequency"] *
            patient_df["Exercise Habit - Duration"]
        )

        patient_df["Total_Muscle_Power"] = (
            patient_df["Muscle Power - UL - Right"] +
            patient_df["Muscle Power - UL - Left"] +
            patient_df["Muscle Power - LL - Right"] +
            patient_df["Muscle Power - LL - Left"]
        )

        # Match training preprocessing

        patient_df["Exercise Habit - Mode"] = (
            patient_df["Exercise Habit - Mode"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({
                "no": "others",
                "jogging+cycling": "others",
                "walking+jogging": "others",
                "": "others"
            })
        )

        bmi_map = {
            "underweight": 1,
            "healthy": 2,
            "overweight": 3,
            "obese": 4
        }

        patient_df["Risk Factor - BMI"] = (
            patient_df["Risk Factor - BMI"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(bmi_map)
            .astype(int)
        )

        ef_map = {
            "reduced": 1,
            "borderline": 2,
            "normal": 3
        }

        patient_df["Risk Factor - ECHO - EF"] = (
            patient_df["Risk Factor - ECHO - EF"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace(ef_map)
            .astype(int)
        )

        numeric_cols = [
            "Age",
            "Exercise Habit - Frequency",
            "Exercise Habit - Duration",
            "Risk Factor - BMI",
            "Risk Factor - ECHO - EF",
            "Target HR (bpm)",
            "Weekly_Exercise_Duration",
            "Total_Muscle_Power",
        ]

        patient_df[numeric_cols] = patient_df[numeric_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Prediction

        missing = [
            c for c in risk_features
            if c not in patient_df.columns
        ]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # -------------------------
        # Risk Prediction
        # -------------------------
        risk_encoded = encode_dataframe(
            patient_df,
            risk_encoder,
            risk_features
        )

        risk_input = risk_encoded[risk_features]

        risk_input = risk_input.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(-1).astype(float)

        print(risk_input)
        print(risk_input.dtypes)

        risk_prediction = risk_model.predict(
            risk_input
        )[0]

        predicted_risk = int(risk_prediction)

        risk_label = {
            0: "Low",
            1: "Moderate",
            2: "High"
        }

        risk_prediction_text = risk_label[predicted_risk]

        # -------------------------
        # Frequency Prediction
        # -------------------------
        frequency_df = patient_df.copy()

        frequency_df["Predicted Risk Level Encoded"] = predicted_risk

        frequency_encoded = encode_dataframe(
            frequency_df,
            frequency_encoder,
            frequency_features
        )

        frequency_input = frequency_encoded[
            frequency_features
        ]

        frequency_input = frequency_input.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(-1).astype(float)

        frequency_prediction = frequency_model.predict(
            frequency_input
        )[0]

        frequency_prediction = (
            frequency_target.inverse_transform(
                [frequency_prediction]
            )[0]
        )

        # -------------------------
        # Intensity Prediction
        # -------------------------
        intensity_df = patient_df.copy()

        intensity_df["Predicted Risk Level Encoded"] = predicted_risk

        intensity_encoded = encode_dataframe(
            intensity_df,
            intensity_encoder,
            intensity_features
        )

        intensity_input = intensity_encoded[
            intensity_features
        ]

        intensity_input = intensity_input.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(-1).astype(float)

        print(intensity_input.dtypes)

        intensity_prediction = intensity_model.predict(
            intensity_input
        )[0]

        intensity_prediction = (
            intensity_target.inverse_transform(
                [intensity_prediction]
            )[0]
        )

        # -------------------------
        # Time Prediction
        # -------------------------
        time_df = patient_df.copy()

        time_df["Predicted Risk Level Encoded"] = predicted_risk

        time_encoded = encode_dataframe(
            time_df,
            time_encoder,
            time_features
        )

        time_input = time_encoded[
            time_features
        ]

        time_input = time_input.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(-1).astype(float)

        time_prediction = time_model.predict(
            time_input
        )[0]

        time_prediction = (
            time_target.inverse_transform(
                [time_prediction]
            )[0]
        )

        time_display = {
            "0-20": "0-20",
            "20.1-40": "20-40",
            "40.1-60": "40-60",
            "60.1-90": "60-90"
        }

        time_prediction_text = time_display.get(
            str(time_prediction),
            str(time_prediction)
        )

        # -------------------------
        # Type Prediction
        # -------------------------
        type_df = patient_df.copy()

        type_encoded = encode_dataframe(
            type_df,
            type_encoder,
            type_features
        )

        type_input = type_encoded[
            type_features
        ]

        type_input = type_input.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(-1).astype(float)

        type_input = type_scaler.transform(
            type_input
        )

        type_prediction = np.argmax(
            type_model.predict(type_input),
            axis=1
        )[0]

        type_prediction = (
            type_target.inverse_transform(
                [type_prediction]
            )[0]
        )
            # Display Results

        st.success("Exercise Prescription Generated Successfully")

        st.divider()

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Risk Level",
                risk_prediction_text
            )

            st.metric(
                "Exercise Frequency",
                f"{frequency_prediction} sessions/week"
            )

            st.metric(
                "Exercise Intensity (Target Heart Rate)",
                f"{intensity_prediction} bpm"
            )

        with col2:

            st.metric(
                "Exercise Duration",
                f"{time_prediction_text} min/session"
            )

            st.metric(
                "Exercise Type",
                str(type_prediction).title()
            )

        st.divider()

        st.subheader("Recommended Exercise Prescription")

        result_df = pd.DataFrame({

            "Component": [
                "Risk Level",
                "Exercise Frequency",
                "Exercise Intensity",
                "Exercise Duration",
                "Exercise Type"
            ],

            "Recommendation": [
                risk_prediction_text,
                f"{frequency_prediction} sessions/week",
                f"{intensity_prediction} bpm",
                f"{time_prediction_text} min/session",
                str(type_prediction).title()
            ]

        })

        st.table(result_df)


if page == "About":

    st.header("About")

    st.write("""
This application predicts a personalised cardiac rehabilitation exercise prescription using trained machine learning models.

### Models Used

- Risk Prediction
- Exercise Frequency Prediction
- Exercise Intensity Prediction
- Exercise Duration Prediction
- Exercise Type Prediction

### Input

Clinical assessment data collected during cardiac rehabilitation.

### Output

- Risk Level
- Exercise Frequency
- Exercise Intensity
- Exercise Duration
- Exercise Type

Developed using:

- Streamlit
- Scikit-learn
- TensorFlow / Keras
- CatBoost
- LightGBM
""")