import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

# Load model, scaler, and feature order
model = load_model('heart_disease_model.h5')
scaler = joblib.load('pipeline.pkl')
feature_order = joblib.load('feature_order.pkl')

# App Title
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the patient data below to predict the likelihood of heart disease.")

# Input Fields
age = st.number_input('Age', min_value=0, max_value=120, value=30)
sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: 'Male' if x == 1 else 'Female')
cp = st.selectbox('Chest Pain Type (cp)', options=[0,1,2,3])
trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=50, max_value=250, value=120)
chol = st.number_input('Cholesterol (chol)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0,1])
restecg = st.selectbox('Resting ECG Results (restecg)', options=[0,1,2])
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=60, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina (exang)', options=[0,1])
oldpeak = st.number_input('Oldpeak (ST depression)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox('Slope of peak exercise ST segment (slope)', options=[0,1,2])
ca = st.selectbox('Number of major vessels colored by fluoroscopy (ca)', options=[0,1,2,3,4])
thal = st.selectbox('Thalassemia (thal)', options=[0,1,2])

# When the user clicks predict
if st.button("Predict"):

    # Prepare input dict
    input_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Handle One-Hot Encoding manually
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Make sure all expected features are present
    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns exactly as training
    input_df = input_df[feature_order]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0][0]

    # Display result
    if prediction >= 0.5:
        st.error(f"⚠️ High chance of heart disease! (Risk Score: {prediction:.2f})")
    else:
        st.success(f"✅ Low chance of heart disease. (Risk Score: {prediction:.2f})")
