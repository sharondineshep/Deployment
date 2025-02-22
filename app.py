import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("Diabetes Prediction")
st.markdown("### Enter customer details to predict subscription outcome.")

# User Inputs (Matching Features from X_train)
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=100.0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0)
age = st.number_input("Age", min_value=10, max_value=100, value=30)

# Create input feature array
user_input = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi, age]])

# Encode categorical features exactly as in training
input_data = pd.get_dummies(input_data, drop_first=True)

# Ensure columns match model input
missing_cols = set(model.feature_names_in_) - set(input_data.columns)


input_data = input_data[model.feature_names_in_]  # Reorder columns

# Make Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_label = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.success(f"Prediction: {prediction_label}")