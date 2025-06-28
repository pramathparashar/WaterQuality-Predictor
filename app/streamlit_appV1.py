import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Water Potability Predictor", layout="centered")

# Title
st.title("💧 Water Potability Predictor")

# Load model and scaler
try:
    model = joblib.load("models/water_potability_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    st.success("✅ Model and Scaler loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model or scaler: {e}")
    st.stop()

# Input section
st.header("Enter Water Quality Parameters")

ph = st.number_input("pH (0.0 - 14.0)", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0)
solids = st.number_input("Solids (ppm)", min_value=0.0, value=500.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=4.0)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=7.0)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=60.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0)

# Predict
if st.button("🔮 Predict Potability"):
    input_df = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity
    }])

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Predict and get probabilities
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    # Interpret prediction correctly: 0 = Potable, 1 = Not Potable
    confidence = probabilities[prediction] * 100
    class_labels = {0: "POTABLE", 1: "NOT POTABLE"}

    if prediction == 0:
        st.success(f"✅ The water is {class_labels[prediction]} (Confidence: {confidence:.2f}%)")
    else:
        st.warning(f"⚠️ The water is {class_labels[prediction]} (Confidence: {confidence:.2f}%)")

    # Show full probabilities
    st.markdown("### 🔍 Class Probabilities:")
    st.json({
        "Not Potable": f"{probabilities[1] * 100:.2f}%",
        "Potable": f"{probabilities[0] * 100:.2f}%"
    })
