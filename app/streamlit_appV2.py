import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="Water Potability Predictor 💧", layout="centered")

# ---------- LOAD MODEL & SCALER ----------
model_path = os.path.join("models", "water_potability_model.pkl")
scaler_path = os.path.join("models", "scaler.pkl")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    model_loaded = False

# ---------- SIDEBAR ----------
st.sidebar.image("docs/water.jpg", width=160)  # Optional logo
st.sidebar.title("📌 About")
st.sidebar.markdown("""
This AI-powered Streamlit app predicts whether water is **potable** or **not potable** using:
- 9 water quality parameters
- A calibrated XGBoost model
- SMOTE + StandardScaler
""")
st.sidebar.info("Created by **Pramath Parashar**")

# ---------- TITLE ----------
st.markdown("<h1 style='text-align: center;'>💧 Water Potability Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------- TABS ----------
tab1, tab2 = st.tabs(["🔮 Predictor", "📈 Model Summary"])

# ========== TAB 1: PREDICTOR ==========
with tab1:

    st.subheader("🧪 Enter Water Sample Characteristics")

    col1, col2 = st.columns(2)

    with col1:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1, value=7.0)
        hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0)
        solids = st.number_input("Solids (ppm)", min_value=0.0, value=10000.0)
        chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=5.0)
        sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0)

    with col2:
        conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0)
        organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=10.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=60.0)
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0)

    st.markdown("---")

    if st.button("🚀 Predict Potability", use_container_width=True):

        if not model_loaded:
            st.error("⚠️ Model not available.")
        else:
            # Prepare input
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

            input_scaled = scaler.transform(input_df)
            proba = model.predict_proba(input_scaled)[0]
            pred = int(np.argmax(proba))
            confidence = round(100 * proba[pred], 2)

            # Result
            if pred == 0:
                st.success(f"✅ The water is **POTABLE** (Confidence: {confidence}%)")
            else:
                st.warning(f"⚠️ The water is **NOT POTABLE** (Confidence: {confidence}%)")

            # Probabilities
            st.markdown("### 🔍 Class Probabilities:")
            st.json({
                "Not Potable": f"{proba[1]*100:.2f}%",
                "Potable": f"{proba[0]*100:.2f}%"
            })

# ========== TAB 2: MODEL SUMMARY ==========
with tab2:
    st.subheader("📊 Model Overview")
    st.markdown("""
This model was trained using:

- 🔄 **SMOTE** to balance classes
- ⚖️ **StandardScaler** for feature scaling
- 🌲 **XGBoost** for classification
- ✅ **CalibratedClassifierCV** to ensure probability accuracy

Final Test Accuracy: **~65%**
    """)
    st.image("docs/architecture_diagram.png", caption="Model Pipeline", use_container_width=True)
    st.image("docs/feature_importance.png", caption="Feature Importance (XGBoost)", use_container_width=True)
