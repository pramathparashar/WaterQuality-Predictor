
# 🛠️ How to Run the Water Potability Predictor (Locally)

This guide walks you through the step-by-step process to run the Streamlit-based app on your local Windows 11 machine using a virtual environment.

---

## 📁 Project Folder Structure

```
WaterQuality-Predictor/
├── app/
│   └── streamlit_appV2.py
├── data/
│   └── water_potability_cleaned.csv
├── models/
│   ├── water_potability_model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── model_training_notebook.ipynb
├── docs/
│   ├── architecture_diagram.png
│   ├── streamlit_ui.png   
│   └── feature_importance.png
├── README.md
├── test_cases.md 
├── how_to_run.md
└── venv/
```

---

## 🧱 Step 1: Open Command Prompt in Project Folder

```bash
cd P:\WaterQuality-Predictor
```

---

## 🐍 Step 2: Activate the Virtual Environment

```bash
venv\Scripts\activate
```

You should now see `(venv)` prefix in terminal.

---

## 📦 Step 3: Install Requirements (if needed)

If starting fresh:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit scikit-learn xgboost imbalanced-learn joblib pandas numpy matplotlib seaborn
```

---

## 🚀 Step 4: Run the App

```bash
streamlit run app/streamlit_app.py
```

---

## 🌐 Step 5: Open in Browser

Once it starts, open browser and go to:

```
http://localhost:8501
```

You’ll see the water potability prediction UI. 🎉

---

## 🔁 Optional: Retrain Model

Use the notebook in `notebooks/model_training_notebook.ipynb` to retrain or fine-tune the model and regenerate `water_potability_model.pkl` and `scaler.pkl`.

---

✅ That’s it! You're all set.
