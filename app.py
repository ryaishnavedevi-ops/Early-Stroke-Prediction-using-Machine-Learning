from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import logging

app = Flask(__name__)

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Load model + training columns
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "stroke_rf_model.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "model_columns.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError(f"Columns file not found: {COLUMNS_PATH}")

model = joblib.load(MODEL_PATH)
model_columns = list(joblib.load(COLUMNS_PATH))

logging.info("Model and columns loaded successfully")

# -----------------------------
# Helper functions
# -----------------------------
def safe_float(value, default=0.0):
    try:
        return float(value)
    except:
        return default


def age_category(age: float) -> str:
    if age < 30:
        return "Young"
    elif age < 50:
        return "Adult"
    elif age < 65:
        return "Senior"
    else:
        return "High_Risk"


def probability_to_risk(prob: float):
    score = prob * 100

    if score < 35:
        return "Low Risk", "low"
    elif score < 55:
        return "Mild Risk", "mild"
    elif score < 75:
        return "Moderate Risk", "moderate"
    else:
        return "Critical Risk", "critical"


def build_feature_row(form_data: dict) -> pd.DataFrame:
    # Safe parsing
    gender = form_data.get("gender", "Male")
    age = safe_float(form_data.get("age"))
    hypertension = int(form_data.get("hypertension", 0))
    heart_disease = int(form_data.get("heart_disease", 0))
    ever_married = form_data.get("ever_married", "No")
    work_type = form_data.get("work_type", "Private")
    residence_type = form_data.get("residence_type", "Urban")
    avg_glucose_level = safe_float(form_data.get("avg_glucose_level"))
    bmi = safe_float(form_data.get("bmi"))
    smoking_status = form_data.get("smoking_status", "Unknown")

    row = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": 1 if ever_married == "Yes" else 0,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    df = pd.DataFrame([row])

    # Feature engineering
    df["age_category"] = df["age"].apply(age_category)
    df["high_glucose"] = df["avg_glucose_level"].apply(lambda x: 1 if x > 140 else 0)
    df["cardio_risk"] = df.apply(
        lambda r: 1 if (r["hypertension"] == 1 or r["heart_disease"] == 1) else 0,
        axis=1
    )
    df["obese"] = df["bmi"].apply(lambda x: 1 if x >= 30 else 0)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    # Convert bools to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Align columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    # Ensure numeric types
    for col in ["age", "avg_glucose_level", "bmi"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {}

    if request.method == "POST":
        try:
            form_values = request.form.to_dict()

            X_input = build_feature_row(request.form)

            prob = float(model.predict_proba(X_input)[0][1])

            # 🔥 Your chosen threshold
            threshold = 0.40
            pred_class = 1 if prob >= threshold else 0

            risk_label, risk_level = probability_to_risk(prob)

            result = {
                "probability": round(prob * 100, 2),
                "prediction": "Stroke Risk Flag Detected" if pred_class == 1 else "No Stroke Risk Flag",
                "risk_label": risk_label,
                "risk_level": risk_level,
                "threshold_used": threshold,
            }

        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            form_values = request.form.to_dict()

    return render_template("index.html", result=result, error=error, form_values=form_values)


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run()
