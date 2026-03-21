from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# -----------------------------
# Load model + training columns
# -----------------------------
MODEL_PATH = "stroke_rf_model.pkl"
COLUMNS_PATH = "model_columns.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError(f"Columns file not found: {COLUMNS_PATH}")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# model_columns may be Index/list/array -> normalize to list
model_columns = list(model_columns)


# -----------------------------
# Helper functions
# -----------------------------
def age_category(age: float) -> str:
    """Match your notebook logic as closely as possible."""
    if age < 30:
        return "Young"
    elif age < 50:
        return "Adult"
    elif age < 65:
        return "Senior"
    else:
        return "High_Risk"   # use underscore to match common dummy column naming style in your screenshots


def build_feature_row(form_data: dict) -> pd.DataFrame:
    """
    Convert raw form inputs into the exact feature format expected by the trained model.
    This reproduces your notebook preprocessing + feature engineering.
    """
    # Parse inputs safely
    gender = form_data.get("gender", "Male")
    age = float(form_data.get("age", 0))
    hypertension = int(form_data.get("hypertension", 0))
    heart_disease = int(form_data.get("heart_disease", 0))
    ever_married = form_data.get("ever_married", "No")   # form sends Yes/No here
    work_type = form_data.get("work_type", "Private")
    residence_type = form_data.get("residence_type", "Urban")
    avg_glucose_level = float(form_data.get("avg_glucose_level", 0))
    bmi = float(form_data.get("bmi", 0))
    smoking_status = form_data.get("smoking_status", "Unknown")

    # Base row (same raw columns as training before get_dummies)
    row = {
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": 1 if ever_married == "Yes" else 0,  # same map({'Yes':1,'No':0})
        "work_type": work_type,
        "Residence_type": residence_type,   # IMPORTANT: same capitalization as dataset
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }

    df = pd.DataFrame([row])

    # -----------------------------
    # Feature engineering (same idea as notebook)
    # -----------------------------
    df["age_category"] = df["age"].apply(age_category)
    df["high_glucose"] = df["avg_glucose_level"].apply(lambda x: 1 if x > 140 else 0)
    df["cardio_risk"] = df.apply(
        lambda r: 1 if (r["hypertension"] == 1 or r["heart_disease"] == 1) else 0,
        axis=1
    )
    df["obese"] = df["bmi"].apply(lambda x: 1 if x >= 30 else 0)

    # One-hot encode exactly like training
    df = pd.get_dummies(df, drop_first=True)

    # Convert boolean columns only (avoid ruining float precision)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Align to training columns exactly
    # Add missing columns = 0
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # Drop any extra columns that training model doesn't know
    df = df[model_columns]

    # Final dtype cleanup (safe)
    for col in ["age", "avg_glucose_level", "bmi"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    return df


def probability_to_risk(prob: float):
    """
    Map probability (0 to 1) to risk category.
    You can tune these thresholds later.
    """
    score = prob * 100

    if score < 35:
        return "Low Risk", "low"
    elif score < 55:
        return "Mild Risk", "mild"
    elif score < 75:
        return "Moderate Risk", "moderate"
    else:
        return "Critical Risk", "critical"


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = {}   # <-- add this

    if request.method == "POST":
        try:
            # Save submitted values so form stays filled after prediction
            form_values = request.form.to_dict()

            # Build model-ready features
            X_input = build_feature_row(request.form)

            # Probability of stroke (class 1)
            prob = float(model.predict_proba(X_input)[0][1])

            threshold = 0.40
            pred_class = 1 if prob >= threshold else 0

            result = {
                "probability": round(prob * 100, 2),
                "prediction": "Stroke Risk Flag Detected" if pred_class == 1 else "No Stroke Risk Flag",
                "threshold_used": threshold,
            }

        except Exception as e:
            error = f"Prediction failed: {str(e)}"
            # Keep values even if prediction fails
            form_values = request.form.to_dict()

    return render_template("index.html", result=result, error=error, form_values=form_values)


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
