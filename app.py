import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

TREATMENTS = [
    "Chemotherapy", "Radiation Therapy", "Immunotherapy",
    "Hormone Therapy", "Surgical Intervention", "Medication / Drug Therapy"
]
FEATURE_COLS = [
    "age", "gender", "bmi", "systolic_bp", "diastolic_bp",
    "blood_glucose", "insulin", "cholesterol", "creatinine", "hemoglobin",
    "has_diabetes", "has_hypertension", "has_heart_disease",
    "has_kidney_disease", "has_liver_disease",
    "allergy_penicillin", "allergy_sulfa", "allergy_nsaid",
    "treatment_type"
]

with open("treatment_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html", treatments=TREATMENTS)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    treatment_type = TREATMENTS.index(data["treatment_name"])

    row = pd.DataFrame([[
        int(data["age"]),        int(data["gender"]),     float(data["bmi"]),
        int(data["systolic_bp"]), int(data["diastolic_bp"]), int(data["blood_glucose"]),
        int(data["insulin"]),    int(data["cholesterol"]), float(data["creatinine"]),
        float(data["hemoglobin"]),
        int(data["has_diabetes"]),       int(data["has_hypertension"]),
        int(data["has_heart_disease"]),  int(data["has_kidney_disease"]),
        int(data["has_liver_disease"]),  int(data["allergy_penicillin"]),
        int(data["allergy_sulfa"]),      int(data["allergy_nsaid"]),
        treatment_type
    ]], columns=FEATURE_COLS)

    proba = model.predict_proba(row)[0]
    pred  = model.predict(row)[0]
    score = round(proba[1] * 100, 1)
    comorbidities = sum([
        int(data["has_diabetes"]), int(data["has_hypertension"]),
        int(data["has_heart_disease"]), int(data["has_kidney_disease"]),
        int(data["has_liver_disease"])
    ])
    risk = "LOW" if score >= 80 and comorbidities <= 1 else ("MODERATE" if score >= 60 else "HIGH")

    return jsonify({
        "feasible": bool(pred),
        "score": score,
        "risk": risk,
        "comorbidities": comorbidities,
        "treatment": data["treatment_name"]
    })

if __name__ == "__main__":
    app.run(debug=True)