import io
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "medcheck_secret_key_change_in_production"

DOMAIN_TREATMENTS = {
    "cancer": [
        "Chemotherapy", "Radiation Therapy", "Immunotherapy",
        "Hormone Therapy", "Surgical Intervention", "Targeted Therapy",
        "CAR-T Cell Therapy", "Bone Marrow Transplant"
    ],
    "cardio": [
        "Antihypertensives", "Beta Blockers", "ACE Inhibitors", "Statins",
        "Anticoagulants", "Angioplasty / Stenting", "Bypass Surgery (CABG)",
        "Cardiac Resynchronization"
    ],
    "neuro": [
        "Anticonvulsants", "Corticosteroids", "Thrombolytics (Stroke)",
        "Deep Brain Stimulation", "Levodopa / Dopaminergics",
        "MS Disease Modifiers", "Intrathecal Chemotherapy", "Neuropathic Pain Agents"
    ],
    "diabetes": [
        "Metformin", "Insulin Therapy", "GLP-1 Agonists", "SGLT-2 Inhibitors",
        "DPP-4 Inhibitors", "Bariatric Surgery",
        "Continuous Glucose Monitoring", "Pancreatic Islet Transplant"
    ],
    "nephro": [
        "ACE Inhibitors / ARBs", "Diuretics", "Erythropoiesis Stimulants",
        "Phosphate Binders", "Hemodialysis", "Peritoneal Dialysis",
        "Kidney Transplant", "Immunosuppressants"
    ],
    "pulmo": [
        "Bronchodilators", "Inhaled Corticosteroids", "Oxygen Therapy",
        "Pulmonary Rehabilitation", "Biologics (Anti-IL)", "Antifibrotics",
        "Non-invasive Ventilation", "Lung Transplant"
    ],
}

ALL_TREATMENTS = [t for treatments in DOMAIN_TREATMENTS.values() for t in treatments]

FEATURE_COLS = [
    "age", "gender", "bmi", "systolic_bp", "diastolic_bp",
    "blood_glucose", "insulin", "cholesterol", "creatinine", "hemoglobin",
    "has_diabetes", "has_hypertension", "has_heart_disease",
    "has_kidney_disease", "has_liver_disease",
    "allergy_penicillin", "allergy_sulfa", "allergy_nsaid",
    "treatment_type"
]

FEATURE_LABELS = {
    "age": "Age", "gender": "Gender", "bmi": "BMI",
    "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP",
    "blood_glucose": "Blood Glucose", "insulin": "Insulin",
    "cholesterol": "Cholesterol", "creatinine": "Creatinine",
    "hemoglobin": "Hemoglobin", "has_diabetes": "Diabetes",
    "has_hypertension": "Hypertension", "has_heart_disease": "Heart Disease",
    "has_kidney_disease": "Kidney Disease", "has_liver_disease": "Liver Disease",
    "allergy_penicillin": "Penicillin Allergy", "allergy_sulfa": "Sulfa Allergy",
    "allergy_nsaid": "NSAID Allergy", "treatment_type": "Treatment Type"
}

MODEL_LABELS = {
    "gradient_boosting":   "Gradient Boosting",
    "random_forest":       "Random Forest",
    "logistic_regression": "Logistic Regression",
    "neural_network":      "Neural Network (MLP)",
}

# Load all 3 models
models = {}
try:
    with open("treatment_model.pkl", "rb") as f:
        models = pickle.load(f)
    print(f"Loaded {len(models)} models: {list(models.keys())}")
except FileNotFoundError:
    print("WARNING: treatment_model.pkl not found. Run train_model.py first.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_row(data, treatment_name):
    treatment_type = ALL_TREATMENTS.index(treatment_name)
    return pd.DataFrame([[
        int(data.get("age", 0)),        int(data.get("gender", 0)),
        float(data.get("bmi", 0)),      int(data.get("systolic_bp", 0)),
        int(data.get("diastolic_bp", 0)), int(data.get("blood_glucose", 0)),
        int(data.get("insulin", 0)),    int(data.get("cholesterol", 0)),
        float(data.get("creatinine", 0)), float(data.get("hemoglobin", 0)),
        int(data.get("has_diabetes", 0)),      int(data.get("has_hypertension", 0)),
        int(data.get("has_heart_disease", 0)), int(data.get("has_kidney_disease", 0)),
        int(data.get("has_liver_disease", 0)), int(data.get("allergy_penicillin", 0)),
        int(data.get("allergy_sulfa", 0)),     int(data.get("allergy_nsaid", 0)),
        treatment_type
    ]], columns=FEATURE_COLS)


def compute_risk(score, comorbidities, domain):
    thresholds = {
        "cancer": (75, 55), "cardio": (80, 60), "neuro": (70, 50),
        "diabetes": (78, 58), "nephro": (72, 52), "pulmo": (75, 55),
    }
    high_t, low_t = thresholds.get(domain, (75, 55))
    if score >= high_t and comorbidities <= 1:
        return "LOW"
    elif score >= low_t:
        return "MODERATE"
    return "HIGH"


def ensemble_predict(row):
    """Run all 3 models and return individual + ensemble scores."""
    individual_scores = {}
    proba_sum = 0.0
    for name, m in models.items():
        p = m.predict_proba(row)[0][1]
        individual_scores[MODEL_LABELS[name]] = round(p * 100, 1)
        proba_sum += p
    ensemble_proba = proba_sum / len(models)
    score = round(ensemble_proba * 100, 1)
    pred  = 1 if ensemble_proba >= 0.5 else 0
    return pred, score, individual_scores


def get_shap_explanation(row):
    try:
        import shap
        m = models.get("gradient_boosting")  # SHAP uses GB (tree-based)
        clf    = m.named_steps["clf"]
        scaler = m.named_steps["scaler"]
        X_scaled = scaler.transform(row)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_scaled)
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
        factors = [{"feature": FEATURE_LABELS[col], "value": float(row.iloc[0][col]), "shap": round(float(vals[i]), 4)}
                   for i, col in enumerate(FEATURE_COLS)]
        factors.sort(key=lambda x: abs(x["shap"]), reverse=True)
        return factors[:6]
    except ImportError:
        raw = row.iloc[0]
        rules = [
            ("creatinine",      raw["creatinine"],      -0.25 if raw["creatinine"] > 1.5 else 0.05),
            ("blood_glucose",   raw["blood_glucose"],   -0.20 if raw["blood_glucose"] > 200 else 0.04),
            ("bmi",             raw["bmi"],             -0.15 if raw["bmi"] > 35 else 0.06),
            ("hemoglobin",      raw["hemoglobin"],      -0.18 if raw["hemoglobin"] < 10 else 0.07),
            ("systolic_bp",     raw["systolic_bp"],     -0.12 if raw["systolic_bp"] > 160 else 0.04),
            ("cholesterol",     raw["cholesterol"],     -0.10 if raw["cholesterol"] > 240 else 0.03),
            ("age",             raw["age"],             -0.08 if raw["age"] > 70 else 0.05),
            ("has_kidney_disease", raw["has_kidney_disease"], -0.20 if raw["has_kidney_disease"] else 0.0),
            ("has_heart_disease",  raw["has_heart_disease"],  -0.15 if raw["has_heart_disease"] else 0.0),
            ("has_diabetes",       raw["has_diabetes"],       -0.10 if raw["has_diabetes"] else 0.0),
        ]
        factors = [{"feature": FEATURE_LABELS[col], "value": float(val), "shap": sv}
                   for col, val, sv in rules if sv != 0.0]
        factors.sort(key=lambda x: abs(x["shap"]), reverse=True)
        return factors[:6]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def root():
    return redirect(url_for("login"))

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/profile")
def profile():
    return render_template("profile.html")

@app.route("/checker")
def checker():
    return render_template("checker.html")


# ── Predict ───────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if not models:
        return jsonify({"error": "Models not loaded. Run train_model.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided."}), 400

    treatment_name = data.get("treatment_name", "")
    domain         = data.get("domain", "cancer")
    patient_name   = data.get("patient_name", "Unknown Patient")

    if treatment_name not in ALL_TREATMENTS:
        return jsonify({"error": f"Unknown treatment: {treatment_name}"}), 400

    try:
        row = build_row(data, treatment_name)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    pred, score, individual_scores = ensemble_predict(row)

    comorbidities = sum([
        int(data.get("has_diabetes", 0)),      int(data.get("has_hypertension", 0)),
        int(data.get("has_heart_disease", 0)), int(data.get("has_kidney_disease", 0)),
        int(data.get("has_liver_disease", 0)),
    ])

    risk = compute_risk(score, comorbidities, domain)
    shap_factors = get_shap_explanation(row)

    result = {
        "feasible":       bool(pred),
        "score":          score,
        "risk":           risk,
        "comorbidities":  comorbidities,
        "treatment":      treatment_name,
        "domain":         domain,
        "patient_name":   patient_name,
        "timestamp":      datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "shap_factors":   shap_factors,
        "model_scores":   individual_scores,
    }

    if "history" not in session:
        session["history"] = []
    history = session["history"]
    history.insert(0, result)
    session["history"] = history[:50]

    return jsonify(result)


# ── Compare ───────────────────────────────────────────────────────────────────

@app.route("/compare", methods=["POST"])
def compare():
    if not models:
        return jsonify({"error": "Models not loaded."}), 503

    data       = request.get_json()
    treatments = data.get("treatments", [])
    domain     = data.get("domain", "cancer")

    if not treatments or len(treatments) < 2:
        return jsonify({"error": "Provide at least 2 treatments to compare."}), 400

    results = []
    for t in treatments:
        if t not in ALL_TREATMENTS:
            continue
        try:
            row = build_row(data, t)
        except (ValueError, TypeError):
            continue

        pred, score, individual_scores = ensemble_predict(row)
        comorbidities = sum([
            int(data.get("has_diabetes", 0)),      int(data.get("has_hypertension", 0)),
            int(data.get("has_heart_disease", 0)), int(data.get("has_kidney_disease", 0)),
            int(data.get("has_liver_disease", 0)),
        ])
        risk = compute_risk(score, comorbidities, domain)
        results.append({
            "treatment":      t,
            "score":          score,
            "feasible":       bool(pred),
            "risk":           risk,
            "model_scores":   individual_scores,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify({"comparisons": results})


# ── History ───────────────────────────────────────────────────────────────────

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify({"history": session.get("history", [])})

@app.route("/history/clear", methods=["POST"])
def clear_history():
    session["history"] = []
    return jsonify({"ok": True})


# ── PDF Export ────────────────────────────────────────────────────────────────

@app.route("/export-pdf", methods=["POST"])
def export_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from flask import send_file

    data          = request.get_json()
    patient_name  = data.get("patient_name", "Unknown Patient")
    treatment     = data.get("treatment", "N/A")
    domain        = data.get("domain", "N/A")
    score         = data.get("score", "--")
    risk          = data.get("risk", "--")
    feasible      = data.get("feasible", False)
    comorbidities = data.get("comorbidities", 0)
    timestamp     = data.get("timestamp", datetime.now().strftime("%d %b %Y, %I:%M %p"))
    shap_factors  = data.get("shap_factors", [])
    model_scores  = data.get("model_scores", {})

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    BLUE  = colors.HexColor("#185FA5")
    LIGHT = colors.HexColor("#E6F1FB")
    MUTED = colors.HexColor("#5F5E5A")
    GREEN = colors.HexColor("#188038")
    RED   = colors.HexColor("#C5221F")
    AMBER = colors.HexColor("#EA8600")
    BLACK = colors.HexColor("#1a1a1a")

    styles = getSampleStyleSheet()
    def sty(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    title_sty   = sty("T",  fontSize=20, textColor=BLUE, fontName="Helvetica-Bold", spaceAfter=2)
    sub_sty     = sty("S",  fontSize=10, textColor=MUTED, spaceAfter=12)
    h2_sty      = sty("H2", fontSize=12, textColor=BLUE, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    body_sty    = sty("B",  fontSize=10, textColor=BLACK, leading=16)
    verdict_sty = sty("V",  fontSize=14, fontName="Helvetica-Bold",
                      textColor=GREEN if feasible else RED, spaceAfter=4)
    risk_color  = {"LOW": GREEN, "MODERATE": AMBER, "HIGH": RED}.get(risk, MUTED)

    story = []

    story.append(Paragraph("MedCheck", title_sty))
    story.append(Paragraph("Treatment Feasibility Assessment Report", sub_sty))
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=12))

    meta = [
        ["Patient",   patient_name,   "Date",    timestamp],
        ["Treatment", treatment,      "Domain",  domain.title()],
    ]
    mt = Table(meta, colWidths=[30*mm, 70*mm, 25*mm, 55*mm])
    mt.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",  (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",  (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (0,-1), MUTED),
        ("TEXTCOLOR", (2,0), (2,-1), MUTED),
        ("TEXTCOLOR", (1,0), (1,-1), BLACK),
        ("TEXTCOLOR", (3,0), (3,-1), BLACK),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(mt)
    story.append(Spacer(1, 8*mm))

    story.append(Paragraph("Assessment Result", h2_sty))
    story.append(Paragraph("Treatment Feasible" if feasible else "Not Recommended", verdict_sty))

    res_data = [
        ["Ensemble Score", "Risk Level", "Comorbidities"],
        [f"{score}%", risk, f"{comorbidities} / 5"],
    ]
    rt = Table(res_data, colWidths=[60*mm, 60*mm, 60*mm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), LIGHT),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",      (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("TEXTCOLOR",     (0,0), (-1,0), BLUE),
        ("TEXTCOLOR",     (0,1), (0,1), BLUE),
        ("TEXTCOLOR",     (1,1), (1,1), risk_color),
        ("TEXTCOLOR",     (2,1), (2,1), BLACK),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#B5D4F4")),
        ("ROWBACKGROUNDS",(0,1), (-1,1),  [colors.HexColor("#F8FAFF")]),
    ]))
    story.append(rt)
    story.append(Spacer(1, 6*mm))

    # Individual model scores table
    if model_scores:
        story.append(Paragraph("Individual Model Scores", h2_sty))
        ms_data = [["Model", "Feasibility Score", "Verdict"]]
        for mname, mscore in model_scores.items():
            ms_data.append([mname, f"{mscore}%", "Feasible" if mscore >= 50 else "Not Feasible"])
        ms_data.append(["Ensemble Average", f"{score}%", "Feasible" if feasible else "Not Feasible"])
        mst = Table(ms_data, colWidths=[80*mm, 55*mm, 45*mm])
        mst.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  LIGHT),
            ("BACKGROUND",    (0,-1),(-1,-1), colors.HexColor("#EBF3FE")),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",      (0,-1),(-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("TEXTCOLOR",     (0,0), (-1,0),  BLUE),
            ("TEXTCOLOR",     (0,-1),(-1,-1), BLUE),
            ("ALIGN",         (1,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#B5D4F4")),
            ("ROWBACKGROUNDS",(0,1), (-1,-2), [colors.white, colors.HexColor("#F8FAFF")]),
        ]))
        story.append(mst)
        story.append(Spacer(1, 6*mm))

    # Patient biomarkers
    story.append(Paragraph("Patient Biomarkers", h2_sty))
    bm_fields = [
        ("Age", data.get("age","--")),
        ("Gender", "Female" if str(data.get("gender","")) == "0" else "Male"),
        ("BMI", data.get("bmi","--")),
        ("Systolic BP", f"{data.get('systolic_bp','--')} mmHg"),
        ("Diastolic BP", f"{data.get('diastolic_bp','--')} mmHg"),
        ("Blood Glucose", f"{data.get('blood_glucose','--')} mg/dL"),
        ("Insulin", f"{data.get('insulin','--')} uU/mL"),
        ("Cholesterol", f"{data.get('cholesterol','--')} mg/dL"),
        ("Creatinine", f"{data.get('creatinine','--')} mg/dL"),
        ("Hemoglobin", f"{data.get('hemoglobin','--')} g/dL"),
    ]
    bm_rows = [bm_fields[i:i+2] for i in range(0, len(bm_fields), 2)]
    bm_data = [["Biomarker", "Value", "Biomarker", "Value"]]
    for row_pair in bm_rows:
        r = []
        for label, val in row_pair:
            r += [label, str(val)]
        if len(r) < 4:
            r += ["", ""]
        bm_data.append(r)
    bmt = Table(bm_data, colWidths=[45*mm, 40*mm, 45*mm, 40*mm])
    bmt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  LIGHT),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (2,1), (2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("TEXTCOLOR",     (0,0), (-1,0),  BLUE),
        ("TEXTCOLOR",     (0,1), (0,-1),  MUTED),
        ("TEXTCOLOR",     (2,1), (2,-1),  MUTED),
        ("ALIGN",         (0,0), (-1,-1), "LEFT"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#B5D4F4")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFF")]),
    ]))
    story.append(bmt)
    story.append(Spacer(1, 6*mm))

    if shap_factors:
        story.append(Paragraph("Key Influencing Factors", h2_sty))
        story.append(Paragraph("Biomarkers with the greatest influence on this prediction:", body_sty))
        story.append(Spacer(1, 3*mm))
        sf_data = [["Biomarker", "Patient Value", "Direction", "Influence"]]
        for f in shap_factors[:6]:
            direction = "Positive" if f["shap"] > 0 else "Negative"
            magnitude = abs(f["shap"])
            level = "High" if magnitude > 0.15 else "Moderate" if magnitude > 0.07 else "Low"
            sf_data.append([f["feature"], str(f["value"]), direction, level])
        sft = Table(sf_data, colWidths=[50*mm, 35*mm, 40*mm, 35*mm])
        sft.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), LIGHT),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("TEXTCOLOR",     (0,0), (-1,0), BLUE),
            ("ALIGN",         (0,0), (-1,-1), "LEFT"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#B5D4F4")),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFF")]),
        ]))
        story.append(sft)
        story.append(Spacer(1, 6*mm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#B5D4F4"), spaceAfter=6))
    story.append(Paragraph(
        "This report is generated by MedCheck AI and is intended as a clinical decision-support tool only. "
        "It does not replace physician judgment. All assessments should be reviewed by a qualified clinician "
        "before any treatment decision is made.",
        sty("disc", fontSize=8, textColor=MUTED, leading=13)
    ))

    doc.build(story)
    buf.seek(0)
    safe_name = patient_name.replace(" ", "_")
    filename  = f"MedCheck_{safe_name}_{treatment.replace('/','-').replace(' ','_')}.pdf"
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="application/pdf")


if __name__ == "__main__":
    app.run(debug=False)