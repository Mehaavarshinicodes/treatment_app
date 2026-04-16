import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# ── All treatments across all domains (order must match app.py) ────────────────
ALL_TREATMENTS = [
    # Cancer
    "Chemotherapy", "Radiation Therapy", "Immunotherapy",
    "Hormone Therapy", "Surgical Intervention", "Targeted Therapy",
    "CAR-T Cell Therapy", "Bone Marrow Transplant",
    # Cardiology
    "Antihypertensives", "Beta Blockers", "ACE Inhibitors", "Statins",
    "Anticoagulants", "Angioplasty / Stenting", "Bypass Surgery (CABG)",
    "Cardiac Resynchronization",
    # Neurology
    "Anticonvulsants", "Corticosteroids", "Thrombolytics (Stroke)",
    "Deep Brain Stimulation", "Levodopa / Dopaminergics",
    "MS Disease Modifiers", "Intrathecal Chemotherapy", "Neuropathic Pain Agents",
    # Diabetes
    "Metformin", "Insulin Therapy", "GLP-1 Agonists", "SGLT-2 Inhibitors",
    "DPP-4 Inhibitors", "Bariatric Surgery",
    "Continuous Glucose Monitoring", "Pancreatic Islet Transplant",
    # Nephrology
    "ACE Inhibitors / ARBs", "Diuretics", "Erythropoiesis Stimulants",
    "Phosphate Binders", "Hemodialysis", "Peritoneal Dialysis",
    "Kidney Transplant", "Immunosuppressants",
    # Pulmonology
    "Bronchodilators", "Inhaled Corticosteroids", "Oxygen Therapy",
    "Pulmonary Rehabilitation", "Biologics (Anti-IL)", "Antifibrotics",
    "Non-invasive Ventilation", "Lung Transplant",
]

FEATURE_COLS = [
    "age", "gender", "bmi", "systolic_bp", "diastolic_bp",
    "blood_glucose", "insulin", "cholesterol", "creatinine", "hemoglobin",
    "has_diabetes", "has_hypertension", "has_heart_disease",
    "has_kidney_disease", "has_liver_disease",
    "allergy_penicillin", "allergy_sulfa", "allergy_nsaid",
    "treatment_type"
]

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("patient_dataset.csv")

# Map treatment names to indices
treatment_map = {t: i for i, t in enumerate(ALL_TREATMENTS)}
df["treatment_type"] = df["treatment_name"].map(treatment_map)

# Drop rows with unknown treatments
unknown = df["treatment_type"].isna().sum()
if unknown > 0:
    print(f"WARNING: {unknown} rows with unmapped treatment names dropped.")
    df = df.dropna(subset=["treatment_type"])

df["treatment_type"] = df["treatment_type"].astype(int)

X = df[FEATURE_COLS]
y = df["feasible"]

print(f"Dataset: {len(df)} rows, class balance: {y.value_counts().to_dict()}")

# ── Train/test split ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Pipeline ───────────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        min_samples_leaf=10,
        random_state=42
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

# ── Evaluation ─────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n── Evaluation on test set ──")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\n── Classification Report ──")
print(classification_report(y_test, y_pred))

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── Save ───────────────────────────────────────────────────────────────────────
with open("treatment_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel saved to treatment_model.pkl")
