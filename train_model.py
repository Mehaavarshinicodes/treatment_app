import warnings
warnings.filterwarnings("ignore")

import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

ALL_TREATMENTS = [
    "Chemotherapy", "Radiation Therapy", "Immunotherapy",
    "Hormone Therapy", "Surgical Intervention", "Targeted Therapy",
    "CAR-T Cell Therapy", "Bone Marrow Transplant",
    "Antihypertensives", "Beta Blockers", "ACE Inhibitors", "Statins",
    "Anticoagulants", "Angioplasty / Stenting", "Bypass Surgery (CABG)",
    "Cardiac Resynchronization",
    "Anticonvulsants", "Corticosteroids", "Thrombolytics (Stroke)",
    "Deep Brain Stimulation", "Levodopa / Dopaminergics",
    "MS Disease Modifiers", "Intrathecal Chemotherapy", "Neuropathic Pain Agents",
    "Metformin", "Insulin Therapy", "GLP-1 Agonists", "SGLT-2 Inhibitors",
    "DPP-4 Inhibitors", "Bariatric Surgery",
    "Continuous Glucose Monitoring", "Pancreatic Islet Transplant",
    "ACE Inhibitors / ARBs", "Diuretics", "Erythropoiesis Stimulants",
    "Phosphate Binders", "Hemodialysis", "Peritoneal Dialysis",
    "Kidney Transplant", "Immunosuppressants",
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

print("=" * 55)
print("  MEDCHECK — TRAINING 4 MODELS")
print("=" * 55)

print("\nLoading dataset...")
df = pd.read_csv("patient_dataset.csv")
df["treatment_type"] = df["treatment_name"].map({t: i for i, t in enumerate(ALL_TREATMENTS)})
unknown = df["treatment_type"].isna().sum()
if unknown > 0:
    print(f"WARNING: {unknown} rows with unmapped treatments dropped.")
    df = df.dropna(subset=["treatment_type"])
df["treatment_type"] = df["treatment_type"].astype(int)

X = df[FEATURE_COLS]
y = df["feasible"]
print(f"Dataset: {len(df)} rows | Class balance: {y.value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

MODELS = {
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        min_samples_leaf=10, random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=6,
        min_samples_leaf=10, random_state=42, n_jobs=-1
    ),
    "logistic_regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=42
    ),
    "neural_network": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ),
}

trained = {}

for name, clf in MODELS.items():
    print(f"\n── Training: {name.replace('_',' ').title()} ──")
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    cv = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
    print(f"  CV AUC   : {cv.mean():.4f} +/- {cv.std():.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Feasible","Feasible"]))
    trained[name] = pipeline

with open("treatment_model.pkl", "wb") as f:
    pickle.dump(trained, f)

print("=" * 55)
print("  All 4 models saved to treatment_model.pkl")
print("  Run: python app.py")
print("=" * 55)