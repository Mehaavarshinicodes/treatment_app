import warnings
warnings.filterwarnings("ignore")
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

df = pd.read_csv("patient_dataset.csv")
df["treatment_type"] = df["treatment_name"].map({t: i for i, t in enumerate(TREATMENTS)})

X = df[FEATURE_COLS]
y = df["feasible"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                        learning_rate=0.05, subsample=0.8, random_state=42))
])
pipeline.fit(X_train, y_train)

with open("treatment_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✔ Model saved to treatment_model.pkl")