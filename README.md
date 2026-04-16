# Treatment Feasibility Predictor

A Flask web app that predicts whether a treatment is feasible for a patient based on their health records using a machine learning model.

# ML Model 
- Algorithm: Gradient Boosting
- Features Used
     Patient Information
        - Age  
        - Gender  
        - BMI  
    Vital Details
        - Systolic Blood Pressure  
        - Diastolic Blood Pressure  
     Lab Test Results
        - Blood Glucose  
        - Insulin  
        - Cholesterol  
        - Creatinine  
        - Hemoglobin  
     Medical Conditions (Binary)
        - Diabetes  
        - Hypertension  
        - Heart Disease  
        - Kidney Disease  
        - Liver Disease  
     Allergies
        - Penicillin  
        - Sulfa  
        - NSAIDs  
     Treatment Information
        - Treatment Type
            - Cancer/Oncology
            - Cardiology
            - Neurology
            - Diabetes/Metabolic
            - Nephrology
            - Pulmonology
            - Medication / Drug Therapy


## Project Structure

```
treatment_app/
├── app.py                  # Flask backend (main application)
├── train_model.py          # Script to train and save ML model
├── treatment_model.pkl     # Trained ML model
├── patient_dataset.csv     # Dataset used for training
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to ignore in Git
├── README.md               # Project documentation
└── templates/              # Frontend HTML pages
    ├── checker.html        # Treatment feasibility checker UI
    ├── dashboard.html      # User dashboard
    ├── login.html          # Login page
    └── profile.html        # User profile page
```

## Setup Instructions

### 1. Clone the repo
```
git clone https://github.com/YOUR_USERNAME/treatment_app.git
cd treatment_app
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Add the dataset
Place `patient_dataset.csv` in the root `treatment_app/` folder.
*(Not included in the repo — share it with teammates separately via email or Google Drive)*

### 4. Train the model
```
python train_model.py
```
This generates `treatment_model.pkl` in the same folder.

### 5. Run the app
```
python app.py
```

### 6. Open in browser
```
http://127.0.0.1:5000
```

## Tech Stack
- Python, Flask

## Screenshots
<img width="1893" height="1017" alt="image" src="https://github.com/user-attachments/assets/4a086910-6fd3-491c-9667-980755c2aa3c" />
<img width="467" height="777" alt="image" src="https://github.com/user-attachments/assets/7fc19424-02e9-4a13-bdfb-9272d66d78fd" />
<img width="1897" height="1015" alt="image" src="https://github.com/user-attachments/assets/3dfe879f-59d2-41f5-9488-3902c1f57f84" />
<img width="1883" height="1021" alt="image" src="https://github.com/user-attachments/assets/c2ab6927-bb49-4f86-aaad-d2e17f0b1c90" />


- scikit-learn (Gradient Boosting Classifier)
- HTML, CSS, JavaScript

## Ways of Treatment
- Single Assessment
- Compare Treatments
