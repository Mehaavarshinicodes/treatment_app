# Treatment Feasibility Predictor

A Flask web app that predicts whether a treatment is feasible for a patient based on their health records using a machine learning model.

# ML Model 
- Algorithms: Gradient Boosting, Logistic Regression, Neural Network(MLP), Random Forest
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
Home Page:
<img width="1902" height="1005" alt="image" src="https://github.com/user-attachments/assets/5f2439d5-7e9c-4100-9ae7-e63ef39a51df" />

<img width="444" height="895" alt="image" src="https://github.com/user-attachments/assets/4c5964e6-6b64-4a33-acf5-c9d03193bb99" />
<img width="577" height="785" alt="image" src="https://github.com/user-attachments/assets/3eadb30b-0324-4be5-89fb-d39af014efee" />
<img width="481" height="883" alt="image" src="https://github.com/user-attachments/assets/d2fc1c72-3065-42cc-b3f4-222b3e064979" />

Downloadable PDF report:
<img width="1027" height="957" alt="image" src="https://github.com/user-attachments/assets/d6deb951-bc5e-4705-bc2e-ce415163ab97" />

- scikit-learn (Gradient Boosting Classifier)
- HTML, CSS, JavaScript

## Ways of Treatment
- Single Assessment
- Compare Treatments
