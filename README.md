# Treatment Feasibility Predictor

A Flask web app that predicts whether a treatment is feasible for a patient based on their health records using a machine learning model.

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

## Treatments Covered
- Cancer/Oncology
- Cardiology
- Neurology
- Diabetes/Metabolic
- Nephrology
- Pulmonology
- Medication / Drug Therapy

## Tech Stack
- Python, Flask
- scikit-learn (Gradient Boosting Classifier)
- HTML, CSS, JavaScript

## Ways of Treatment
- Single Assessment
- Compare Treatments
