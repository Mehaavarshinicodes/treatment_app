# Treatment Feasibility Predictor

A Flask web app that predicts whether a treatment is feasible for a patient based on their health records using a machine learning model.

## Project Structure

```
treatment_app/
├── app.py              # Flask backend
├── train_model.py      # Script to train and save the model
├── requirements.txt    # Python dependencies
└── templates/
    └── index.html      # Frontend UI
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
- Chemotherapy
- Radiation Therapy
- Immunotherapy
- Hormone Therapy
- Surgical Intervention
- Medication / Drug Therapy

## Tech Stack
- Python, Flask
- scikit-learn (Gradient Boosting Classifier)
- HTML, CSS, JavaScript
