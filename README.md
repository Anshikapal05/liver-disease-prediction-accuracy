# Liver Disease Prediction & Mortality Risk Predictor

Streamlit app for **two-stage ICU risk prediction** on PhysioNet MIMIC-II biomarkers (n=1750, 19.1% mortality):

1. **Liver disease** — Random Forest (ROC-AUC 0.8635, 11 features)
2. **Mortality risk** — Random Forest (ROC-AUC 0.7886, 14 features) → Low / Medium / High using 33rd/67th percentile thresholds (~23.9% / ~40.3%)

> **Disclaimer:** For research purposes only. Not for clinical use.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Verify models:

```bash
python predictor.py
```

Use **Load Sample Patient** in the sidebar to test the UI.

## Project layout

```
├── app.py                    # Streamlit entry point
├── streamlit_app.py          # Main UI (single patient, batch CSV, model info)
├── predictor.py              # Load models, predict, validate inputs
├── Copy_of_Final_Liver.ipynb # Training notebook (Colab)
├── requirements.txt
├── sample_patients.csv       # Example batch file (3 rows)
└── saved_models/             # joblib artifacts from notebook
    ├── liver_disease_model.pkl
    ├── mortality_model.pkl
    ├── scaler_liver.pkl
    ├── scaler_mortality.pkl
    ├── thresholds.pkl
    ├── feature_names_liver.pkl
    └── feature_names_mortality.pkl
```

**Liver model features (11):** BUN, GCS, Urine, HCO3, Glucose, Age, Creatinine, NIMAP, Platelets, NIDiasABP, HR

**Mortality model features (14):** above plus Bilirubin, ALP, Albumin

## Batch prediction

Upload a CSV with the 14 mortality-model columns. Download adds: `Liver_Prob`, `Liver_Label`, `Mortality_Prob`, `Risk_Category`.

## Training notebook

Model development: `Copy_of_Final_Liver.ipynb`. Re-run the final save cell to regenerate `saved_models/`. If you download from Colab, rename files to match the names above (remove browser suffixes like `(1)`).
