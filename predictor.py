"""
ICU Liver Disease & Mortality Risk — prediction module.
Loads pre-trained models from saved_models/ and runs the two-stage pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import streamlit as st

# Mortality model uses the full feature set; liver model uses a subset.
LIVER_FEATURES = [
    "BUN",
    "GCS",
    "Urine",
    "HCO3",
    "Glucose",
    "Age",
    "Creatinine",
    "NIMAP",
    "Platelets",
    "NIDiasABP",
    "HR",
]

MORTALITY_FEATURES = [
    "BUN",
    "GCS",
    "Urine",
    "HCO3",
    "Bilirubin",
    "ALP",
    "Glucose",
    "Age",
    "Creatinine",
    "NIMAP",
    "Albumin",
    "Platelets",
    "NIDiasABP",
    "HR",
]

# All inputs required for batch CSV and the UI form.
FEATURES = MORTALITY_FEATURES

# Clinical reference ranges: (min, max, unit)
NORMAL_RANGES: dict[str, tuple[float, float, str]] = {
    "BUN": (7.0, 20.0, "mg/dL"),
    "GCS": (13.0, 15.0, "points"),
    "Urine": (500.0, 2000.0, "mL/24h"),
    "HCO3": (22.0, 29.0, "mEq/L"),
    "Bilirubin": (0.2, 1.2, "mg/dL"),
    "ALP": (44.0, 120.0, "U/L"),
    "Glucose": (70.0, 140.0, "mg/dL"),
    "Age": (18.0, 100.0, "years"),
    "Creatinine": (0.6, 1.2, "mg/dL"),
    "NIMAP": (70.0, 100.0, "mmHg"),
    "Albumin": (3.5, 5.0, "g/dL"),
    "Platelets": (150.0, 400.0, "x10³/µL"),
    "NIDiasABP": (60.0, 90.0, "mmHg"),
    "HR": (60.0, 100.0, "bpm"),
}

RISK_COLORS = {
    "Low Risk": "#2ecc71",
    "Medium Risk": "#f39c12",
    "High Risk": "#e74c3c",
}

SAMPLE_PATIENTS = {
    "A — Stable ICU Patient": {
        "BUN": 14,
        "GCS": 15,
        "Urine": 1500,
        "HCO3": 25,
        "Bilirubin": 0.8,
        "ALP": 70,
        "Glucose": 110,
        "Age": 45,
        "Creatinine": 0.9,
        "NIMAP": 85,
        "Albumin": 4.0,
        "Platelets": 220,
        "NIDiasABP": 75,
        "HR": 78,
    },
    "B — Moderate Risk": {
        "BUN": 32,
        "GCS": 12,
        "Urine": 800,
        "HCO3": 20,
        "Bilirubin": 3.5,
        "ALP": 280,
        "Glucose": 165,
        "Age": 62,
        "Creatinine": 1.8,
        "NIMAP": 72,
        "Albumin": 2.8,
        "Platelets": 110,
        "NIDiasABP": 58,
        "HR": 98,
    },
    "C — High Risk / Multi-organ": {
        "BUN": 58,
        "GCS": 8,
        "Urine": 350,
        "HCO3": 14,
        "Bilirubin": 8.5,
        "ALP": 520,
        "Glucose": 220,
        "Age": 76,
        "Creatinine": 3.5,
        "NIMAP": 58,
        "Albumin": 2.0,
        "Platelets": 55,
        "NIDiasABP": 48,
        "HR": 115,
    },
}

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "saved_models"


def _find_pkl(model_dir: Path, prefix: str) -> Path:
    """Resolve a .pkl file whose name starts with prefix (handles '(1)' suffixes)."""
    matches = sorted(model_dir.glob(f"{prefix}*.pkl"))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{prefix}*.pkl' in {model_dir}. "
            f"Found: {[p.name for p in model_dir.glob('*.pkl')]}"
        )
    return matches[0]


def _load_artifacts(model_dir: Path | str | None = None) -> dict[str, Any]:
    """Load all joblib artifacts from disk (uncached)."""
    base = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    thresholds = joblib.load(_find_pkl(base, "thresholds"))

    liver_features = list(
        thresholds.get("feature_names_liver")
        or joblib.load(_find_pkl(base, "feature_names_liver"))
    )
    mortality_features = list(
        thresholds.get("feature_names_mortality")
        or joblib.load(_find_pkl(base, "feature_names_mortality"))
    )

    return {
        "liver_model": joblib.load(_find_pkl(base, "liver_disease_model")),
        "mortality_model": joblib.load(_find_pkl(base, "mortality_model")),
        "scaler_liver": joblib.load(_find_pkl(base, "scaler_liver")),
        "scaler_mortality": joblib.load(_find_pkl(base, "scaler_mortality")),
        "liver_features": liver_features,
        "mortality_features": mortality_features,
        "low_thresh": float(thresholds["low_thresh"]),
        "high_thresh": float(thresholds["high_thresh"]),
    }


@st.cache_resource
def load_models() -> dict[str, Any]:
    """Load and cache models (call once per Streamlit session)."""
    return _load_artifacts()


def get_risk_color(risk_category: str) -> str:
    """Return hex color for a risk category label."""
    return RISK_COLORS.get(risk_category, "#95a5a6")


def validate_inputs(patient_data: dict[str, float]) -> list[dict[str, str]]:
    """
    Return warnings for biomarkers outside normal reference ranges.
    Each item: {feature, message, severity} where severity is 'borderline' or 'critical'.
    """
    warnings: list[dict[str, str]] = []
    for feature in FEATURES:
        if feature not in patient_data:
            continue
        value = float(patient_data[feature])
        lo, hi, unit = NORMAL_RANGES[feature]
        width = hi - lo
        if lo <= value <= hi:
            continue
        if value < lo:
            margin = lo - value
            severity = "borderline" if margin <= 0.2 * width else "critical"
            warnings.append(
                {
                    "feature": feature,
                    "message": (
                        f"{feature} = {value} {unit} is below normal "
                        f"({lo}–{hi} {unit})"
                    ),
                    "severity": severity,
                }
            )
        else:
            margin = value - hi
            severity = "borderline" if margin <= 0.2 * width else "critical"
            warnings.append(
                {
                    "feature": feature,
                    "message": (
                        f"{feature} = {value} {unit} is above normal "
                        f"({lo}–{hi} {unit})"
                    ),
                    "severity": severity,
                }
            )
    return warnings


def is_outside_normal(feature: str, value: float) -> bool:
    """True if value is outside the reference range for feature."""
    lo, hi, _ = NORMAL_RANGES[feature]
    return value < lo or value > hi


def predict_patient(
    patient_data: dict[str, float],
    artifacts: dict[str, Any] | None = None,
) -> tuple[float, int, float, str]:
    """
    Two-stage prediction on raw (unscaled) clinical values.

    Returns:
        liv_prob   — liver disease probability (0–1)
        liv_label  — 0 = no disease, 1 = disease
        mort_prob  — mortality probability (0–1)
        risk       — 'Low Risk' | 'Medium Risk' | 'High Risk'
    """
    if artifacts is None:
        artifacts = load_models()

    liver_features = artifacts["liver_features"]
    mortality_features = artifacts["mortality_features"]
    low_t = artifacts["low_thresh"]
    high_t = artifacts["high_thresh"]

    X_liv = np.array([[patient_data[f] for f in liver_features]])
    X_mort = np.array([[patient_data[f] for f in mortality_features]])

    X_liv = artifacts["scaler_liver"].transform(X_liv)
    X_mort = artifacts["scaler_mortality"].transform(X_mort)

    liver_model = artifacts["liver_model"]
    mortality_model = artifacts["mortality_model"]

    liv_prob = float(liver_model.predict_proba(X_liv)[0][1])
    liv_label = int(liver_model.predict(X_liv)[0])
    mort_prob = float(mortality_model.predict_proba(X_mort)[0][1])

    if mort_prob < low_t:
        risk = "Low Risk"
    elif mort_prob < high_t:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return liv_prob, liv_label, mort_prob, risk


def predict_batch(
    df,
    artifacts: dict[str, Any] | None = None,
):
    """Run predict_patient on each row; return DataFrame with result columns."""
    import pandas as pd

    if artifacts is None:
        artifacts = load_models()

    rows = []
    for _, row in df.iterrows():
        patient = {f: float(row[f]) for f in FEATURES}
        liv_prob, liv_label, mort_prob, risk = predict_patient(patient, artifacts)
        rows.append(
            {
                "Liver_Prob": round(liv_prob * 100, 2),
                "Liver_Label": liv_label,
                "Mortality_Prob": round(mort_prob * 100, 2),
                "Risk_Category": risk,
            }
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


if __name__ == "__main__":
    artifacts = _load_artifacts()
    print("Loaded models from:", DEFAULT_MODEL_DIR)
    print(f"Liver features ({len(artifacts['liver_features'])}): {artifacts['liver_features']}")
    print(
        f"Mortality features ({len(artifacts['mortality_features'])}): "
        f"{artifacts['mortality_features']}"
    )
    print(
        f"Thresholds: low={artifacts['low_thresh']:.4f}, "
        f"high={artifacts['high_thresh']:.4f}\n"
    )

    for name, patient in SAMPLE_PATIENTS.items():
        liv_prob, liv_label, mort_prob, risk = predict_patient(patient, artifacts)
        print(f"{name}")
        print(f"  Liver: {liv_prob * 100:.2f}%  label={liv_label}")
        print(f"  Mortality: {mort_prob * 100:.2f}%  risk={risk}")
        print()
