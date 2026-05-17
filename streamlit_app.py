"""
Liver Disease & Mortality Risk Predictor — Streamlit application.
Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from predictor import (
    FEATURES,
    NORMAL_RANGES,
    SAMPLE_PATIENTS,
    get_risk_color,
    is_outside_normal,
    load_models,
    predict_batch,
    predict_patient,
    validate_inputs,
)

DISCLAIMER = (
    "**Disclaimer:** For research purposes only. "
    "Model trained on PhysioNet MIMIC-II ICU dataset (n=1750, 19.1% mortality)."
)

INPUT_SPECS: dict[str, dict] = {
    "BUN": {"min": 0.0, "max": 150.0, "step": 1.0, "format": "%.0f"},
    "GCS": {"min": 3.0, "max": 15.0, "step": 1.0, "format": "%.0f"},
    "Urine": {"min": 0.0, "max": 5000.0, "step": 10.0, "format": "%.0f"},
    "HCO3": {"min": 5.0, "max": 45.0, "step": 1.0, "format": "%.0f"},
    "Bilirubin": {"min": 0.0, "max": 30.0, "step": 0.1, "format": "%.1f"},
    "ALP": {"min": 0.0, "max": 800.0, "step": 1.0, "format": "%.0f"},
    "Glucose": {"min": 0.0, "max": 600.0, "step": 1.0, "format": "%.0f"},
    "Age": {"min": 0.0, "max": 120.0, "step": 1.0, "format": "%.0f"},
    "Creatinine": {"min": 0.0, "max": 15.0, "step": 0.1, "format": "%.1f"},
    "NIMAP": {"min": 40.0, "max": 150.0, "step": 1.0, "format": "%.0f"},
    "Albumin": {"min": 0.0, "max": 6.0, "step": 0.1, "format": "%.1f"},
    "Platelets": {"min": 0.0, "max": 600.0, "step": 1.0, "format": "%.0f"},
    "NIDiasABP": {"min": 30.0, "max": 120.0, "step": 1.0, "format": "%.0f"},
    "HR": {"min": 20.0, "max": 220.0, "step": 1.0, "format": "%.0f"},
}

FORM_SECTIONS = {
    "Vitals & Hemodynamics": ["HR", "NIMAP", "NIDiasABP", "GCS"],
    "Renal & Fluid": ["BUN", "Creatinine", "Urine"],
    "Metabolic": ["HCO3", "Glucose"],
    "Liver Panel": ["Bilirubin", "ALP", "Albumin"],
    "Hematology": ["Platelets"],
    "Demographics": ["Age"],
}

LIVER_MODEL_TABLE = [
    ["Random Forest", "0.8635", "0.815", "0.9319±0.0128"],
    ["Naive Bayes", "0.8391", "0.708", "0.8831±0.0202"],
    ["SVM", "0.8326", "0.847", "0.9326±0.0144"],
    ["Logistic Regression", "0.8566", "0.750", "0.8649±0.0120"],
    ["XGBoost", "0.8551", "0.841", "0.9535±0.0105"],
    ["KNN", "0.7681", "0.659", "0.9558±0.0031"],
]

MORTALITY_MODEL_TABLE = [
    ["Random Forest", "0.7886", "0.478", "0.7814±0.0328"],
    ["Logistic Regression", "0.7785", "0.687", "0.7893±0.0288"],
    ["XGBoost", "0.7711", "0.478", "0.7789±0.0238"],
    ["Naive Bayes", "0.7468", "0.448", "0.7609±0.0453"],
    ["SVM", "0.7462", "0.433", "0.7432±0.0278"],
    ["KNN", "0.6820", "0.239", "0.7070±0.0361"],
]


def _init_session_defaults() -> None:
    default = SAMPLE_PATIENTS["A — Stable ICU Patient"]
    if "inputs" not in st.session_state:
        st.session_state.inputs = dict(default)
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    for feature in FEATURES:
        key = f"input_{feature}"
        if key not in st.session_state:
            st.session_state[key] = float(st.session_state.inputs.get(feature, default[feature]))


def _help_text(feature: str) -> str:
    lo, hi, unit = NORMAL_RANGES[feature]
    return f"Normal: {lo}–{hi} {unit}"


def _render_disclaimer() -> None:
    st.markdown("---")
    st.caption(DISCLAIMER.replace("**Disclaimer:** ", ""))


def _number_input(feature: str, container) -> float:
    spec = INPUT_SPECS[feature]
    lo, hi, unit = NORMAL_RANGES[feature]
    value = container.number_input(
        label=f"{feature} ({unit})",
        min_value=spec["min"],
        max_value=spec["max"],
        step=spec["step"],
        format=spec["format"],
        help=_help_text(feature),
        key=f"input_{feature}",
    )
    st.session_state.inputs[feature] = value
    if is_outside_normal(feature, value):
        container.markdown(
            f'<p style="color:#e74c3c;font-size:0.85rem;margin-top:-0.5rem;">'
            f"Outside normal range ({lo}–{hi} {unit})</p>",
            unsafe_allow_html=True,
        )
    return value


def _render_input_form() -> dict[str, float]:
    patient: dict[str, float] = {}
    for section, features in FORM_SECTIONS.items():
        with st.expander(section, expanded=True):
            cols = st.columns(2 if len(features) > 2 else len(features))
            for i, feature in enumerate(features):
                with cols[i % len(cols)]:
                    patient[feature] = _number_input(feature, st)
    return patient


def _render_liver_result(liv_prob: float, liv_label: int) -> None:
    liv_pct = liv_prob * 100
    color = "#e74c3c" if liv_label else "#2ecc71"
    text = "Liver Disease Detected" if liv_label else "No Liver Disease"
    st.markdown(
        f'<div style="background:{color}22;border-left:6px solid {color};'
        f'padding:1rem 1.25rem;border-radius:8px;margin-bottom:1rem;">'
        f'<h3 style="color:{color};margin:0;">{text}</h3>'
        f'<p style="font-size:1.4rem;margin:0.5rem 0 0 0;">'
        f"Probability: <strong>{liv_pct:.2f}%</strong></p></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(liv_prob, 1.0))
    st.caption(f"Liver disease probability: {liv_pct:.2f}%")


def _render_mortality_result(mort_prob: float, risk: str, low_t: float, high_t: float) -> None:
    mort_pct = mort_prob * 100
    color = get_risk_color(risk)
    st.markdown(
        f'<div style="background:{color}22;border-left:6px solid {color};'
        f'padding:1rem 1.25rem;border-radius:8px;margin-bottom:1rem;">'
        f'<h3 style="color:{color};margin:0;">{risk}</h3>'
        f'<p style="font-size:1.4rem;margin:0.5rem 0 0 0;">'
        f"Mortality probability: <strong>{mort_pct:.2f}%</strong></p></div>",
        unsafe_allow_html=True,
    )
    st.progress(min(mort_prob, 1.0))
    st.markdown(
        f"""
**Risk stratification (data-driven thresholds):**
- **Low Risk:** &lt; {low_t * 100:.1f}% predicted mortality
- **Medium Risk:** {low_t * 100:.1f}% – {high_t * 100:.1f}% predicted mortality
- **High Risk:** &gt; {high_t * 100:.1f}% predicted mortality

*Thresholds derived from 33rd/67th percentile of the model's predicted probability distribution.*
        """
    )


def _render_clinical_alerts(patient: dict[str, float]) -> None:
    warnings = validate_inputs(patient)
    if not warnings:
        st.success("All biomarkers within normal reference ranges.")
        return
    for w in warnings:
        color = "#e74c3c" if w["severity"] == "critical" else "#f39c12"
        icon = "🔴" if w["severity"] == "critical" else "🟠"
        st.markdown(
            f'<p style="color:{color};margin:0.25rem 0;">{icon} {w["message"]} '
            f'(<em>{w["severity"]}</em>)</p>',
            unsafe_allow_html=True,
        )


def page_single_patient(artifacts: dict) -> None:
    _init_session_defaults()
    low_t = artifacts["low_thresh"]
    high_t = artifacts["high_thresh"]

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Patient Biomarkers")
        patient = _render_input_form()
        if st.button("Predict", type="primary", use_container_width=True):
            liv_prob, liv_label, mort_prob, risk = predict_patient(patient, artifacts)
            st.session_state.prediction = {
                "patient": patient,
                "liv_prob": liv_prob,
                "liv_label": liv_label,
                "mort_prob": mort_prob,
                "risk": risk,
            }

    with col_out:
        st.subheader("Prediction Results")
        if st.session_state.prediction is None:
            st.info("Enter biomarkers and click **Predict** to see results.")
        else:
            pred = st.session_state.prediction
            st.markdown("#### Stage 1 — Liver Disease")
            _render_liver_result(pred["liv_prob"], pred["liv_label"])
            st.markdown("#### Stage 2 — Mortality Risk")
            _render_mortality_result(
                pred["mort_prob"], pred["risk"], low_t, high_t
            )
            st.markdown("#### Clinical Alerts")
            _render_clinical_alerts(pred["patient"])
            st.markdown("#### Disclaimer")
            st.warning(
                "For research purposes only. Not for clinical use."
            )

    _render_disclaimer()


def page_batch(artifacts: dict) -> None:
    st.subheader("Batch Prediction")
    st.markdown(
        "Upload a CSV with exactly these 14 columns (raw unscaled values): "
        f"`{', '.join(FEATURES)}`"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to run batch predictions.")
        _render_disclaimer()
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        _render_disclaimer()
        return

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        _render_disclaimer()
        return

    df = df[FEATURES].copy()
    with st.spinner("Running predictions…"):
        results = predict_batch(df, artifacts)

    st.success(f"Processed {len(results)} patients.")
    st.dataframe(results, use_container_width=True, height=400)

    st.markdown("#### Summary Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Liver Prob (%)", f"{results['Liver_Prob'].mean():.2f}")
    c2.metric("Mean Mortality Prob (%)", f"{results['Mortality_Prob'].mean():.2f}")
    c3.metric("Liver Disease (+)", int(results["Liver_Label"].sum()))
    risk_counts = results["Risk_Category"].value_counts()
    c4.metric("High Risk Count", int(risk_counts.get("High Risk", 0)))

    st.markdown("**Risk distribution:**")
    st.bar_chart(results["Risk_Category"].value_counts())

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV",
        data=csv_bytes,
        file_name="liver_mortality_predictions.csv",
        mime="text/csv",
        type="primary",
    )
    _render_disclaimer()


def page_model_info(artifacts: dict) -> None:
    st.subheader("Model Comparison")
    st.markdown("#### Liver Disease Models")
    st.table(
        pd.DataFrame(
            LIVER_MODEL_TABLE,
            columns=["Model", "ROC-AUC", "PR-AUC", "CV Score"],
        )
    )
    st.markdown("#### Mortality Models")
    st.table(
        pd.DataFrame(
            MORTALITY_MODEL_TABLE,
            columns=["Model", "ROC-AUC", "PR-AUC", "CV Score"],
        )
    )

    st.subheader("Feature Importance (SHAP)")
    st.markdown(
        """
**Liver Disease (11 features, label-creators excluded) — top SHAP:**  
GCS, Platelets, HCO3, BUN, Creatinine

**Mortality (14 features) — top SHAP:**  
BUN, GCS, Urine, Glucose, Bilirubin
        """
    )

    st.subheader("Risk Stratification")
    low_t = artifacts["low_thresh"]
    high_t = artifacts["high_thresh"]
    st.markdown(
        f"""
Mortality risk categories use fixed thresholds from the test-set distribution:

| Category | Predicted mortality probability |
|----------|--------------------------------|
| Low Risk | &lt; {low_t * 100:.2f}% ({low_t:.4f}) |
| Medium Risk | {low_t * 100:.2f}% – {high_t * 100:.2f}% |
| High Risk | ≥ {high_t * 100:.2f}% ({high_t:.4f}) |

**Deployed models:** Liver = RandomForest (200 trees), Mortality = RandomForest (300 trees, balanced)
        """
    )
    _render_disclaimer()


def main() -> None:
    st.set_page_config(
        page_title="Liver Disease & Mortality Risk Predictor",
        page_icon="🏥",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp { font-family: 'Segoe UI', system-ui, sans-serif; }
        div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1117 0%, #161b22 100%); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("ICU Risk Predictor")
        st.markdown(
            "Two-stage pipeline for **liver disease detection** and "
            "**ICU mortality risk stratification** using MIMIC-II biomarkers."
        )
        st.markdown("---")
        st.markdown("**Load Sample Patient**")
        for label, data in SAMPLE_PATIENTS.items():
            if st.button(label, use_container_width=True, key=f"sample_{label}"):
                st.session_state.inputs = dict(data)
                st.session_state.prediction = None
                for f, v in data.items():
                    st.session_state[f"input_{f}"] = float(v)
                st.rerun()

        st.markdown("---")
        st.markdown(
            """
**About**
- Liver Model: RandomForest (11 features)
- Mortality: RandomForest (14 features)
- Dataset: PhysioNet MIMIC-II (n=1750)
- Mortality rate: 19.1%
            """
        )

    artifacts = load_models()

    page = st.sidebar.radio(
        "Navigation",
        ["Single Patient", "Batch Prediction", "Model Info"],
        label_visibility="collapsed",
    )

    st.title("Liver Disease & Mortality Risk Predictor")

    if page == "Single Patient":
        page_single_patient(artifacts)
    elif page == "Batch Prediction":
        page_batch(artifacts)
    else:
        page_model_info(artifacts)


if __name__ == "__main__":
    main()
