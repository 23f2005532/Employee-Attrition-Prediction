import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO

# --------------- CONFIG ---------------
st.set_page_config(
    page_title="Employee Attrition – Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------- STYLES (Corporate) ---------------
st.markdown("""
<style>
body { background-color: #f4f6f9; }
.card { background:#fff; padding:22px; border:1px solid #e5e7eb; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,0.05); }
.metric-box { background:#fff; padding:16px; border:1px solid #d1d5db; border-radius:10px; text-align:center; }
h1, h2, h3, h4 { color:#0f172a; }
.small { color:#475569; font-size:13px; }
.kpi-bg { height: 12px; width: 100%; background-color: #e5e7eb; border-radius: 6px; }
.kpi-bar { height: 12px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# --------------- LOAD ARTIFACTS ---------------
import os
MODEL_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))


# Try to load training CSV to compute medians & reference distributions (optional but recommended)
def try_load_training_data():
    try:
        df = pd.read_csv(os.path.join(MODEL_DIR, "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"))
        # standardize column names to match your training pipeline
        df.columns = df.columns.str.lower().str.replace(" ", "")
        return df
    except Exception:
        return None

train_df_raw = try_load_training_data()

# Known categorical sets (from IBM HR attrition dataset)
CAT_CHOICES = {
    "overtime": ["No", "Yes"],
    "jobrole": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative"
    ],
    "department": ["Human Resources", "Research & Development", "Sales"],
    "maritalstatus": ["Divorced", "Married", "Single"],
}

# Helper to encode categoricals consistently (alphabetical codes, like LabelEncoder default)
def encode_with_vocab(series: pd.Series, vocab):
    cat = pd.Categorical(series, categories=sorted(vocab))
    return pd.Series(cat.codes, index=series.index).fillna(-1).astype(int)

# Build a full feature row in the exact order model expects; fill missing with medians (or 0 fallback)
def build_full_row(partial: pd.DataFrame, feature_order, ref_medians: dict | None):
    full = pd.DataFrame(columns=feature_order)
    # start with zeros
    for col in feature_order:
        full[col] = [0]

    # overwrite provided numeric fields directly (they already match training spellings below)
    for col in partial.columns:
        if col in full.columns:
            full[col] = partial[col].values

    # For any expected numeric feature not provided, use ref medians if available
    if ref_medians:
        for col in feature_order:
            if col not in partial.columns and col in ref_medians:
                full[col] = ref_medians[col]

    # Ensure numeric dtypes
    full = full.apply(pd.to_numeric, errors="coerce").fillna(0)
    return full

# Compute reference medians in training column spellings used during modeling
def compute_reference_medians(train_df):
    # Transform to your modeling spellings: during training you lowercased and removed spaces/underscores
    # Your model feature names are already those final spellings (e.g., monthlyincome, yearsatcompany, etc.)
    ref = {}
    if train_df is None:
        return ref
    for col in train_df.columns:
        if col in feature_names and pd.api.types.is_numeric_dtype(train_df[col]):
            ref[col] = float(train_df[col].median())
    return ref

REF_MEDIANS = compute_reference_medians(train_df_raw)

# Department median incomes for interpretation (optional)
def department_medians(train_df):
    if train_df is None:
        return {}
    if "department" not in train_df.columns or "monthlyincome" not in train_df.columns:
        return {}
    return train_df.groupby("department")["monthlyincome"].median().to_dict()

DEPT_MEDIANS = department_medians(train_df_raw)


def department_percentiles(train_df):
    if train_df is None:
        return {}
    if "department" not in train_df.columns or "monthlyincome" not in train_df.columns:
        return {}
    pct_map = {}
    for dep, group in train_df.groupby("department"):
        # compute percentiles once per department
        incomes = group["monthlyincome"].sort_values().values
        pct_map[dep] = incomes
    return pct_map

DEPT_PCT = department_percentiles(train_df_raw)

# ------------ INTERPRETATION (rule-based, research-friendly) ------------
def interpret_row(row_dict, risk_pct: float):
    """
    row_dict keys are user level inputs using training spellings:
      age, monthlyincome, totalworkingyears, yearsatcompany, jobsatisfaction, environmentsatisfaction,
      overtime, jobrole, department, maritalstatus
    """
    notes = []

    # 1) Compensation vs dept median (if available)
    dep = row_dict.get("department")
    income = row_dict.get("monthlyincome", 0)
    # Salary comparison: median + percentile position
    if DEPT_MEDIANS and DEPT_PCT and dep in DEPT_MEDIANS and dep in DEPT_PCT:
        dep_med = DEPT_MEDIANS[dep]
        incomes = DEPT_PCT[dep]

        # Percentile: rank in sorted list
        rank = (incomes < income).sum()
        pct = round((rank / len(incomes)) * 100, 1)

        # Narrative
        notes.append(f"Monthly income is at the **{pct}th percentile** for the {dep} department (USD).")

        # Additional interpretation thresholds
        if pct < 25:
            notes.append("Compensation is relatively low compared to department peers (potential dissatisfaction).")
        elif pct > 75:
            notes.append("Compensation is relatively high compared to peers (retention positive signal).")


    # 2) Satisfaction
    js = row_dict.get("jobsatisfaction", 3)
    es = row_dict.get("environmentsatisfaction", 3)
    if js <= 2:
        notes.append("Job satisfaction is low (≤2) — disengagement risk.")
    if es <= 2:
        notes.append("Environment satisfaction is low — workspace/managerial friction possible.")

    # 3) Tenure dynamics
    years_company = row_dict.get("yearsatcompany", 0)
    total_years = row_dict.get("totalworkingyears", 0)
    if years_company <= 1 and total_years <= 3:
        notes.append("Very early tenure — churn risk is naturally higher.")
    if years_company >= 7 and js <= 2:
        notes.append("Long tenure with low satisfaction — stagnation risk.")

    # 4) Overtime
    if row_dict.get("overtime") == "Yes":
        notes.append("Frequent overtime — burnout risk.")

    # 5) Role context (sales roles trend higher)
    if row_dict.get("jobrole") in ["Sales Representative", "Sales Executive"]:
        notes.append("Sales-facing role — historically higher attrition segment.")

    # Synthesize view by risk bucket
    if risk_pct >= 70:
        headline = "Critical risk of attrition. Immediate retention action recommended."
    elif risk_pct >= 50:
        headline = "High risk of attrition. Prioritize targeted intervention."
    elif risk_pct >= 25:
        headline = "Moderate risk. Monitor and address contributing factors."
    else:
        headline = "Low risk. Maintain engagement practices."

    return headline, notes


# ------------ SIDEBAR ------------
st.sidebar.title("Research Dashboard")
st.sidebar.markdown("Model: **XGBoost** · Objective: binary attrition · Focus: *recall on minority class*")
st.sidebar.markdown("This app demonstrates **probability-based risk scoring** and a **transparent explanation layer** without external explainer libs (paper-friendly).")

# ------------ TABS ------------
tab_pred, tab_interp, tab_compare, tab_batch, tab_card = st.tabs(
    ["Risk Prediction", "Interpretation", "Factor Comparisons", "Batch Scoring (CSV)", "Model Card"]
)

# ===== TAB: PREDICTION =====
with tab_pred:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Single Employee – Risk Estimator")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 60, 30)
        totalworkingyears = st.number_input("Total Working Years", 0, 40, 5)
        yearsatcompany = st.number_input("Years at Company", 0, 30, 2)
    with c2:
        monthlyincome = st.number_input("Monthly Income (USD)", 1000, 25000, 5000, step=250)
        jobsatisfaction = st.selectbox("Job Satisfaction (1 low – 10 high)", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], index=2)
        environmentsatisfaction = st.selectbox("Environment Satisfaction (1 low – 10 high)", [1, 2, 3, 4], index=2)
    with c3:
        overtime = st.selectbox("OverTime", CAT_CHOICES["overtime"])
        jobrole = st.selectbox("Job Role", CAT_CHOICES["jobrole"])
        department = st.selectbox("Department", CAT_CHOICES["department"])
    maritalstatus = st.selectbox("Marital Status", CAT_CHOICES["maritalstatus"])

    run = st.button("Run Risk Analysis")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction Result")

    if run:
        # Build the partial row with training spellings used by your model/features
        # (Your training pipeline used lowercased/no-space names, e.g., monthlyincome, yearsatcompany)
        partial = pd.DataFrame([{
            "age": age,
            "monthlyincome": monthlyincome,
            "totalworkingyears": totalworkingyears,
            "yearsatcompany": yearsatcompany,
            "jobsatisfaction": jobsatisfaction,
            "environmentsatisfaction": environmentsatisfaction,
            "overtime": encode_with_vocab(pd.Series([overtime]), CAT_CHOICES["overtime"]),
            "jobrole": encode_with_vocab(pd.Series([jobrole]), CAT_CHOICES["jobrole"]),
            "department": encode_with_vocab(pd.Series([department]), CAT_CHOICES["department"]),
            "maritalstatus": encode_with_vocab(pd.Series([maritalstatus]), CAT_CHOICES["maritalstatus"]),
        }])

        # Fill full feature row in correct order
        full_row = build_full_row(partial, feature_names, REF_MEDIANS)

        # Predict probability (attrition=1)
        prob = float(model.predict_proba(full_row)[0][1])
        risk = round(prob * 100, 1)

        # Risk label & color
        if risk < 25:
            label, color = "Very Low", "#15803d"
        elif risk < 50:
            label, color = "Moderate", "#ca8a04"
        elif risk < 70:
            label, color = "High", "#ea580c"
        else:
            label, color = "Critical", "#b91c1c"

        # KPI
        k1, k2 = st.columns([1, 3])
        with k1:
            st.markdown(f"""
            <div class="metric-box">
                <h2 style='color:{color}; margin-bottom:0;'>{risk}%</h2>
                <p style='font-weight:600; color:{color}; margin-top:0;'>Attrition Risk ({label})</p>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-bg">
              <div class="kpi-bar" style="width:{risk}%; background:{color};"></div>
            </div>
            """, unsafe_allow_html=True)

        # Stash latest inputs for interpretation tab
        st.session_state["_latest_inputs_user"] = {
            "age": age,
            "monthlyincome": monthlyincome,
            "totalworkingyears": totalworkingyears,
            "yearsatcompany": yearsatcompany,
            "jobsatisfaction": jobsatisfaction,
            "environmentsatisfaction": environmentsatisfaction,
            "overtime": overtime,
            "jobrole": jobrole,
            "department": department,
            "maritalstatus": maritalstatus,
            "risk_pct": risk
        }

    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("All monetary values represent **monthly income in USD** as per the original IBM HR dataset.")


# ===== TAB: INTERPRETATION (rule-based) =====
with tab_interp:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Interpretation – Research Narrative")

    if "_latest_inputs_user" not in st.session_state:
        st.info("Run a prediction in the **Risk Prediction** tab first.")
    else:
        ui = st.session_state["_latest_inputs_user"].copy()
        headline, notes = interpret_row(ui, ui["risk_pct"])
        st.subheader(headline)
        st.write("**Key contributing factors (rule-based):**")
        if notes:
            for n in notes:
                st.write(f"- {n}")
        else:
            st.write("- No strong negative drivers detected from the provided indicators.")

        st.caption("This interpretation uses transparent, rule-based logic aligned with observed dataset patterns. It is reproducible and paper-friendly (no external explainer dependencies).")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("All monetary values are **monthly income in USD** based on the original IBM HR dataset. Percentile comparisons are computed within employee's department.")


# ===== TAB: FACTOR COMPARISONS =====
with tab_compare:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Factor Comparisons (Reference from Training Data)")
    if train_df_raw is None:
        st.info("Training CSV not found at `../data/WA_Fn-UseC_-HR-Employee-Attrition.csv`. "
                "Place it there to enable comparison charts.")
    else:
        # Make tidy copies
        df = train_df_raw.copy()
        # Small set of simple charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Attrition Rate by Job Role")
            tmp = df.copy()
            # original dataset has 'attrition' as 'Yes'/'No'
            if "attrition" in tmp.columns:
                tmp["attrition_flag"] = (tmp["attrition"].astype(str).str.lower() == "yes").astype(int)
                chart = tmp.groupby("jobrole")["attrition_flag"].mean().sort_values()
                st.bar_chart(chart)
        with c2:
            st.subheader("Income vs. Attrition (by Department)")
            tmp = df.copy()
            if "attrition" in tmp.columns:
                tmp["attrition_flag"] = (tmp["attrition"].astype(str).str.lower() == "yes").astype(int)
                dept_means = tmp.groupby("department")[["monthlyincome","attrition_flag"]].mean()
                st.dataframe(dept_means.style.format({"monthlyincome":"{:,.0f}", "attrition_flag":"{:.2f}"}))

    st.markdown("</div>", unsafe_allow_html=True)

# ===== TAB: BATCH SCORING =====
with tab_batch:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Batch Scoring – Upload CSV")

    st.write("""
    **Expected columns (any order; others optional):**  
    `age, monthlyincome, totalworkingyears, yearsatcompany, jobsatisfaction, environmentsatisfaction, overtime, jobrole, department, maritalstatus`
    """)

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        raw = pd.read_csv(file)
        df = raw.copy()

        # Normalize column names to training spellings
        df.columns = (df.columns
                        .str.lower()
                        .str.replace(" ", "")
                        .str.replace("_", ""))

        # Ensure required columns exist
        required = ["age","monthlyincome","totalworkingyears","yearsatcompany",
                    "jobsatisfaction","environmentsatisfaction","overtime",
                    "jobrole","department","maritalstatus"]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan

        # Encode categoricals consistently
        df["overtime"] = encode_with_vocab(df["overtime"].astype(str), CAT_CHOICES["overtime"])
        df["jobrole"] = encode_with_vocab(df["jobrole"].astype(str), CAT_CHOICES["jobrole"])
        df["department"] = encode_with_vocab(df["department"].astype(str), CAT_CHOICES["department"])
        df["maritalstatus"] = encode_with_vocab(df["maritalstatus"].astype(str), CAT_CHOICES["maritalstatus"])

        # Fill missing numeric inputs with medians from training if available
        for col in ["age","monthlyincome","totalworkingyears","yearsatcompany","jobsatisfaction","environmentsatisfaction"]:
            if col in df.columns:
                if df[col].isna().any():
                    fill_val = REF_MEDIANS.get(col, df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0)
                    df[col] = df[col].fillna(fill_val)

        # Build full matrix with model's expected features
        X_full = pd.DataFrame(columns=feature_names)
        for col in feature_names:
            X_full[col] = 0
        for col in df.columns:
            if col in X_full.columns:
                X_full[col] = df[col]

        X_full = X_full.apply(pd.to_numeric, errors="coerce").fillna(0)

        probs = model.predict_proba(X_full)[:, 1]
        risks = (probs * 100).round(1)
        out = raw.copy()
        out["attrition_risk_pct"] = risks

        # Risk labels
        bins = [-1, 25, 50, 70, 101]
        labels = ["Very Low","Moderate","High","Critical"]
        out["risk_label"] = pd.cut(risks, bins=bins, labels=labels)

        st.subheader("Scored Results (Top Risk First)")
        st.dataframe(out.sort_values("attrition_risk_pct", ascending=False))

        # Download scored CSV
        csv_bytes = out.to_csv(index=False).encode()
        st.download_button("Download Scored CSV", data=csv_bytes, file_name="scored_attrition.csv", mime="text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

# ===== TAB: MODEL CARD =====
with tab_card:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Model Card (Research Summary)")
    st.markdown("""
**Objective:** Predict probability of employee attrition (Yes/No).  
**Model:** XGBoost (trained with SMOTE on the minority class), evaluated on a stratified hold-out set.  
**Primary Metric:** Recall on attrition class; also report precision, accuracy.

**Inputs Used in UI:** age, monthlyincome, totalworkingyears, yearsatcompany, jobsatisfaction, environmentsatisfaction, overtime, jobrole, department, maritalstatus.  
Other engineered/expected features are filled with medians from training for stability.

**Assumptions & Limits:**
- Encodings are alphabetically ordered and fixed; unseen categories map to -1.
- Batch scoring expects the same semantic columns; mismatched schemas may yield degraded predictions.
- This is an observational model; it **does not** establish causality.
- Probability is calibrated by the model’s loss, not post-calibration; treat thresholds as decision policy, not truth.

**Ethics & Fairness:**
- Do not use model outputs in isolation for employment decisions.
- Use explanations + manager review + consented HR policy.
- Evaluate subgroup performance (gender/department) before operational use.

**Recommended Use:** Early risk flagging → retention conversation → track outcomes.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
