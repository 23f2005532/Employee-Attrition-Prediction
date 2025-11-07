
# Employee Attrition Risk & Workforce Stability Analysis

This project develops a probability-based **employee attrition prediction and interpretation system** using the IBM HR Analytics dataset.  
Rather than treating attrition prediction as a simple classification task, the project focuses on **understanding the underlying drivers** of employee turnover and presenting risk insights in a **transparent, research-oriented, and decision-support format**.

The final application is a **Streamlit dashboard** that supports:
- Single-employee attrition risk assessment
- Evidence-based interpretation of contributing factors
- Department-level benchmarking of compensation
- Batch risk scoring of multiple employees via CSV upload
- Research-aligned model documentation (Model Card)

---

## 1. Problem Context

Employee attrition is costly:
- **Financial cost** of rehiring and onboarding
- **Loss of organizational knowledge**
- **Reduced morale and productivity**

The goal is not just to **predict** attrition, but to:
> **Identify key drivers of turnover and support early retention interventions.**

---

## 2. Dataset

Dataset: *IBM HR Analytics Employee Attrition & Performance*  
Units: All compensation is in **USD**, as per the original dataset.

The dataset is **class-imbalanced**:
- Majority: Employees who stay
- Minority: Employees who leave (attrition = Yes)

To address this, the model is trained using **SMOTE** to improve recall on the minority class.

---

## 3. Methodology

### Preprocessing
- Standardized feature names and encoding
- Balanced minority class with **SMOTE**
- Train-test split (stratified)

### Model
- **XGBoost classifier**, selected for interpretability and imbalanced-class performance
- Evaluation focused on:
  - **Recall on attrition class**
  - Precision
  - Accuracy

### Key Performance Summary
| Metric | Value | Interpretation |
|-------|-------|----------------|
| Recall (Attrition) | ~0.36 | The model captures ~36% of employees likely to leave |
| Accuracy | ~0.82 | Good overall predictive stability |
| Precision (Attrition) | Moderate | Balanced signaling of true risk vs false alarms |

---

## 4. Application Overview

The Streamlit dashboard provides a **practical and research-oriented interface**:

### **Tab 1 — Risk Prediction**
- Enter key employee attributes (age, role, income, satisfaction, etc.)
- Model returns **attrition probability (%)**
- Risk categorized as:
  - Very Low
  - Moderate
  - High
  - Critical

### **Tab 2 — Interpretation**
A rule-based narrative explains *why* the risk level was assigned, using:

- Income **percentile positioning within department**
- Job satisfaction levels
- Overtime workload signals
- Tenure and role characteristics

### **Tab 3 — Factor Comparisons**
Visual summaries using original dataset distributions:
- Attrition rate by job role
- Income vs attrition patterns across departments

### **Tab 4 — Batch Scoring (CSV)**
Upload a CSV to score **multiple employees at once**, enabling:
- Workforce-wide heatmaps
- Retention prioritization

### **Tab 5 — Model Card**
Transparent documentation of:
- Modeling assumptions
- Limitations
- Fair and responsible use guidelines

---

## 5. Ethical & Practical Use

This model is intended for:
- Early **risk flagging**
- Supporting **retention discussions**

It should *not* be used for:
- Hiring or firing decisions
- Automated personnel judgment

### Responsible Use Recommendations
- Always pair model output with **human review**
- Maintain transparency with employees where appropriate
- Monitor subgroup performance for fairness

---

## 6. How to Run

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

(Optional) Place the training dataset at:
```
data/WA_Fn-UseC_-HR-Employee-Attrition.csv
```
to enable department benchmarking charts.

---

## 7. Future Work
- SHAP or IG-based causal interpretation (advanced explainability)
- Time-based attrition forecasting
- Intervention outcome tracking module

---

## Author
Md Ehtesham Ansari

BS in Data Science & Applications (IIT Madras)
