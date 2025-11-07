
# Employee Attrition Prediction Dashboard

This project provides an interactive dashboard to understand and predict employee attrition risk. 
It is designed to support HR departments, managers, and organizational researchers in identifying 
early signs of employee disengagement and turnover.

🔗 **Live App:**  
https://employee-attrition-prediction-pdg6y4uxrqus3beemdt6ms.streamlit.app/

---

## Understanding the Project

Employee attrition refers to employees leaving an organization over time. High attrition affects:
- Organizational stability
- Hiring & training costs
- Team morale
- Productivity and long-term planning

**Goal of this Dashboard:**  
To estimate the likelihood that an employee may leave and to explain *why* that risk is high or low.

---

## Key Capabilities

| Feature | What it Does | Why it Matters |
|--------|--------------|----------------|
| **Risk Prediction** | Calculates a percentage chance of attrition | Helps identify employees needing attention |
| **Clear Explanations** | Provides plain-language reasons behind each prediction | Avoids black-box decisions |
| **Salary Percentile Comparison** | Shows where employee salary stands relative to peers | Helps identify compensation-related dissatisfaction |
| **Batch CSV Scoring** | Allows scoring multiple employees at once | Useful for organization-wide analysis |
| **Model Card (Ethical Note)** | Explains model assumptions and cautions | Encourages responsible and fair usage |

---

## How the Model Works

- The model is built using **XGBoost**, a widely used machine learning algorithm.
- Class imbalance (few employees leave vs many who stay) is handled before training.
- The model’s main focus is **recall** for detecting actual attrition cases.
- To keep explanations transparent, the dashboard uses **clear rule-based interpretation** rather than opaque explainers.

This means the tool not only predicts **what** may happen, but also explains **why**.

---

## Data Used

This project uses the **IBM HR Employee Attrition dataset**, a common open dataset used for HR analytics learning and research.

Values like salary are presented in **USD**, as in the original dataset.

---

## Running Locally

```bash
git clone <your-repo>
cd employee-attrition-prediction
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Project Structure

```
├── app
│   ├── app.py
│   ├── feature_names.pkl
│   ├── label_encoder.pkl
│   └── model.pkl
├── data
│   ├── archive
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── archive.zip
├── notebooks
│   └── 01_attrition_analysis_and_model.ipynb
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Ethical Use Disclaimer

This application is intended for **learning, research, and HR decision support**, not for automated hiring or termination.

Final decisions should always involve:
- Human judgment
- Contextual understanding
- Transparent conversation with employees

---

## Author

**Md Ehtesham Ansari**  
BS in Data Science & Applications — IIT Madras  

---

If you find this project useful, consider ⭐ starring the repository!
