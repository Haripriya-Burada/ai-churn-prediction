"""
pages/about.py
----------------
About page — project information, architecture, and viva preparation.
"""

import streamlit as st


def render():
    st.markdown('<div class="section-header">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
## 🧠 AI-Powered Customer Intelligence & Churn Prediction System

### 📋 Project Overview

This is an end-to-end **Data Science application** that enables businesses to:
- **Analyse** customer behaviour through interactive dashboards
- **Predict** customer churn using machine learning
- **Segment** customers using unsupervised learning
- **Explain** predictions using SHAP (Explainable AI)
- **Generate** AI-powered business recommendations

---

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                    │
│   Dashboard │ EDA │ Prediction │ Segmentation │ SHAP    │
├─────────────────────────────────────────────────────────┤
│                  Business Logic Layer                    │
│  Data Loader │ Preprocessing │ Feature Eng │ AI Insights│
├─────────────────────────────────────────────────────────┤
│                   ML Engine Layer                        │
│  LogReg │ Decision Tree │ Random Forest │ XGBoost       │
├─────────────────────────────────────────────────────────┤
│                   Data Layer                             │
│          SQLite Database │ CSV Files │ Models            │
└─────────────────────────────────────────────────────────┘
```

---

### 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Visualisation** | Plotly, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Explainable AI** | SHAP |
| **Database** | SQLite |
| **AI Integration** | Claude API (Anthropic) |
| **Language** | Python 3.11+ |

---

### 🤖 Machine Learning Models

| Model | Strengths | Handles Imbalance |
|-------|-----------|-------------------|
| **Logistic Regression** | Fast, interpretable, linear baseline | `class_weight="balanced"` |
| **Decision Tree** | Interpretable rules | `class_weight="balanced"` |
| **Random Forest** | Robust ensemble, reduces overfitting | `class_weight="balanced"` |
| **XGBoost** | Best for tabular data, gradient boosting | `scale_pos_weight` |

---

### 📊 Evaluation Metrics

- **Accuracy** — Overall correctness
- **Precision** — Of predicted churns, how many actually churned?
- **Recall** — Of actual churns, how many did we catch?
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Model's ability to distinguish churn from non-churn

**We prioritise ROC-AUC and F1** because churn is a class-imbalanced problem 
where accuracy alone is misleading.

---

### 👨‍🎓 Key Concepts for Viva

1. **Data Leakage**: We fit the preprocessing pipeline ONLY on training data to 
   prevent information from the test set leaking into the model.

2. **Class Imbalance**: We use `class_weight="balanced"` and `scale_pos_weight` 
   so the model doesn't just predict "no churn" for everyone.

3. **SHAP Values**: Based on Shapley values from game theory — they fairly 
   distribute the "credit" for a prediction among all features.

4. **K-Means Clustering**: Groups similar customers together based on distance 
   in feature space. We scale features first so no single feature dominates.

5. **Feature Engineering**: We create new features like TenureGroup, 
   EngagementScore, and SupportIntensity to help the model find patterns.

---

### 🚀 How to Run

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Generate sample dataset
python generate_sample_data.py

# Run the application
streamlit run app.py
```

---

### 👤 Author

Built as a Mini Project for Data Science / Machine Learning coursework.
    """)

    st.markdown("---")

    # Viva Questions
    with st.expander("🎓 10 Likely Viva Questions & Answers"):
        viva_qa = [
            (
                "What is customer churn and why does it matter?",
                "Customer churn is when a customer stops using a company's product or service. "
                "It matters because acquiring a new customer costs 5-7x more than retaining an existing one. "
                "Predicting churn allows businesses to intervene proactively."
            ),
            (
                "Why did you use multiple ML models instead of just one?",
                "Different models have different strengths. Logistic Regression is interpretable, "
                "Random Forest is robust, and XGBoost often gives the best accuracy. By comparing them, "
                "we can select the best model for our specific data."
            ),
            (
                "What is data leakage and how did you prevent it?",
                "Data leakage occurs when information from the test set influences model training. "
                "We prevent it by fitting the preprocessing pipeline (imputation, scaling, encoding) "
                "ONLY on the training set, then transforming the test set with the same fitted pipeline."
            ),
            (
                "Why is accuracy not sufficient for evaluating churn models?",
                "If only 20% of customers churn, a model that always predicts 'no churn' gets 80% accuracy "
                "but catches zero actual churners. We use ROC-AUC and F1 Score which account for both "
                "precision and recall, especially important for the minority class."
            ),
            (
                "What is SHAP and why is it important?",
                "SHAP (SHapley Additive exPlanations) uses game theory to explain individual predictions. "
                "It tells us which features pushed a prediction toward churn and by how much. "
                "This is crucial for business users who need to understand WHY a customer is at risk."
            ),
            (
                "How does K-Means clustering work?",
                "K-Means groups customers into K clusters by minimising the distance between each point "
                "and its cluster centroid. We scale features first so that features with larger ranges "
                "don't dominate the distance calculation."
            ),
            (
                "What is a ColumnTransformer and why did you use it?",
                "ColumnTransformer applies different preprocessing steps to different column types — "
                "numerical columns get imputed and scaled, categorical columns get imputed and one-hot "
                "encoded. It ensures consistent preprocessing in a single pipeline."
            ),
            (
                "How does the recommendation engine work?",
                "It's a rule-based system that maps customer attributes and risk factors to specific "
                "business actions. For example, a high-risk customer with a month-to-month contract "
                "gets a recommendation to offer an annual contract discount."
            ),
            (
                "What is class_weight='balanced' and why is it used?",
                "When classes are imbalanced (e.g., 80% non-churn, 20% churn), the model may ignore "
                "the minority class. class_weight='balanced' automatically adjusts sample weights "
                "inversely proportional to class frequencies, forcing the model to pay equal attention."
            ),
            (
                "How does the Claude API integration work and what happens if it fails?",
                "We send anonymised, aggregate-level analytics (churn rate, top drivers, segment stats) "
                "to Claude API for generating business insights. If the API key is missing or the call "
                "fails, we fall back to a rule-based engine that generates insights from predefined templates."
            ),
        ]

        for i, (q, a) in enumerate(viva_qa, 1):
            st.markdown(f"**Q{i}: {q}**")
            st.markdown(f"> {a}")
            st.markdown("")
