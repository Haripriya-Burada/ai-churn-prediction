# 🧠 AI-Powered Customer Intelligence & Churn Prediction System

## 📋 Project Overview

An end-to-end **Data Science application** that enables businesses to understand customer behaviour, predict customer churn, segment customers, and generate AI-powered business recommendations — all through an interactive dashboard.

## ❓ Problem Statement

Customer churn (attrition) is one of the biggest challenges facing subscription-based businesses. Acquiring a new customer costs **5–7× more** than retaining an existing one. This project builds a predictive analytics platform that helps businesses:

1. **Identify** customers likely to churn before they leave
2. **Understand** the reasons behind churn
3. **Segment** the customer base for targeted strategies
4. **Take action** with data-driven retention recommendations

## 🎯 Objectives

- Build a complete, working ML pipeline from data loading to prediction
- Create an interactive Streamlit dashboard for non-technical users
- Implement Explainable AI so decisions are transparent
- Generate actionable business insights using Claude AI (with fallback)

## ✨ Features

| Module | Description |
|--------|-------------|
| **Data Upload** | Upload CSV, preview, clean, and download datasets |
| **EDA** | Interactive charts, KPI cards, filters, and correlation analysis |
| **Feature Engineering** | Automated feature creation (tenure groups, engagement scores) |
| **ML Training** | 4 models trained and compared (LogReg, DTree, RF, XGBoost) |
| **Churn Prediction** | Individual and batch prediction with risk scoring |
| **Segmentation** | K-Means clustering with automatic segment labelling |
| **Explainability** | SHAP feature importance (global and individual) |
| **AI Insights** | Claude API integration with rule-based fallback |
| **Recommendations** | Context-aware retention strategy engine |
| **Database** | SQLite persistence for customers, predictions, and metrics |

## 🏗️ Architecture

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

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Visualisation**: Plotly, Matplotlib, Seaborn
- **ML**: Scikit-learn, XGBoost
- **Explainable AI**: SHAP
- **Database**: SQLite
- **AI**: Claude API (Anthropic SDK)

## 🤖 Machine Learning Algorithms

| Model | Type | Key Parameters |
|-------|------|---------------|
| Logistic Regression | Linear | `class_weight="balanced"` |
| Decision Tree | Tree-based | `max_depth=8, class_weight="balanced"` |
| Random Forest | Ensemble | `n_estimators=200, class_weight="balanced"` |
| XGBoost | Gradient Boosting | `n_estimators=200, scale_pos_weight=3` |

## 📊 Dataset Description

The sample dataset contains **5,000 synthetic customers** with 14 columns:

| Column | Type | Description |
|--------|------|-------------|
| CustomerID | String | Unique identifier |
| Age | Integer | Customer age (18–72) |
| Gender | Categorical | Male / Female |
| Tenure | Integer | Months as customer (1–72) |
| MonthlyCharges | Float | Monthly bill amount |
| TotalCharges | Float | Cumulative charges |
| Contract | Categorical | Month-to-month / One year / Two year |
| PaymentMethod | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| InternetService | Categorical | Fiber optic / DSL / No |
| SupportCalls | Integer | Number of support interactions |
| NumProducts | Integer | Number of subscribed products |
| UsageFrequency | Integer | Days per month of usage |
| SatisfactionScore | Integer | Customer satisfaction (1–5) |
| Churn | Binary | 0 = Active, 1 = Churned |

**Churn labels are realistic** — generated via a logistic function of business-meaningful features.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd customer-intelligence
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Environment Setup

```bash
copy .env.example .env
```

Edit `.env` and add your Anthropic API key (optional):
```
ANTHROPIC_API_KEY=your_actual_key_here
```

### 6. Generate Sample Dataset

```bash
python generate_sample_data.py
```

### 7. Run the Application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

## 📖 How to Use the Dashboard

1. **Start** → The dashboard loads with the sample dataset automatically
2. **Data Upload** → Upload your own CSV or use the sample data
3. **EDA** → Explore interactive charts, filter data, view correlations
4. **Model Performance** → Train 4 ML models and compare them
5. **Churn Prediction** → Enter customer details to get churn probability
6. **Segmentation** → Cluster customers into meaningful segments
7. **Explainability** → Understand why the model makes its predictions
8. **AI Insights** → Generate AI-powered business recommendations

## 📈 Model Evaluation

Models are evaluated using:
- **Accuracy** — Overall correctness
- **Precision** — Of predicted churns, how many truly churned?
- **Recall** — Of actual churns, how many did we catch?
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — Discrimination ability across all thresholds

**Best model selection**: ROC-AUC (primary), F1 (tiebreaker)

## 🔮 Future Enhancements

- [ ] PDF report generation
- [ ] Model hyperparameter tuning (GridSearchCV)
- [ ] Deep learning model (neural network)
- [ ] Real-time data integration
- [ ] Multi-language support
- [ ] A/B testing framework for retention strategies
- [ ] Time-series churn prediction

## ⚠️ Limitations

- The sample dataset is synthetic (not real customer data)
- SHAP may be slow on very large datasets (>50K rows)
- Claude AI requires an API key (rule-based fallback is available)
- K-Means assumes spherical clusters — may not capture all segment shapes

## 👤 Author

Built as a Mini Project for Data Science / Machine Learning coursework.

## 📄 Licence

This project is for educational purposes.
