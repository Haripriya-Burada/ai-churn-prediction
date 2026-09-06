"""
app.py — Main Streamlit application for the Customer Intelligence Platform.

Run with:
    streamlit run app.py
"""

import streamlit as st
import os
import sys

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import init_db

# ──────────────── PAGE CONFIG ──────────────── #

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────── CUSTOM CSS ──────────────── #

st.markdown("""
<style>
    /* KPI card styling */
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .kpi-card h3 {
        margin: 0;
        font-size: 14px;
        opacity: 0.9;
        font-weight: 500;
    }
    .kpi-card h1 {
        margin: 8px 0 0 0;
        font-size: 28px;
        font-weight: 700;
    }
    .kpi-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .kpi-red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .kpi-blue { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
    .kpi-orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .kpi-purple { background: linear-gradient(135deg, #7f00ff 0%, #e100ff 100%); }

    /* Risk badges */
    .risk-high { color: #e74c3c; font-weight: bold; }
    .risk-medium { color: #f39c12; font-weight: bold; }
    .risk-low { color: #2ecc71; font-weight: bold; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }

    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────── INITIALISE DATABASE ──────────────── #

init_db()

# ──────────────── SIDEBAR NAVIGATION ──────────────── #

st.sidebar.markdown("## 🧠 Customer Intelligence")
st.sidebar.markdown("---")

PAGES = {
    "📊 Dashboard": "dashboard",
    "📁 Data Upload": "data_upload",
    "📈 Exploratory Analysis": "eda",
    "🤖 Model Performance": "model_performance",
    "🔮 Churn Prediction": "churn_prediction",
    "👥 Customer Segmentation": "segmentation",
    "🔍 Explainability": "explainability",
    "💡 AI Insights": "ai_insights",
    "ℹ️ About": "about",
}

selected_page = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
page_module = PAGES[selected_page]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color: #888;'>Built with ❤️ using Streamlit, Scikit-learn, XGBoost & SHAP</small>",
    unsafe_allow_html=True,
)

# ──────────────── LOAD & RENDER PAGE ──────────────── #

# Import page modules dynamically
from views import (
    dashboard,
    data_upload,
    eda,
    model_performance,
    churn_prediction,
    segmentation,
    explainability,
    ai_insights,
    about,
)

PAGE_MAP = {
    "dashboard": dashboard,
    "data_upload": data_upload,
    "eda": eda,
    "model_performance": model_performance,
    "churn_prediction": churn_prediction,
    "segmentation": segmentation,
    "explainability": explainability,
    "ai_insights": ai_insights,
    "about": about,
}

PAGE_MAP[page_module].render()
