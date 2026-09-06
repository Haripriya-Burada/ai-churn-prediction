"""
pages/dashboard.py
-------------------
Main dashboard — shows KPIs, churn trends, top drivers, and segments
at a glance.
"""

import streamlit as st
import pandas as pd
import os

from src.data_loader import load_csv, get_sample_data_path, normalize_churn_column
from src.eda import compute_kpis, churn_distribution_chart, churn_by_category, churn_by_numerical
from src.feature_engineering import engineer_features
from src.prediction import predict_batch, classify_risk
from src.model_training import load_model
from src.explainability import get_global_importance
from src.database import save_customers


def _load_dashboard_data() -> pd.DataFrame:
    """Load the dataset from session state or the sample CSV."""
    if "df" in st.session_state and st.session_state["df"] is not None:
        return st.session_state["df"]

    sample_path = get_sample_data_path()
    if os.path.exists(sample_path):
        df = load_csv(sample_path)
        df, _ = normalize_churn_column(df)
        st.session_state["df"] = df
        return df

    return pd.DataFrame()


def render():
    st.markdown('<div class="section-header">📊 Customer Intelligence Dashboard</div>', unsafe_allow_html=True)

    df = _load_dashboard_data()

    if df.empty:
        st.warning("No data loaded. Go to **Data Upload** to load a dataset or generate the sample data.")
        return

    # ─── KPI Row ─── #
    kpis = compute_kpis(df)

    col1, col2, col3, col4, col5 = st.columns(5)

    total_val = kpis.get("total_customers", 0)
    total_str = f"{total_val:,}" if isinstance(total_val, (int, float)) else str(total_val)
    active_val = kpis.get("active", "N/A")
    active_str = f"{active_val:,}" if isinstance(active_val, (int, float)) else str(active_val)

    with col1:
        st.markdown(
            f'<div class="kpi-card kpi-blue"><h3>Total Customers</h3><h1>{total_str}</h1></div>',
            unsafe_allow_html=True,
        )
    with col2:
        churn_rate = kpis.get("churn_rate", "N/A")
        css = "kpi-red" if isinstance(churn_rate, (int, float)) and churn_rate > 20 else "kpi-orange"
        st.markdown(
            f'<div class="kpi-card {css}"><h3>Churn Rate</h3><h1>{churn_rate}%</h1></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="kpi-card kpi-green"><h3>Active Customers</h3><h1>{active_str}</h1></div>',
            unsafe_allow_html=True,
        )
    with col4:
        avg_mc = kpis.get("avg_monthly_charges", "N/A")
        st.markdown(
            f'<div class="kpi-card kpi-purple"><h3>Avg Monthly Charges</h3><h1>₹{avg_mc}</h1></div>',
            unsafe_allow_html=True,
        )
    with col5:
        avg_t = kpis.get("avg_tenure", "N/A")
        st.markdown(
            f'<div class="kpi-card kpi-orange"><h3>Avg Tenure</h3><h1>{avg_t} mo</h1></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ─── Risk Distribution (if model is trained) ─── #
    model, preprocessor, model_name = load_model()

    if model is not None and "Churn" in df.columns:
        try:
            from src.preprocessing import identify_feature_columns
            numerical_cols, categorical_cols = identify_feature_columns(df)
            risk_df = predict_batch(model, preprocessor, df, numerical_cols, categorical_cols)

            rcol1, rcol2, rcol3 = st.columns(3)
            high_risk = int((risk_df["RiskLevel"] == "HIGH").sum())
            med_risk = int((risk_df["RiskLevel"] == "MEDIUM").sum())
            low_risk = int((risk_df["RiskLevel"] == "LOW").sum())

            with rcol1:
                st.markdown(
                    f'<div class="kpi-card kpi-red"><h3>🔴 High Risk</h3><h1>{high_risk:,}</h1></div>',
                    unsafe_allow_html=True,
                )
            with rcol2:
                st.markdown(
                    f'<div class="kpi-card kpi-orange"><h3>🟡 Medium Risk</h3><h1>{med_risk:,}</h1></div>',
                    unsafe_allow_html=True,
                )
            with rcol3:
                st.markdown(
                    f'<div class="kpi-card kpi-green"><h3>🟢 Low Risk</h3><h1>{low_risk:,}</h1></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("---")
        except Exception:
            pass

    # ─── Charts ─── #
    if "Churn" in df.columns:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.plotly_chart(churn_distribution_chart(df), use_container_width=True)

        with chart_col2:
            if "Contract" in df.columns:
                st.plotly_chart(churn_by_category(df, "Contract"), use_container_width=True)
            elif "Gender" in df.columns:
                st.plotly_chart(churn_by_category(df, "Gender"), use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            if "Tenure" in df.columns:
                st.plotly_chart(churn_by_numerical(df, "Tenure"), use_container_width=True)
        with chart_col4:
            if "MonthlyCharges" in df.columns:
                st.plotly_chart(churn_by_numerical(df, "MonthlyCharges"), use_container_width=True)

    # ─── Top Churn Drivers ─── #
    if model is not None:
        st.markdown("### 🏆 Top Churn Drivers")
        try:
            training = st.session_state.get("training_results")
            if training and "feature_names" in training:
                from src.explainability import compute_shap_values, get_global_importance, plot_global_importance
                shap_vals, _, method = compute_shap_values(
                    model, training["X_test"], training["feature_names"], max_samples=100,
                )
                imp_df = get_global_importance(model, shap_vals, training["feature_names"], method)
                st.plotly_chart(plot_global_importance(imp_df, top_n=10), use_container_width=True)
        except Exception:
            st.info("Train a model to see top churn drivers.")
    else:
        st.info("💡 Train a model on the **Model Performance** page to unlock risk scores and churn drivers.")
