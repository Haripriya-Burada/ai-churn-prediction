"""
pages/eda.py
-------------
Exploratory Data Analysis page with KPI cards, interactive charts,
and filtering capabilities.
"""

import streamlit as st
import pandas as pd

from src.eda import (
    compute_kpis,
    churn_distribution_chart,
    churn_by_category,
    churn_by_numerical,
    correlation_heatmap,
    numerical_distributions,
    categorical_distributions,
)


def render():
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Please load a dataset on the **Data Upload** page first.")
        return

    df = st.session_state["df"]
    has_churn = "Churn" in df.columns

    # ─── Filters ─── #
    with st.expander("🔍 Filter Data", expanded=False):
        filtered_df = _apply_filters(df)

    # ─── KPI Cards ─── #
    kpis = compute_kpis(filtered_df)

    cols = st.columns(6)
    kpi_items = [
        ("Total Customers", f"{kpis['total_customers']:,}", "kpi-blue"),
        ("Churned", f"{kpis.get('churned', 'N/A'):,}" if isinstance(kpis.get('churned'), int) else "N/A", "kpi-red"),
        ("Active", f"{kpis.get('active', 'N/A'):,}" if isinstance(kpis.get('active'), int) else "N/A", "kpi-green"),
        ("Churn Rate", f"{kpis.get('churn_rate', 'N/A')}%", "kpi-orange"),
        ("Avg Monthly", f"₹{kpis.get('avg_monthly_charges', 'N/A')}", "kpi-purple"),
        ("Avg Tenure", f"{kpis.get('avg_tenure', 'N/A')} mo", ""),
    ]

    for col, (label, value, css) in zip(cols, kpi_items):
        with col:
            st.markdown(
                f'<div class="kpi-card {css}"><h3>{label}</h3><h1>{value}</h1></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    if not has_churn:
        st.warning("Churn column not found. Showing general distributions only.")

    # ─── Churn Charts ─── #
    if has_churn:
        st.markdown("### Churn Analysis")

        tab1, tab2, tab3 = st.tabs(["📊 By Category", "📈 By Numerical", "🔗 Correlations"])

        with tab1:
            cat_cols = [c for c in ["Gender", "Contract", "PaymentMethod", "InternetService", "TenureGroup"]
                        if c in filtered_df.columns]
            if cat_cols:
                for i in range(0, len(cat_cols), 2):
                    row_cols = st.columns(2)
                    for j, col_name in enumerate(cat_cols[i:i+2]):
                        with row_cols[j]:
                            st.plotly_chart(
                                churn_by_category(filtered_df, col_name),
                                use_container_width=True,
                            )
            else:
                st.info("No categorical columns available for churn analysis.")

        with tab2:
            num_chart_cols = [c for c in ["Tenure", "MonthlyCharges", "TotalCharges", "Age",
                                          "SupportCalls", "SatisfactionScore", "UsageFrequency"]
                             if c in filtered_df.columns]
            if num_chart_cols:
                for i in range(0, len(num_chart_cols), 2):
                    row_cols = st.columns(2)
                    for j, col_name in enumerate(num_chart_cols[i:i+2]):
                        with row_cols[j]:
                            st.plotly_chart(
                                churn_by_numerical(filtered_df, col_name),
                                use_container_width=True,
                            )
            else:
                st.info("No numerical columns available for churn analysis.")

        with tab3:
            st.plotly_chart(correlation_heatmap(filtered_df), use_container_width=True)

    # ─── General Distributions ─── #
    st.markdown("### Feature Distributions")

    dist_tab1, dist_tab2 = st.tabs(["🔢 Numerical", "📝 Categorical"])

    with dist_tab1:
        num_cols = list(filtered_df.select_dtypes(include=["number"]).columns)
        if num_cols:
            st.plotly_chart(numerical_distributions(filtered_df, num_cols), use_container_width=True)
        else:
            st.info("No numerical columns.")

    with dist_tab2:
        cat_cols = list(filtered_df.select_dtypes(include=["object", "category"]).columns)
        if cat_cols:
            st.plotly_chart(categorical_distributions(filtered_df, cat_cols), use_container_width=True)
        else:
            st.info("No categorical columns.")


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar-style filters to the DataFrame."""
    filtered = df.copy()

    filter_cols = st.columns(4)

    # Gender filter
    if "Gender" in df.columns:
        with filter_cols[0]:
            gender = st.multiselect("Gender", df["Gender"].dropna().unique(), default=list(df["Gender"].dropna().unique()))
            if gender:
                filtered = filtered[filtered["Gender"].isin(gender)]

    # Contract filter
    if "Contract" in df.columns:
        with filter_cols[1]:
            contracts = st.multiselect("Contract", df["Contract"].dropna().unique(), default=list(df["Contract"].dropna().unique()))
            if contracts:
                filtered = filtered[filtered["Contract"].isin(contracts)]

    # Tenure range
    if "Tenure" in df.columns:
        with filter_cols[2]:
            min_t, max_t = int(df["Tenure"].min()), int(df["Tenure"].max())
            tenure_range = st.slider("Tenure Range", min_t, max_t, (min_t, max_t))
            filtered = filtered[(filtered["Tenure"] >= tenure_range[0]) & (filtered["Tenure"] <= tenure_range[1])]

    # Monthly Charges range
    if "MonthlyCharges" in df.columns:
        with filter_cols[3]:
            min_c, max_c = float(df["MonthlyCharges"].min()), float(df["MonthlyCharges"].max())
            charge_range = st.slider("Monthly Charges", min_c, max_c, (min_c, max_c))
            filtered = filtered[(filtered["MonthlyCharges"] >= charge_range[0]) & (filtered["MonthlyCharges"] <= charge_range[1])]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} customers after filtering.")
    return filtered
