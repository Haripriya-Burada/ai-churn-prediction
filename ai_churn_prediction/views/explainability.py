"""
pages/explainability.py
------------------------
SHAP explainability page — global and individual explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.model_training import load_model
from src.explainability import (
    compute_shap_values,
    get_global_importance,
    plot_global_importance,
    get_individual_explanation,
)
from src.preprocessing import identify_feature_columns


def render():
    st.markdown('<div class="section-header">🔍 Model Explainability (SHAP)</div>', unsafe_allow_html=True)

    # Check for trained model
    tr = st.session_state.get("training_results")
    model, preprocessor, model_name = load_model()

    if tr:
        model = tr["best_model"]
        preprocessor = tr["preprocessor"]
        model_name = tr["best_model_name"]
        feature_names = tr["feature_names"]
        X_test = tr["X_test"]
        numerical_cols = tr["numerical_cols"]
        categorical_cols = tr["categorical_cols"]
    elif model is not None:
        feature_names = None
        X_test = None
        numerical_cols = None
        categorical_cols = None
    else:
        st.warning("⚠️ No trained model found. Please train models on the **Model Performance** page first.")
        _show_shap_explanation()
        return

    tab1, tab2 = st.tabs(["🌍 Global Explanations", "👤 Individual Explanations"])

    with tab1:
        _global_explanations(model, X_test, feature_names, model_name)

    with tab2:
        _individual_explanations(model, preprocessor, feature_names, numerical_cols, categorical_cols, X_test)


def _global_explanations(model, X_test, feature_names, model_name):
    """Display global feature importance using SHAP or fallback."""
    st.markdown(f"### Global Feature Importance — {model_name}")

    if X_test is None or feature_names is None:
        st.info("Global explanations require training results. Please retrain models on the Model Performance page.")
        return

    if st.button("🔍 Compute Feature Importance", type="primary"):
        with st.spinner("Computing SHAP values... This may take a moment."):
            shap_values, explainer, method = compute_shap_values(
                model, X_test, feature_names, max_samples=200,
            )

            if method == "shap":
                st.success("✅ SHAP values computed successfully.")
            else:
                st.info("ℹ️ Using model's built-in feature importances (SHAP was not compatible with this model).")

            importance_df = get_global_importance(model, shap_values, feature_names, method)
            st.session_state["importance_df"] = importance_df
            st.session_state["shap_method"] = method

    # Display cached results
    if "importance_df" in st.session_state:
        importance_df = st.session_state["importance_df"]
        method = st.session_state.get("shap_method", "feature_importances")

        st.plotly_chart(plot_global_importance(importance_df), use_container_width=True)

        with st.expander("📋 Full Feature Importance Table"):
            st.dataframe(importance_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🧠 Interpretation")
        top5 = importance_df.head(5)["Feature"].tolist()
        st.markdown(
            f"The top 5 most important features for predicting churn are: "
            f"**{', '.join(top5)}**. "
            f"These features have the greatest influence on whether a customer "
            f"is predicted to churn or remain active."
        )


def _individual_explanations(model, preprocessor, feature_names, numerical_cols, categorical_cols, X_background):
    """Explain a single prediction."""
    st.markdown("### Individual Prediction Explanation")

    if feature_names is None or numerical_cols is None:
        st.info("Individual explanations require training results. Please retrain on the Model Performance page.")
        return

    df = st.session_state.get("df")
    if df is None:
        st.info("Load a dataset to use individual explanations.")
        return

    # Let user pick a customer
    if "CustomerID" in df.columns:
        customer_ids = df["CustomerID"].tolist()[:100]
        selected_id = st.selectbox("Select a Customer", customer_ids)
        customer_row = df[df["CustomerID"] == selected_id].iloc[0]
        customer_data = {col: customer_row[col] for col in numerical_cols + categorical_cols if col in customer_row.index}
    else:
        st.info("Select a row number to explain.")
        row_idx = st.number_input("Row Index", min_value=0, max_value=len(df)-1, value=0)
        customer_row = df.iloc[row_idx]
        customer_data = {col: customer_row[col] for col in numerical_cols + categorical_cols if col in customer_row.index}

    if st.button("🔍 Explain This Prediction"):
        with st.spinner("Computing explanation..."):
            explanation = get_individual_explanation(
                model, preprocessor, customer_data,
                feature_names, numerical_cols, categorical_cols,
                X_background=X_background,
            )

            if explanation["drivers"]:
                st.markdown(f"**Method used:** {explanation['method'].upper()}")
                st.markdown("#### Key Factors")

                for i, driver in enumerate(explanation["drivers"], 1):
                    icon = "🔴" if driver["direction"] == "increases churn" else "🟢"
                    st.markdown(
                        f"{i}. {icon} **{driver['feature']}** — "
                        f"{driver['direction']} (impact: {driver['shap_value']:.4f})"
                    )
            else:
                st.info("Could not compute individual explanation for this model type.")

    _show_shap_explanation()


def _show_shap_explanation():
    """Educational explanation of SHAP."""
    with st.expander("📖 Understanding SHAP"):
        st.markdown("""
**SHAP (SHapley Additive exPlanations)** is an approach from game theory that 
explains the output of any ML model.

**Key Concepts:**
- Each feature gets a **SHAP value** for each prediction.
- A **positive** SHAP value means the feature **pushes the prediction toward churn**.
- A **negative** value means it **pushes away from churn**.
- The sum of all SHAP values equals the difference between the model's prediction 
  and the average prediction.

**Why is this useful?**
- **Global importance** shows which features matter most across all customers.
- **Individual explanations** show why a specific customer was predicted to churn.
- This helps business users **trust** the model and take **targeted action**.

**Example:**
> "Customer CUST-00123 has HIGH churn risk because:
> 1. Month-to-month contract (+0.35)
> 2. High monthly charges (+0.22)
> 3. Short tenure (+0.18)
> 4. Low satisfaction score (+0.15)"
        """)
