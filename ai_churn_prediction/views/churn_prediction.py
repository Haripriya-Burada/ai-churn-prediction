"""
pages/churn_prediction.py
--------------------------
Individual customer churn prediction form and risk dashboard.
"""

import streamlit as st
import pandas as pd

from src.model_training import load_model
from src.prediction import predict_single_customer, predict_batch, classify_risk
from src.recommendations import generate_recommendations
from src.preprocessing import identify_feature_columns
from src.database import save_predictions


def render():
    st.markdown('<div class="section-header">🔮 Churn Prediction</div>', unsafe_allow_html=True)

    # Load model
    model, preprocessor, model_name = load_model()

    if model is None:
        # Check session state for a freshly trained model
        tr = st.session_state.get("training_results")
        if tr:
            model = tr["best_model"]
            preprocessor = tr["preprocessor"]
            model_name = tr["best_model_name"]
        else:
            st.warning(
                "⚠️ No trained model found. Please train models on the **Model Performance** page first."
            )
            return

    tab1, tab2 = st.tabs(["👤 Individual Prediction", "📋 Batch Risk Dashboard"])

    with tab1:
        _individual_prediction(model, preprocessor, model_name)

    with tab2:
        _batch_dashboard(model, preprocessor, model_name)


def _individual_prediction(model, preprocessor, model_name):
    """Form-based single customer prediction."""
    st.markdown("### Enter Customer Details")

    # Determine which columns the model expects
    df = st.session_state.get("df")
    if df is not None:
        numerical_cols, categorical_cols = identify_feature_columns(df)
    else:
        numerical_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges", "SupportCalls",
                          "NumProducts", "UsageFrequency", "SatisfactionScore"]
        categorical_cols = ["Gender", "Contract", "PaymentMethod", "InternetService"]

    # Build the form
    with st.form("prediction_form"):
        form_cols = st.columns(3)
        customer_data = {}

        # Numerical inputs
        num_defaults = {
            "Age": (18, 80, 35),
            "Tenure": (0, 72, 12),
            "MonthlyCharges": (0.0, 200.0, 65.0),
            "TotalCharges": (0.0, 10000.0, 800.0),
            "SupportCalls": (0, 15, 2),
            "NumProducts": (1, 5, 2),
            "UsageFrequency": (1, 30, 15),
            "SatisfactionScore": (1, 5, 3),
        }

        for idx, col in enumerate(numerical_cols):
            with form_cols[idx % 3]:
                if col in num_defaults:
                    mn, mx, default = num_defaults[col]
                    if isinstance(mn, float):
                        customer_data[col] = st.number_input(col, min_value=mn, max_value=mx, value=default, step=0.01)
                    else:
                        customer_data[col] = st.number_input(col, min_value=mn, max_value=mx, value=default)
                else:
                    customer_data[col] = st.number_input(col, value=0.0)

        # Categorical inputs
        cat_options = {
            "Gender": ["Male", "Female"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            "InternetService": ["Fiber optic", "DSL", "No"],
        }

        for idx, col in enumerate(categorical_cols):
            with form_cols[idx % 3]:
                if col in cat_options:
                    customer_data[col] = st.selectbox(col, cat_options[col])
                elif df is not None and col in df.columns:
                    options = list(df[col].dropna().unique())
                    customer_data[col] = st.selectbox(col, options)
                else:
                    customer_data[col] = st.text_input(col)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

    if submitted:
        try:
            # Get feature names from training
            tr = st.session_state.get("training_results")
            feature_names = tr["feature_names"] if tr else None

            result = predict_single_customer(
                model, preprocessor, customer_data,
                feature_names=feature_names,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
            )

            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")

            res_cols = st.columns(3)

            with res_cols[0]:
                st.markdown(
                    f'<div class="kpi-card" style="background:{result["risk_color"]}">'
                    f'<h3>Churn Probability</h3>'
                    f'<h1>{result["probability_pct"]}</h1></div>',
                    unsafe_allow_html=True,
                )
            with res_cols[1]:
                st.markdown(
                    f'<div class="kpi-card" style="background:{result["risk_color"]}">'
                    f'<h3>Risk Level</h3>'
                    f'<h1>{result["risk_level"]}</h1></div>',
                    unsafe_allow_html=True,
                )
            with res_cols[2]:
                st.markdown(
                    f'<div class="kpi-card kpi-blue">'
                    f'<h3>Prediction</h3>'
                    f'<h1>{result["predicted_label"]}</h1></div>',
                    unsafe_allow_html=True,
                )

            # Top factors
            if result["top_factors"]:
                st.markdown("### 🔍 Top Contributing Factors")
                for i, factor in enumerate(result["top_factors"], 1):
                    st.markdown(f"**{i}. {factor['feature']}** — importance: {factor['importance']:.4f}")

            # Recommendations
            st.markdown("### 💡 Recommended Actions")
            recs = generate_recommendations(
                result["risk_level"], customer_data, result["top_factors"],
            )
            for rec in recs:
                priority_color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(
                    rec["priority"], "⚪"
                )
                with st.expander(f"{priority_color} {rec['title']} (Priority: {rec['priority']})"):
                    st.write(rec["description"])

            # Save to DB
            save_predictions([{
                "customer_id": "MANUAL",
                "churn_probability": result["probability"],
                "predicted_churn": result["predicted_class"],
                "risk_level": result["risk_level"],
            }], model_name=model_name if model_name else "")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


def _batch_dashboard(model, preprocessor, model_name):
    """Batch prediction risk dashboard."""
    df = st.session_state.get("df")
    if df is None or "Churn" not in df.columns:
        st.info("Load a dataset with a churn column to see the risk dashboard.")
        return

    st.markdown("### Customer Risk Overview")

    try:
        numerical_cols, categorical_cols = identify_feature_columns(df)
        risk_df = predict_batch(model, preprocessor, df, numerical_cols, categorical_cols)

        # KPI row
        kpi_cols = st.columns(4)
        total = len(risk_df)
        high = int((risk_df["RiskLevel"] == "HIGH").sum())
        medium = int((risk_df["RiskLevel"] == "MEDIUM").sum())
        low = int((risk_df["RiskLevel"] == "LOW").sum())

        with kpi_cols[0]:
            st.metric("Total Customers", f"{total:,}")
        with kpi_cols[1]:
            st.metric("🔴 High Risk", f"{high:,}", delta=f"{high/total*100:.1f}%")
        with kpi_cols[2]:
            st.metric("🟡 Medium Risk", f"{medium:,}", delta=f"{medium/total*100:.1f}%")
        with kpi_cols[3]:
            st.metric("🟢 Low Risk", f"{low:,}", delta=f"{low/total*100:.1f}%")

        st.markdown("---")

        # Filter by risk level
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "HIGH", "MEDIUM", "LOW"])
        if risk_filter != "All":
            display_df = risk_df[risk_df["RiskLevel"] == risk_filter]
        else:
            display_df = risk_df

        # Sort by churn probability
        display_df = display_df.sort_values("ChurnProbability", ascending=False)

        # Select display columns
        show_cols = ["CustomerID", "ChurnProbability", "RiskLevel", "PredictedChurn"]
        for c in ["Tenure", "MonthlyCharges", "Contract", "SupportCalls", "SatisfactionScore"]:
            if c in display_df.columns:
                show_cols.append(c)

        available_cols = [c for c in show_cols if c in display_df.columns]
        st.dataframe(display_df[available_cols].head(200), use_container_width=True, hide_index=True)

        # Download
        csv = display_df[available_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Risk Report (CSV)",
            data=csv,
            file_name="customer_risk_report.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Batch prediction failed: {str(e)}")
