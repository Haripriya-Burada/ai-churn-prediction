"""
pages/model_performance.py
----------------------------
Model training, evaluation, and comparison page.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

from src.model_training import train_and_evaluate, build_comparison_table
from src.database import save_model_metrics


def render():
    st.markdown('<div class="section-header">🤖 Model Training & Performance</div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Please load a dataset on the **Data Upload** page first.")
        return

    df = st.session_state["df"]

    if "Churn" not in df.columns:
        st.error(
            "❌ Churn column not found. Supervised model training requires a "
            "target column named Churn. Please upload a dataset with a churn label."
        )
        return

    # ─── Training Controls ─── #
    st.markdown("### ⚙️ Training Configuration")

    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20, 5) / 100
    with config_col2:
        random_state = st.number_input("Random Seed", value=42, min_value=0)
    with config_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        train_clicked = st.button("🚀 Train All Models", type="primary", use_container_width=True)

    # ─── Train ─── #
    if train_clicked:
        with st.spinner("Training models... This may take a moment."):
            try:
                results = train_and_evaluate(df, test_size=test_size, random_state=random_state)
                st.session_state["training_results"] = results

                # Save metrics to DB
                save_model_metrics(results["results"], results["best_model_name"])

                st.success(
                    f"✅ Training complete! Best model: **{results['best_model_name']}** "
                    f"(ROC-AUC: {results['results'][results['best_model_name']]['roc_auc']:.4f})"
                )
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                return

    # ─── Display Results ─── #
    if "training_results" not in st.session_state:
        st.info("Click **Train All Models** to start training. Models: Logistic Regression, Decision Tree, Random Forest, XGBoost.")
        _show_model_explanations()
        return

    results = st.session_state["training_results"]
    model_results = results["results"]
    best_name = results["best_model_name"]

    # Comparison Table
    st.markdown("### 📊 Model Comparison")
    comp_table = build_comparison_table(model_results)
    st.dataframe(
        comp_table.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            color="#a8f0c6",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"🏆 **Best Model: {best_name}** — Selected based on highest ROC-AUC "
        f"(primary) and F1 Score (tiebreaker). These metrics are preferred for "
        f"churn prediction because they account for class imbalance and balance "
        f"precision with recall."
    )

    st.markdown("---")

    # ─── Detailed Metrics ─── #
    tab1, tab2, tab3, tab4 = st.tabs(["📈 ROC Curves", "📉 PR Curves", "🔢 Confusion Matrices", "📊 Metric Comparison"])

    with tab1:
        _plot_roc_curves(model_results)

    with tab2:
        _plot_pr_curves(model_results)

    with tab3:
        _plot_confusion_matrices(model_results)

    with tab4:
        _plot_metric_comparison(comp_table)

    # ─── Model Explanations ─── #
    with st.expander("📖 Understanding the Models"):
        _show_model_explanations()


def _plot_roc_curves(results: dict):
    """Plot ROC curves for all models."""
    fig = go.Figure()
    for name, m in results.items():
        fpr, tpr, _ = m["roc_curve"]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name} (AUC={m['roc_auc']:.3f})",
            mode="lines",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random",
        line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_pr_curves(results: dict):
    """Plot Precision-Recall curves."""
    fig = go.Figure()
    for name, m in results.items():
        precision, recall, _ = m["pr_curve"]
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name=name,
            mode="lines",
        ))
    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_confusion_matrices(results: dict):
    """Plot confusion matrices for all models."""
    models = list(results.keys())
    cols = st.columns(2)
    for idx, name in enumerate(models):
        cm = np.array(results[name]["confusion_matrix"])
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Active", "Churned"],
            y=["Active", "Churned"],
            color_continuous_scale="Blues",
            title=f"{name}",
        )
        fig.update_layout(height=350)
        with cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)


def _plot_metric_comparison(comp_table: pd.DataFrame):
    """Grouped bar chart of all metrics by model."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=comp_table["Model"],
            y=comp_table[metric],
            text=comp_table[metric].round(3),
            textposition="auto",
        ))
    fig.update_layout(
        title="Model Metrics Comparison",
        barmode="group",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def _show_model_explanations():
    """Show brief explanations of each ML model."""
    explanations = {
        "Logistic Regression": (
            "A linear model that estimates the probability of churn using a "
            "sigmoid function. Fast, interpretable, and works well when the "
            "relationship between features and churn is approximately linear."
        ),
        "Decision Tree": (
            "A tree-structured model that splits data on feature thresholds. "
            "Highly interpretable ('if tenure < 6 months AND contract = monthly → churn'). "
            "Can overfit if the tree is too deep."
        ),
        "Random Forest": (
            "An ensemble of many decision trees, each trained on a random subset of "
            "data. Reduces overfitting compared to a single tree. One of the most "
            "reliable general-purpose classifiers."
        ),
        "XGBoost": (
            "Gradient-boosted trees — builds trees sequentially, each correcting the "
            "errors of the previous one. Often achieves the highest accuracy for "
            "tabular data. The `scale_pos_weight` parameter helps with class imbalance."
        ),
    }

    for name, desc in explanations.items():
        st.markdown(f"**{name}**: {desc}")
