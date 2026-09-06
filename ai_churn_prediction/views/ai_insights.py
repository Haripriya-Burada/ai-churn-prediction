"""
pages/ai_insights.py
----------------------
AI-powered business insights using Claude API with a complete fallback.
"""

import streamlit as st
import pandas as pd

from src.ai_insights import get_ai_insights, build_analytics_summary
from src.eda import compute_kpis


def render():
    st.markdown('<div class="section-header">💡 AI-Powered Business Insights</div>', unsafe_allow_html=True)

    df = st.session_state.get("df")
    if df is None:
        st.info("Please load a dataset on the **Data Upload** page first.")
        return

    # Build analytics summary
    kpis = compute_kpis(df)

    # Gather additional info from session state
    importance_df = st.session_state.get("importance_df")
    clustered_df = st.session_state.get("clustered_df")
    cluster_labels = st.session_state.get("cluster_labels", {})

    # Build segment info
    segments = {}
    if clustered_df is not None and cluster_labels:
        for cid, label in cluster_labels.items():
            count = int((clustered_df["Cluster"] == cid).sum())
            segments[label] = {"count": count}

    # Check for risk data
    risk_df = None
    tr = st.session_state.get("training_results")
    if tr:
        try:
            from src.preprocessing import identify_feature_columns
            from src.prediction import predict_batch
            numerical_cols, categorical_cols = identify_feature_columns(df)
            risk_df = predict_batch(tr["best_model"], tr["preprocessor"], df, numerical_cols, categorical_cols)
        except Exception:
            pass

    summary = build_analytics_summary(
        df=df,
        kpis=kpis,
        importance_df=importance_df,
        risk_df=risk_df,
        segments=segments,
    )

    st.markdown("### 📋 Analytics Summary Sent to AI")
    with st.expander("View Data Summary"):
        for key, value in summary.items():
            if key != "segments":
                st.markdown(f"- **{key}**: {value}")
            else:
                st.markdown(f"- **segments**: {value}")

    st.markdown("---")

    # Generate insights
    if st.button("🧠 Generate AI Insights", type="primary", use_container_width=True):
        with st.spinner("Generating business insights..."):
            result = get_ai_insights(summary)

            st.session_state["ai_insights"] = result

            if result["source"] == "claude":
                st.success("✅ Insights generated using Claude AI.")
            else:
                st.info(
                    "ℹ️ Insights generated using the built-in rule-based engine. "
                    "Set your `ANTHROPIC_API_KEY` in the `.env` file to use Claude AI."
                )

    # Display cached insights
    if "ai_insights" in st.session_state:
        result = st.session_state["ai_insights"]

        source_badge = "🤖 Claude AI" if result["source"] == "claude" else "📊 Rule-Based Engine"
        st.markdown(f"**Source:** {source_badge}")
        st.markdown("---")

        st.markdown(result["content"])

        # Download
        st.markdown("---")
        st.download_button(
            "⬇️ Download Insights Report",
            data=result["content"],
            file_name="ai_business_insights.md",
            mime="text/markdown",
        )
    else:
        st.info("Click **Generate AI Insights** to create a business intelligence report.")

    # Tips
    with st.expander("💡 Tips for Better Insights"):
        st.markdown("""
1. **Train models first** — the AI generates better insights when it has access to 
   model performance data and feature importances.
2. **Run segmentation** — segment information helps generate targeted recommendations.
3. **Use Claude API** — set `ANTHROPIC_API_KEY` in your `.env` file for more nuanced, 
   context-aware insights.
4. **Privacy** — only anonymised, aggregate-level statistics are sent to the API. 
   No individual customer data is shared.
        """)
