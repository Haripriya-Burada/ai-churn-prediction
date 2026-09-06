"""
pages/segmentation.py
-----------------------
Customer segmentation page using K-Means clustering.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.clustering import (
    run_kmeans,
    auto_label_clusters,
    cluster_summary,
    cluster_scatter,
    cluster_distribution_chart,
)
from src.recommendations import generate_segment_recommendations
from src.database import save_segments


def render():
    st.markdown('<div class="section-header">👥 Customer Segmentation</div>', unsafe_allow_html=True)

    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Please load a dataset on the **Data Upload** page first.")
        return

    df = st.session_state["df"]

    # Identify numerical columns for clustering
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    # Remove target and ID-like columns
    exclude = {"Churn", "PredictedChurn", "Cluster"}
    num_cols = [c for c in num_cols if c not in exclude and "id" not in c.lower()]

    if len(num_cols) < 2:
        st.error("Not enough numerical columns for clustering (need at least 2).")
        return

    # Default features
    default_features = [c for c in ["Tenure", "MonthlyCharges", "TotalCharges", "UsageFrequency", "NumProducts"]
                        if c in num_cols]
    if not default_features:
        default_features = num_cols[:3]

    # ─── Configuration ─── #
    st.markdown("### ⚙️ Clustering Configuration")

    config_col1, config_col2 = st.columns([2, 1])

    with config_col1:
        selected_features = st.multiselect(
            "Select Features for Clustering",
            num_cols,
            default=default_features,
            help="Choose numerical features that best describe customer segments.",
        )

    with config_col2:
        n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4)

    if len(selected_features) < 2:
        st.warning("Please select at least 2 features.")
        return

    if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
        with st.spinner("Running K-Means clustering..."):
            try:
                clustered_df, kmeans_model, scaler = run_kmeans(
                    df, selected_features, n_clusters=n_clusters,
                )
                labels = auto_label_clusters(clustered_df, selected_features)

                st.session_state["clustered_df"] = clustered_df
                st.session_state["cluster_labels"] = labels
                st.session_state["cluster_features"] = selected_features

                # Save to DB
                save_segments(clustered_df, labels)

                st.success(f"✅ Segmentation complete! {n_clusters} clusters identified.")

            except Exception as e:
                st.error(f"Clustering failed: {str(e)}")
                return

    # ─── Results ─── #
    if "clustered_df" not in st.session_state:
        st.info("Configure the settings above and click **Run Segmentation** to start.")
        _show_explanation()
        return

    clustered_df = st.session_state["clustered_df"]
    labels = st.session_state["cluster_labels"]
    selected_features = st.session_state["cluster_features"]

    # Segment KPI cards
    st.markdown("### 📊 Segments Overview")
    n_seg = max(1, len(labels))
    seg_cols = st.columns(n_seg)
    colors = ["kpi-blue", "kpi-green", "kpi-orange", "kpi-purple", "kpi-red", "kpi-blue", "kpi-green", "kpi-orange"]

    for idx, (cid, label) in enumerate(labels.items()):
        count = int((clustered_df["Cluster"] == cid).sum())
        css = colors[idx % len(colors)]
        with seg_cols[idx]:
            st.markdown(
                f'<div class="kpi-card {css}"><h3>{label}</h3><h1>{count:,}</h1></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Charts
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔵 Scatter Plot", "📋 Summary"])

    with tab1:
        st.plotly_chart(cluster_distribution_chart(clustered_df, labels), use_container_width=True)

    with tab2:
        if len(selected_features) >= 2:
            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                x_feat = st.selectbox("X Axis", selected_features, index=0)
            with sc_col2:
                y_feat = st.selectbox("Y Axis", selected_features, index=min(1, len(selected_features)-1))
            st.plotly_chart(cluster_scatter(clustered_df, x_feat, y_feat, labels), use_container_width=True)

    with tab3:
        summary = cluster_summary(clustered_df, selected_features)
        summary["Segment"] = summary["Cluster"].map(labels)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # ─── Segment Recommendations ─── #
    st.markdown("### 💡 Segment Recommendations")
    for cid, label in labels.items():
        cluster_data = clustered_df[clustered_df["Cluster"] == cid]
        stats = cluster_data[selected_features].mean().to_dict()
        recs = generate_segment_recommendations(label, stats)
        with st.expander(f"📌 {label} ({len(cluster_data):,} customers)"):
            for rec in recs:
                st.markdown(f"- {rec}")


def _show_explanation():
    """Brief explanation of K-Means for students."""
    with st.expander("📖 How K-Means Clustering Works"):
        st.markdown("""
**K-Means** is an unsupervised machine learning algorithm that groups customers into **K clusters** 
based on similarity in their features.

**Steps:**
1. **Choose K** — the number of segments you want.
2. **Initialise** — randomly place K centroids in the feature space.
3. **Assign** — each customer is assigned to the nearest centroid.
4. **Update** — move each centroid to the mean of its assigned customers.
5. **Repeat** steps 3–4 until centroids stop moving.

**Why scale the data?**
Features like Monthly Charges (₹20–120) and Tenure (1–72 months) have different ranges. 
Without scaling, the feature with the largest range would dominate the distance calculation. 
We use **StandardScaler** to normalise all features to mean=0, std=1.

**Choosing K:**
- Start with K=3–5 for business interpretability.
- The **Elbow Method** and **Silhouette Score** can help find the optimal K.
        """)
