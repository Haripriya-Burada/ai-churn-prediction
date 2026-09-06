"""
pages/data_upload.py
---------------------
Data upload page — CSV upload, preview, summary, and cleaning.
"""

import streamlit as st
import pandas as pd
import os

from src.data_loader import (
    load_csv,
    load_uploaded_file,
    normalize_churn_column,
    get_dataset_summary,
    get_sample_data_path,
    clean_dataset,
)
from src.feature_engineering import engineer_features, get_engineered_feature_descriptions
from src.database import save_customers


def render():
    st.markdown('<div class="section-header">📁 Data Upload & Exploration</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📤 Upload Data", "📊 Dataset Summary"])

    with tab1:
        _upload_section()

    with tab2:
        _summary_section()


def _upload_section():
    """Handle file upload or sample data loading."""

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="Upload your customer dataset in CSV format.",
        )

    with col2:
        st.markdown("#### Or use sample data")
        use_sample = st.button("📦 Load Sample Dataset", use_container_width=True)

    # Process uploaded file
    if uploaded_file is not None:
        try:
            df = load_uploaded_file(uploaded_file)
            df, has_churn = normalize_churn_column(df)

            if not has_churn:
                st.warning(
                    "⚠️ **Churn column not found.** Supervised churn prediction requires a target column. "
                    "Supported names: Churn, Exited, Customer_Churn, Is_Churn. "
                    "EDA and segmentation will still work."
                )

            st.session_state["df"] = df
            st.session_state["has_churn"] = has_churn
            st.success(f"✅ Dataset loaded successfully! ({len(df):,} rows, {len(df.columns)} columns)")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return

    # Load sample data
    if use_sample:
        sample_path = get_sample_data_path()
        if not os.path.exists(sample_path):
            st.warning("Sample dataset not found. Generating it now...")
            try:
                from generate_sample_data import generate_customer_dataset
                os.makedirs(os.path.dirname(sample_path), exist_ok=True)
                df = generate_customer_dataset()
                df.to_csv(sample_path, index=False)
                st.success("✅ Sample dataset generated!")
            except Exception as e:
                st.error(f"Error generating sample data: {str(e)}")
                return
        else:
            df = load_csv(sample_path)

        df, has_churn = normalize_churn_column(df)
        st.session_state["df"] = df
        st.session_state["has_churn"] = has_churn
        st.success(f"✅ Sample dataset loaded! ({len(df):,} rows, {len(df.columns)} columns)")

    # Preview
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]

        st.markdown("### 👀 Data Preview")
        st.dataframe(df.head(20), use_container_width=True, height=400)

        # Feature Engineering
        with st.expander("🔧 Apply Feature Engineering"):
            if st.button("Engineer Features"):
                df_eng = engineer_features(df)
                new_cols = [c for c in df_eng.columns if c not in df.columns]
                if new_cols:
                    st.session_state["df"] = df_eng
                    st.success(f"✅ Created {len(new_cols)} new features: {', '.join(new_cols)}")

                    descs = get_engineered_feature_descriptions()
                    for col in new_cols:
                        if col in descs:
                            st.info(f"**{col}**: {descs[col]}")
                else:
                    st.info("No new features could be created from the available columns.")

        # Clean & Download
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🧹 Clean Dataset"):
                df_clean = clean_dataset(df)
                removed = len(df) - len(df_clean)
                st.session_state["df"] = df_clean
                st.success(f"✅ Cleaned! Removed {removed} duplicate rows.")

        with col_b:
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Dataset (CSV)",
                data=csv_data,
                file_name="customer_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # Save to DB
        with st.expander("💾 Save to Database"):
            if st.button("Save to SQLite"):
                try:
                    rows = save_customers(df)
                    st.success(f"✅ {rows:,} rows saved to database.")
                except Exception as e:
                    st.error(f"Database error: {str(e)}")


def _summary_section():
    """Show dataset summary statistics."""
    if "df" not in st.session_state or st.session_state["df"] is None:
        st.info("Load a dataset first to see the summary.")
        return

    df = st.session_state["df"]
    summary = get_dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{summary['num_rows']:,}")
    with col2:
        st.metric("Columns", summary["num_columns"])
    with col3:
        st.metric("Missing Values", f"{summary['total_missing']:,}")
    with col4:
        st.metric("Duplicate Rows", f"{summary['duplicate_rows']:,}")

    st.markdown("---")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("#### 📐 Numerical Columns")
        if summary["numerical_columns"]:
            st.write(", ".join(summary["numerical_columns"]))
        else:
            st.write("None detected")

    with info_col2:
        st.markdown("#### 📝 Categorical Columns")
        if summary["categorical_columns"]:
            st.write(", ".join(summary["categorical_columns"]))
        else:
            st.write("None detected")

    st.markdown("---")

    st.markdown("#### 📋 Column Details")
    detail_df = pd.DataFrame({
        "Column": summary["columns"],
        "Data Type": [summary["dtypes"][c] for c in summary["columns"]],
        "Missing": [summary["missing_values"][c] for c in summary["columns"]],
        "Missing %": [
            f"{summary['missing_values'][c] / summary['num_rows'] * 100:.1f}%"
            for c in summary["columns"]
        ],
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 📊 Descriptive Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)
