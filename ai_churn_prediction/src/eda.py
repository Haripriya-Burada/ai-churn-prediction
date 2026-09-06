"""
eda.py
------
Exploratory Data Analysis helpers — builds Plotly figures and computes
KPI metrics for the Streamlit EDA page.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional


# ─────────────────────── KPI METRICS ─────────────────────── #

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute key performance indicator metrics from the dataset."""
    kpis: Dict[str, Any] = {}
    kpis["total_customers"] = len(df)

    if "Churn" in df.columns:
        kpis["churned"] = int(df["Churn"].sum())
        kpis["active"] = kpis["total_customers"] - kpis["churned"]
        kpis["churn_rate"] = round(df["Churn"].mean() * 100, 2)
    else:
        kpis["churned"] = "N/A"
        kpis["active"] = "N/A"
        kpis["churn_rate"] = "N/A"

    if "MonthlyCharges" in df.columns:
        kpis["avg_monthly_charges"] = round(df["MonthlyCharges"].mean(), 2)
    if "Tenure" in df.columns:
        kpis["avg_tenure"] = round(df["Tenure"].mean(), 1)
    if "TotalCharges" in df.columns:
        kpis["avg_total_charges"] = round(df["TotalCharges"].mean(), 2)

    return kpis


# ─────────────────── CHART BUILDERS ──────────────────── #

def churn_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Pie chart showing churn vs. active split."""
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    counts["Label"] = counts["Churn"].map({0: "Active", 1: "Churned"})
    fig = px.pie(
        counts, names="Label", values="Count",
        color="Label",
        color_discrete_map={"Active": "#2ecc71", "Churned": "#e74c3c"},
        title="Churn Distribution",
        hole=0.4,
    )
    fig.update_layout(template="plotly_white")
    return fig


def churn_by_category(df: pd.DataFrame, col: str, title: Optional[str] = None) -> go.Figure:
    """Grouped bar chart of churn rate by a categorical column."""
    if col not in df.columns or "Churn" not in df.columns:
        return _empty_figure(f"{col} not available")

    cross = df.groupby(col)["Churn"].value_counts(normalize=True).unstack(fill_value=0)
    cross = cross.reset_index()
    if 1 not in cross.columns:
        cross[1] = 0.0

    cross["Churn Rate (%)"] = (cross[1] * 100).round(1)
    fig = px.bar(
        cross, x=col, y="Churn Rate (%)",
        title=title or f"Churn Rate by {col}",
        text="Churn Rate (%)",
        color="Churn Rate (%)",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(template="plotly_white")
    return fig


def churn_by_numerical(df: pd.DataFrame, col: str) -> go.Figure:
    """Box / violin plot of a numerical column split by churn."""
    if col not in df.columns or "Churn" not in df.columns:
        return _empty_figure(f"{col} not available")

    temp = df.copy()
    temp["Churn Label"] = temp["Churn"].map({0: "Active", 1: "Churned"})
    fig = px.box(
        temp, x="Churn Label", y=col,
        color="Churn Label",
        color_discrete_map={"Active": "#2ecc71", "Churned": "#e74c3c"},
        title=f"{col} Distribution by Churn",
    )
    fig.update_layout(template="plotly_white")
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of all numerical columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return _empty_figure("Not enough numerical columns for correlation")

    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white")
    return fig


def numerical_distributions(df: pd.DataFrame, cols: list) -> go.Figure:
    """Histograms for a list of numerical columns."""
    cols_available = [c for c in cols if c in df.columns]
    if not cols_available:
        return _empty_figure("No numerical columns to plot")

    n = len(cols_available)
    rows = (n + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=cols_available)

    for idx, col in enumerate(cols_available):
        r = idx // 2 + 1
        c = idx % 2 + 1
        fig.add_trace(
            go.Histogram(x=df[col].dropna(), name=col, marker_color="#3498db"),
            row=r, col=c,
        )

    fig.update_layout(
        title_text="Numerical Feature Distributions",
        showlegend=False,
        template="plotly_white",
        height=300 * rows,
    )
    return fig


def categorical_distributions(df: pd.DataFrame, cols: list) -> go.Figure:
    """Bar charts for categorical columns."""
    cols_available = [c for c in cols if c in df.columns]
    if not cols_available:
        return _empty_figure("No categorical columns to plot")

    n = len(cols_available)
    rows = (n + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=cols_available)

    for idx, col in enumerate(cols_available):
        r = idx // 2 + 1
        c = idx % 2 + 1
        counts = df[col].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=counts.index.astype(str), y=counts.values, name=col, marker_color="#9b59b6"),
            row=r, col=c,
        )

    fig.update_layout(
        title_text="Categorical Feature Distributions",
        showlegend=False,
        template="plotly_white",
        height=300 * rows,
    )
    return fig


# ─────────────────── HELPERS ──────────────────── #

def _empty_figure(message: str) -> go.Figure:
    """Return an empty figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        template="plotly_white", height=300,
    )
    return fig
