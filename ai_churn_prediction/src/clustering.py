"""
clustering.py
--------------
K-Means customer segmentation with automatic segment labelling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple


def run_kmeans(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Run K-Means clustering on selected numerical features.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list of numerical columns to use.
    n_clusters : int
    random_state : int

    Returns
    -------
    (df_with_clusters, kmeans_model, scaler)
    """
    data = df[feature_cols].copy()
    data = data.fillna(data.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)

    result = df.copy()
    result["Cluster"] = labels
    return result, kmeans, scaler


def auto_label_clusters(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[int, str]:
    """
    Automatically generate meaningful labels based on cluster statistics.

    The logic compares each cluster's mean to the overall mean on key
    dimensions and picks the most descriptive label.
    """
    overall = df[feature_cols].mean()
    labels: Dict[int, str] = {}

    for cluster_id in sorted(df["Cluster"].unique()):
        cluster_data = df[df["Cluster"] == cluster_id]
        cluster_means = cluster_data[feature_cols].mean()

        # Score various "archetypes"
        scores: Dict[str, float] = {}

        # High Value — above-average on spending
        spend_cols = [c for c in feature_cols if "charge" in c.lower() or "spending" in c.lower() or "total" in c.lower()]
        if spend_cols:
            scores["High Value"] = sum(
                (cluster_means[c] - overall[c]) / (overall[c] + 1e-9) for c in spend_cols
            )

        # At Risk — high support calls or low satisfaction
        risk_cols = [c for c in feature_cols if "support" in c.lower()]
        if risk_cols:
            scores["At Risk"] = sum(
                (cluster_means[c] - overall[c]) / (overall[c] + 1e-9) for c in risk_cols
            )

        # Loyal — long tenure
        tenure_cols = [c for c in feature_cols if "tenure" in c.lower()]
        if tenure_cols:
            scores["Loyal"] = sum(
                (cluster_means[c] - overall[c]) / (overall[c] + 1e-9) for c in tenure_cols
            )

        # Price Sensitive — low spending
        if spend_cols:
            scores["Price Sensitive"] = -scores.get("High Value", 0)

        # Low Engagement — low usage / products
        engage_cols = [c for c in feature_cols if "usage" in c.lower() or "product" in c.lower() or "engagement" in c.lower()]
        if engage_cols:
            scores["Low Engagement"] = -sum(
                (cluster_means[c] - overall[c]) / (overall[c] + 1e-9) for c in engage_cols
            )

        if scores:
            best_label = max(scores, key=lambda k: scores[k])
        else:
            best_label = f"Segment {cluster_id}"

        # Deduplicate labels
        if best_label in labels.values():
            best_label = f"{best_label} (Group {cluster_id})"

        labels[cluster_id] = best_label

    return labels


def cluster_summary(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Summary statistics per cluster."""
    if df.empty or not feature_cols or "Cluster" not in df.columns:
        return pd.DataFrame()

    counts = df.groupby("Cluster").size().rename("Customers")
    means = df.groupby("Cluster")[feature_cols].mean().round(2)
    summary = pd.concat([counts, means], axis=1).reset_index()
    return summary


def cluster_scatter(df: pd.DataFrame, x_col: str, y_col: str, labels: Dict[int, str]) -> go.Figure:
    """2D scatter plot coloured by cluster."""
    temp = df.copy()
    temp["Segment"] = temp["Cluster"].map(labels)
    fig = px.scatter(
        temp, x=x_col, y=y_col,
        color="Segment",
        title=f"Customer Segments: {x_col} vs {y_col}",
        opacity=0.7,
    )
    fig.update_layout(template="plotly_white")
    return fig


def cluster_distribution_chart(df: pd.DataFrame, labels: Dict[int, str]) -> go.Figure:
    """Pie chart of cluster sizes."""
    temp = df.copy()
    temp["Segment"] = temp["Cluster"].map(labels)
    counts = temp["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig = px.pie(counts, names="Segment", values="Count", title="Cluster Distribution", hole=0.4)
    fig.update_layout(template="plotly_white")
    return fig
