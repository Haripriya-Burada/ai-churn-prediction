"""
feature_engineering.py
-----------------------
Creates new features from existing columns when the required source columns
are present. Each engineered feature is explained in its docstring / comments.
"""

import pandas as pd
import numpy as np
from typing import List


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all applicable feature engineering transformations.
    Only creates features when the required source columns exist.

    Parameters
    ----------
    df : pd.DataFrame
        The raw (or partially cleaned) customer DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional engineered columns.
    """
    df = df.copy()

    df = _add_tenure_group(df)
    df = _add_avg_monthly_spending(df)
    df = _add_monthly_to_total_ratio(df)
    df = _add_engagement_score(df)
    df = _add_support_intensity(df)

    return df


# ---------- Individual Feature Functions ---------- #

def _add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tenure Group — buckets tenure into business-meaningful segments.
    Short-tenure customers tend to churn more frequently.
    """
    if "Tenure" not in df.columns:
        return df
    bins = [0, 6, 12, 24, 48, np.inf]
    labels = ["0-6 mo", "6-12 mo", "1-2 yr", "2-4 yr", "4+ yr"]
    df["TenureGroup"] = pd.cut(df["Tenure"], bins=bins, labels=labels, right=True)
    return df


def _add_avg_monthly_spending(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average Monthly Spending = TotalCharges / Tenure.
    A divergence between this and MonthlyCharges may indicate recent plan
    changes — a potential churn signal.
    """
    if "TotalCharges" in df.columns and "Tenure" in df.columns:
        df["AvgMonthlySpending"] = (
            df["TotalCharges"] / df["Tenure"].replace(0, np.nan)
        ).round(2)
    return df


def _add_monthly_to_total_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    MonthlyToTotalRatio = MonthlyCharges / TotalCharges.
    A high ratio means the customer is relatively new or spending has
    increased recently.
    """
    if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
        df["MonthlyToTotalRatio"] = (
            df["MonthlyCharges"] / df["TotalCharges"].replace(0, np.nan)
        ).round(4)
    return df


def _add_engagement_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engagement Score — a composite metric from usage, products, and
    satisfaction (if available). Higher is better.
    Each component is min-max normalised to [0, 1] and then averaged.
    """
    components: List[pd.Series] = []

    if "UsageFrequency" in df.columns:
        s = df["UsageFrequency"]
        components.append((s - s.min()) / (s.max() - s.min() + 1e-9))

    if "NumProducts" in df.columns:
        s = df["NumProducts"]
        components.append((s - s.min()) / (s.max() - s.min() + 1e-9))

    if "SatisfactionScore" in df.columns:
        s = df["SatisfactionScore"]
        components.append((s - s.min()) / (s.max() - s.min() + 1e-9))

    if components:
        df["EngagementScore"] = (
            pd.concat(components, axis=1).mean(axis=1).round(3)
        )
    return df


def _add_support_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Support Intensity = SupportCalls / Tenure.
    A high value means the customer contacts support very frequently
    relative to how long they have been with the company.
    """
    if "SupportCalls" in df.columns and "Tenure" in df.columns:
        df["SupportIntensity"] = (
            df["SupportCalls"] / df["Tenure"].replace(0, np.nan)
        ).round(4)
    return df


def get_engineered_feature_descriptions() -> dict:
    """Return human-readable descriptions of all engineered features."""
    return {
        "TenureGroup": "Customer tenure bucketed into segments (0-6 mo, 6-12 mo, 1-2 yr, 2-4 yr, 4+ yr).",
        "AvgMonthlySpending": "Average monthly spending = TotalCharges / Tenure.",
        "MonthlyToTotalRatio": "Ratio of current monthly charges to total charges.",
        "EngagementScore": "Composite score (0-1) from usage, products, and satisfaction.",
        "SupportIntensity": "Support calls per month of tenure.",
    }
