"""
prediction.py
--------------
Single-customer churn prediction, risk classification, and
top contributing factor identification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


# ─────────── RISK THRESHOLDS ─────────── #

def classify_risk(probability: float) -> str:
    """
    Map a churn probability to a human-readable risk level.

    0–30 %  → LOW
    30–60 % → MEDIUM
    60–100% → HIGH
    """
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    else:
        return "HIGH"


def risk_color(level: str) -> str:
    """Return a CSS colour for each risk level."""
    return {"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}.get(level, "gray")


# ─────────── SINGLE PREDICTION ─────────── #

def predict_single_customer(
    model,
    preprocessor,
    customer_data: dict,
    feature_names: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Predict churn for one customer.

    Parameters
    ----------
    model : trained estimator
    preprocessor : fitted ColumnTransformer
    customer_data : dict with feature values
    feature_names : names after preprocessing (for importance)
    numerical_cols : list of numerical column names used during training
    categorical_cols : list of categorical column names used during training

    Returns
    -------
    dict with probability, predicted_class, risk_level, risk_color, top_factors
    """
    # Build DataFrame from dict
    input_df = pd.DataFrame([customer_data])

    # Apply feature engineering to derive any engineered features automatically
    from src.feature_engineering import engineer_features
    input_df = engineer_features(input_df)

    # Ensure column order matches training
    if numerical_cols and categorical_cols:
        all_cols = numerical_cols + categorical_cols
        for col in all_cols:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[all_cols]

    # Transform
    X_processed = preprocessor.transform(input_df)

    # Predict
    proba = model.predict_proba(X_processed)[0][1]
    pred_class = int(proba >= 0.5)
    risk = classify_risk(proba)

    # Top contributing factors (from model feature importances)
    top_factors = _get_top_factors(model, X_processed, feature_names)

    return {
        "probability": round(float(proba), 4),
        "probability_pct": f"{proba * 100:.1f}%",
        "predicted_class": pred_class,
        "predicted_label": "Churned" if pred_class else "Active",
        "risk_level": risk,
        "risk_color": risk_color(risk),
        "top_factors": top_factors,
    }


def _get_top_factors(
    model, X_processed: np.ndarray, feature_names: Optional[List[str]], top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Extract the features with the highest absolute contribution for this
    prediction, using model feature importances as a proxy.
    """
    if feature_names is None:
        return []

    # Try tree-based feature_importances_ first
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None or len(importances) != len(feature_names):
        return []

    # Weight importances by feature values to get contribution direction
    weighted = importances * np.abs(X_processed[0])
    indices = np.argsort(weighted)[::-1][:top_n]

    factors = []
    for i in indices:
        factors.append({
            "feature": feature_names[i],
            "importance": round(float(importances[i]), 4),
        })
    return factors


# ─────────── BATCH PREDICTION ─────────── #

def predict_batch(
    model,
    preprocessor,
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """
    Predict churn probability for all rows in a DataFrame.
    Adds columns: ChurnProbability, PredictedChurn, RiskLevel.
    """
    all_cols = numerical_cols + categorical_cols
    X = df[all_cols].copy()
    X_processed = preprocessor.transform(X)

    probas = model.predict_proba(X_processed)[:, 1]

    result = df.copy()
    result["ChurnProbability"] = np.round(probas, 4)
    result["PredictedChurn"] = (probas >= 0.5).astype(int)
    result["RiskLevel"] = result["ChurnProbability"].apply(classify_risk)
    return result
