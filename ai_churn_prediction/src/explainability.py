"""
explainability.py
------------------
SHAP-based model explanations with a reliable fallback when SHAP
is incompatible with the model or crashes.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Tuple
import warnings


def compute_shap_values(
    model,
    X_test: np.ndarray,
    feature_names: List[str],
    max_samples: int = 200,
) -> Tuple[Optional[np.ndarray], Optional[Any], str]:
    """
    Compute SHAP values for the given model.

    Falls back to model feature importances if SHAP fails.

    Parameters
    ----------
    model : trained estimator
    X_test : preprocessed test data
    feature_names : feature names after preprocessing
    max_samples : subsample to avoid long compute times

    Returns
    -------
    (shap_values, explainer, method)
        shap_values may be None on failure.
        method ∈ {"shap", "feature_importances", "none"}
    """
    # Subsample for speed
    if X_test.shape[0] > max_samples:
        indices = np.random.RandomState(42).choice(X_test.shape[0], max_samples, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test

    # Attempt SHAP
    try:
        import shap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Use TreeExplainer for tree-based models
            model_type = type(model).__name__
            if model_type in ("RandomForestClassifier", "DecisionTreeClassifier",
                              "XGBClassifier", "GradientBoostingClassifier"):
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_sample)
                # For binary classification, some models return a list [class0, class1]
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # class 1 = churn
                elif hasattr(shap_vals, "values"):
                    shap_vals = shap_vals.values
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1]
                return shap_vals, explainer, "shap"
            else:
                # Use KernelExplainer for others (like LogisticRegression)
                # Use a small background sample
                bg = X_sample[:50]
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                shap_vals = explainer.shap_values(X_sample[:50])
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
                elif hasattr(shap_vals, "values"):
                    shap_vals = shap_vals.values
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    shap_vals = shap_vals[:, :, 1]
                return shap_vals, explainer, "shap"
    except Exception:
        pass

    # Fallback: feature importances
    return None, None, "feature_importances"


def get_global_importance(
    model,
    shap_values: Optional[np.ndarray],
    feature_names: List[str],
    method: str,
) -> pd.DataFrame:
    """
    Return a DataFrame of feature → importance, sorted descending.
    Uses SHAP mean absolute values if available, else model importances.
    """
    if method == "shap" and shap_values is not None:
        importance = np.abs(shap_values).mean(axis=0)
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame({"Feature": feature_names, "Importance": [0] * len(feature_names)})

    if len(importance) != len(feature_names):
        # length mismatch — return what we can
        n = min(len(importance), len(feature_names))
        importance = importance[:n]
        feature_names = feature_names[:n]

    df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


def plot_global_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    top = importance_df.head(top_n).sort_values("Importance")
    fig = go.Figure(go.Bar(
        x=top["Importance"],
        y=top["Feature"],
        orientation="h",
        marker_color="#3498db",
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, top_n * 30),
    )
    return fig


def get_individual_explanation(
    model,
    preprocessor,
    customer_data: dict,
    feature_names: List[str],
    numerical_cols: List[str],
    categorical_cols: List[str],
    X_background: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Explain a single prediction — returns top positive/negative drivers.
    """
    input_df = pd.DataFrame([customer_data])
    all_cols = numerical_cols + categorical_cols
    for col in all_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df = input_df[all_cols]

    X_processed = preprocessor.transform(input_df)

    # Try SHAP
    try:
        import shap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model_type = type(model).__name__
            if model_type in ("RandomForestClassifier", "DecisionTreeClassifier",
                              "XGBClassifier", "GradientBoostingClassifier"):
                explainer = shap.TreeExplainer(model)
            elif X_background is not None:
                explainer = shap.KernelExplainer(model.predict_proba, X_background[:50])
            else:
                raise RuntimeError("No background data")

            sv = explainer.shap_values(X_processed)
            if isinstance(sv, list):
                sv = sv[1]
            elif hasattr(sv, "values"):
                sv = sv.values
            if isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
            shap_row = sv[0]

            drivers = []
            sorted_idx = np.argsort(np.abs(shap_row))[::-1]
            for i in sorted_idx[:8]:
                drivers.append({
                    "feature": feature_names[i],
                    "shap_value": round(float(shap_row[i]), 4),
                    "direction": "increases churn" if shap_row[i] > 0 else "decreases churn",
                })
            return {"method": "shap", "drivers": drivers}

    except Exception:
        pass

    # Fallback — use feature importances as proxy
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is not None and len(importances) == len(feature_names):
        weighted = importances * np.abs(X_processed[0])
        sorted_idx = np.argsort(weighted)[::-1]
        drivers = []
        for i in sorted_idx[:8]:
            drivers.append({
                "feature": feature_names[i],
                "shap_value": round(float(weighted[i]), 4),
                "direction": "contributes to prediction",
            })
        return {"method": "feature_importances", "drivers": drivers}

    return {"method": "none", "drivers": []}
