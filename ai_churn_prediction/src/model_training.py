"""
model_training.py
------------------
Trains multiple classifiers, evaluates them, selects the best model,
and persists the artefacts.

Key design choices
------------------
* class_weight="balanced" is used where supported, so the model
  pays attention to the minority (churned) class.
* The preprocessing pipeline is fitted ONLY on the training split.
* Model selection uses ROC-AUC as the primary metric, with F1 as a
  tiebreaker.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.preprocessing import (
    build_preprocessing_pipeline,
    identify_feature_columns,
    get_feature_names_from_pipeline,
)


# ─────────────── MODEL DEFINITIONS ─────────────── #

def get_models() -> Dict[str, Any]:
    """Return a dict of model name → untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=3,  # approximate class imbalance correction
            random_state=42,
            eval_metric="logloss",
        ),
    }


# ─────────────── TRAINING PIPELINE ─────────────── #

def train_and_evaluate(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    End-to-end training pipeline.

    Steps
    -----
    1. Separate features / target.
    2. Train/test split (stratified).
    3. Build & fit preprocessing pipeline on train data only.
    4. Train all models.
    5. Evaluate on the test set.
    6. Select the best model (ROC-AUC primary, F1 tiebreaker).
    7. Save artefacts to disk.

    Returns
    -------
    dict with keys:
        results       — per-model metrics dict
        best_model    — (name, estimator)
        preprocessor  — fitted ColumnTransformer
        feature_names — list of feature names after transformation
        X_test, y_test, X_test_raw — for downstream use
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    numerical_cols, categorical_cols = identify_feature_columns(df, target=target)

    X = df[numerical_cols + categorical_cols]
    y = df[target]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )

    # Build & fit preprocessing on TRAINING data only
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = get_feature_names_from_pipeline(
        preprocessor, numerical_cols, categorical_cols,
    )

    # Train & evaluate each model
    models = get_models()
    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        model.fit(X_train_processed, y_train)

        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "roc_curve": roc_curve(y_test, y_proba),
            "pr_curve": precision_recall_curve(y_test, y_proba),
            "model": model,
        }
        results[name] = metrics

    # Select best model (primary: ROC-AUC, tiebreaker: F1)
    best_name = max(
        results,
        key=lambda n: (results[n]["roc_auc"], results[n]["f1"]),
    )
    best_model = results[best_name]["model"]

    # Save artefacts
    _save_artefacts(best_model, preprocessor, best_name)

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": best_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "X_test": X_test_processed,
        "X_test_raw": X_test,
        "y_test": y_test,
    }


# ─────────────── PERSISTENCE ─────────────── #

def _save_artefacts(model, preprocessor, model_name: str) -> None:
    """Save model and pipeline to disk."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model, os.path.join(models_dir, "best_model.pkl"))
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessing_pipeline.pkl"))
    # Save model name
    with open(os.path.join(models_dir, "model_name.txt"), "w") as f:
        f.write(model_name)


def load_model():
    """Load saved model and preprocessor."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    model_path = os.path.join(models_dir, "best_model.pkl")
    pipeline_path = os.path.join(models_dir, "preprocessing_pipeline.pkl")
    name_path = os.path.join(models_dir, "model_name.txt")

    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        return None, None, None

    model = joblib.load(model_path)
    preprocessor = joblib.load(pipeline_path)
    name = ""
    if os.path.exists(name_path):
        with open(name_path) as f:
            name = f.read().strip()
    return model, preprocessor, name


def build_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Build a tidy comparison DataFrame from the results dict."""
    rows = []
    for name, m in results.items():
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1"],
            "ROC-AUC": m["roc_auc"],
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
