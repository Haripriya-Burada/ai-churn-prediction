"""
preprocessing.py
-----------------
Robust preprocessing pipeline using scikit-learn Pipeline / ColumnTransformer.
The pipeline is fitted ONLY on training data to avoid data leakage.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def identify_feature_columns(
    df: pd.DataFrame,
    target: str = "Churn",
    id_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Separate numerical and categorical feature columns, excluding the target
    and ID columns.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
        Name of the target column.
    id_cols : list[str] or None
        Columns to exclude (e.g. CustomerID).

    Returns
    -------
    (numerical_cols, categorical_cols)
    """
    if id_cols is None:
        # Auto-detect ID-like columns
        id_cols = [c for c in df.columns if "id" in c.lower() and df[c].nunique() == len(df)]

    exclude = set(id_cols) | {target}
    feature_cols = [c for c in df.columns if c not in exclude]

    numerical_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    return numerical_cols, categorical_cols


def build_preprocessing_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Imputes missing numerics with the median, then scales.
    - Imputes missing categoricals with the most-frequent value, then one-hot encodes.

    Parameters
    ----------
    numerical_cols : list[str]
    categorical_cols : list[str]

    Returns
    -------
    ColumnTransformer
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",  # drop ID / target / other columns
    )
    return preprocessor


def get_feature_names_from_pipeline(
    preprocessor: ColumnTransformer,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """
    Extract readable feature names after fitting the ColumnTransformer.
    """
    try:
        raw_names = list(preprocessor.get_feature_names_out())
        clean_names = []
        for name in raw_names:
            if name.startswith("num__"):
                clean_names.append(name[5:])
            elif name.startswith("cat__"):
                clean_names.append(name[5:])
            else:
                clean_names.append(name)
        return clean_names
    except Exception:
        feature_names = list(numerical_cols)
        if categorical_cols:
            try:
                cat_pipeline = preprocessor.named_transformers_["cat"]
                ohe = cat_pipeline.named_steps["encoder"]
                cat_features = list(ohe.get_feature_names_out(categorical_cols))
                feature_names.extend(cat_features)
            except Exception:
                pass
        return feature_names


def prepare_data(
    df: pd.DataFrame,
    target: str = "Churn",
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Separate features and target, identify column types.

    Returns
    -------
    (X, y, numerical_cols, categorical_cols)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    numerical_cols, categorical_cols = identify_feature_columns(df, target=target)

    X = df[numerical_cols + categorical_cols].copy()
    y = df[target].copy()

    return X, y, numerical_cols, categorical_cols
