"""
data_loader.py
--------------
Handles loading CSV data, detecting churn columns, and basic validation.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional, List

# Common names that indicate a churn target column
CHURN_COLUMN_NAMES = [
    "Churn", "churn", "CHURN",
    "Exited", "exited",
    "Customer_Churn", "customer_churn",
    "Is_Churn", "is_churn",
    "Churned", "churned",
    "Attrition", "attrition",
]


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with basic error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")
    return df


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Load a Streamlit UploadedFile object into a DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")
        return df
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded file is empty or not a valid CSV.")
    except pd.errors.ParserError:
        raise ValueError("Could not parse the file. Please upload a valid CSV.")


def detect_churn_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect the churn target column by matching against known churn column names.

    Returns the original column name if found, or None.
    """
    for col in df.columns:
        if col.strip() in CHURN_COLUMN_NAMES:
            return col.strip()
    return None


def normalize_churn_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Find and rename the churn column to 'Churn'. Convert values to 0/1.

    Returns
    -------
    (df, has_churn) : Tuple[pd.DataFrame, bool]
        The modified DataFrame and whether a churn column was found.
    """
    churn_col = detect_churn_column(df)
    if churn_col is None:
        return df, False

    df = df.copy()
    if churn_col != "Churn":
        df.rename(columns={churn_col: "Churn"}, inplace=True)

    # Normalize values to 0/1
    unique_vals = df["Churn"].dropna().unique()
    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        df["Churn"] = df["Churn"].astype(int)
    elif set(str(v).lower() for v in unique_vals).issubset({"yes", "no"}):
        df["Churn"] = df["Churn"].map(lambda x: 1 if str(x).lower() == "yes" else 0)
    elif set(str(v).lower() for v in unique_vals).issubset({"true", "false"}):
        df["Churn"] = df["Churn"].map(lambda x: 1 if str(x).lower() == "true" else 0)
    else:
        # Attempt numeric conversion
        try:
            df["Churn"] = pd.to_numeric(df["Churn"]).astype(int)
        except (ValueError, TypeError):
            pass

    return df, True


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a summary dictionary of the dataset."""
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
    }


def get_sample_data_path() -> str:
    """Return the absolute path to the bundled sample dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "customers.csv")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic dataset cleaning:
    - Drop exact duplicate rows.
    - Strip whitespace from string columns.
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
    return df
