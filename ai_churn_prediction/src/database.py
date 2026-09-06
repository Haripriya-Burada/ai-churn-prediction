"""
database.py
------------
SQLite database layer for persisting customers, predictions,
segments, and model metrics.
"""

import os
import sqlite3
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database")
DB_PATH = os.path.join(DB_DIR, "customer_intelligence.db")


def _get_connection() -> sqlite3.Connection:
    """Get a SQLite connection, creating the DB dir if needed."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create all required tables if they do not exist."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            age INTEGER,
            gender TEXT,
            tenure INTEGER,
            monthly_charges REAL,
            total_charges REAL,
            contract TEXT,
            payment_method TEXT,
            internet_service TEXT,
            support_calls INTEGER,
            num_products INTEGER,
            usage_frequency INTEGER,
            satisfaction_score INTEGER,
            churn INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            churn_probability REAL,
            predicted_churn INTEGER,
            risk_level TEXT,
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            cluster_id INTEGER,
            segment_label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            accuracy REAL,
            precision_score REAL,
            recall REAL,
            f1_score REAL,
            roc_auc REAL,
            is_best INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ─────────── INSERT OPERATIONS ─────────── #

def save_customers(df: pd.DataFrame) -> int:
    """Save customer data to the database. Returns rows inserted."""
    conn = _get_connection()

    col_map = {
        "CustomerID": "customer_id",
        "Age": "age",
        "Gender": "gender",
        "Tenure": "tenure",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "Contract": "contract",
        "PaymentMethod": "payment_method",
        "InternetService": "internet_service",
        "SupportCalls": "support_calls",
        "NumProducts": "num_products",
        "UsageFrequency": "usage_frequency",
        "SatisfactionScore": "satisfaction_score",
        "Churn": "churn",
    }

    db_cols = [v for k, v in col_map.items() if k in df.columns]
    df_cols = [k for k in col_map.keys() if k in df.columns]

    subset = df[df_cols].copy()
    subset.columns = db_cols

    subset.to_sql("customers", conn, if_exists="replace", index=False)
    rows = len(subset)
    conn.close()
    return rows


def save_predictions(predictions: List[Dict[str, Any]], model_name: str = "") -> int:
    """Save prediction records."""
    conn = _get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    for p in predictions:
        cursor.execute(
            """INSERT INTO predictions
               (customer_id, churn_probability, predicted_churn, risk_level, model_name, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                p.get("customer_id", ""),
                p.get("churn_probability", 0),
                p.get("predicted_churn", 0),
                p.get("risk_level", ""),
                model_name,
                now,
            ),
        )

    conn.commit()
    rows = len(predictions)
    conn.close()
    return rows


def save_model_metrics(metrics: Dict[str, Dict], best_model_name: str) -> None:
    """Save evaluation metrics for all models."""
    conn = _get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    # Clear previous metrics
    cursor.execute("DELETE FROM model_metrics")

    for name, m in metrics.items():
        cursor.execute(
            """INSERT INTO model_metrics
               (model_name, accuracy, precision_score, recall, f1_score, roc_auc, is_best, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                m.get("accuracy", 0),
                m.get("precision", 0),
                m.get("recall", 0),
                m.get("f1", 0),
                m.get("roc_auc", 0),
                1 if name == best_model_name else 0,
                now,
            ),
        )

    conn.commit()
    conn.close()


def save_segments(df: pd.DataFrame, labels: Dict[int, str]) -> None:
    """Save segmentation results."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM segments")

    for _, row in df.iterrows():
        cid = row.get("CustomerID", "")
        cluster = int(row.get("Cluster", 0))
        label = labels.get(cluster, f"Segment {cluster}")
        cursor.execute(
            "INSERT INTO segments (customer_id, cluster_id, segment_label) VALUES (?, ?, ?)",
            (str(cid), cluster, label),
        )

    conn.commit()
    conn.close()


# ─────────── QUERY OPERATIONS ─────────── #

def get_prediction_history(limit: int = 100) -> pd.DataFrame:
    """Retrieve recent predictions."""
    conn = _get_connection()
    df = pd.read_sql_query(
        f"SELECT * FROM predictions ORDER BY created_at DESC LIMIT {limit}", conn,
    )
    conn.close()
    return df


def get_model_metrics() -> pd.DataFrame:
    """Retrieve saved model metrics."""
    conn = _get_connection()
    df = pd.read_sql_query("SELECT * FROM model_metrics ORDER BY roc_auc DESC", conn)
    conn.close()
    return df
