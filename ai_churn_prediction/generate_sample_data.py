"""
generate_sample_data.py
-----------------------
Generates a realistic synthetic customer dataset with 5,000 rows.
Churn labels are NOT random — they are driven by realistic business logic
so that the ML model can learn meaningful patterns.
"""

import pandas as pd
import numpy as np
import os


def generate_customer_dataset(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic customer dataset with realistic churn patterns.

    The churn probability for each customer is computed from a logistic function
    of several business-meaningful features so that the downstream ML model
    can discover interpretable relationships.

    Parameters
    ----------
    n_customers : int
        Number of customer rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with customer attributes and a binary Churn column.
    """
    rng = np.random.RandomState(seed)

    # ---- CustomerID ----
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]

    # ---- Demographics ----
    ages = rng.randint(18, 72, size=n_customers)
    genders = rng.choice(["Male", "Female"], size=n_customers)

    # ---- Account features ----
    tenures = rng.exponential(scale=24, size=n_customers).clip(1, 72).astype(int)

    contracts = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_customers,
        p=[0.50, 0.30, 0.20],
    )

    payment_methods = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_customers,
        p=[0.35, 0.20, 0.25, 0.20],
    )

    internet_services = rng.choice(
        ["Fiber optic", "DSL", "No"],
        size=n_customers,
        p=[0.45, 0.35, 0.20],
    )

    # ---- Usage / Behaviour ----
    # Monthly charges depend on internet service and contract
    base_charge = np.where(
        internet_services == "Fiber optic",
        rng.normal(85, 15, n_customers),
        np.where(
            internet_services == "DSL",
            rng.normal(55, 12, n_customers),
            rng.normal(25, 8, n_customers),
        ),
    ).clip(18, 120).round(2)

    monthly_charges = base_charge.astype(float)
    total_charges = (monthly_charges * tenures + rng.normal(0, 50, n_customers)).clip(0).round(2)

    support_calls = rng.poisson(lam=2, size=n_customers).clip(0, 15)
    num_products = rng.choice([1, 2, 3, 4, 5], size=n_customers, p=[0.25, 0.30, 0.25, 0.15, 0.05])
    usage_frequency = rng.randint(1, 31, size=n_customers)  # days per month
    satisfaction_scores = rng.choice([1, 2, 3, 4, 5], size=n_customers, p=[0.10, 0.15, 0.30, 0.25, 0.20])

    # ---- Compute realistic churn probability ----
    # Higher churn when: month-to-month, short tenure, high charges, many support
    # calls, low satisfaction, electronic check, fiber optic
    churn_score = np.zeros(n_customers, dtype=float)

    # Contract effect — strongest driver
    churn_score += np.where(contracts == "Month-to-month", 1.5, 0.0)
    churn_score += np.where(contracts == "One year", 0.3, 0.0)
    churn_score += np.where(contracts == "Two year", -0.8, 0.0)

    # Tenure — new customers churn more
    churn_score += np.where(tenures < 6, 1.0, 0.0)
    churn_score += np.where(tenures < 12, 0.4, 0.0)
    churn_score += np.where(tenures >= 36, -0.6, 0.0)

    # Monthly charges — higher charges → more churn
    churn_score += (monthly_charges - monthly_charges.mean()) / monthly_charges.std() * 0.5

    # Support calls
    churn_score += (support_calls - 2) * 0.25

    # Satisfaction — low satisfaction drives churn
    churn_score += (3 - satisfaction_scores) * 0.4

    # Payment method
    churn_score += np.where(payment_methods == "Electronic check", 0.5, 0.0)

    # Internet service
    churn_score += np.where(internet_services == "Fiber optic", 0.3, 0.0)

    # Add noise
    churn_score += rng.normal(0, 0.8, n_customers)

    # Convert to probability via sigmoid
    churn_prob = 1 / (1 + np.exp(-churn_score))

    # Sample churn labels
    churn = (rng.random(n_customers) < churn_prob).astype(int)

    # ---- Build DataFrame ----
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": ages,
        "Gender": genders,
        "Tenure": tenures,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contracts,
        "PaymentMethod": payment_methods,
        "InternetService": internet_services,
        "SupportCalls": support_calls,
        "NumProducts": num_products,
        "UsageFrequency": usage_frequency,
        "SatisfactionScore": satisfaction_scores,
        "Churn": churn,
    })

    # Introduce a small number of missing values to make it realistic
    missing_indices = rng.choice(n_customers, size=int(n_customers * 0.02), replace=False)
    df.loc[missing_indices[:25], "TotalCharges"] = np.nan
    df.loc[missing_indices[25:50], "SatisfactionScore"] = np.nan
    df.loc[missing_indices[50:75], "SupportCalls"] = np.nan

    return df


if __name__ == "__main__":
    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    df = generate_customer_dataset()
    output_path = os.path.join(data_dir, "customers.csv")
    df.to_csv(output_path, index=False)

    print(f"Dataset generated: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    print(f"\nColumn dtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
