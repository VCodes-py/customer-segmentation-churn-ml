"""
data_loader.py - Synthetic e-commerce transaction & customer dataset generator.

Generates realistic customer purchase histories and customer-level attributes
with realistic statistical distributions and behavioral correlations:
- Support ticket frequency correlated with churn risk
- Contract type (Month-to-month vs 1-year / 2-year) driving churn tendencies
- Purchase recency and monetary spend influencing retention
- Reproducible via random_state for portfolio and testing reliability
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_customers: int = 2500,
    n_transactions: int = 15000,
    random_state: int = 42,
    end_date: str = "2026-11-01",
) -> pd.DataFrame:
    """
    Generate synthetic e-commerce transaction and customer dataset.

    Parameters
    ----------
    n_customers : int
        Number of unique customers to generate.
    n_transactions : int
        Total number of purchase transactions across all customers (must be >= n_customers).
    random_state : int
        Random seed for reproducibility.
    end_date : str
        Snapshot date (YYYY-MM-DD) representing 'today' for calculating recency.

    Returns
    -------
    pd.DataFrame
        DataFrame containing transaction records with customer demographics,
        behavioral metrics, and churn status.
    """
    if n_transactions < n_customers:
        raise ValueError("n_transactions must be at least equal to n_customers.")

    np.random.seed(random_state)
    ref_date = datetime.strptime(end_date, "%Y-%m-%d")

    # 1. Customer Demographics and Behavioral Attributes
    customer_ids = [f"CUST_{i+1:05d}" for i in range(n_customers)]

    # Contract type distribution: 55% month-to-month, 25% one-year, 20% two-year
    contract_choices = ["Month-to-month", "One-year", "Two-year"]
    contract_probs = [0.55, 0.25, 0.20]
    contract_types = np.random.choice(contract_choices, size=n_customers, p=contract_probs)

    # Customer tenure in months: 1 to 72 months
    tenures = np.clip(np.random.exponential(scale=20, size=n_customers).astype(int) + 1, 1, 72)

    # Support tickets: Poisson distributed, higher for month-to-month contracts
    support_base = np.where(contract_types == "Month-to-month", 2.3, 1.1)
    support_tickets = np.random.poisson(lam=support_base, size=n_customers)

    # Activity level / transaction weighting per customer (Gamma distribution)
    customer_weights = np.random.gamma(shape=2.0, scale=1.5, size=n_customers)
    customer_weights /= customer_weights.sum()

    # 2. Transaction Generation (ensuring each customer has at least 1 transaction)
    base_indices = np.arange(n_customers)
    remaining_count = n_transactions - n_customers
    extra_indices = np.random.choice(
        n_customers, size=remaining_count, p=customer_weights, replace=True
    )
    all_assigned = np.concatenate([base_indices, extra_indices])
    np.random.shuffle(all_assigned)

    tx_cust_ids = [customer_ids[i] for i in all_assigned]
    tx_tenures = tenures[all_assigned]
    tx_contracts = contract_types[all_assigned]
    tx_tickets = support_tickets[all_assigned]

    # Purchase dates: random day within customer's active tenure (up to 720 days back)
    days_back = np.random.uniform(1, np.minimum(tx_tenures * 30, 720))
    invoice_dates = [ref_date - timedelta(days=float(d)) for d in days_back]

    # Order amounts: Lognormal distribution (average order ~$75 with long tail)
    order_amounts = np.round(np.random.lognormal(mean=3.9, sigma=0.65, size=n_transactions), 2)
    order_amounts = np.clip(order_amounts, 9.99, 1200.0)

    df_tx = pd.DataFrame(
        {
            "customer_id": tx_cust_ids,
            "invoice_date": invoice_dates,
            "order_amount": order_amounts,
            "tenure_months": tx_tenures,
            "support_tickets": tx_tickets,
            "contract_type": tx_contracts,
        }
    )

    # 3. Derive Ground Truth Churn Label per Customer
    last_tx_per_cust = df_tx.groupby("customer_id")["invoice_date"].max()
    cust_recency = {
        cid: (ref_date - max_date).days for cid, max_date in last_tx_per_cust.items()
    }
    cust_spend = df_tx.groupby("customer_id")["order_amount"].sum().to_dict()
    cust_freq = df_tx.groupby("customer_id")["order_amount"].count().to_dict()

    churn_labels = {}
    for i, cid in enumerate(customer_ids):
        c_type = contract_types[i]
        c_tenure = tenures[i]
        c_tickets = support_tickets[i]
        c_rec = cust_recency.get(cid, 180)
        c_freq = cust_freq.get(cid, 1)

        # Base log-odds calibrated for ~22-26% churn rate (realistic industry baseline)
        log_odds = -1.25

        # Contract effect: Month-to-month contracts have substantially higher churn risk
        if c_type == "Month-to-month":
            log_odds += 1.35
        elif c_type == "One-year":
            log_odds -= 0.60
        else:  # Two-year
            log_odds -= 1.65

        # Support ticket effect: frequent complaints indicate dissatisfaction
        log_odds += 0.35 * c_tickets

        # Inactivity effect: higher recency = haven't purchased in a long time
        if c_rec > 120:
            log_odds += 1.40
        elif c_rec > 60:
            log_odds += 0.55
        else:
            log_odds -= 0.85

        # Tenure effect: longer tenure builds brand loyalty
        log_odds -= 0.032 * c_tenure

        # Order frequency effect: high engagement reduces churn
        log_odds -= 0.075 * min(c_freq, 20)

        # Add realistic unexplained variance (noise)
        noise = np.random.normal(loc=0.0, scale=0.40)
        prob = 1.0 / (1.0 + np.exp(-(log_odds + noise)))

        churn_labels[cid] = int(prob > 0.50)

    # Map churn label back to transaction records
    df_tx["churned"] = df_tx["customer_id"].map(churn_labels)

    # Sort chronologically by invoice_date
    df_tx = df_tx.sort_values(by="invoice_date").reset_index(drop=True)
    df_tx["invoice_date"] = df_tx["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df_tx


def save_dataset(df: pd.DataFrame, filepath: str = "data/customer_transactions.csv") -> str:
    """Save dataset to CSV file, ensuring target directories exist."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


def load_dataset(
    filepath: str = "data/customer_transactions.csv",
    fallback_generate: bool = True,
    **gen_kwargs,
) -> pd.DataFrame:
    """
    Load dataset from CSV if available, or generate synthetic dataset on the fly.
    """
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        return df
    elif fallback_generate:
        df = generate_synthetic_data(**gen_kwargs)
        save_dataset(df, filepath)
        df["invoice_date"] = pd.to_datetime(df["invoice_date"])
        return df
    else:
        raise FileNotFoundError(f"Dataset not found at {filepath}")


if __name__ == "__main__":
    print("Generating synthetic customer transactions...")
    df = generate_synthetic_data()
    saved_path = save_dataset(df)
    unique_custs = df["customer_id"].nunique()
    churn_rate = df.groupby("customer_id")["churned"].first().mean()
    print(f"Generated {len(df):,} transactions for {unique_custs:,} unique customers.")
    print(f"Overall customer churn rate: {churn_rate:.2%}")
    print(f"Saved to: {saved_path}")
