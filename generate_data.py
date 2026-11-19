"""
generate_data.py - Standalone CLI script to generate reproducible synthetic dataset.

Usage:
    python generate_data.py --customers 2500 --transactions 15000 --seed 42 --output data/customer_transactions.csv
"""

from __future__ import annotations

import argparse
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import generate_synthetic_data, save_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic e-commerce customer transaction dataset."
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=2500,
        help="Number of unique customer records (default: 2500)",
    )
    parser.add_argument(
        "--transactions",
        type=int,
        default=15000,
        help="Total number of transaction rows (default: 15000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/customer_transactions.csv",
        help="Destination path for generated CSV (default: data/customer_transactions.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("  CUSTOMER DATASET GENERATOR")
    print("=" * 60)
    print(f"  Target Customers    : {args.customers:,}")
    print(f"  Target Transactions : {args.transactions:,}")
    print(f"  Random Seed         : {args.seed}")
    print(f"  Output Destination  : {args.output}")
    print("-" * 60)

    df = generate_synthetic_data(
        n_customers=args.customers,
        n_transactions=args.transactions,
        random_state=args.seed,
    )
    out_path = save_dataset(df, args.output)

    unique_custs = df["customer_id"].nunique()
    churn_rate = df.groupby("customer_id")["churned"].first().mean()
    total_revenue = df["order_amount"].sum()

    print("[SUCCESS] Dataset generated and saved successfully!")
    print(f"  Total Transactions : {len(df):,}")
    print(f"  Unique Customers   : {unique_custs:,}")
    print(f"  Customer Churn Rate: {churn_rate:.2%}")
    print(f"  Gross Revenue      : ${total_revenue:,.2f}")
    print(f"  File Path          : {os.path.abspath(out_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
