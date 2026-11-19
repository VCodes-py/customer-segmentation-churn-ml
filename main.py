"""
main.py - End-to-End Customer Segmentation & Churn Prediction Pipeline.

Orchestrates:
1. Data loading / generation (data/customer_transactions.csv)
2. RFM Feature Engineering & K-Means Clustering (outputs/elbow_curve.png, outputs/rfm_clusters.png)
3. Churn Classifier Training & Comparison (outputs/model_comparison.csv, outputs/roc_curve.png)
4. Model Explainability & Driver Extraction (outputs/top_churn_drivers.png)

Usage:
    python main.py
    python main.py --customers 3000 --transactions 18000 --seed 42 --n-clusters 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_dataset, save_dataset, generate_synthetic_data
from src.segmentation import perform_segmentation
from src.churn_model import run_churn_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-End Customer Segmentation & Churn Prediction ML Pipeline"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/customer_transactions.csv",
        help="Path to customer transaction CSV (default: data/customer_transactions.csv)",
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=2500,
        help="Number of customers if generating synthetic data (default: 2500)",
    )
    parser.add_argument(
        "--transactions",
        type=int,
        default=15000,
        help="Number of transactions if generating synthetic data (default: 15000)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of K-Means clusters for customer segmentation (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save plots and summary artifacts (default: outputs)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of dataset even if CSV exists",
    )
    return parser.parse_args()


def print_banner():
    banner = r"""
================================================================================
   CUSTOMER SEGMENTATION & CHURN PREDICTION MACHINE LEARNING PIPELINE
   RFM K-Means Segmentation  |  Ensemble Churn Modeling  |  Portfolio Ready
================================================================================
"""
    print(banner)


def main():
    start_time = time.time()
    args = parse_args()
    print_banner()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STEP 1: Data Ingestion / Generation
    # ---------------------------------------------------------
    print("[1/4] DATA INGESTION & SYNTHESIS")
    print(f"      Target CSV: {args.data_path}")

    if args.regenerate or not os.path.exists(args.data_path):
        print(f"      Generating fresh synthetic dataset ({args.customers:,} customers, {args.transactions:,} transactions)...")
        tx_df = generate_synthetic_data(
            n_customers=args.customers,
            n_transactions=args.transactions,
            random_state=args.seed,
        )
        save_dataset(tx_df, args.data_path)
    else:
        print("      Loading existing dataset from disk...")
        tx_df = load_dataset(args.data_path)

    n_cust = tx_df["customer_id"].nunique()
    n_tx = len(tx_df)
    churn_rate = tx_df.groupby("customer_id")["churned"].first().mean()
    print(f"      [OK] Loaded {n_tx:,} transactions across {n_cust:,} unique customers.")
    print(f"      [OK] Overall ground-truth customer churn rate: {churn_rate:.2%}\n")

    # ---------------------------------------------------------
    # STEP 2: RFM Feature Engineering & K-Means Clustering
    # ---------------------------------------------------------
    print("[2/4] RFM FEATURE ENGINEERING & K-MEANS SEGMENTATION")
    print(f"      Evaluating clusters & fitting K-Means (k={args.n_clusters})...")
    rfm_df, cluster_summary, kmeans, scaler = perform_segmentation(
        df=tx_df,
        n_clusters=args.n_clusters,
        use_log_transform=True,
        random_state=args.seed,
        output_dir=args.output_dir,
    )

    print("      [OK] Cluster profiling complete:")
    print("-" * 78)
    # Format and display summary table
    display_summary = cluster_summary.copy()
    for col in display_summary.columns:
        if "churn_rate" in col:
            display_summary[col] = display_summary[col].apply(lambda x: f"{x:.1%}")
        elif "monetary" in col:
            display_summary[col] = display_summary[col].apply(lambda x: f"${x:,.2f}")
        elif "recency" in col or "frequency" in col:
            display_summary[col] = display_summary[col].apply(lambda x: f"{x:.1f}")
    print(display_summary.to_string())
    print("-" * 78)
    print(f"      [Artifact] Saved elbow curve: {os.path.join(args.output_dir, 'elbow_curve.png')}")
    print(f"      [Artifact] Saved cluster plots: {os.path.join(args.output_dir, 'rfm_clusters.png')}")
    print(f"      [Artifact] Saved summary table: {os.path.join(args.output_dir, 'cluster_summary.csv')}\n")

    # ---------------------------------------------------------
    # STEP 3: Churn Classification & Imbalance Modeling
    # ---------------------------------------------------------
    print("[3/4] CHURN CLASSIFIER TRAINING & CROSS-VALIDATION")
    print("      Benchmarking: Logistic Regression, Random Forest, Gradient Boosting...")
    comparison_df, models, fi_df = run_churn_pipeline(
        rfm_df=rfm_df,
        output_dir=args.output_dir,
        random_state=args.seed,
    )

    print("      [OK] Model Benchmark Scorecard:")
    print("-" * 88)
    print(comparison_df.to_string(index=False))
    print("-" * 88)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_auc = comparison_df.iloc[0]["ROC-AUC"]
    best_f1 = comparison_df.iloc[0]["F1-Score"]
    print(f"      [Champion Model] {best_model_name} (ROC-AUC: {best_auc:.4f}, F1-Score: {best_f1:.4f})\n")

    # ---------------------------------------------------------
    # STEP 4: Churn Drivers & Explainability
    # ---------------------------------------------------------
    print("[4/4] EXPLAINABILITY & TOP CHURN DRIVERS")
    print("      Top 5 Most Influential Features Driving Churn:")
    for idx, row in fi_df.head(5).iterrows():
        print(f"        {idx+1}. {row['Feature']:<25} (Importance: {row['Importance']:.4f})")

    print(f"\n      [Artifact] Saved ROC Curve plot: {os.path.join(args.output_dir, 'roc_curve.png')}")
    print(f"      [Artifact] Saved Churn Drivers plot: {os.path.join(args.output_dir, 'top_churn_drivers.png')}")
    print(f"      [Artifact] Saved Feature Importances: {os.path.join(args.output_dir, 'feature_importances.csv')}")
    print(f"      [Artifact] Saved Model Comparison: {os.path.join(args.output_dir, 'model_comparison.csv')}\n")

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"[SUCCESS] Pipeline executed successfully in {elapsed:.2f} seconds!")
    print(f"All portfolio artifacts ready in directory: '{os.path.abspath(args.output_dir)}'")
    print("=" * 80)


if __name__ == "__main__":
    main()
