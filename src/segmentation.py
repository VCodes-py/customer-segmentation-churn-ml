"""
segmentation.py - RFM Feature Engineering, K-Means Clustering & Customer Profiling.

Computes Recency, Frequency, and Monetary (RFM) metrics from raw transactions,
evaluates optimal cluster counts using the Elbow Method and Silhouette Analysis,
fits K-Means segmentation, maps clusters to human-readable business personas,
and exports publication-grade visual diagnostics and summary tables.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def compute_rfm_features(
    df: pd.DataFrame,
    reference_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary (RFM) features per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction records containing customer_id, invoice_date, and order_amount.
    reference_date : Optional[str]
        Snapshot date for recency calculation. If None, max(invoice_date) + 1 day is used.

    Returns
    -------
    pd.DataFrame
        Customer-level dataframe with RFM features and behavioral attributes.
    """
    tx_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(tx_df["invoice_date"]):
        tx_df["invoice_date"] = pd.to_datetime(tx_df["invoice_date"])

    if reference_date is None:
        snap_date = tx_df["invoice_date"].max() + pd.Timedelta(days=1)
    else:
        snap_date = pd.to_datetime(reference_date)

    # Aggregate transactions per customer
    rfm = tx_df.groupby("customer_id").agg(
        recency=("invoice_date", lambda x: (snap_date - x.max()).days),
        frequency=("invoice_date", "count"),
        monetary=("order_amount", "sum"),
        avg_order_value=("order_amount", "mean"),
    ).reset_index()

    # If customer metadata is present in transactions, carry it forward
    meta_cols = ["tenure_months", "support_tickets", "contract_type", "churned"]
    existing_meta = [c for c in meta_cols if c in tx_df.columns]
    if existing_meta:
        meta_df = tx_df.groupby("customer_id")[existing_meta].first().reset_index()
        rfm = pd.merge(rfm, meta_df, on="customer_id", how="left")

    return rfm


def evaluate_clusters(
    rfm_scaled: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42,
    output_path: str = "outputs/elbow_curve.png",
) -> Tuple[List[float], List[float], int]:
    """
    Compute Elbow (Inertia) and Silhouette scores across candidate k values,
    and save a diagnostic dual-axis plot.

    Parameters
    ----------
    rfm_scaled : np.ndarray
        Standardized RFM feature array.
    k_range : range
        Candidate range of clusters k to evaluate (e.g. 2 to 8).
    random_state : int
        Seed for reproducibility.
    output_path : str
        File path to save the generated diagnostic curve.

    Returns
    -------
    Tuple[List[float], List[float], int]
        (inertias, silhouette_scores, recommended_k)
    """
    inertias: List[float] = []
    sil_scores: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        km.fit(rfm_scaled)
        inertias.append(km.inertia_)
        score = silhouette_score(rfm_scaled, km.labels_)
        sil_scores.append(score)

    # Pick recommended k with highest silhouette score (or elbow heuristic)
    best_k_idx = int(np.argmax(sil_scores))
    recommended_k = list(k_range)[best_k_idx]

    # Plot Elbow & Silhouette curves
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    color_elbow = "#1f77b4"
    ax1.set_xlabel("Number of Clusters (k)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", color=color_elbow, fontsize=12, fontweight="bold")
    l1 = ax1.plot(list(k_range), inertias, marker="o", color=color_elbow, linewidth=2, label="Inertia (Elbow)")
    ax1.tick_params(axis="y", labelcolor=color_elbow)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color_sil = "#2ca02c"
    ax2.set_ylabel("Silhouette Score", color=color_sil, fontsize=12, fontweight="bold")
    l2 = ax2.plot(list(k_range), sil_scores, marker="s", color=color_sil, linewidth=2, linestyle="--", label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color_sil)

    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)

    plt.title("Cluster Evaluation: Elbow Method & Silhouette Analysis", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return inertias, sil_scores, recommended_k


def assign_business_personas(
    rfm_df: pd.DataFrame,
    cluster_col: str = "cluster",
) -> Dict[int, str]:
    """
    Map numeric cluster IDs to human-readable marketing personas based on
    relative rank of median Recency, Frequency, and Monetary spend:
    - Champions: Lowest Recency, Highest Frequency & Monetary
    - Loyal Customers: Moderate Recency, Strong Frequency & Monetary
    - At-Risk: High Recency, Moderate/High Historical Frequency
    - Hibernating: High Recency, Low Frequency, Low Monetary
    """
    summary = rfm_df.groupby(cluster_col).agg(
        med_r=("recency", "median"),
        med_f=("frequency", "median"),
        med_m=("monetary", "median"),
    )

    k = len(summary)
    label_map = {}

    if k == 4:
        # Standard 4-persona mapping
        # Rank clusters:
        # 1. Champions: Low R, High F, High M
        # 2. Loyal Customers: Medium-Low R, Moderate F, Moderate M
        # 3. At-Risk: High R, Medium F/M
        # 4. Hibernating: High R, Lowest F & M
        scores = {}
        for c in summary.index:
            r = summary.loc[c, "med_r"]
            f = summary.loc[c, "med_f"]
            m = summary.loc[c, "med_m"]
            # Engagement score: higher is better
            scores[c] = (f * 1.5 + np.log1p(m) * 2.0) - (r * 0.1)

        sorted_by_engagement = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        # Highest engagement -> Champions
        label_map[sorted_by_engagement[0]] = "Champions"
        label_map[sorted_by_engagement[1]] = "Loyal Customers"

        # Between the remaining two, distinguish At-Risk (higher historical spend/freq) vs Hibernating
        rem1, rem2 = sorted_by_engagement[2], sorted_by_engagement[3]
        if summary.loc[rem1, "med_m"] >= summary.loc[rem2, "med_m"]:
            label_map[rem1] = "At-Risk"
            label_map[rem2] = "Hibernating"
        else:
            label_map[rem2] = "At-Risk"
            label_map[rem1] = "Hibernating"
    else:
        # Generic descriptive fallback for other values of k
        sorted_by_monetary = summary.sort_values(by="med_m", ascending=False).index.tolist()
        persona_pool = ["Champions", "Loyal Customers", "Promising", "Needs Attention", "At-Risk", "Hibernating"]
        for idx, c in enumerate(sorted_by_monetary):
            label_map[c] = persona_pool[idx] if idx < len(persona_pool) else f"Cluster {c}"

    return label_map


def plot_rfm_clusters(
    rfm_df: pd.DataFrame,
    cluster_label_col: str = "cluster_label",
    output_path: str = "outputs/rfm_clusters.png",
) -> None:
    """
    Generate a 4-panel multi-dimensional visualization of RFM customer clusters.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)

    palette = sns.color_palette("Set2", n_colors=rfm_df[cluster_label_col].nunique())

    # Panel 1: Recency vs Monetary Spend
    sns.scatterplot(
        data=rfm_df,
        x="recency",
        y="monetary",
        hue=cluster_label_col,
        palette=palette,
        alpha=0.75,
        s=45,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Recency vs. Monetary Spend", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Recency (Days since last purchase)", fontsize=10)
    axes[0, 0].set_ylabel("Monetary Spend ($)", fontsize=10)

    # Panel 2: Frequency vs Monetary Spend
    sns.scatterplot(
        data=rfm_df,
        x="frequency",
        y="monetary",
        hue=cluster_label_col,
        palette=palette,
        alpha=0.75,
        s=45,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Order Frequency vs. Monetary Spend", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Frequency (Order Count)", fontsize=10)
    axes[0, 1].set_ylabel("Monetary Spend ($)", fontsize=10)

    # Panel 3: Recency vs Frequency
    sns.scatterplot(
        data=rfm_df,
        x="recency",
        y="frequency",
        hue=cluster_label_col,
        palette=palette,
        alpha=0.75,
        s=45,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Recency vs. Order Frequency", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Recency (Days)", fontsize=10)
    axes[1, 0].set_ylabel("Frequency (Order Count)", fontsize=10)

    # Panel 4: Monetary Spend Boxplot by Segment
    sns.boxplot(
        data=rfm_df,
        x=cluster_label_col,
        y="monetary",
        hue=cluster_label_col,
        legend=False,
        palette=palette,
        ax=axes[1, 1],
        showfliers=False,
    )
    axes[1, 1].set_title("Monetary Spend Distribution by Segment", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Customer Segment", fontsize=10)
    axes[1, 1].set_ylabel("Monetary Spend ($)", fontsize=10)
    axes[1, 1].tick_params(axis="x", rotation=15)

    plt.suptitle("Customer Segmentation: K-Means RFM Personas", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def perform_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 4,
    use_log_transform: bool = True,
    random_state: int = 42,
    output_dir: str = "outputs",
) -> Tuple[pd.DataFrame, pd.DataFrame, KMeans, StandardScaler]:
    """
    End-to-end customer segmentation pipeline:
    1. Aggregates transactions into RFM features.
    2. Scales features using log-transform + StandardScaler.
    3. Runs Elbow and Silhouette evaluation.
    4. Fits K-Means clustering.
    5. Maps business persona labels and exports summary + plots.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, KMeans, StandardScaler]
        (rfm_df_with_clusters, cluster_summary_table, fitted_kmeans, fitted_scaler)
    """
    os.makedirs(output_dir, exist_ok=True)
    rfm_df = compute_rfm_features(df)

    rfm_features = ["recency", "frequency", "monetary"]
    X = rfm_df[rfm_features].copy()

    # Preprocessing: log1p transformation for skewness mitigation
    if use_log_transform:
        X_trans = np.log1p(X)
    else:
        X_trans = X.copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trans)

    # Diagnostic curves (Elbow & Silhouette)
    elbow_path = os.path.join(output_dir, "elbow_curve.png")
    evaluate_clusters(X_scaled, k_range=range(2, 8), random_state=random_state, output_path=elbow_path)

    # Fit final K-Means model
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=15, random_state=random_state)
    rfm_df["cluster"] = kmeans.fit_predict(X_scaled)

    # Assign business personas
    label_map = assign_business_personas(rfm_df, cluster_col="cluster")
    rfm_df["cluster_label"] = rfm_df["cluster"].map(label_map)

    # Cluster Summary Statistics
    summary_aggs = {
        "customer_id": "count",
        "recency": ["mean", "median"],
        "frequency": ["mean", "median"],
        "monetary": ["mean", "median"],
    }
    if "churned" in rfm_df.columns:
        summary_aggs["churned"] = "mean"

    summary = rfm_df.groupby("cluster_label").agg(summary_aggs)
    summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns]
    summary = summary.rename(columns={"customer_id_count": "customer_count"})
    if "churned_mean" in summary.columns:
        summary = summary.rename(columns={"churned_mean": "churn_rate"})

    summary_path = os.path.join(output_dir, "cluster_summary.csv")
    summary.to_csv(summary_path)

    # Plot Cluster Visualizations
    clusters_path = os.path.join(output_dir, "rfm_clusters.png")
    plot_rfm_clusters(rfm_df, cluster_label_col="cluster_label", output_path=clusters_path)

    return rfm_df, summary, kmeans, scaler
