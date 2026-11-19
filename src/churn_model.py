"""
churn_model.py - Churn Classification, Imbalance Handling & Feature Importance Analysis.

Trains and rigorously compares multiple classifiers:
- LogisticRegression (with balanced class weights & standard scaling)
- RandomForestClassifier (balanced ensemble trees)
- GradientBoostingClassifier (sequential gradient boosting)

Evaluates cross-validation and test set performance using F1-Score and ROC-AUC,
handles class imbalance, exports publication-grade ROC curves, and extracts
the top behavioral drivers of churn.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_features(
    rfm_df: pd.DataFrame,
    target_col: str = "churned",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Construct the ML feature matrix and target vector from segmented customer data.

    Encodes categorical attributes (contract_type, cluster_label) and combines
    them with RFM metrics and behavioral tenure/support ticket features.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        (X, y, feature_names)
    """
    df = rfm_df.copy()

    # Base continuous features
    num_features = ["recency", "frequency", "monetary", "avg_order_value", "tenure_months", "support_tickets"]
    avail_num = [c for c in num_features if c in df.columns]

    # Categorical features for one-hot encoding
    cat_features = ["contract_type", "cluster_label"]
    avail_cat = [c for c in cat_features if c in df.columns]

    if avail_cat:
        df_encoded = pd.get_dummies(df[avail_cat], prefix=avail_cat, drop_first=False, dtype=float)
        X = pd.concat([df[avail_num], df_encoded], axis=1)
    else:
        X = df[avail_num].copy()

    # Fill any missing values if present
    X = X.fillna(X.median(numeric_only=True))

    y = df[target_col].astype(int)
    feature_names = list(X.columns)

    return X, y, feature_names


def train_and_evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data with stratification, train candidate classifiers, and compute
    cross-validation and held-out test performance metrics.

    Returns
    -------
    Tuple[Dict[str, Any], pd.DataFrame, Tuple]
        (models_dict, comparison_df, (X_train, X_test, y_train, y_test))
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Standard scaling for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    # Class imbalance is addressed via balanced weighting and cross-entropy optimization
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state),
            "use_scaled": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=7,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            "use_scaled": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                n_estimators=120,
                learning_rate=0.08,
                max_depth=4,
                random_state=random_state,
            ),
            "use_scaled": False,
        },
    }

    metrics_records = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, config in models.items():
        clf = config["model"]
        is_scaled = config["use_scaled"]

        train_data = X_train_scaled if is_scaled else X_train.values
        test_data = X_test_scaled if is_scaled else X_test.values

        # 5-fold Stratified Cross-Validation on Train split
        cv_roc = cross_val_score(clf, train_data, y_train, cv=cv, scoring="roc_auc")
        cv_f1 = cross_val_score(clf, train_data, y_train, cv=cv, scoring="f1")

        # Fit model on entire training set
        clf.fit(train_data, y_train)

        # Predictions on held-out test set
        y_pred = clf.predict(test_data)
        y_prob = clf.predict_proba(test_data)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        metrics_records.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1-Score": round(f1, 4),
                "ROC-AUC": round(roc, 4),
                "CV ROC-AUC (Mean)": round(float(np.mean(cv_roc)), 4),
                "CV ROC-AUC (Std)": round(float(np.std(cv_roc)), 4),
                "CV F1 (Mean)": round(float(np.mean(cv_f1)), 4),
            }
        )

        config["fitted_clf"] = clf
        config["test_prob"] = y_prob
        config["test_pred"] = y_pred

    comparison_df = pd.DataFrame(metrics_records).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    splits = (X_train, X_test, y_train, y_test, scaler)

    return models, comparison_df, splits


def plot_roc_curves(
    models: Dict[str, Any],
    y_test: np.ndarray,
    output_path: str = "outputs/roc_curve.png",
) -> None:
    """
    Plot and save multi-model ROC Curves with AUC comparison.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)

    colors = {"Logistic Regression": "#2b5c8f", "Random Forest": "#2e7d32", "Gradient Boosting": "#d95f02"}

    for name, config in models.items():
        y_prob = config["test_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        color = colors.get(name, None)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2.2, color=color)

    plt.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.500)", alpha=0.6, linewidth=1.5)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_churn_drivers(
    rf_model: Any,
    lr_model: Any,
    feature_names: List[str],
    output_path: str = "outputs/top_churn_drivers.png",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Extract and visualize top drivers of customer churn using Random Forest
    Gini importance and Logistic Regression standardized coefficients.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1. Random Forest Feature Importance
    importances = rf_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # 2. Logistic Regression Coefficients (Log-Odds Impact)
    lr_coefs = lr_model.coef_[0]
    lr_df = pd.DataFrame({"Feature": feature_names, "Coefficient": lr_coefs})
    lr_df["Abs_Coef"] = lr_df["Coefficient"].abs()
    lr_df = lr_df.sort_values(by="Abs_Coef", ascending=False).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # Left plot: Random Forest Feature Importance
    top_fi = fi_df.head(top_n).sort_values(by="Importance", ascending=True)
    axes[0].barh(top_fi["Feature"], top_fi["Importance"], color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("Relative Importance (MDI)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"Top {top_n} Churn Drivers (Random Forest)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="x")

    # Right plot: Logistic Regression Standardized Coefficients
    top_lr = lr_df.head(top_n).sort_values(by="Coefficient", ascending=True)
    colors = ["#d95f02" if c > 0 else "#2ca02c" for c in top_lr["Coefficient"]]
    axes[1].barh(top_lr["Feature"], top_lr["Coefficient"], color=colors, edgecolor="black", alpha=0.85)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Standardized Log-Odds Impact (Coefficients)", fontsize=11, fontweight="bold")
    axes[1].set_title(f"Top {top_n} Directional Drivers (Logistic Regression)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="x")

    # Legend note on directional impact
    plt.suptitle("Model Explainability: Key Drivers of Customer Churn", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return fi_df


def run_churn_pipeline(
    rfm_df: pd.DataFrame,
    output_dir: str = "outputs",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """
    End-to-end churn classification pipeline:
    1. Prepares feature matrix from RFM + demographics + clusters.
    2. Trains and cross-validates LogisticRegression, RandomForest, GradientBoosting.
    3. Exports model comparison table (outputs/model_comparison.csv).
    4. Plots ROC curves (outputs/roc_curve.png).
    5. Extracts and plots top churn drivers (outputs/top_churn_drivers.png).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]
        (comparison_df, models_dict, feature_importances_df)
    """
    os.makedirs(output_dir, exist_ok=True)
    X, y, feature_names = prepare_features(rfm_df)

    models, comparison_df, splits = train_and_evaluate_models(
        X, y, test_size=0.25, random_state=random_state
    )
    X_train, X_test, y_train, y_test, scaler = splits

    # Save metrics comparison table
    comp_path = os.path.join(output_dir, "model_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)

    # ROC Curves
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plot_roc_curves(models, y_test.values, output_path=roc_path)

    # Churn Drivers Plot
    rf_clf = models["Random Forest"]["fitted_clf"]
    lr_clf = models["Logistic Regression"]["fitted_clf"]
    drivers_path = os.path.join(output_dir, "top_churn_drivers.png")
    fi_df = plot_churn_drivers(rf_clf, lr_clf, feature_names, output_path=drivers_path)

    fi_path = os.path.join(output_dir, "feature_importances.csv")
    fi_df.to_csv(fi_path, index=False)

    return comparison_df, models, fi_df
