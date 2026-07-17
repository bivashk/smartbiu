"""
Model Evaluation Suite for Credit Risk Scorecards
Implements Kolmogorov-Smirnov (KS) statistic, Gini coefficient, ROC-AUC, Brier score,
and Decile Gains/Lift tables standard in Indian NBFC risk governance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix

def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, int, pd.DataFrame]:
    """
    Computes the Kolmogorov-Smirnov (KS) statistic and returns the 10-decile scorecard table.
    
    KS = max(Cumulative % Bads - Cumulative % Goods)
    Industry Standard:
      < 20%: Very weak model (reject)
      20% - 40%: Acceptable
      40% - 60%: Excellent separation
      > 65%: Overfitting or target leakage warning
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    
    # Divide into 10 deciles based on predicted default probability (Decile 1 = Highest Risk)
    # Using rank to avoid binning collisions
    df["rank"] = df["y_prob"].rank(ascending=False, method="first")
    df["decile"] = pd.qcut(df["rank"], q=10, labels=range(1, 11))
    
    table = df.groupby("decile", observed=False).agg(
        total_accounts=("y_true", "count"),
        bads=("y_true", "sum"),
        min_prob=("y_prob", "min"),
        max_prob=("y_prob", "max")
    ).reset_index()
    
    table["goods"] = table["total_accounts"] - table["bads"]
    table["bad_rate_pct"] = (table["bads"] / table["total_accounts"]) * 100
    
    total_bads = table["bads"].sum() or 1
    total_goods = table["goods"].sum() or 1
    
    table["pct_bads"] = (table["bads"] / total_bads) * 100
    table["pct_goods"] = (table["goods"] / total_goods) * 100
    
    table["cum_pct_bads"] = table["pct_bads"].cumsum()
    table["cum_pct_goods"] = table["pct_goods"].cumsum()
    
    table["ks"] = table["cum_pct_bads"] - table["cum_pct_goods"]
    
    ks_value = table["ks"].max()
    max_ks_decile = int(table.loc[table["ks"].idxmax(), "decile"])
    
    # Calculate Lift: (Cumulative % Bads / Cumulative % Population)
    table["cum_pct_pop"] = (np.arange(1, 11) / 10.0) * 100
    table["lift"] = table["cum_pct_bads"] / table["cum_pct_pop"]
    
    return float(ks_value), max_ks_decile, table

def evaluate_credit_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.25) -> Dict[str, Any]:
    """
    Computes complete statistical evaluation metrics for risk governance.
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    gini = 2 * roc_auc - 1
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    brier = brier_score_loss(y_true, y_prob)
    
    ks_val, ks_decile, decile_table = compute_ks_statistic(y_true, y_prob)
    
    # Binary predictions at threshold
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sens_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (prec * sens_recall) / (prec + sens_recall) if (prec + sens_recall) > 0 else 0
    
    return {
        "roc_auc": round(float(roc_auc), 4),
        "gini_coefficient": round(float(gini), 4),
        "pr_auc": round(float(pr_auc), 4),
        "ks_statistic_pct": round(float(ks_val), 2),
        "max_ks_decile": int(ks_decile),
        "brier_score": round(float(brier), 4),
        "optimal_threshold": threshold,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "accuracy": round(float(accuracy), 4),
        "sensitivity_recall": round(float(sens_recall), 4),
        "specificity": round(float(spec), 4),
        "precision": round(float(prec), 4),
        "f1_score": round(float(f1), 4),
        "decile_table": decile_table.to_dict(orient="records")
    }

if __name__ == "__main__":
    # Smoke test on synthetic distribution
    np.random.seed(42)
    sample_y = np.random.binomial(1, 0.05, 5000)
    sample_probs = np.clip(sample_y * 0.4 + np.random.beta(1, 15, 5000), 0, 1)
    metrics = evaluate_credit_model(sample_y, sample_probs)
    print("Smoke Test Metrics:")
    print(f"  ROC-AUC: {metrics['roc_auc']}")
    print(f"  Gini: {metrics['gini_coefficient']}")
    print(f"  KS-Statistic: {metrics['ks_statistic_pct']}% at Decile {metrics['max_ks_decile']}")
