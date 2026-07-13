"""
Feature Engineering Pipeline for Credit Risk Modeling
Extracts financial ratios, bureau features, and repayment behavior from SQLite.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "retail_credit.db")

FEATURE_COLUMNS = [
    "cibil_score",
    "monthly_income",
    "existing_active_loans",
    "loan_amount",
    "tenure_months",
    "interest_rate",
    "monthly_emi",
    "ltv",
    "foir",
    "installment_to_income",
    "bounces_last_3m",
    "bounces_last_6m",
    "total_bounces",
    "product_type_Affordable Housing",
    "product_type_Digital Personal Loan",
    "product_type_Loan Against Property",
    "product_type_Secured MSME",
    "product_type_Used Car Loan",
    "tier_Tier-1",
    "tier_Tier-2",
    "tier_Tier-3",
    "employment_type_Salaried",
    "employment_type_Self-Employed Business",
    "employment_type_Self-Employed Professional"
]

def load_feature_dataset(db_path: str = DB_PATH) -> pd.DataFrame:
    """Loads consolidated raw features joining customers, loans, and delinquencies."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            l.loan_id,
            l.customer_id,
            c.age,
            c.tier,
            c.employment_type,
            c.monthly_income,
            c.cibil_score,
            c.existing_active_loans,
            l.product_type,
            l.loan_amount,
            l.tenure_months,
            l.interest_rate,
            l.monthly_emi,
            l.ltv,
            l.foir,
            d.total_bounces,
            d.bounces_last_3m,
            d.bounces_last_6m,
            d.is_90_plus_dpd AS target_default
        FROM loans l
        JOIN customers c ON l.customer_id = c.customer_id
        JOIN delinquency_summary d ON l.loan_id = d.loan_id;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs engineered financial ratios and encodes categorical variables.
    """
    data = df.copy()
    
    # 1. Domain-specific financial engineering
    # Installment to Income ratio (%)
    data["installment_to_income"] = (data["monthly_emi"] / data["monthly_income"].clip(lower=1000)) * 100
    
    # One-hot encode categoricals
    cat_cols = ["product_type", "tier", "employment_type"]
    data_encoded = pd.get_dummies(data, columns=cat_cols, dtype=float)
    
    # Ensure all expected columns exist
    for col in FEATURE_COLUMNS:
        if col not in data_encoded.columns:
            data_encoded[col] = 0.0
            
    X = data_encoded[FEATURE_COLUMNS]
    y = data_encoded["target_default"]
    
    return X, y

def calculate_woe_iv(df: pd.DataFrame, feature: str, target: str, bins: int = 5) -> Dict[str, Any]:
    """
    Computes Weight of Evidence (WOE) and Information Value (IV) for credit scorecard explainability.
    IV Interpretation:
      < 0.02: Not predictive
      0.02 to 0.1: Weak predictor
      0.1 to 0.3: Medium predictor
      0.3 to 0.5: Strong predictor
      > 0.5: Suspicious / Too good
    """
    data = df[[feature, target]].dropna().copy()
    if data[feature].nunique() > bins:
        data["bin"] = pd.qcut(data[feature], q=bins, duplicates="drop")
    else:
        data["bin"] = data[feature]
        
    grouped = data.groupby("bin", observed=False)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bads"]
    grouped["goods"] = grouped["total"] - grouped["bads"]
    
    total_goods = grouped["goods"].sum() or 1
    total_bads = grouped["bads"].sum() or 1
    
    grouped["dist_goods"] = grouped["goods"] / total_goods
    grouped["dist_bads"] = grouped["bads"] / total_bads
    
    # Avoid zero division
    grouped["dist_goods"] = grouped["dist_goods"].replace(0, 0.0001)
    grouped["dist_bads"] = grouped["dist_bads"].replace(0, 0.0001)
    
    grouped["woe"] = np.log(grouped["dist_goods"] / grouped["dist_bads"])
    grouped["iv_component"] = (grouped["dist_goods"] - grouped["dist_bads"]) * grouped["woe"]
    total_iv = grouped["iv_component"].sum()
    
    return {
        "feature": feature,
        "information_value": round(float(total_iv), 4),
        "woe_table": grouped[["total", "goods", "bads", "woe", "iv_component"]].to_dict(orient="index")
    }

if __name__ == "__main__":
    raw_df = load_feature_dataset()
    print(f"Loaded dataset: {raw_df.shape[0]} rows, default rate: {raw_df['target_default'].mean():.2%}")
    X, y = preprocess_features(raw_df)
    print(f"Engineered feature matrix: {X.shape}")
    cibil_iv = calculate_woe_iv(raw_df, "cibil_score", "target_default")
    print(f"CIBIL Information Value (IV): {cibil_iv['information_value']}")
