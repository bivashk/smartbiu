"""
Credit Risk Explainability & Adverse Action Engine
Translates model predictions into human-interpretable risk drivers and RBI-compliant reason codes.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import joblib
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_COLUMNS, preprocess_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_default_xgb.joblib")
METADATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_metadata.joblib")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "retail_credit.db")

class CreditRiskExplainer:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")
        self.model = joblib.load(MODEL_PATH)
        self.metadata = joblib.load(METADATA_PATH)
        self.feature_columns = self.metadata["feature_columns"]
        
    def explain_loan(self, loan_id: str, db_path: str = DB_PATH) -> Dict[str, Any]:
        """
        Evaluates a specific loan account and returns default probability, risk tier,
        and top contributing adverse factors.
        """
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT 
                l.loan_id,
                l.customer_id,
                c.name,
                c.age,
                c.gender,
                c.city,
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
                l.disbursement_date,
                l.loan_status,
                d.total_bounces,
                d.bounces_last_3m,
                d.bounces_last_6m,
                d.max_dpd,
                d.current_dpd,
                d.is_90_plus_dpd AS target_default
            FROM loans l
            JOIN customers c ON l.customer_id = c.customer_id
            JOIN delinquency_summary d ON l.loan_id = d.loan_id
            WHERE l.loan_id = '{loan_id}';
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {"error": f"Loan ID '{loan_id}' not found in database."}
            
        row = df.iloc[0]
        X_encoded, _ = preprocess_features(df)
        
        # Predict probability of default
        prob_default = float(self.model.predict_proba(X_encoded[self.feature_columns])[0, 1])
        
        # Risk Tier Classification
        if prob_default < 0.05:
            risk_tier = "Low Risk (Prime)"
            recommendation = "APPROVE / NORMAL MONITORING"
        elif prob_default < 0.20:
            risk_tier = "Moderate Risk (Watchlist)"
            recommendation = "CONDITIONAL APPROVAL / INTENSIVE SOFT COLLECTIONS"
        else:
            risk_tier = "High Risk (Subprime / Distress)"
            recommendation = "REJECT / HARD COLLECTIONS & RESTRUCTURING"
            
        # Adverse Action Reason Codes (Rule & Contribution-based)
        reasons: List[Dict[str, Any]] = []
        
        if row["bounces_last_3m"] > 0:
            reasons.append({
                "factor": "Recent EMI Bounces",
                "severity": "HIGH",
                "detail": f"{int(row['bounces_last_3m'])} bounce(s) in last 90 days indicates severe liquidity distress.",
                "action": "Trigger immediate tele-calling reminder before next billing cycle."
            })
            
        if row["foir"] > 55.0:
            reasons.append({
                "factor": "Elevated Debt-to-Income (FOIR)",
                "severity": "MEDIUM",
                "detail": f"FOIR is {row['foir']:.1f}% (exceeds conservative 50% prudent lending threshold).",
                "action": "Cap additional credit exposure."
            })
            
        if row["cibil_score"] < 680:
            reasons.append({
                "factor": "Bureau Score Below Cutoff",
                "severity": "HIGH" if row["cibil_score"] < 620 else "MEDIUM",
                "detail": f"CIBIL score of {int(row['cibil_score'])} reflects prior delinquency with other institutions.",
                "action": "Request additional collateral or co-guarantor."
            })
            
        if row["ltv"] > 80.0 and row["product_type"] != "Digital Personal Loan":
            reasons.append({
                "factor": "High Loan-to-Value (LTV)",
                "severity": "MEDIUM",
                "detail": f"LTV is {row['ltv']:.1f}%, leaving low collateral cushion against market devaluation.",
                "action": "Re-evaluate property/asset valuation."
            })
            
        if row["existing_active_loans"] >= 3:
            reasons.append({
                "factor": "Multiple Active Credit Lines",
                "severity": "LOW",
                "detail": f"Borrower holds {int(row['existing_active_loans'])} concurrent active loans.",
                "action": "Monitor credit bureau scrub reports monthly."
            })
            
        if not reasons:
            reasons.append({
                "factor": "Pristine Credit Profile",
                "severity": "POSITIVE",
                "detail": f"CIBIL {int(row['cibil_score'])}, zero bounces, FOIR at {row['foir']:.1f}%.",
                "action": "Eligible for pre-approved top-up or rate discount."
            })
            
        return {
            "loan_id": loan_id,
            "customer_name": row["name"],
            "product_type": row["product_type"],
            "city_tier": f"{row['city']} ({row['tier']})",
            "loan_amount": float(row["loan_amount"]),
            "cibil_score": int(row["cibil_score"]),
            "foir": float(row["foir"]),
            "ltv": float(row["ltv"]),
            "total_bounces": int(row["total_bounces"]),
            "current_dpd": int(row["current_dpd"]),
            "probability_of_default": round(prob_default, 4),
            "probability_of_default_pct": round(prob_default * 100, 2),
            "risk_tier": risk_tier,
            "recommendation": recommendation,
            "adverse_action_factors": reasons
        }

if __name__ == "__main__":
    explainer = CreditRiskExplainer()
    sample_res = explainer.explain_loan("LN_000001")
    print(f"Loan ID: {sample_res['loan_id']}")
    print(f"Customer: {sample_res['customer_name']} | Risk: {sample_res['risk_tier']}")
    print(f"PD: {sample_res['probability_of_default_pct']}% | Recommendation: {sample_res['recommendation']}")
    print("Risk Factors:")
    for f in sample_res["adverse_action_factors"]:
        print(f"  - [{f['severity']}] {f['factor']}: {f['detail']}")
