"""
Credit Risk Analytics Engine for Retail Finance BIU
Implements Portfolio At Risk (PAR), Vintage (Cohort) Curves, and Roll-Rate Migration Matrices.
"""

import sqlite3
from typing import Dict, List, Any
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "retail_credit.db")

def get_portfolio_kpis(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Calculates high-level executive portfolio health metrics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Total Active AUM, Total Disbursed Volume, Average Ticket Size
    cursor.execute("""
        SELECT 
            COUNT(loan_id) AS total_loans,
            SUM(loan_amount) AS total_disbursed_aum,
            AVG(loan_amount) AS avg_ticket_size,
            AVG(interest_rate) AS weighted_avg_rate,
            AVG(ltv) AS avg_ltv,
            AVG(foir) AS avg_foir
        FROM loans;
    """)
    totals = cursor.fetchone()
    
    # 2. Delinquency & NPA Metrics (PAR-30, PAR-60, PAR-90 / Gross NPA)
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN d.is_30_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) AS par_30_amount,
            SUM(CASE WHEN d.is_60_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) AS par_60_amount,
            SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) AS gnpa_amount,
            COUNT(CASE WHEN d.is_90_plus_dpd = 1 THEN 1 END) AS npa_count
        FROM loans l
        JOIN delinquency_summary d ON l.loan_id = d.loan_id;
    """)
    delinq = cursor.fetchone()
    
    # 3. Average CIBIL Score
    cursor.execute("SELECT AVG(cibil_score) FROM customers;")
    avg_cibil = cursor.fetchone()[0]
    
    total_aum = totals[1] or 1.0
    par_30_pct = ((delinq[0] or 0) / total_aum) * 100
    par_60_pct = ((delinq[1] or 0) / total_aum) * 100
    gnpa_pct = ((delinq[2] or 0) / total_aum) * 100
    
    conn.close()
    
    return {
        "total_loans": totals[0],
        "total_aum": round(totals[1], 2),
        "avg_ticket_size": round(totals[2], 2),
        "weighted_avg_rate": round(totals[3], 2),
        "avg_ltv": round(totals[4], 2),
        "avg_foir": round(totals[5], 2),
        "avg_cibil": round(avg_cibil, 1),
        "par_30_amount": round(delinq[0] or 0, 2),
        "par_30_pct": round(par_30_pct, 2),
        "par_60_amount": round(delinq[1] or 0, 2),
        "par_60_pct": round(par_60_pct, 2),
        "gnpa_amount": round(delinq[2] or 0, 2),
        "gnpa_pct": round(gnpa_pct, 2),
        "npa_count": delinq[3]
    }

def get_product_breakdown(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Product-level AUM, Average Ticket, and Gross NPA %."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            l.product_type,
            COUNT(l.loan_id) AS loan_count,
            SUM(l.loan_amount) AS total_aum,
            AVG(l.loan_amount) AS avg_ticket,
            AVG(l.interest_rate) AS avg_roi,
            AVG(l.ltv) AS avg_ltv,
            AVG(l.foir) AS avg_foir,
            ROUND(100.0 * SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) / SUM(l.loan_amount), 2) AS gnpa_pct,
            ROUND(100.0 * SUM(CASE WHEN d.is_30_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) / SUM(l.loan_amount), 2) AS par_30_pct
        FROM loans l
        JOIN delinquency_summary d ON l.loan_id = d.loan_id
        GROUP BY l.product_type
        ORDER BY total_aum DESC;
    """)
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def get_tier_breakdown(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Geographic Bharat Tier Breakdown (Tier-1, Tier-2, Tier-3)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            c.tier,
            COUNT(l.loan_id) AS loan_count,
            SUM(l.loan_amount) AS total_aum,
            AVG(c.cibil_score) AS avg_cibil,
            AVG(c.monthly_income) AS avg_income,
            ROUND(100.0 * SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) / SUM(l.loan_amount), 2) AS gnpa_pct,
            ROUND(100.0 * SUM(CASE WHEN d.is_30_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) / SUM(l.loan_amount), 2) AS par_30_pct
        FROM loans l
        JOIN customers c ON l.customer_id = c.customer_id
        JOIN delinquency_summary d ON l.loan_id = d.loan_id
        GROUP BY c.tier
        ORDER BY c.tier ASC;
    """)
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def get_vintage_analysis(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """
    Vintage / Cohort Analysis:
    Groups loans by Origination Quarter (e.g. 2024-Q1, 2024-Q2, 2024-Q3...)
    and calculates delinquency curve across Months on Book (MOB).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Calculate origination cohort and MOB default rates
    cursor.execute("""
        WITH LoanCohorts AS (
            SELECT 
                loan_id,
                product_type,
                strftime('%Y', disbursement_date) || '-Q' || ((CAST(strftime('%m', disbursement_date) AS INTEGER) + 2) / 3) AS cohort_quarter,
                loan_amount
            FROM loans
        ),
        RepaymentMaturity AS (
            SELECT 
                r.loan_id,
                lc.cohort_quarter,
                r.installment_no AS mob,
                r.dpd,
                CASE WHEN r.dpd >= 30 THEN 1 ELSE 0 END AS is_30_plus
            FROM repayments r
            JOIN LoanCohorts lc ON r.loan_id = lc.loan_id
            WHERE r.installment_no IN (3, 6, 9, 12, 18, 24)
        )
        SELECT 
            cohort_quarter,
            mob,
            COUNT(DISTINCT loan_id) AS active_accounts,
            ROUND(100.0 * SUM(is_30_plus) / COUNT(loan_id), 2) AS delinq_30_plus_pct
        FROM RepaymentMaturity
        GROUP BY cohort_quarter, mob
        ORDER BY cohort_quarter ASC, mob ASC;
    """)
    
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def get_roll_rate_matrix(db_path: str = DB_PATH) -> Dict[str, Dict[str, float]]:
    """
    Computes Roll-Rate Transition Probability Matrix between consecutive installments:
    0 DPD -> 0 DPD (Current)
    0 DPD -> 30+ DPD
    30+ DPD -> 60+ DPD
    60+ DPD -> 90+ DPD (Default/NPA)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        WITH RankedRepayments AS (
            SELECT 
                loan_id,
                installment_no,
                dpd,
                CASE 
                    WHEN dpd = 0 THEN 'Current (0 DPD)'
                    WHEN dpd BETWEEN 1 AND 29 THEN '1-29 DPD'
                    WHEN dpd BETWEEN 30 AND 59 THEN '30-59 DPD'
                    WHEN dpd BETWEEN 60 AND 89 THEN '60-89 DPD'
                    ELSE '90+ DPD (NPA)'
                END AS bucket,
                LEAD(dpd, 1) OVER (PARTITION BY loan_id ORDER BY installment_no) AS next_dpd
            FROM repayments
        ),
        Transitions AS (
            SELECT 
                bucket AS from_bucket,
                CASE 
                    WHEN next_dpd = 0 THEN 'Current (0 DPD)'
                    WHEN next_dpd BETWEEN 1 AND 29 THEN '1-29 DPD'
                    WHEN next_dpd BETWEEN 30 AND 59 THEN '30-59 DPD'
                    WHEN next_dpd BETWEEN 60 AND 89 THEN '60-89 DPD'
                    ELSE '90+ DPD (NPA)'
                END AS to_bucket
            FROM RankedRepayments
            WHERE next_dpd IS NOT NULL
        )
        SELECT 
            from_bucket,
            to_bucket,
            COUNT(*) AS count
        FROM Transitions
        GROUP BY from_bucket, to_bucket
        ORDER BY from_bucket, to_bucket;
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    # Normalize transitions into percentages
    matrix: Dict[str, Dict[str, float]] = {}
    totals: Dict[str, int] = {}
    
    for from_b, to_b, cnt in rows:
        if from_b not in matrix:
            matrix[from_b] = {}
            totals[from_b] = 0
        matrix[from_b][to_b] = cnt
        totals[from_b] += cnt
        
    normalized_matrix: Dict[str, Dict[str, float]] = {}
    for from_b, targets in matrix.items():
        total = totals[from_b] or 1
        normalized_matrix[from_b] = {to_b: round((cnt / total) * 100, 2) for to_b, cnt in targets.items()}
        
    return normalized_matrix

if __name__ == "__main__":
    kpis = get_portfolio_kpis()
    print("Portfolio KPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
