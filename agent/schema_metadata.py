"""
Semantic Metadata, Domain Glossary, and Few-Shot SQL Exemplars
Used by the SmartBIU Agent for Schema Linking and Few-Shot In-Context Learning.
"""

SCHEMA_DOCUMENTATION = """
### Database Tables & Semantic Meanings:

1. `customers` (Borrower Demographics & Baseline Profile)
   - `customer_id` (TEXT, PK): Unique borrower ID (e.g. 'CUST_000001')
   - `name` (TEXT): Full name of the borrower
   - `age` (INTEGER): Age in years (21 - 60)
   - `gender` (TEXT): 'Male' or 'Female'
   - `city` (TEXT): Residential city (e.g. 'Mumbai', 'Jaipur', 'Alwar')
   - `tier` (TEXT): 'Tier-1' (Metros), 'Tier-2' (Growing urban), 'Tier-3' (Semi-urban / Rural Bharat)
   - `state` (TEXT): Indian state name
   - `employment_type` (TEXT): 'Salaried', 'Self-Employed Professional', 'Self-Employed Business'
   - `monthly_income` (REAL): Monthly declared net income in INR
   - `cibil_score` (INTEGER): Credit bureau score (300 to 900). Prime is >= 750, Subprime is < 650.
   - `existing_active_loans` (INTEGER): Number of active loans prior to this application

2. `loans` (Loan Accounts & Underwriting Terms)
   - `loan_id` (TEXT, PK): Unique loan identifier (e.g. 'LN_000001')
   - `customer_id` (TEXT, FK -> customers.customer_id)
   - `product_type` (TEXT): 'Affordable Housing', 'Loan Against Property', 'Secured MSME', 'Digital Personal Loan', 'Used Car Loan'
   - `loan_amount` (REAL): Disbursed principal amount in INR
   - `collateral_value` (REAL): Market valuation of pledged security (0 for unsecured Digital Personal Loans)
   - `ltv` (REAL): Loan to Value percentage (e.g. 75.0 = 75%)
   - `tenure_months` (INTEGER): Loan duration in months (12 to 240)
   - `interest_rate` (REAL): Annual interest rate % (e.g. 11.5)
   - `monthly_emi` (REAL): Monthly installment amount in INR
   - `foir` (REAL): Fixed Obligation to Income Ratio % (total monthly EMI / monthly income * 100)
   - `disbursement_date` (TEXT): YYYY-MM-DD format
   - `maturity_date` (TEXT): YYYY-MM-DD format
   - `loan_status` (TEXT): 'Active', 'Closed', 'Written-Off'

3. `repayments` (Monthly Repayment Cash Flows & DPD History)
   - `repayment_id` (TEXT, PK): e.g. 'RP_00000001'
   - `loan_id` (TEXT, FK -> loans.loan_id)
   - `installment_no` (INTEGER): Month number since disbursement (1, 2, 3...)
   - `due_date` (TEXT): YYYY-MM-DD
   - `paid_date` (TEXT): YYYY-MM-DD or NULL if unpaid
   - `amount_due` (REAL): EMI due
   - `amount_paid` (REAL): Amount actually collected
   - `dpd` (INTEGER): Days Past Due on this specific installment
   - `bounce_flag` (INTEGER): 1 if installment bounced (NACH / ECS failure), 0 if cleared

4. `delinquency_summary` (Aggregated Account Risk Snapshot)
   - `loan_id` (TEXT, PK, FK -> loans.loan_id)
   - `max_dpd` (INTEGER): Peak historical days past due across all installments
   - `current_dpd` (INTEGER): Latest installment days past due
   - `total_bounces` (INTEGER): Total count of bounced payments
   - `bounces_last_3m` (INTEGER): Bounces in recent 90 days (leading indicator of default)
   - `bounces_last_6m` (INTEGER): Bounces in recent 180 days
   - `is_30_plus_dpd` (INTEGER): 1 if current_dpd >= 30, else 0
   - `is_60_plus_dpd` (INTEGER): 1 if current_dpd >= 60, else 0
   - `is_90_plus_dpd` (INTEGER): 1 if current_dpd >= 90 (RBI NPA definition: Non-Performing Asset)
   - `last_status_update` (TEXT): YYYY-MM-DD
"""

BUSINESS_GLOSSARY = {
    "NPA": "Non-Performing Asset: An account with >= 90 Days Past Due (is_90_plus_dpd = 1 or current_dpd >= 90).",
    "Gross NPA %": "(Total loan_amount of NPA accounts / Total portfolio loan_amount) * 100",
    "PAR-30": "Portfolio At Risk 30+: Total loan_amount where current_dpd >= 30",
    "PAR-30 %": "(Total loan_amount of PAR-30 accounts / Total portfolio loan_amount) * 100",
    "FOIR": "Fixed Obligation to Income Ratio (% of borrower income servicing debt)",
    "LTV": "Loan to Value ratio (% of collateral value financed)",
    "MOB": "Months On Book: Installment number representing loan age",
    "Bounce Rate": "% of installments where bounce_flag = 1"
}

FEW_SHOT_EXEMPLARS = [
    {
        "question": "What is the total AUM and average loan size by product type?",
        "sql": """
SELECT 
    product_type,
    COUNT(loan_id) AS total_accounts,
    ROUND(SUM(loan_amount), 2) AS total_aum,
    ROUND(AVG(loan_amount), 2) AS avg_loan_size,
    ROUND(AVG(interest_rate), 2) AS avg_interest_rate
FROM loans
GROUP BY product_type
ORDER BY total_aum DESC;
        """.strip()
    },
    {
        "question": "Show me the Gross NPA percentage and total NPA amount across Bharat tiers.",
        "sql": """
SELECT 
    c.tier,
    COUNT(l.loan_id) AS total_loans,
    ROUND(SUM(l.loan_amount), 2) AS total_aum,
    ROUND(SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN l.loan_amount ELSE 0 END), 2) AS gnpa_amount,
    ROUND(100.0 * SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) / SUM(l.loan_amount), 2) AS gnpa_pct
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
JOIN delinquency_summary d ON l.loan_id = d.loan_id
GROUP BY c.tier
ORDER BY gnpa_pct DESC;
        """.strip()
    },
    {
        "question": "Which top 5 cities have the highest number of bounced installments?",
        "sql": """
SELECT 
    c.city,
    c.tier,
    c.state,
    SUM(d.total_bounces) AS total_bounces,
    COUNT(l.loan_id) AS total_loans,
    ROUND(1.0 * SUM(d.total_bounces) / COUNT(l.loan_id), 2) AS bounces_per_loan
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
JOIN delinquency_summary d ON l.loan_id = d.loan_id
GROUP BY c.city, c.tier, c.state
ORDER BY total_bounces DESC
LIMIT 5;
        """.strip()
    },
    {
        "question": "Find the average CIBIL score and average FOIR for defaulted vs non-defaulted loans.",
        "sql": """
SELECT 
    CASE WHEN d.is_90_plus_dpd = 1 THEN 'Defaulted (NPA)' ELSE 'Performing (Standard)' END AS loan_category,
    COUNT(l.loan_id) AS loan_count,
    ROUND(AVG(c.cibil_score), 1) AS avg_cibil,
    ROUND(AVG(l.foir), 2) AS avg_foir,
    ROUND(AVG(l.ltv), 2) AS avg_ltv
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
JOIN delinquency_summary d ON l.loan_id = d.loan_id
GROUP BY d.is_90_plus_dpd;
        """.strip()
    },
    {
        "question": "List high-risk loans with more than 2 bounces in the last 3 months and current DPD over 30.",
        "sql": """
SELECT 
    l.loan_id,
    c.name AS borrower_name,
    l.product_type,
    c.city,
    c.tier,
    l.loan_amount,
    d.bounces_last_3m,
    d.current_dpd,
    l.foir
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
JOIN delinquency_summary d ON l.loan_id = d.loan_id
WHERE d.bounces_last_3m > 2 AND d.current_dpd > 30
ORDER BY d.current_dpd DESC
LIMIT 20;
        """.strip()
    }
]
