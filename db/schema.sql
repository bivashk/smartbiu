-- Schema for Retail Credit Risk & Portfolio Intelligence Data Warehouse

DROP TABLE IF EXISTS delinquency_summary;
DROP TABLE IF EXISTS repayments;
DROP TABLE IF EXISTS loans;
DROP TABLE IF EXISTS customers;

-- 1. Customers Table (Demographics & Baseline Financial Profile)
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    city TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN ('Tier-1', 'Tier-2', 'Tier-3')),
    state TEXT NOT NULL,
    employment_type TEXT NOT NULL CHECK (employment_type IN ('Salaried', 'Self-Employed Professional', 'Self-Employed Business')),
    monthly_income REAL NOT NULL,
    cibil_score INTEGER NOT NULL,
    existing_active_loans INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

-- 2. Loans Table (Account Level Terms & Underwriting Ratios)
CREATE TABLE loans (
    loan_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product_type TEXT NOT NULL CHECK (product_type IN (
        'Affordable Housing', 
        'Loan Against Property', 
        'Secured MSME', 
        'Digital Personal Loan', 
        'Used Car Loan'
    )),
    loan_amount REAL NOT NULL,
    collateral_value REAL NOT NULL,
    ltv REAL NOT NULL,                    -- Loan to Value Ratio (%)
    tenure_months INTEGER NOT NULL,
    interest_rate REAL NOT NULL,         -- Annual Interest Rate (%)
    monthly_emi REAL NOT NULL,
    foir REAL NOT NULL,                   -- Fixed Obligation to Income Ratio (%)
    disbursement_date TEXT NOT NULL,
    maturity_date TEXT NOT NULL,
    loan_status TEXT NOT NULL CHECK (loan_status IN ('Active', 'Closed', 'Written-Off'))
);

-- 3. Repayments Table (Monthly Cash Flow & DPD Tracking)
CREATE TABLE repayments (
    repayment_id TEXT PRIMARY KEY,
    loan_id TEXT NOT NULL REFERENCES loans(loan_id),
    installment_no INTEGER NOT NULL,
    due_date TEXT NOT NULL,
    paid_date TEXT,
    amount_due REAL NOT NULL,
    amount_paid REAL NOT NULL DEFAULT 0,
    dpd INTEGER NOT NULL DEFAULT 0,       -- Days Past Due on this installment
    bounce_flag INTEGER NOT NULL DEFAULT 0 CHECK (bounce_flag IN (0, 1))
);

-- 4. Delinquency Summary Table (Aggregated Risk Profile & NPA Classification)
CREATE TABLE delinquency_summary (
    loan_id TEXT PRIMARY KEY REFERENCES loans(loan_id),
    max_dpd INTEGER NOT NULL DEFAULT 0,
    current_dpd INTEGER NOT NULL DEFAULT 0,
    total_bounces INTEGER NOT NULL DEFAULT 0,
    bounces_last_3m INTEGER NOT NULL DEFAULT 0,
    bounces_last_6m INTEGER NOT NULL DEFAULT 0,
    is_30_plus_dpd INTEGER NOT NULL DEFAULT 0,
    is_60_plus_dpd INTEGER NOT NULL DEFAULT 0,
    is_90_plus_dpd INTEGER NOT NULL DEFAULT 0,  -- NPA definition (RBI 90+ DPD mandate)
    last_status_update TEXT NOT NULL
);

-- Optimized Analytical Indexes
CREATE INDEX idx_customers_tier ON customers(tier);
CREATE INDEX idx_customers_cibil ON customers(cibil_score);
CREATE INDEX idx_loans_customer ON loans(customer_id);
CREATE INDEX idx_loans_product ON loans(product_type);
CREATE INDEX idx_loans_disbursed ON loans(disbursement_date);
CREATE INDEX idx_repayments_loan ON repayments(loan_id);
CREATE INDEX idx_repayments_due ON repayments(due_date);
CREATE INDEX idx_delinq_npa ON delinquency_summary(is_90_plus_dpd);
CREATE INDEX idx_delinq_current_dpd ON delinquency_summary(current_dpd);
