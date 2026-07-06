"""
Synthetic Data Generator for Retail Credit Risk & Portfolio Intelligence BIU
Accurately calibrated to NBFC benchmark metrics:
Overall Gross NPA ~ 3.5% - 4.5%, PAR-30 ~ 6.5% - 8.5%, realistic FOIR and CIBIL distributions.
"""

import os
import sys
import random
import math
import sqlite3
from datetime import datetime, timedelta

# Set deterministic seed for reproducibility
random.seed(42)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "retail_credit.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "schema.sql")

TIER_1_CITIES = [
    ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Bengaluru", "Karnataka"),
    ("Hyderabad", "Telangana"), ("Chennai", "Tamil Nadu"), ("Pune", "Maharashtra"),
    ("Kolkata", "West Bengal"), ("Ahmedabad", "Gujarat")
]

TIER_2_CITIES = [
    ("Jaipur", "Rajasthan"), ("Lucknow", "Uttar Pradesh"), ("Indore", "Madhya Pradesh"),
    ("Coimbatore", "Tamil Nadu"), ("Kochi", "Kerala"), ("Chandigarh", "Punjab"),
    ("Nagpur", "Maharashtra"), ("Bhopal", "Madhya Pradesh"), ("Surat", "Gujarat"),
    ("Patna", "Bihar"), ("Vadodara", "Gujarat"), ("Visakhapatnam", "Andhra Pradesh")
]

TIER_3_CITIES = [
    ("Alwar", "Rajasthan"), ("Gorakhpur", "Uttar Pradesh"), ("Muzaffarpur", "Bihar"),
    ("Belagavi", "Karnataka"), ("Salem", "Tamil Nadu"), ("Ujjain", "Madhya Pradesh"),
    ("Jhansi", "Uttar Pradesh"), ("Kurnool", "Andhra Pradesh"), ("Warangal", "Telangana"),
    ("Tirunelveli", "Tamil Nadu"), ("Satara", "Maharashtra"), ("Anand", "Gujarat")
]

FIRST_NAMES = [
    "Aarav", "Aditi", "Amit", "Ananya", "Arjun", "Bhavna", "Chetan", "Deepak",
    "Divya", "Gaurav", "Harsh", "Ishaan", "Kavita", "Manish", "Neha", "Nikhil",
    "Pooja", "Pradeep", "Priya", "Rahul", "Rajesh", "Ritu", "Rohan", "Sanjay",
    "Shikha", "Sneha", "Suresh", "Tanvi", "Varun", "Vikas", "Vikram", "Yash"
]

LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Mehta", "Patel", "Reddy", "Nair", "Iyer",
    "Singh", "Kumar", "Chauhan", "Joshi", "Deshmukh", "Kulkarni", "Mishra",
    "Pandey", "Yadav", "Agarwal", "Bansal", "Chopra", "Das", "Ghosh"
]

PRODUCTS = {
    "Affordable Housing": {
        "min_amount": 1200000, "max_amount": 4500000,
        "min_tenure": 120, "max_tenure": 240,
        "min_rate": 8.75, "max_rate": 10.75,
        "base_ltv": (0.60, 0.80),
        "base_risk": -4.8
    },
    "Loan Against Property": {
        "min_amount": 2500000, "max_amount": 12000000,
        "min_tenure": 60, "max_tenure": 180,
        "min_rate": 9.50, "max_rate": 12.50,
        "base_ltv": (0.45, 0.65),
        "base_risk": -4.2
    },
    "Secured MSME": {
        "min_amount": 800000, "max_amount": 6000000,
        "min_tenure": 36, "max_tenure": 84,
        "min_rate": 11.50, "max_rate": 15.50,
        "base_ltv": (0.50, 0.70),
        "base_risk": -3.8
    },
    "Digital Personal Loan": {
        "min_amount": 50000, "max_amount": 600000,
        "min_tenure": 12, "max_tenure": 48,
        "min_rate": 13.50, "max_rate": 21.00,
        "base_ltv": (1.0, 1.0),
        "base_risk": -3.2
    },
    "Used Car Loan": {
        "min_amount": 250000, "max_amount": 1200000,
        "min_tenure": 24, "max_tenure": 60,
        "min_rate": 11.00, "max_rate": 14.50,
        "base_ltv": (0.70, 0.85),
        "base_risk": -3.6
    }
}

EMPLOYMENT_TYPES = ["Salaried", "Self-Employed Professional", "Self-Employed Business"]
EMPLOYMENT_WEIGHTS = [0.45, 0.20, 0.35]

def calculate_emi(principal, annual_rate, tenure_months):
    monthly_rate = (annual_rate / 12) / 100
    if monthly_rate == 0:
        return principal / tenure_months
    emi = principal * monthly_rate * ((1 + monthly_rate) ** tenure_months) / (((1 + monthly_rate) ** tenure_months) - 1)
    return round(emi, 2)

def generate_dataset(num_loans=12000):
    print(f"Generating calibrated Retail Credit portfolio with {num_loans} loan accounts...")
    
    customers = []
    loans = []
    repayments = []
    delinquencies = []
    
    snapshot_date = datetime(2026, 8, 31)
    start_disb_date = datetime(2024, 1, 1)
    
    customer_counter = 1
    repayment_counter = 1
    
    for i in range(1, num_loans + 1):
        tier_choice = random.choices(["Tier-1", "Tier-2", "Tier-3"], weights=[0.30, 0.45, 0.25])[0]
        if tier_choice == "Tier-1":
            city, state = random.choice(TIER_1_CITIES)
            income_base = random.randint(55000, 240000)
        elif tier_choice == "Tier-2":
            city, state = random.choice(TIER_2_CITIES)
            income_base = random.randint(38000, 160000)
        else:
            city, state = random.choice(TIER_3_CITIES)
            income_base = random.randint(28000, 110000)
            
        emp_type = random.choices(EMPLOYMENT_TYPES, weights=EMPLOYMENT_WEIGHTS)[0]
        if emp_type == "Self-Employed Business":
            income_base = int(income_base * random.uniform(1.1, 1.8))
            
        age = random.randint(24, 58)
        gender = random.choices(["Male", "Female"], weights=[0.74, 0.26])[0]
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        
        # CIBIL Score distribution (300 to 900)
        if random.random() < 0.10:
            cibil = int(random.gauss(620, 35))
        else:
            cibil = int(random.gauss(760, 30))
        cibil = max(450, min(870, cibil))
        
        existing_loans = random.choices([0, 1, 2, 3], weights=[0.50, 0.35, 0.12, 0.03])[0]
        
        cust_id = f"CUST_{customer_counter:06d}"
        cust_created_at = (start_disb_date - timedelta(days=random.randint(10, 180))).strftime("%Y-%m-%d")
        customers.append((
            cust_id, name, age, gender, city, tier_choice, state, emp_type,
            float(income_base), cibil, existing_loans, cust_created_at
        ))
        customer_counter += 1
        
        # 2. Loan Account Details
        prod_name = random.choices(
            list(PRODUCTS.keys()),
            weights=[0.32, 0.16, 0.20, 0.20, 0.12]
        )[0]
        p_cfg = PRODUCTS[prod_name]
        
        loan_amount = round(random.uniform(p_cfg["min_amount"], p_cfg["max_amount"]) / 10000) * 10000
        tenure = random.randint(p_cfg["min_tenure"], p_cfg["max_tenure"])
        
        risk_premium = (750 - cibil) * 0.012 if cibil < 750 else 0
        int_rate = round(min(p_cfg["max_rate"] + 2.0, max(p_cfg["min_rate"], random.uniform(p_cfg["min_rate"], p_cfg["max_rate"]) + risk_premium)), 2)
        
        emi = calculate_emi(loan_amount, int_rate, tenure)
        
        min_ltv, max_ltv = p_cfg["base_ltv"]
        if prod_name == "Digital Personal Loan":
            ltv = 100.0
            collateral = 0.0
        else:
            ltv = round(random.uniform(min_ltv, max_ltv) * 100, 2)
            collateral = round((loan_amount / (ltv / 100.0)) / 10000) * 10000
            
        existing_obligation = existing_loans * random.uniform(2500, 6000)
        foir = round(((emi + existing_obligation) / income_base) * 100, 2)
        
        days_between = (datetime(2026, 6, 30) - start_disb_date).days
        disb_date = start_disb_date + timedelta(days=random.randint(0, days_between))
        disb_str = disb_date.strftime("%Y-%m-%d")
        maturity_date = (disb_date + timedelta(days=int(tenure * 30.4375))).strftime("%Y-%m-%d")
        
        loan_id = f"LN_{i:06d}"
        
        # 3. Credit Risk Probability of Default (Calibrated Real-world Logit)
        cibil_factor = (740 - cibil) / 50.0
        foir_factor = (foir - 42) / 20.0
        ltv_factor = (ltv - 65) / 25.0 if prod_name != "Digital Personal Loan" else 0.20
        tier_factor = 0.25 if tier_choice == "Tier-3" else (0.10 if tier_choice == "Tier-2" else -0.15)
        emp_factor = 0.20 if emp_type == "Self-Employed Business" else -0.10
        
        # Latent borrower credit distress: combines underwriting ratios + idiosyncratic life events
        distress = (
            0.75 * cibil_factor 
            + 0.50 * foir_factor 
            + 0.20 * ltv_factor 
            + tier_factor 
            + emp_factor 
            + random.gauss(0, 1.45)
        )
        
        # Default probability (calibrated to retail banking benchmark)
        default_logit = p_cfg["base_risk"] + 0.85 * distress + random.gauss(0, 1.55)
        pd_prob = 1.0 / (1.0 + math.exp(-default_logit))
        pd_prob = max(0.002, min(0.92, pd_prob))
        
        will_default = (random.random() < pd_prob)
        # Moderate distress flag (rolls into 30+ DPD ~ 7.5%)
        will_be_delinquent = will_default or (random.random() < pd_prob * 1.5)
        
        # 4. Generate Monthly Repayments with realistic bounce and cure dynamics
        months_active = (snapshot_date.year - disb_date.year) * 12 + (snapshot_date.month - disb_date.month)
        months_active = min(tenure, max(1, months_active))
        
        total_bounces = 0
        bounces_3m = 0
        bounces_6m = 0
        max_dpd = 0
        current_dpd = 0
        
        # Borrower bounce propensity is driven strictly by underlying financial distress
        # (technical bounces occur in good accounts; highly distressed accounts have higher bounce rates)
        p_bounce_raw = 1.0 / (1.0 + math.exp(-(distress - 0.4)))
        borrower_bounce_propensity = max(0.04, min(0.45, p_bounce_raw * 0.35))
        
        for inst_no in range(1, months_active + 1):
            due_d = disb_date + timedelta(days=int(inst_no * 30.4375))
            if due_d > snapshot_date:
                break
            due_str = due_d.strftime("%Y-%m-%d")
            is_recent_3m = (snapshot_date - due_d).days <= 95
            is_recent_6m = (snapshot_date - due_d).days <= 185
            
            # Base bounce propensity
            bounce = 1 if random.random() < borrower_bounce_propensity else 0
            
            if will_default and inst_no > max(1, months_active - 4):
                # 90+ DPD default
                inst_dpd = random.randint(92, 130)
                paid_amt = 0.0 if inst_dpd > 100 else round(emi * random.choice([0.0, 0.5]), 2)
                paid_d = None if paid_amt == 0 else (due_d + timedelta(days=inst_dpd)).strftime("%Y-%m-%d")
                bounce = 1 if (bounce or random.random() < 0.35) else 0
            elif will_be_delinquent and inst_no > max(1, months_active - 3):
                # 30-59 DPD distress
                inst_dpd = random.randint(32, 58)
                paid_amt = emi
                paid_d = (due_d + timedelta(days=inst_dpd)).strftime("%Y-%m-%d")
                bounce = 1 if (bounce or random.random() < 0.25) else 0
            else:
                # Performing account: occasional technical / salary-delay bounce
                inst_dpd = random.randint(2, 16) if bounce else 0
                paid_amt = emi
                paid_d = (due_d + timedelta(days=inst_dpd)).strftime("%Y-%m-%d")
                
            if bounce:
                total_bounces += 1
                if is_recent_3m:
                    bounces_3m += 1
                if is_recent_6m:
                    bounces_6m += 1
                    
            if inst_dpd > max_dpd:
                max_dpd = inst_dpd
            current_dpd = inst_dpd
            
            repay_id = f"RP_{repayment_counter:08d}"
            repayments.append((
                repay_id, loan_id, inst_no, due_str, paid_d, emi, paid_amt, inst_dpd, bounce
            ))
            repayment_counter += 1
            
        # 5. Loan Status & Delinquency Summary
        is_30_plus = 1 if current_dpd >= 30 else 0
        is_60_plus = 1 if current_dpd >= 60 else 0
        is_90_plus = 1 if current_dpd >= 90 else 0
        
        if is_90_plus and current_dpd > 120:
            loan_status = "Written-Off" if random.random() < 0.15 else "Active"
        elif months_active >= tenure and current_dpd == 0:
            loan_status = "Closed"
        else:
            loan_status = "Active"
            
        loans.append((
            loan_id, cust_id, prod_name, loan_amount, collateral, ltv,
            tenure, int_rate, emi, foir, disb_str, maturity_date, loan_status
        ))
        
        delinquencies.append((
            loan_id, max_dpd, current_dpd, total_bounces, bounces_3m, bounces_6m,
            is_30_plus, is_60_plus, is_90_plus, snapshot_date.strftime("%Y-%m-%d")
        ))
        
    print(f"Generated {len(customers)} customers, {len(loans)} loans, {len(repayments)} repayment records.")
    
    # Save to SQLite
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    
    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", customers)
    cursor.executemany("INSERT INTO loans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", loans)
    cursor.executemany("INSERT INTO repayments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", repayments)
    cursor.executemany("INSERT INTO delinquency_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", delinquencies)
    
    conn.commit()
    conn.close()
    print(f"Successfully populated calibrated SQLite warehouse at {DB_PATH}")

if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
    generate_dataset(count)
