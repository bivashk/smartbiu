"""
Autonomous BIU Agent with Self-Healing Feedback Loop & Risk Explainer Routing
Translates natural language questions into safe, verified SQL queries and routes loan-level diagnostics.
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import re
import time
import sqlite3
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.schema_metadata import SCHEMA_DOCUMENTATION, FEW_SHOT_EXEMPLARS, BUSINESS_GLOSSARY
from agent.sql_validator import validate_and_sanitize_sql
from agent.llm_client import LLMClient
from ml.explainer import CreditRiskExplainer

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db", "retail_credit.db")

class SmartBIUAgent:
    def __init__(
        self,
        db_path: str = DB_PATH,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.db_path = db_path
        self.explainer = CreditRiskExplainer()
        self.llm = LLMClient(api_key=api_key, provider=provider, model=model)
        
        # Build prompt context for live LLM mode
        exemplars_str = "\n\n".join([
            f"Q: {e['question']}\nSQL:\n{e['sql']}" for e in FEW_SHOT_EXEMPLARS[:5]
        ])
        self.system_prompt = f"""You are the SmartBIU Autonomous SQL Copilot for Retail Finance & Portfolio Risk Intelligence.
Your task is to write accurate, read-only SQLite SQL queries to answer business and risk management questions.

DATABASE SCHEMA:
{SCHEMA_DOCUMENTATION}

FEW-SHOT EXAMPLES:
{exemplars_str}

RULES:
1. Return ONLY valid SQLite SQL.
2. Use aggregations and ROUND(..., 2) for currency and percentage fields.
3. Apply appropriate table aliases (l. for loans, c. for customers, d. for delinquency_summary, r. for repayments).
4. Enforce read-only SELECT or WITH statements.
"""
        
    def route_and_execute(self, user_query: str) -> Dict[str, Any]:
        """
        Main Agent Entrypoint:
        1. Checks for specific Loan ID diagnosis queries -> routes to Credit Risk Explainer.
        2. Routes to Self-Healing Text-to-SQL engine for portfolio analytics.
        """
        start_time = time.time()
        
        # 1. Check for specific Loan ID pattern (e.g., LN_000123)
        loan_match = re.search(r"\b(LN_\d{6})\b", user_query, re.IGNORECASE)
        if loan_match and any(w in user_query.lower() for w in ["explain", "why", "score", "risk", "status", "adverse", "profile"]):
            loan_id = loan_match.group(1).upper()
            explanation = self.explainer.explain_loan(loan_id, db_path=self.db_path)
            latency_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "query_type": "loan_risk_explanation",
                "loan_id": loan_id,
                "data": explanation,
                "latency_ms": latency_ms,
                "engine": "credit_risk_explainer",
                "status": "SUCCESS"
            }
            
        # 2. Portfolio / Analytical Query -> Text-to-SQL with Dual Engine & Self-Correction Loop
        return self._execute_text_to_sql(user_query, start_time)
        
    def _execute_text_to_sql(self, user_query: str, start_time: float) -> Dict[str, Any]:
        """
        Autonomous Text-to-SQL pipeline featuring:
        - Dual-Engine Generation (Live LLM with few-shot schema linking OR Semantic Fallback)
        - AST Security & Validation (sqlglot)
        - Self-Healing feedback loop with LLM reflection upon error
        """
        query_history = []
        max_attempts = 3
        
        # Generate initial SQL: First try live LLM if active, else fall back to semantic pattern
        engine_used = "SEMANTIC_FALLBACK"
        current_sql = None
        if self.llm.is_available():
            llm_sql = self.llm.generate_sql(user_query, self.system_prompt)
            if llm_sql:
                current_sql = llm_sql
                engine_used = f"LIVE_LLM ({self.llm.provider}:{self.llm.model})"
                
        if not current_sql:
            current_sql = self._generate_sql(user_query)
        
        for attempt in range(1, max_attempts + 1):
            query_history.append({"attempt": attempt, "sql": current_sql, "engine": engine_used})
            
            # Step A: Validate AST and Security
            is_valid, sanitized_sql, val_err = validate_and_sanitize_sql(current_sql)
            if not is_valid:
                # Self-Correction: try LLM reflection first, then heuristic syntax healer
                healed = None
                if self.llm.is_available():
                    healed = self.llm.heal_sql(current_sql, f"AST Validation Error: {val_err}", SCHEMA_DOCUMENTATION, user_query)
                current_sql = healed if healed else self._heal_sql_syntax(current_sql, val_err, user_query)
                continue
                
            # Step B: Execute against SQLite
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(sanitized_sql)
                rows = [dict(r) for r in cursor.fetchall()]
                conn.close()
                
                # Execution Succeeded!
                latency_ms = round((time.time() - start_time) * 1000, 2)
                summary_text = self._synthesize_summary(user_query, rows, sanitized_sql)
                
                return {
                    "query_type": "analytical_sql",
                    "user_query": user_query,
                    "final_sql": sanitized_sql,
                    "rows": rows,
                    "row_count": len(rows),
                    "attempts": attempt,
                    "healed": attempt > 1,
                    "engine": engine_used,
                    "llm_info": self.llm.get_info(),
                    "query_history": query_history,
                    "latency_ms": latency_ms,
                    "summary": summary_text,
                    "status": "SUCCESS"
                }
                
            except sqlite3.OperationalError as db_err:
                err_msg = str(db_err)
                # Self-Healing: Feed SQLite error back to LLM reflection or heuristic healer
                healed = None
                if self.llm.is_available():
                    healed = self.llm.heal_sql(current_sql, err_msg, SCHEMA_DOCUMENTATION, user_query)
                current_sql = healed if healed else self._heal_sql_runtime_error(current_sql, err_msg, user_query)
                
            except Exception as e:
                err_msg = str(e)
                healed = None
                if self.llm.is_available():
                    healed = self.llm.heal_sql(current_sql, err_msg, SCHEMA_DOCUMENTATION, user_query)
                current_sql = healed if healed else self._heal_sql_runtime_error(current_sql, err_msg, user_query)
                
        # If all attempts exhausted
        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "query_type": "analytical_sql",
            "user_query": user_query,
            "final_sql": current_sql,
            "status": "FAILED",
            "error": "Exceeded maximum self-healing attempts.",
            "attempts": max_attempts,
            "engine": engine_used,
            "llm_info": self.llm.get_info(),
            "query_history": query_history,
            "latency_ms": latency_ms
        }
        
    def _generate_sql(self, query: str) -> str:
        """
        Translates natural language to SQL using semantic pattern resolution
        and few-shot schema linking.
        """
        q = query.lower()
        
        # Pattern: Product level breakdown / AUM / ticket size
        if any(w in q for w in ["product", "product-wise", "product type"]) and any(w in q for w in ["aum", "volume", "count", "disbursed", "ticket"]):
            return """
SELECT 
    product_type,
    COUNT(loan_id) AS total_loans,
    ROUND(SUM(loan_amount), 2) AS total_aum,
    ROUND(AVG(loan_amount), 2) AS avg_ticket_size,
    ROUND(AVG(interest_rate), 2) AS avg_interest_rate
FROM loans
GROUP BY product_type
ORDER BY total_aum DESC;
            """.strip()
            
        # Pattern: NPA / Gross NPA across Bharat tiers
        if any(w in q for w in ["tier", "geography", "bharat"]) and any(w in q for w in ["npa", "delinquent", "default", "par", "risk"]):
            return """
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
            
        # Pattern: Product-wise NPA rate
        if any(w in q for w in ["product", "product-wise"]) and any(w in q for w in ["npa", "default rate", "gnpa", "bad loan"]):
            return """
SELECT 
    l.product_type,
    COUNT(l.loan_id) AS loan_count,
    ROUND(SUM(l.loan_amount), 2) AS total_aum,
    ROUND(SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN l.loan_amount ELSE 0 END), 2) AS gnpa_amount,
    ROUND(100.0 * SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) / SUM(l.loan_amount), 2) AS gnpa_pct,
    ROUND(100.0 * SUM(CASE WHEN d.is_30_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) / SUM(l.loan_amount), 2) AS par_30_pct
FROM loans l
JOIN delinquency_summary d ON l.loan_id = d.loan_id
GROUP BY l.product_type
ORDER BY gnpa_pct DESC;
            """.strip()

        # Pattern: Top cities with bounces or defaults
        if "city" in q or "cities" in q:
            return """
SELECT 
    c.city,
    c.tier,
    c.state,
    COUNT(l.loan_id) AS loan_count,
    SUM(d.total_bounces) AS total_bounces,
    ROUND(100.0 * SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN 1 ELSE 0 END) / COUNT(l.loan_id), 2) AS default_rate_pct
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
JOIN delinquency_summary d ON l.loan_id = d.loan_id
GROUP BY c.city, c.tier, c.state
ORDER BY total_bounces DESC
LIMIT 10;
            """.strip()

        # Pattern: Vintage analysis / Cohort
        if "vintage" in q or "cohort" in q or "mob" in q:
            return """
WITH LoanCohorts AS (
    SELECT 
        loan_id,
        strftime('%Y', disbursement_date) || '-Q' || ((CAST(strftime('%m', disbursement_date) AS INTEGER) + 2) / 3) AS cohort_quarter
    FROM loans
)
SELECT 
    lc.cohort_quarter,
    r.installment_no AS mob,
    COUNT(DISTINCT r.loan_id) AS accounts,
    ROUND(100.0 * SUM(CASE WHEN r.dpd >= 30 THEN 1 ELSE 0 END) / COUNT(r.loan_id), 2) AS delinq_30_plus_pct
FROM repayments r
JOIN LoanCohorts lc ON r.loan_id = lc.loan_id
WHERE r.installment_no IN (3, 6, 9, 12)
GROUP BY lc.cohort_quarter, r.installment_no
ORDER BY lc.cohort_quarter, mob;
            """.strip()

        # Pattern: CIBIL or FOIR comparison
        if any(w in q for w in ["cibil", "foir", "score", "income"]) and any(w in q for w in ["default", "npa", "performing"]):
            return """
SELECT 
    CASE WHEN d.is_90_plus_dpd = 1 THEN 'Defaulted (NPA)' ELSE 'Standard (Performing)' END AS account_status,
    COUNT(l.loan_id) AS account_count,
    ROUND(AVG(c.cibil_score), 1) AS avg_cibil,
    ROUND(AVG(l.foir), 2) AS avg_foir,
    ROUND(AVG(l.ltv), 2) AS avg_ltv,
    ROUND(AVG(c.monthly_income), 2) AS avg_monthly_income
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
JOIN delinquency_summary d ON l.loan_id = d.loan_id
GROUP BY d.is_90_plus_dpd;
            """.strip()

        # Pattern: List recent bounce / high risk accounts
        if any(w in q for w in ["high risk", "bounced", "delinquent accounts", "watch list"]):
            return """
SELECT 
    l.loan_id,
    c.name,
    l.product_type,
    c.city,
    c.tier,
    l.loan_amount,
    d.bounces_last_3m,
    d.current_dpd,
    l.foir,
    c.cibil_score
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
JOIN delinquency_summary d ON l.loan_id = d.loan_id
WHERE d.bounces_last_3m > 1 OR d.current_dpd >= 30
ORDER BY d.current_dpd DESC, d.bounces_last_3m DESC
LIMIT 20;
            """.strip()

        # Default fallback: General portfolio summary
        return """
SELECT 
    COUNT(l.loan_id) AS total_loans,
    ROUND(SUM(l.loan_amount), 2) AS total_disbursed_aum,
    ROUND(AVG(l.loan_amount), 2) AS avg_ticket_size,
    ROUND(AVG(l.interest_rate), 2) AS avg_interest_rate,
    ROUND(100.0 * SUM(CASE WHEN d.is_90_plus_dpd = 1 THEN l.loan_amount ELSE 0 END) / SUM(l.loan_amount), 2) AS gnpa_pct
FROM loans l
JOIN delinquency_summary d ON l.loan_id = d.loan_id;
        """.strip()

    def _heal_sql_syntax(self, failed_sql: str, val_err: str, query: str) -> str:
        """Heals queries failing AST parsing (e.g. trailing semicolons, missing clauses)."""
        cleaned = re.sub(r";+\s*$", "", failed_sql).strip()
        if "select" not in cleaned.lower():
            return f"SELECT * FROM loans LIMIT 20;"
        return cleaned

    def _heal_sql_runtime_error(self, failed_sql: str, db_err: str, query: str) -> str:
        """
        Self-Healing Engine:
        Analyzes SQLite runtime error (e.g. 'no such column', 'ambiguous column')
        and reconstructs valid joins and column aliases.
        """
        healed_sql = failed_sql
        
        # Case 1: Ambiguous column name (e.g. 'loan_id' without table qualifier)
        if "ambiguous column name" in db_err.lower():
            col = db_err.split(":")[-1].strip()
            # Prefix with 'l.' for loans table
            healed_sql = re.sub(rf"\b{col}\b", f"l.{col}", healed_sql)
            return healed_sql
            
        # Case 2: Missing table JOIN (e.g. referencing customer or delinquency column without JOIN)
        if "no such column" in db_err.lower():
            missing_col = db_err.split(":")[-1].strip().split(".")[-1]
            if missing_col in ["tier", "cibil_score", "monthly_income", "city", "state", "employment_type"]:
                if "JOIN customers" not in healed_sql:
                    healed_sql = healed_sql.replace("FROM loans", "FROM loans l JOIN customers c ON l.customer_id = c.customer_id")
            elif missing_col in ["is_90_plus_dpd", "is_30_plus_dpd", "current_dpd", "total_bounces", "bounces_last_3m"]:
                if "JOIN delinquency_summary" not in healed_sql:
                    healed_sql = healed_sql.replace("FROM loans", "FROM loans l JOIN delinquency_summary d ON l.loan_id = d.loan_id")
            return healed_sql
            
        # Case 3: Syntax error near token -> Fallback to safe semantic template
        return self._generate_sql(query)

    def _synthesize_summary(self, user_query: str, rows: List[Dict[str, Any]], sql: str) -> str:
        """Generates executive insight summary from returned database rows."""
        if not rows:
            return "The query executed successfully but returned 0 records matching the criteria."
            
        row_count = len(rows)
        first_row = rows[0]
        
        if "product_type" in first_row and "total_aum" in first_row:
            top_prod = rows[0]
            return f"Retrieved metrics across {row_count} loan products. **{top_prod['product_type']}** represents the largest AUM segment at ₹{top_prod['total_aum']:,.2f}."
            
        if "tier" in first_row and "gnpa_pct" in first_row:
            highest_risk_tier = max(rows, key=lambda x: x.get("gnpa_pct", 0))
            return f"Portfolio risk segmented across {row_count} tiers. **{highest_risk_tier['tier']}** exhibits the highest Gross NPA at **{highest_risk_tier['gnpa_pct']}%**."
            
        if "total_disbursed_aum" in first_row:
            return f"Overall portfolio AUM stands at ₹{first_row['total_disbursed_aum']:,.2f} across {first_row.get('total_loans', 0):,} loans with Gross NPA at {first_row.get('gnpa_pct', 0)}%."
            
        return f"Query returned {row_count} record(s) matching your analytical criteria."

if __name__ == "__main__":
    agent = SmartBIUAgent()
    
    print("--- Test 1: Portfolio Breakdown Query ---")
    res1 = agent.route_and_execute("Show me product-wise AUM and average loan ticket size")
    print("Status:", res1["status"])
    print("SQL:", res1.get("final_sql"))
    print("Summary:", res1.get("summary"))
    
    print("\n--- Test 2: Loan Level Diagnosis ---")
    res2 = agent.route_and_execute("Explain why loan LN_000010 has risk")
    print("Status:", res2["status"])
    print("Risk Tier:", res2["data"].get("risk_tier"))
    print("Adverse Factors:", res2["data"].get("adverse_action_factors"))
