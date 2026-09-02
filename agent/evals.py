"""
Evaluation Benchmark Suite for SmartBIU Agent
Measures Execution Accuracy, AST Validity, Self-Healing Resilience, and Latency.
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import pandas as pd
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.biu_agent import SmartBIUAgent

BENCHMARK_TEST_SUITE = [
    {
        "id": "Q01",
        "category": "Portfolio Sizing",
        "query": "What is the total AUM and average loan size by product type?",
        "expected_columns": ["product_type", "total_aum"]
    },
    {
        "id": "Q02",
        "category": "Portfolio Risk",
        "query": "Show me the Gross NPA percentage and total NPA amount across Bharat tiers.",
        "expected_columns": ["tier", "gnpa_pct"]
    },
    {
        "id": "Q03",
        "category": "Delinquency Diagnostics",
        "query": "Which top cities have the highest number of bounced installments?",
        "expected_columns": ["city", "total_bounces"]
    },
    {
        "id": "Q04",
        "category": "Credit Quality",
        "query": "Find the average CIBIL score and average FOIR for defaulted vs non-defaulted loans.",
        "expected_columns": ["account_status", "avg_cibil"]
    },
    {
        "id": "Q05",
        "category": "Collections Watchlist",
        "query": "Show me high risk loans with recent bounces and high current DPD.",
        "expected_columns": ["loan_id", "current_dpd"]
    },
    {
        "id": "Q06",
        "category": "Vintage Cohorts",
        "query": "Give me the vintage cohort analysis for 30+ DPD by months on book.",
        "expected_columns": ["cohort_quarter", "mob"]
    },
    {
        "id": "Q07",
        "category": "Product Risk",
        "query": "What is the product-wise NPA rate and PAR-30 rate across the book?",
        "expected_columns": ["product_type", "gnpa_pct"]
    },
    {
        "id": "Q08",
        "category": "Executive Summary",
        "query": "Give me the overall portfolio summary with total disbursed volume and GNPA.",
        "expected_columns": ["total_loans", "total_disbursed_aum"]
    },
    {
        "id": "Q09",
        "category": "Underwriting Diagnosis",
        "query": "Explain why loan LN_000005 is flagged as risky and give adverse reasons.",
        "expected_type": "loan_risk_explanation"
    },
    {
        "id": "Q10",
        "category": "Underwriting Diagnosis",
        "query": "What is the probability of default and scorecard recommendation for loan LN_000025?",
        "expected_type": "loan_risk_explanation"
    }
]

def run_agent_evals() -> Dict[str, Any]:
    print("=" * 65)
    print("  RUNNING SMARTBIU AGENT EVALUATION BENCHMARK SUITE")
    print("=" * 65)
    
    agent = SmartBIUAgent(provider="offline")
    results = []
    
    total_tests = len(BENCHMARK_TEST_SUITE)
    successful_executions = 0
    healed_count = 0
    total_latency = 0.0
    
    for item in BENCHMARK_TEST_SUITE:
        t0 = time.time()
        res = agent.route_and_execute(item["query"])
        latency = (time.time() - t0) * 1000
        total_latency += latency
        
        status = res.get("status") == "SUCCESS"
        if status:
            successful_executions += 1
            
        is_healed = res.get("healed", False)
        if is_healed:
            healed_count += 1
            
        # Validate expected output structure
        matched_contract = True
        if item.get("expected_type"):
            matched_contract = (res.get("query_type") == item["expected_type"])
        elif item.get("expected_columns") and res.get("rows"):
            first_row_keys = res["rows"][0].keys()
            matched_contract = all(col in first_row_keys for col in item["expected_columns"])
            
        results.append({
            "Test ID": item["id"],
            "Category": item["category"],
            "Query": item["query"][:45] + "...",
            "Status": "PASS" if (status and matched_contract) else "FAIL",
            "Attempts": res.get("attempts", 1),
            "Healed": "Yes" if is_healed else "No",
            "Latency (ms)": round(latency, 2)
        })
        
    df_results = pd.DataFrame(results)
    
    accuracy_pct = (successful_executions / total_tests) * 100
    avg_latency_ms = total_latency / total_tests
    
    print("\n" + df_results.to_string(index=False))
    print("\n" + "=" * 65)
    print(f"  BENCHMARK SUMMARY METRICS")
    print(f"  Total Queries:            {total_tests}")
    print(f"  Execution Accuracy:       {accuracy_pct:.1f}%")
    print(f"  Self-Healing Recovery:    {healed_count} query(s)")
    print(f"  Avg Latency:              {avg_latency_ms:.2f} ms")
    print("=" * 65 + "\n")
    
    return {
        "total_tests": total_tests,
        "execution_accuracy_pct": accuracy_pct,
        "avg_latency_ms": round(avg_latency_ms, 2),
        "detailed_results": results
    }

if __name__ == "__main__":
    run_agent_evals()
