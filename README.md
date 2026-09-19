# SmartBIU: Autonomous Credit Risk & Portfolio Intelligence Platform

An autonomous credit risk intelligence, portfolio analytics, and AI underwriting platform engineered for **Retail Lending & NBFC Business Intelligence Units (BIU)**.

## Key Features

1. **Realistic Bharat Loan Portfolio (SQLite Warehouse):**
   - 12,000 retail loan accounts and 194,000+ monthly repayment records.
   - Products: *Affordable Housing, Loan Against Property (LAP), Secured MSME, Digital Personal Loans, Used Car Loans*.
   - Geographic coverage: *Tier-1 (Metros), Tier-2 (Emerging urban), Tier-3 (Semi-urban Bharat)*.
2. **Predictive Credit Scorecard & Risk Engine:**
   - Production Gradient Boosting default probability ($PD$) classifier.
   - Rigorous banking evaluation: **ROC-AUC (0.864)**, **Gini Coefficient (0.728)**, **KS-Statistic (56.75% at Decile 3)**, and **Brier Score (0.070)**.
   - Built-in GBDT feature importance identifying top default drivers (Recent Bounces, FOIR, Installment-to-Income, CIBIL).
   - Zero target-leakage calibration: realistic non-default technical bounces and gradual distress progression.
3. **Underwriting Explainer & Adverse Action Codes:**
   - Instant borrower risk tiering (Prime / Watchlist / Subprime).
   - Generates human-interpretable reason codes compliant with RBI fair lending mandates.
4. **Agentic Text-to-SQL Copilot with Dual-Engine Architecture & Self-Healing Loop:**
   - **Dual-Engine:** Supports live LLMs (**Groq / Llama-3.3-70B**, **Google Gemini REST**, **OpenAI / GPT-4o-mini**, or local **Ollama**) with zero-dependency HTTP calls + resilient **Semantic Fallback** for offline environments.
   - Schema linking and few-shot exemplar injection with banking glossary.
   - AST security validation via `sqlglot` (enforces read-only `SELECT` and row bounds).
   - Autonomous self-healing feedback loop featuring LLM reflection upon SQLite errors.
5. **Interactive Executive Dashboard (Streamlit & Plotly):**
   - **Tab 1: Portfolio Executive Overview & Vintage Curves** (AUM, GNPA %, PAR-30, Vintage curves by MOB, Roll-Rate transition matrix).
   - **Tab 2: Underwriting Inspector** (Scorecard, probability of default, adverse action factors).
   - **Tab 3: Conversational Copilot Chat** (Interactive natural language query execution with live LLM / semantic engine toggle, generated SQL, and data tables).
   - **Tab 4: Model Governance & Evals** (KS curve, ROC, Decile gains table, and live agent benchmark runner).

---

## Directory Structure

```
smartbiu-credit-risk-agent/
├── data/
│   └── generate_loan_data.py       # Simulates 12,000+ realistic retail loans (Tier 1/2/3 Bharat)
├── db/
│   ├── schema.sql                  # Star-schema tables (customers, loans, repayments, delinquencies)
│   ├── database.py                 # SQLite connector and schema introspection
│   └── retail_credit.db            # SQLite database warehouse
├── ml/
│   ├── features.py                 # Financial ratios (FOIR, LTV, bounce velocity, WOE/IV)
│   ├── train_model.py              # Scorecard training & decile generation
│   ├── evaluate.py                 # KS-statistic, Gini coefficient, ROC-AUC, Brier score
│   ├── explainer.py                # Adverse action reason code generator
│   └── credit_default_xgb.joblib   # Trained model artifact
├── analytics/
│   └── risk_metrics.py             # SQL analytics: Vintage analysis, Roll rates, PAR-30/60/90
├── agent/
│   ├── llm_client.py               # Unified LLM caller (Groq, Gemini, OpenAI, Ollama)
│   ├── schema_metadata.py          # Semantic data dictionary and few-shot exemplars
│   ├── sql_validator.py            # AST validation and security enforcement using sqlglot
│   ├── biu_agent.py                # Agent core with dual engine & self-healing feedback loop
│   └── evals.py                    # Evaluation benchmark suite (10 test queries, latency & accuracy)
├── app/
│   └── dashboard.py                # Streamlit UI: Executive View, Underwriting Inspector, Copilot Chat
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart Guide

### 1. Installation & Environment Setup
```bash
git clone https://github.com/bivashk/smartbiu.git
cd smartbiu

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Analytics & Scorecard
```bash
# 1. Run Portfolio Risk Analytics (PAR-30/60/90, Roll Rates, Vintages)
python analytics/risk_metrics.py

# 2. Retrain and Evaluate Credit Risk Scorecard
python ml/train_model.py

# 3. Run Agent Text-to-SQL Benchmark Evaluation (10 Test Queries)
python agent/evals.py
```

### 3. Launch the Interactive Dashboard
```bash
streamlit run app/dashboard.py
```
Open your browser at `http://localhost:8501`.

---

## Technical Architecture & Agent Design

### Dual-Engine Text-to-SQL Pipeline
1. **Schema Linking & Domain Injection:** Embeds the SQLite schema, table relationships, and standard banking metric formulas directly into LLM context.
2. **AST Security & Validation (`sqlglot`):** Guarantees zero side effects by permitting only `SELECT`/`WITH` queries, blocking any mutation attempt (`DROP`, `DELETE`, `INSERT`), and automatically enforcing defensible row limits.
3. **Self-Healing Execution Loop with LLM Reflection:** If SQLite raises an operational or syntax error (e.g. column ambiguity, missing joins), the agent intercepts the traceback, re-prompts the LLM with the error context, and re-executes the auto-repaired query.
4. **Resilient Fallback Mode:** When operating offline or without API keys, the agent seamlessly executes deterministic AST queries.

### Credit Scorecard & Regulatory Compliance
- **Discriminatory Separation:** 56.75% KS-statistic at Decile 3 and 0.864 ROC-AUC adhering to Basel II / retail banking standards.
- **Probability Calibration:** 0.070 Brier score ensuring reliable Probability of Default ($PD$) estimates for Expected Loss ($EL$) calculations.
- **Adverse Action Explanations:** Compliant with fair lending mandates by surfacing primary debt-to-income and bureau risk drivers with prescriptive underwriting actions.
