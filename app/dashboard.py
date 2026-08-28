"""
Streamlit Web Application: SmartBIU - Retail Credit Risk & Portfolio Intelligence Platform
Executive Portfolio View, Scorecard Underwriting, Text-to-SQL Agent Chat & Model Governance Evals.
"""

import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analytics.risk_metrics import get_portfolio_kpis, get_product_breakdown, get_tier_breakdown, get_vintage_analysis, get_roll_rate_matrix
from ml.explainer import CreditRiskExplainer
from ml.features import FEATURE_COLUMNS
from agent.biu_agent import SmartBIUAgent
from agent.evals import run_agent_evals

# Page Config
st.set_page_config(
    page_title="SmartBIU | Credit Risk & Portfolio Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3c72;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 5px solid #1e3c72;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .badge-prime {
        background-color: #e6f7ec;
        color: #0e8a16;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-watch {
        background-color: #fff8e1;
        color: #b78103;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-distress {
        background-color: #fde8e8;
        color: #cf222e;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 14px 18px; border-radius: 8px; color: white; margin-bottom: 15px;">
    <h3 style="margin:0; color:white; font-size:1.4rem;">🏦 SmartBIU</h3>
    <p style="margin:3px 0 0 0; font-size:0.85rem; opacity:0.9;">Retail Credit Risk Intelligence</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("### **Portfolio Intelligence Copilot**")
st.sidebar.markdown("**Unit:** Retail Finance & Lending (BIU)")
st.sidebar.markdown("**Focus:** Bharat Lending Portfolio (Tier 1/2/3)")
st.sidebar.markdown("---")

# LLM Engine Configuration
st.sidebar.markdown("⚙️ **Copilot AI Engine Mode:**")
engine_choice = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Semantic Fallback (Offline)", "Groq (Llama-3.3-70B)", "Google Gemini (REST)", "OpenAI (GPT-4o-mini)", "Local Ollama"],
    index=0
)

api_key_override = None
provider_map = {
    "Semantic Fallback (Offline)": "offline",
    "Groq (Llama-3.3-70B)": "groq",
    "Google Gemini (REST)": "gemini",
    "OpenAI (GPT-4o-mini)": "openai",
    "Local Ollama": "ollama"
}
selected_provider = provider_map[engine_choice]

if selected_provider in ["groq", "gemini", "openai"]:
    api_key_override = st.sidebar.text_input(
        f"Enter {selected_provider.upper()} API Key:",
        type="password",
        help="Leave blank to use environment variable if configured"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **System Capabilities:**")
st.sidebar.markdown("- 📊 Portfolio Risk & Vintage Curves")
st.sidebar.markdown("- 🤖 Agentic Text-to-SQL with Self-Healing")
st.sidebar.markdown("- 🔍 AI Scorecard & Adverse Action Reason Codes")
st.sidebar.markdown("- ⚖️ Basel/RBI Model Governance & Evals")
st.sidebar.markdown("---")

# Header
st.markdown('<p class="main-header">SmartBIU — Retail Credit Risk Intelligence Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Autonomous Credit Risk Analytics, Portfolio Diagnostics & Agentic Copilot for Retail Lending</p>', unsafe_allow_html=True)

# Load Portfolio KPIs
kpis = get_portfolio_kpis()

# KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Disbursed AUM", f"₹{kpis['total_aum']/1e7:,.1f} Cr", help="Total active principal disbursed")
with c2:
    st.metric("Gross NPA %", f"{kpis['gnpa_pct']:.2f}%", delta=f"{kpis['npa_count']} loans", delta_color="inverse", help="RBI 90+ DPD mandate")
with c3:
    st.metric("PAR-30 Rate", f"{kpis['par_30_pct']:.2f}%", help="Portfolio At Risk 30+ DPD")
with c4:
    st.metric("Active Accounts", f"{kpis['total_loans']:,}", help="Total serviced borrowers")
with c5:
    st.metric("Avg CIBIL Score", f"{kpis['avg_cibil']}", help="Average portfolio bureau score")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Portfolio & Vintage Curves",
    "🔍 Underwriting & Scorecard Inspector",
    "💬 SmartBIU Conversational Copilot",
    "⚖️ Model Governance & Evals"
])

# ==============================================================================
# TAB 1: EXECUTIVE PORTFOLIO & VINTAGE CURVES
# ==============================================================================
with tab1:
    st.subheader("Retail Loan Book Composition & Delinquency Dynamics")
    
    col_left, col_right = st.columns(2)
    
    # 1. Product Breakdown
    prod_data = get_product_breakdown()
    df_prod = pd.DataFrame(prod_data)
    
    with col_left:
        fig_prod = px.bar(
            df_prod,
            x="product_type",
            y="total_aum",
            color="gnpa_pct",
            color_continuous_scale="Reds",
            title="AUM & Gross NPA % by Product Category",
            labels={"total_aum": "AUM (₹)", "product_type": "Product", "gnpa_pct": "GNPA %"},
            text_auto=False
        )
        fig_prod.update_layout(xaxis_tickangle=-25, margin=dict(t=40, b=40, l=40, r=20))
        st.plotly_chart(fig_prod, use_container_width=True)
        
    # 2. Tier Breakdown
    tier_data = get_tier_breakdown()
    df_tier = pd.DataFrame(tier_data)
    
    with col_right:
        fig_tier = px.pie(
            df_tier,
            names="tier",
            values="total_aum",
            title="Portfolio Exposure Across Bharat (Tiers)",
            hole=0.45,
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )
        fig_tier.update_layout(margin=dict(t=40, b=40, l=40, r=20))
        st.plotly_chart(fig_tier, use_container_width=True)
        
    st.markdown("---")
    
    # 3. Vintage / Cohort Delinquency Curves
    st.subheader("Vintage (Cohort) Analysis: 30+ DPD Progression by Months On Book (MOB)")
    st.caption("Tracks how credit cohorts deteriorate over time. A steep upward curve in recent vintages flags loosening credit policy.")
    
    vintage_data = get_vintage_analysis()
    df_vintage = pd.DataFrame(vintage_data)
    
    fig_vintage = px.line(
        df_vintage,
        x="mob",
        y="delinq_30_plus_pct",
        color="cohort_quarter",
        markers=True,
        title="Vintage Curves: Cumulative 30+ DPD Rate (%) vs Months on Book (MOB)",
        labels={"mob": "Months On Book (MOB)", "delinq_30_plus_pct": "30+ DPD Rate (%)", "cohort_quarter": "Disbursement Cohort"}
    )
    fig_vintage.update_layout(xaxis=dict(tickmode="linear", tick0=3, dtick=3))
    st.plotly_chart(fig_vintage, use_container_width=True)
    
    # 4. Roll-Rate Migration Matrix
    st.subheader("Roll-Rate Transition Probability Matrix")
    st.caption("Transition probability of loan accounts moving between delinquency buckets from one month to the next.")
    
    matrix = get_roll_rate_matrix()
    df_matrix = pd.DataFrame(matrix).fillna(0.0).T
    
    fig_matrix = px.imshow(
        df_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Transition To Bucket", y="Current Bucket", color="Probability (%)"),
        title="Month-over-Month DPD Roll Rates (%)"
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

# ==============================================================================
# TAB 2: UNDERWRITING & SCORECARD INSPECTOR
# ==============================================================================
with tab2:
    st.subheader("Credit Scorecard & Adverse Action Reason Code Generator")
    st.caption("Evaluates individual loan applications using the trained Gradient Boosting model and provides explainable underwriting decisions.")
    
    c_in1, c_in2 = st.columns([2, 1])
    with c_in1:
        selected_loan_id = st.text_input("Enter or Search Loan ID:", value="LN_000025")
    with c_in2:
        st.write("")
        st.write("")
        if st.button("🎲 Pick Random Risky Loan"):
            conn = sqlite3.connect(os.path.join(PROJECT_ROOT, "db", "retail_credit.db"))
            cursor = conn.cursor()
            cursor.execute("SELECT loan_id FROM delinquency_summary WHERE is_90_plus_dpd = 1 ORDER BY RANDOM() LIMIT 1;")
            random_row = cursor.fetchone()
            conn.close()
            if random_row:
                selected_loan_id = random_row[0]
                st.session_state["selected_loan_id"] = selected_loan_id
                
    if "selected_loan_id" in st.session_state:
        selected_loan_id = st.session_state["selected_loan_id"]
        
    explainer = CreditRiskExplainer()
    res = explainer.explain_loan(selected_loan_id)
    
    if "error" in res:
        st.error(res["error"])
    else:
        # Decision Banner
        pd_val = res["probability_of_default_pct"]
        tier = res["risk_tier"]
        badge_class = "badge-prime" if "Low" in tier else ("badge-watch" if "Moderate" in tier else "badge-distress")
        
        st.markdown(f"""
        <div style="background:#f8f9fa; border:1px solid #e1e4e8; border-radius:10px; padding:18px 24px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="margin:0; color:#24292f;">{res['customer_name']} &nbsp; <span style="font-size:0.9rem; color:#6a737d;">({res['loan_id']})</span></h3>
                    <p style="margin:5px 0 0 0; color:#57606a;">Product: <b>{res['product_type']}</b> | City: <b>{res['city_tier']}</b> | Loan Amount: <b>₹{res['loan_amount']:,.0f}</b></p>
                </div>
                <div style="text-align:right;">
                    <span class="{badge_class}" style="font-size:1.1rem;">{tier}</span>
                    <h2 style="margin:8px 0 0 0; color:{'#0e8a16' if pd_val < 5 else ('#b78103' if pd_val < 20 else '#cf222e')};">PD: {pd_val:.2f}%</h2>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Borrower Financial Profile
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("CIBIL Score", res["cibil_score"])
        p2.metric("FOIR (Debt-to-Income)", f"{res['foir']:.1f}%")
        p3.metric("Loan-to-Value (LTV)", f"{res['ltv']:.1f}%")
        p4.metric("Recent 3M Bounces", res["total_bounces"])
        
        st.markdown("### 📋 Adverse Action Risk Drivers & Recommendation")
        st.info(f"**Underwriting Action:** {res['recommendation']}")
        
        for idx, reason in enumerate(res["adverse_action_factors"], 1):
            severity = reason["severity"]
            icon = "🔴" if severity == "HIGH" else ("🟡" if severity == "MEDIUM" else ("🟢" if severity == "POSITIVE" else "🔵"))
            with st.expander(f"{icon} Factor {idx}: {reason['factor']} [{severity}]", expanded=True):
                st.write(f"**Finding:** {reason['detail']}")
                st.write(f"**Prescribed Action:** {reason['action']}")

# ==============================================================================
# TAB 3: SMARTBIU CONVERSATIONAL COPILOT
# ==============================================================================
with tab3:
    st.subheader("SmartBIU Autonomous Conversational Copilot")
    st.caption("Ask natural language portfolio questions or loan diagnostics. The agent performs Schema Linking, AST validation, and auto-corrects SQL upon execution errors.")
    
    agent = SmartBIUAgent(api_key=api_key_override, provider=selected_provider)
    llm_info = agent.llm.get_info()
    
    if llm_info["active"] == "LIVE_LLM":
        st.success(f"🤖 Active Engine: **Live LLM ({llm_info['provider'].upper()} - {llm_info['model']})**")
    else:
        st.info("⚡ Active Engine: **Semantic Fallback (Deterministic Schema-Linked AST)**")
    
    sample_queries = [
        "What is the total AUM and average loan size by product type?",
        "Show me the Gross NPA percentage and total NPA amount across Bharat tiers.",
        "Which top cities have the highest number of bounced installments?",
        "Find the average CIBIL score and average FOIR for defaulted vs non-defaulted loans.",
        "Explain why loan LN_000005 is flagged as risky and give adverse reasons."
    ]
    
    selected_sample = st.selectbox("📌 Or choose a standard BIU query prompt:", [""] + sample_queries)
    
    user_input = st.text_input("Enter your business question in plain English:", value=selected_sample if selected_sample else "")
    
    if st.button("🚀 Run SmartBIU Agent", type="primary"):
        if not user_input.strip():
            st.warning("Please type a question or select a prompt above.")
        else:
            with st.spinner("SmartBIU Agent is reasoning, linking schema, and validating SQL..."):
                response = agent.route_and_execute(user_input)
                
            st.success(f"Agent executed successfully via **{response.get('engine', 'SmartBIU')}** in **{response.get('latency_ms', 0)} ms**!")
            
            if response["query_type"] == "loan_risk_explanation":
                st.markdown("#### 👤 Loan-Level Diagnostic Result")
                st.json(response["data"])
            else:
                # Analytical SQL Result
                st.markdown("#### 💡 Executive Summary")
                st.info(response.get("summary", "Query completed."))
                
                with st.expander("🛠️ Generated & Validated SQL Query", expanded=True):
                    st.code(response.get("final_sql"), language="sql")
                    if response.get("healed"):
                        st.warning(f"⚠️ Query required {response.get('attempts')} self-healing cycles to correct runtime schema syntax.")
                        
                # Data Table
                rows = response.get("rows", [])
                if rows:
                    st.markdown(f"#### 📊 Result Data ({len(rows)} rows)")
                    df_res = pd.DataFrame(rows)
                    st.dataframe(df_res, use_container_width=True)
                    
                    # Auto-chart if numeric columns exist
                    num_cols = df_res.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = df_res.select_dtypes(exclude=[np.number]).columns.tolist()
                    if cat_cols and num_cols:
                        st.markdown("#### 📈 Visual Representation")
                        chart_fig = px.bar(df_res, x=cat_cols[0], y=num_cols[0], title=f"{num_cols[0]} by {cat_cols[0]}")
                        st.plotly_chart(chart_fig, use_container_width=True)

# ==============================================================================
# TAB 4: MODEL GOVERNANCE & EVALS
# ==============================================================================
with tab4:
    st.subheader("Model Governance, Scorecard Performance & Agent Evals")
    st.caption("Quantitative audit metrics adhering to RBI and Basel Committee on Banking Supervision (BCBS) standards.")
    
    meta_path = os.path.join(PROJECT_ROOT, "ml", "model_metadata.joblib")
    if os.path.exists(meta_path):
        import joblib
        meta = joblib.load(meta_path)
        evals = meta["eval_results"]
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{evals['roc_auc']:.4f}", help="Area under ROC Curve")
        m2.metric("Gini Index", f"{evals['gini_coefficient']:.4f}", help="2 * AUC - 1 (Discriminatory power)")
        m3.metric("KS-Statistic", f"{evals['ks_statistic_pct']:.2f}%", help=f"Max separation at Decile {evals['max_ks_decile']}")
        m4.metric("Brier Score", f"{evals['brier_score']:.4f}", help="Probability Calibration Error")
        
        st.markdown("---")
        
        # Decile Scorecard Table
        st.subheader("10-Decile Scorecard Table (Gains & Separation)")
        st.caption("Industry standard for evaluating whether defaults are concentrated in top score deciles.")
        decile_df = pd.DataFrame(evals["decile_table"])
        st.dataframe(decile_df.style.format({
            "min_prob": "{:.4f}",
            "max_prob": "{:.4f}",
            "bad_rate_pct": "{:.2f}%",
            "cum_pct_bads": "{:.2f}%",
            "cum_pct_goods": "{:.2f}%",
            "ks": "{:.2f}%",
            "lift": "{:.2f}x"
        }), use_container_width=True)
        
        # KS Chart
        st.subheader("Kolmogorov-Smirnov (KS) Separation Plot")
        ks_fig = go.Figure()
        ks_fig.add_trace(go.Scatter(x=decile_df["decile"], y=decile_df["cum_pct_bads"], mode="lines+markers", name="Cumulative % Bads", line=dict(color="red", width=3)))
        ks_fig.add_trace(go.Scatter(x=decile_df["decile"], y=decile_df["cum_pct_goods"], mode="lines+markers", name="Cumulative % Goods", line=dict(color="blue", width=3)))
        ks_fig.add_trace(go.Scatter(x=decile_df["decile"], y=decile_df["ks"], mode="lines+markers", name="KS Statistic", line=dict(color="green", dash="dash", width=2)))
        ks_fig.update_layout(title="KS Curve Across Score Deciles", xaxis_title="Score Decile (1 = Highest Risk)", yaxis_title="Cumulative Percentage (%)")
        st.plotly_chart(ks_fig, use_container_width=True)
        
        # Feature Importance
        st.subheader("Top Risk Drivers (Gradient Boosting Feature Importance)")
        imp_df = pd.DataFrame(list(meta["importances"].items()), columns=["Feature", "Importance"]).head(10)
        fig_imp = px.bar(imp_df[::-1], x="Importance", y="Feature", orientation="h", title="Top Predictive Features in Credit Default Model")
        st.plotly_chart(fig_imp, use_container_width=True)
        
    st.markdown("---")
    st.subheader("🤖 Agentic Text-to-SQL Benchmark Evaluation")
    if st.button("Run Live Agent Benchmark Suite"):
        with st.spinner("Running 10 benchmark queries through the agent..."):
            eval_res = run_agent_evals()
            st.success(f"Benchmark finished! Execution Accuracy: **{eval_res['execution_accuracy_pct']:.1f}%**, Avg Latency: **{eval_res['avg_latency_ms']} ms**")
            st.dataframe(pd.DataFrame(eval_res["detailed_results"]), use_container_width=True)
