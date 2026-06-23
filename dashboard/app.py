"""
dashboard/app.py  v5.0  -  IndiaCommerce Analytics (Stateless Viewer)
Coupled entirely to FastAPI REST engine and Celery background workers.
"""
import io
import os
import sys

# Initialize sys.path with project root so local modules are visible
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import time
import logging
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config variables
import core.config as cfg
from data.connectors import (
    SHOPIFY_MOCK_SCHEMA, AMAZON_MOCK_SCHEMA, WOOCOMMERCE_MOCK_SCHEMA,
    SIMULATION_CONFIG, generate_simulation, validate_schema,
)

# API Endpoint Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title=cfg.APP_NAME, page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

# CSS Styling (Identical custom theme)
CSS = """
<style>
html,body,[class*="css"],.stApp,.main,.block-container,
p,span,div,label,li,td,th,h1,h2,h3,h4,h5,h6,
.stMarkdown,.stMarkdown p,.stMarkdown span,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
.stSelectbox label,.stMultiSelect label,
.stSlider label,.stFileUploader label,
.stCheckbox label,.stRadio label,
.stExpander summary,.stExpander p,
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stCaptionContainer"] p {
  color:#0f172a !important;
}
[data-testid="stAppViewContainer"]{background:#f1f5f9 !important}
.main .block-container{padding-top:1.5rem}
[data-testid="stSidebar"]{background:#0f172a !important}
[data-testid="stSidebar"] *,[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div{color:#e2e8f0 !important}
[data-testid="stSidebar"] h3{color:#a5b4fc !important;font-size:1rem !important}
[data-testid="stSidebar"] hr{border-color:#1e293b !important}
[data-testid="stSidebar"] .stButton>button{
  background:#1e293b !important;color:#e2e8f0 !important;
  border:1px solid #334155 !important;border-radius:8px !important}
[data-testid="stSidebar"] .stButton>button:hover{background:#334155 !important}
[data-testid="metric-container"]{
  background:#ffffff !important;border:1px solid #cbd5e1 !important;
  border-radius:12px !important;padding:16px 20px !important;
  box-shadow:0 2px 8px rgba(0,0,0,.08) !important}
[data-testid="stMetricValue"]{font-size:1.5rem !important;font-weight:700 !important;color:#0f172a !important}
[data-testid="stMetricLabel"]{font-size:.85rem !important;font-weight:600 !important;color:#475569 !important}
.stTabs [data-baseweb="tab-list"]{
  background:#ffffff !important;border-radius:10px !important;
  padding:4px !important;border:1px solid #cbd5e1 !important;gap:2px !important}
.stTabs [data-baseweb="tab"]{
  border-radius:8px !important;font-weight:600 !important;
  color:#334155 !important;padding:6px 14px !important}
.stTabs [aria-selected="true"]{background:#4f46e5 !important;color:#ffffff !important}
.stButton>button[kind="primary"]{
  background:#4f46e5 !important;border:none !important;
  border-radius:8px !important;font-weight:600 !important;
  color:#ffffff !important;padding:8px 20px !important}
.stButton>button[kind="primary"]:hover{background:#4338ca !important}
.stButton>button[kind="secondary"]{
  border:1px solid #cbd5e1 !important;color:#0f172a !important;
  border-radius:8px !important;font-weight:500 !important}
[data-testid="stDataFrame"] td,[data-testid="stDataFrame"] th{color:#0f172a !important;font-size:.9rem !important}
[data-testid="stExpander"]{
  background:#ffffff !important;border:1px solid #e2e8f0 !important;border-radius:10px !important}
[data-testid="stExpander"] summary{color:#0f172a !important;font-weight:600 !important}
[data-testid="stCaptionContainer"] p{color:#475569 !important;font-size:.82rem !important}
.card{
  background:#ffffff;border:1px solid #cbd5e1;border-radius:12px;
  padding:18px 22px;margin:8px 0;box-shadow:0 2px 6px rgba(0,0,0,.07);
  color:#0f172a;font-size:.95rem;line-height:1.6}
.card-blue{border-left:5px solid #4f46e5;background:#eef2ff;color:#1e1b4b}
.card-red{border-left:5px solid #dc2626;background:#fef2f2;color:#7f1d1d}
.card-green{border-left:5px solid #16a34a;background:#f0fdf4;color:#14532d}
.card-amber{border-left:5px solid #d97706;background:#fffbeb;color:#78350f}
.card strong,.card b{color:inherit;font-weight:700}
.rec-card{
  background:#ffffff;border:1px solid #cbd5e1;border-radius:12px;
  padding:18px 22px;margin:10px 0;box-shadow:0 2px 6px rgba(0,0,0,.07)}
.rec-card .rec-title{font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:6px}
.rec-card .rec-action{font-size:.95rem;color:#1e293b;margin-bottom:6px}
.rec-card .rec-meta{font-size:.82rem;color:#475569}
.section-title{
  font-size:1.1rem;font-weight:700;color:#0f172a;
  margin:20px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}
.headline-card{
  background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
  color:#ffffff !important;border-radius:14px;padding:22px 28px;
  margin-bottom:20px;font-size:1.15rem;font-weight:600;line-height:1.5;
  box-shadow:0 4px 16px rgba(79,70,229,.3)}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --- REST Helper Methods ---

def _get_headers() -> dict:
    token = st.session_state.get("access_token", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_get(endpoint: str, params: dict = None) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}{endpoint}", headers=_get_headers(), params=params, timeout=15)
        if resp.status_code == 401:
            st.session_state.pop("access_token", None)
            st.rerun()
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception as e:
        logger.error("API GET %s failed: %s", endpoint, e)
        return None


def api_post(endpoint: str, json_data: dict = None, files: dict = None) -> dict | None:
    try:
        if files:
            resp = requests.post(f"{API_URL}{endpoint}", headers=_get_headers(), files=files, timeout=60)
        else:
            resp = requests.post(f"{API_URL}{endpoint}", headers=_get_headers(), json=json_data, timeout=15)
        if resp.status_code == 401:
            st.session_state.pop("access_token", None)
            st.rerun()
        if resp.status_code in (200, 201):
            return resp.json()
        return None
    except Exception as e:
        logger.error("API POST %s failed: %s", endpoint, e)
        return None


def wait_for_task(task_id: str):
    """Render a spinner and block until a background Celery task completes."""
    with st.spinner("Processing computation on background worker..."):
        while True:
            res = api_get(f"/tasks/status/{task_id}")
            if not res:
                st.error("Error connecting to Celery task monitor")
                break
            state = res.get("status")
            if state == "SUCCESS":
                st.success("Task completed successfully!")
                time.sleep(1.5)
                break
            elif state in ("FAILURE", "REVOKED"):
                st.error(f"Task execution failed: {res.get('result')}")
                break
            time.sleep(1)


# --- USER AUTHENTICATION SCREEN ---

if not st.session_state.get("access_token"):
    st.markdown("<h2 style='text-align: center;'>Welcome to IndiaCommerce Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Stateless Multi-Tenant Relational Insights Platform</p>", unsafe_allow_html=True)
    
    auth_mode = st.tabs(["Login", "Sign Up"])
    
    with auth_mode[0]:
        with st.form("login_form"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", type="primary")
            if submitted:
                res = api_post("/auth/login", json_data={"email": email, "password": password})
                if res and "access_token" in res:
                    st.session_state["access_token"] = res["access_token"]
                    st.session_state["email"] = res["email"]
                    st.session_state["user_id"] = res["user_id"]
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Authentication failed. Invalid email or password.")
                    
    with auth_mode[1]:
        with st.form("signup_form"):
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                res = api_post("/auth/signup", json_data={"email": new_email, "password": new_password})
                if res and res.get("status") == "success":
                    st.success("Registration complete! You can now log in.")
                else:
                    st.error("Failed to register tenant.")
    st.stop()


# --- LOGGED IN STATE ---

# Load profile
profile = api_get("/profile")
tenant_id = st.session_state["user_id"]
tenant_email = st.session_state["email"]

# Load database orders metadata
kpis = api_get("/analytics/kpis") or {}
total_orders = kpis.get("total_orders", 0)

# Load macro signals (exchangerate, world bank, google trends)
# (Decoupled macro fetching handled inside API)
macro_res = api_get("/health")  # checks connection


# --- SIDEBAR & INGESTION ---

with st.sidebar:
    st.markdown(f"### {cfg.APP_NAME}")
    st.caption(f"Authenticated as: **{tenant_email}**")
    st.caption(f"Tenant ID: `{tenant_id}`")
    st.markdown("---")

    # Ingestion Controls
    st.markdown("### Ingest Transaction Data")
    if total_orders > 0:
        st.success(f"{total_orders:,} transaction rows loaded in PostgreSQL")
        if st.button("Ingest new file", use_container_width=True):
            # Simulated clear (doesn't wipe db, just allows uploading another file)
            total_orders = 0
            st.rerun()
    else:
        up = st.file_uploader(
            "Upload dataset",
            type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
            help="Supported: CSV, TSV, Excel, JSON, Parquet"
        )
        if up:
            try:
                # Upload directly to API
                files = {"file": (up.name, up.read())}
                res = api_post("/orders/upload", files=files)
                if res and "task_id" in res:
                    wait_for_task(res["task_id"])
                    st.rerun()
                else:
                    st.error("Failed to upload orders.")
            except Exception as e:
                st.error(f"Upload error: {e}")

    st.markdown("---")
    
    # Recalculation commands (Triggers worker tasks)
    st.markdown("### Recalculate Operations")
    col1, col2 = st.columns(2)
    if col1.button("Pricing", use_container_width=True):
        res = api_post("/analytics/price-optimizer/recalculate")
        if res and "task_id" in res:
            wait_for_task(res["task_id"])
            st.rerun()
            
    if col2.button("CLV & Churn", use_container_width=True):
        res = api_post("/analytics/clv/recalculate")
        if res and "task_id" in res:
            wait_for_task(res["task_id"])
            st.rerun()

    col3, col4 = st.columns(2)
    if col3.button("Anomalies", use_container_width=True):
        res = api_post("/analytics/anomalies/recalculate")
        if res and "task_id" in res:
            wait_for_task(res["task_id"])
            st.rerun()
            
    if col4.button("Model Drift", use_container_width=True):
        res = api_post("/analytics/drift/recalculate")
        if res and "task_id" in res:
            wait_for_task(res["task_id"])
            st.rerun()

    st.markdown("---")
    if st.button("Sign Out", type="primary", use_container_width=True):
        st.session_state.pop("access_token", None)
        st.rerun()


# --- HEADER & KEY KPIs ---

st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
  <span style="font-size:1.75rem;font-weight:800;color:#1e293b">{cfg.APP_NAME}</span>
</div>
<p style="color:#64748b;font-size:.9rem;margin-bottom:20px">
  Multi-tenant Relational DB Mode | Isolated via Row-Level Security | Distributed Celery Workers
</p>
""", unsafe_allow_html=True)


# Render primary DB metrics
if total_orders > 0:
    rev_cr = kpis.get("total_revenue", 0.0) / 1e7
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Revenue", f"Rs{rev_cr:.1f} Cr")
    k2.metric("Total Orders", f"{kpis.get('total_orders', 0):,}")
    k3.metric("Avg Order Value", f"Rs{kpis.get('aov', 0):,.0f}")
    k4.metric("Avg Discount", f"{kpis.get('avg_discount', 0):.1f}%")
    k5.metric("Avg Units / Order", f"{kpis.get('avg_units_sold', 0):.1f}")
    st.markdown("---")
else:
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px;background:#fff;border-radius:16px;
         border:1px solid #e2e8f0;margin-top:20px">
      <h2 style="color:#1e293b;margin:12px 0 8px">Ingest a dataset via the sidebar to get started</h2>
      <p style="color:#64748b;font-size:1.05rem;max-width:500px;margin:0 auto 20px">
        Upload order logs in CSV, Parquet, or Excel format. Or run simulated sandbox data inside the Data Connector Matrix.
      </p>
    </div>""", unsafe_allow_html=True)


# --- TABS (15 total, including Observability) ---

tabs_names = [
    "Data Connector Matrix", "Executive Summary", "Price Optimizer", 
    "At-Risk Customers", "Model Drift", "Revenue Trends", "Categories", 
    "Regional", "Inventory", "CLV", "Anomalies", "Cohort", "Pareto", 
    "Operational Actions", "Observability (OpenTelemetry)"
]

tabs = st.tabs(tabs_names)


# --- TAB 0: DATA CONNECTOR MATRIX ---

with tabs[0]:
    st.subheader("Data Connector Matrix")
    st.caption("Secure cryptographic webhooks. Payloads are verified using tenant signature secrets.")
    col_sh, col_am, col_woo, col_sim = st.columns(4)
    with col_sh:
        st.markdown('<div class="card card-blue"><strong>Shopify Webhook</strong><br><small>POST /ingest/shopify</small></div>', unsafe_allow_html=True)
    with col_am:
        st.markdown('<div class="card card-blue"><strong>Amazon Ingestion</strong><br><small>POST /ingest/amazon</small></div>', unsafe_allow_html=True)
    with col_woo:
        st.markdown('<div class="card card-blue"><strong>WooCommerce Webhook</strong><br><small>POST /ingest/woocommerce</small></div>', unsafe_allow_html=True)
    with col_sim:
        st.markdown('<div class="card card-amber"><strong>Simulation Sandbox</strong><br><small>Celery task queuing</small></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    connector = st.selectbox("Select connector info", ["Shopify Webhooks", "Amazon Ingestion", "WooCommerce Webhooks", "Simulation Sandbox"])
    
    # Show tenant secret and endpoint paths
    tenant_sec = profile.get("webhook_secret") if profile else "loading..."
    
    if connector == "Shopify Webhooks":
        st.markdown("<p class='section-title'>Shopify Webhook Cryptographic Details</p>", unsafe_allow_html=True)
        st.info(f"Endpoint: `POST {API_URL}/ingest/shopify?tenant_id={tenant_id}`")
        st.info(f"Verification Secret (add to Shopify settings): `{tenant_sec}`")
        st.json(SHOPIFY_MOCK_SCHEMA["sample"])
    elif connector == "WooCommerce Webhooks":
        st.markdown("<p class='section-title'>WooCommerce Webhook Cryptographic Details</p>", unsafe_allow_html=True)
        st.info(f"Endpoint: `POST {API_URL}/ingest/woocommerce?tenant_id={tenant_id}`")
        st.info(f"Verification Secret (add to WooCommerce settings): `{tenant_sec}`")
        st.json(WOOCOMMERCE_MOCK_SCHEMA["sample"])
    elif connector == "Amazon Ingestion":
        st.markdown("<p class='section-title'>Amazon Ingestion Cryptographic Details</p>", unsafe_allow_html=True)
        st.info(f"Endpoint: `POST {API_URL}/ingest/amazon?tenant_id={tenant_id}`")
        st.info(f"Verification Secret (add to Amazon config): `{tenant_sec}`")
        st.json(AMAZON_MOCK_SCHEMA["sample"])
    elif connector == "Simulation Sandbox":
        st.markdown('<div class="card card-amber"><strong>Simulation Sandbox</strong> - Generates synthetic macro-calibrated transaction logs.</div>', unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns(3)
        sim_rows = sc1.slider("Rows", 500, 10000, 3000, 500)
        sim_months = sc2.slider("Months of history", 6, 36, 24)
        sim_seed = sc3.number_input("Seed", value=42, step=1)
        
        if st.button("Generate & Ingest Simulation Dataset", type="primary"):
            with st.spinner("Generating and posting to relational database..."):
                # Call connector local function, convert to CSV, post to REST /orders/upload
                sim_df = generate_simulation(int(sim_rows), int(sim_months), 0.07, 0.05, 84.0, int(sim_seed))
                csv_buf = io.BytesIO()
                sim_df.to_csv(csv_buf, index=False)
                csv_buf.seek(0)
                
                files = {"file": ("simulation.csv", csv_buf.read())}
                res = api_post("/orders/upload", files=files)
                if res and "task_id" in res:
                    wait_for_task(res["task_id"])
                    st.rerun()


# Stop tab rendering if database is empty
if total_orders == 0:
    st.stop()


# --- TAB 1: EXECUTIVE SUMMARY ---

with tabs[1]:
    summary_data = api_get("/analytics/executive-summary")
    if summary_data:
        summary = summary_data.get("summary", {})
        recs = summary_data.get("recommendations", [])
        
        # Headline card
        st.markdown(f'<div class="headline-card">{summary.get("headline", "")}</div>', unsafe_allow_html=True)
        
        ex1, ex2, _ = st.columns([1, 1, 5])
        with ex1:
            st.markdown(
                f'<a href="{API_URL}/analytics/export/excel" target="_blank" style="text-decoration:none;">'
                f'<button style="background-color:#4f46e5;color:white;padding:8px 16px;border:none;border-radius:8px;font-weight:600;width:100%;cursor:pointer;">'
                f'Export Excel</button></a>',
                unsafe_allow_html=True
            )
        with ex2:
            st.markdown(
                f'<a href="{API_URL}/analytics/export/pdf" target="_blank" style="text-decoration:none;">'
                f'<button style="border:1px solid #cbd5e1;color:#0f172a;padding:8px 16px;border-radius:8px;font-weight:500;width:100%;cursor:pointer;">'
                f'Export PDF</button></a>',
                unsafe_allow_html=True
            )
            
        st.markdown("---")
        cl, cr = st.columns(2)
        
        with cl:
            st.markdown('<p class="section-title">Key Performance Indicators</p>', unsafe_allow_html=True)
            kpi_df = pd.DataFrame(list(summary.get("kpis", {}).items()), columns=["Metric", "Value"])
            st.dataframe(kpi_df, use_container_width=True, hide_index=True)
            
            st.markdown('<p class="section-title">Top Insights</p>', unsafe_allow_html=True)
            for ins in summary.get("top_insights", []):
                st.markdown(f'<div class="card card-blue">{ins}</div>', unsafe_allow_html=True)
                
        with cr:
            st.markdown('<p class="section-title">Risks</p>', unsafe_allow_html=True)
            for r in summary.get("risks", []):
                cls = "card-red" if any(w in r for w in ["Warning", "above", "declined", "High", "risk"]) else "card-green"
                st.markdown(f'<div class="card {cls}">{r}</div>', unsafe_allow_html=True)
                
            st.markdown('<p class="section-title">Opportunities</p>', unsafe_allow_html=True)
            for o in summary.get("opportunities", []):
                st.markdown(f'<div class="card card-green">{o}</div>', unsafe_allow_html=True)
                
        st.markdown("---")
        st.markdown('<p class="section-title">Prioritised Recommendations</p>', unsafe_allow_html=True)
        _p_colours = {
            "High":   ("#dc2626", "#fef2f2"),
            "Medium": ("#d97706", "#fffbeb"),
            "Low":    ("#16a34a", "#f0fdf4"),
        }
        for rec in recs:
            key = "High" if "High" in rec["priority"] else ("Medium" if "Medium" in rec["priority"] else "Low")
            bc, bg = _p_colours[key]
            st.markdown(
                f'<div class="rec-card" style="border-left:5px solid {bc};background:{bg}">'
                f'<div class="rec-title">{rec["priority"]} | {rec["category"]}</div>'
                f'<div class="rec-action">{rec["action"]}</div>'
                f'<div class="rec-meta">Impact: <strong>{rec["impact"]}</strong>'
                f' &nbsp;|&nbsp; Effort: <strong>{rec["effort"]}</strong>'
                f' &nbsp;|&nbsp; Metric: <strong>{rec["metric"]}</strong></div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# --- TAB 2: PRICE OPTIMIZER ---

with tabs[2]:
    st.subheader("Dynamic Price Optimizer")
    st.caption("Price elasticity of demand (Lerner Index) computed via backend statsmodels OLS.")
    
    price_data = api_get("/analytics/price-optimizer")
    if not price_data:
        st.info("No cached pricing recommendations available. Trigger pricing recalculation via the sidebar.")
    else:
        price_recs = pd.DataFrame(price_data)
        
        # Plot optimal discount bar chart
        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            name="Current Discount",
            x=price_recs["category"],
            y=price_recs["current_discount"],
            marker_color="#94a3b8",
        ))
        fig_p.add_trace(go.Bar(
            name="Optimal Discount",
            x=price_recs["category"],
            y=price_recs["optimal_discount"],
            marker_color=["#22c55e" if d == "decrease" else "#ef4444" if d == "increase" else "#4f46e5"
                          for d in price_recs["direction"]],
        ))
        fig_p.update_layout(
            barmode="group",
            title="Current vs Optimal Discount by Category",
            xaxis_title="Category",
            yaxis_title="Discount %",
            template="plotly_white",
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_p, use_container_width=True)
        
        total_impact = price_recs["revenue_impact_pct"].sum()
        if total_impact > 0:
            st.success(f"Applying recommendations can yield **+{total_impact:.1f}%** revenue impact.")
        st.dataframe(price_recs, use_container_width=True, hide_index=True)


# --- TAB 3: AT-RISK CUSTOMERS ---

with tabs[3]:
    st.subheader("At-Risk Customer Automation")
    st.caption("RFM Churn scoring database cache output.")
    
    at_risk_data = api_get("/analytics/at-risk")
    if not at_risk_data:
        st.info("No cached customer risk metrics available. Trigger CLV & Churn scoring via the sidebar.")
    else:
        at_risk_df = pd.DataFrame(at_risk_data)
        
        c1, c2 = st.columns(2)
        # Churn score distribution
        c1.plotly_chart(px.histogram(
            at_risk_df, x="churn_risk_score", color="risk_label",
            title="Churn Risk Score Distribution", template="plotly_white"
        ), use_container_width=True)
        
        # Risk label share
        c2.plotly_chart(px.pie(
            at_risk_df, names="risk_label", values="total_revenue",
            title="Revenue Exposure by Churn Risk", hole=0.4, template="plotly_white"
        ), use_container_width=True)
        
        st.dataframe(at_risk_df, use_container_width=True, hide_index=True)


# --- TAB 4: MODEL DRIFT ---

with tabs[4]:
    st.subheader("Model Drift Monitoring")
    st.caption("Population Stability Index (PSI) values computed asynchronously.")
    
    drift_data = api_get("/analytics/drift")
    if not drift_data or drift_data.get("status") == "not_computed":
        st.info("No data drift monitoring reports found. Trigger drift recalculation via sidebar.")
    else:
        st.markdown("**Drift Performance Parameters**")
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Features Drifted", drift_data.get("features_drifted", 0))
        pm2.metric("Max PSI", round(drift_data.get("max_psi", 0.0), 3))
        pm3.metric("R2 prediction drop", f"{drift_data.get('pred_r2_drop', 0.0):.3f}")
        
        if drift_data.get("drift_alert"):
            st.error("Significant data distribution shift detected. Model retraining is highly recommended.")
        else:
            st.success("Feature metrics are within normal drift bounds.")


# --- TAB 5: REVENUE TRENDS ---

with tabs[5]:
    st.subheader("Revenue Trends")
    trends = api_get("/analytics/charts/revenue-trends")
    if trends:
        m = pd.DataFrame(trends["monthly_revenue"])
        avg_disc = pd.DataFrame(trends["monthly_discount"])
        zone_rev = pd.DataFrame(trends["zone_revenue"])
        brand_rev = pd.DataFrame(trends["brand_revenue"])
        
        st.plotly_chart(px.line(
            m, x="year_month", y="revenue", markers=True,
            title="Total Monthly Revenue", labels={"revenue": "Revenue (Rs)"},
            template="plotly_white", color_discrete_sequence=["#4f46e5"]
        ), use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(
            avg_disc, x="year_month", y="discount_percent", markers=True,
            title="Average Discount Trend", template="plotly_white", color_discrete_sequence=["#ef4444"]
        ), use_container_width=True)
        
        c2.plotly_chart(px.line(
            zone_rev, x="year_month", y="revenue", color="zone", markers=True,
            title="Revenue Share by Zone", template="plotly_white"
        ), use_container_width=True)


# --- TAB 6: CATEGORIES ---

with tabs[6]:
    st.subheader("Category & Brand Analysis")
    cats_data = api_get("/analytics/charts/category-analysis")
    if cats_data:
        cat_df = pd.DataFrame(cats_data["category_revenue"])
        brand_df = pd.DataFrame(cats_data["brand_revenue"])
        event_df = pd.DataFrame(cats_data["event_revenue"])
        
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(
            cat_df, names="category", values="revenue", title="Revenue by Category",
            hole=0.4, template="plotly_white"
        ), use_container_width=True)
        
        c2.plotly_chart(px.pie(
            brand_df, names="brand_type", values="revenue", title="Mass vs Premium Mix",
            hole=0.4, template="plotly_white"
        ), use_container_width=True)
        
        st.plotly_chart(px.bar(
            event_df, x="year_month", y="revenue", color="sales_event",
            title="Festival vs Normal Revenue", template="plotly_white", barmode="group"
        ), use_container_width=True)


# --- TAB 7: REGIONAL ---

with tabs[7]:
    st.subheader("Regional Analysis")
    reg_data = api_get("/analytics/charts/regional-analysis")
    if reg_data:
        state_df = pd.DataFrame(reg_data["state_revenue"])
        zone_df = pd.DataFrame(reg_data["zone_revenue"])
        units_df = pd.DataFrame(reg_data["zone_units"])
        
        st.plotly_chart(px.bar(
            state_df, x="revenue", y="state", orientation="h",
            title="Top 15 States by Revenue", template="plotly_white",
            color="revenue", color_continuous_scale="Blues"
        ), use_container_width=True)


# --- TAB 8: INVENTORY ---

with tabs[8]:
    st.subheader("Inventory Alert System")
    inv_data = api_get("/analytics/charts/inventory-alerts")
    if inv_data:
        alerts = pd.DataFrame(inv_data)
        cmap = {
            "CRITICAL - Reorder Now":    "#ef4444",
            "HIGH - Monitor Closely":    "#f97316",
            "CLEARANCE - Excess Stock":  "#eab308",
            "SLOW MOVER - Review Listing":"#3b82f6",
            "HEALTHY":                   "#22c55e",
        }
        st.plotly_chart(px.scatter(
            alerts, x="avg_discount", y="avg_units_sold", color="alert_level",
            size="high_pressure_pct", hover_data=["category", "zone", "recommendation"],
            color_discrete_map=cmap, title="Inventory Alert Scatter Map", template="plotly_white"
        ), use_container_width=True)
        
        st.dataframe(alerts, use_container_width=True, hide_index=True)


# --- TAB 9: CLV ---

with tabs[9]:
    st.subheader("Customer Lifetime Value (BG/NBD)")
    clv_data = api_get("/analytics/clv")
    if not clv_data or isinstance(clv_data, dict) and clv_data.get("status") == "not_computed":
        st.info("No CLV calculations found. Trigger CLV & Churn via sidebar.")
    elif isinstance(clv_data, list):
        clv_df = pd.DataFrame(clv_data)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(
            clv_df, x="clv", color="clv_tier", title="CLV Histogram Distribution",
            template="plotly_white"
        ), use_container_width=True)
        
        c2.plotly_chart(px.scatter(
            clv_df, x="frequency", y="clv", color="clv_tier", size="monetary",
            title="Frequency vs CLV", template="plotly_white"
        ), use_container_width=True)


# --- TAB 10: ANOMALIES ---

with tabs[10]:
    st.subheader("Anomaly Detection (Isolation Forest)")
    anom_data = api_get("/analytics/anomalies")
    if not anom_data or isinstance(anom_data, dict) and anom_data.get("status") == "not_computed":
        st.info("No anomaly computations found. Trigger anomalies calculations via the sidebar.")
    elif isinstance(anom_data, list):
        anom_df = pd.DataFrame(anom_data)
        st.plotly_chart(px.scatter(
            anom_df, x="log_units_sold", y="log_revenue", color="confirmed_anomaly",
            color_discrete_map={True: "#ef4444", False: "#94a3b8"}, opacity=0.5,
            title="log(Units Sold) vs log(Revenue)", template="plotly_white"
        ), use_container_width=True)


# --- TAB 11: COHORT ---

with tabs[11]:
    st.subheader("Cohort Analysis Matrix")
    cohort_data = api_get("/analytics/charts/cohort-analysis")
    if cohort_data:
        c1, c2 = st.columns(2)
        
        cc = cohort_data["cohort_counts"]
        cc_df = pd.DataFrame(cc["values"], index=cc["index"], columns=cc["columns"])
        # retention pct
        cc_pct = (cc_df.div(cc_df.iloc[:, 0], axis=0) * 100).round(1)
        c1.plotly_chart(px.imshow(
            cc_pct, color_continuous_scale="Blues", title="Retention Rate (%) - Orders",
            text_auto=".0f", template="plotly_white"
        ), use_container_width=True)
        
        cr = cohort_data["cohort_revenue"]
        cr_df = pd.DataFrame(cr["values"], index=cr["index"], columns=cr["columns"])
        cr_pct = (cr_df.div(cr_df.iloc[:, 0], axis=0) * 100).round(1)
        c2.plotly_chart(px.imshow(
            cr_pct, color_continuous_scale="Greens", title="Revenue Retention Rate (%)",
            text_auto=".0f", template="plotly_white"
        ), use_container_width=True)


# --- TAB 12: PARETO ---

with tabs[12]:
    st.subheader("Pareto Analysis & Lorenz Curve")
    pareto_data = api_get("/analytics/charts/pareto-analysis")
    if pareto_data:
        agg = pd.DataFrame(pareto_data["pareto_categories"])
        lorenz = pd.DataFrame(pareto_data["lorenz_curve"])
        gini = pareto_data.get("lorenz_gini", 0.0)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=agg["category"], y=agg["revenue"], name="Revenue", marker_color="#4f46e5"), secondary_y=False)
        fig.add_trace(go.Scatter(x=agg["category"], y=agg["cum_pct"], mode="lines+markers", name="Cumulative %", line=dict(color="#ef4444", width=2.5)), secondary_y=True)
        fig.update_layout(title="Pareto Category Contribution", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#94a3b8"), name="Equality Line"))
        fig2.add_trace(go.Scatter(x=lorenz["x"], y=lorenz["y"], mode="lines", fill="tozeroy", name=f"Lorenz (Gini={gini})", line=dict(color="#4f46e5", width=2.5)))
        fig2.update_layout(title=f"Lorenz Curve (Gini Coefficient = {gini})", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)


# --- TAB 13: OPERATIONAL ACTIONS ---

with tabs[13]:
    st.subheader("Operational Actions")
    st.caption("Approve pricing recommendations or audit past worker logs.")
    
    act1, act2 = st.columns(2)
    with act1:
        st.markdown('<p class="section-title">Approve Price Optimisation</p>', unsafe_allow_html=True)
        recs = api_get("/analytics/price-optimizer")
        if recs:
            st.dataframe(pd.DataFrame(recs)[["category", "current_discount", "optimal_discount", "direction", "revenue_impact_pct"]].head(10), use_container_width=True, hide_index=True)
            if st.button("Approve All Price Adjustments", type="primary", use_container_width=True):
                api_post("/operational-actions", json_data={
                    "action_type": "price_approval",
                    "payload": {"count": len(recs), "items": recs},
                    "status": "approved"
                })
                st.success("Pricing adjustments logged as approved.")
        else:
            st.info("No pricing adjustments pending approval.")
            
    with act2:
        st.markdown('<p class="section-title">Export At-Risk Cohort</p>', unsafe_allow_html=True)
        alerts = api_get("/analytics/at-risk")
        if alerts:
            st.metric("Total high-risk customers", len(alerts))
            if st.button("Log Export to Audits", use_container_width=True):
                api_post("/operational-actions", json_data={
                    "action_type": "at_risk_export",
                    "payload": {"exported_count": len(alerts)},
                    "status": "exported"
                })
                st.success("Export event logged successfully.")
        else:
            st.info("No at-risk cohort alerts generated.")
            
    st.markdown("---")
    st.markdown('<p class="section-title">Pending Actions Log</p>', unsafe_allow_html=True)
    pending = api_get("/operational-actions/pending")
    if pending:
        pending_df = pd.DataFrame(pending)
        st.dataframe(pending_df[["id", "action_type", "status", "source", "created_at"]], use_container_width=True, hide_index=True)
        
        col_id, col_stat = st.columns(2)
        aid = col_id.number_input("Action ID to update", min_value=1, step=1)
        ns = col_stat.selectbox("New status", ["approved", "dismissed", "exported"])
        if st.button("Update Status"):
            api_post("/operational-actions/update", json_data={"action_id": int(aid), "status": ns})
            st.success(f"Status of action {aid} updated to {ns}")
            st.rerun()
    else:
        st.info("No pending actions in operational logs.")


# --- TAB 14: OBSERVABILITY (OPENTELEMETRY) ---

with tabs[14]:
    st.subheader("System Observability (OpenTelemetry)")
    st.caption("Real-time telemetry retrieved from FastAPI instrumentation.")
    
    telemetry = api_get("/analytics/telemetry")
    if telemetry:
        # KPI metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("API Requests Recorded", f"{telemetry.get('requests_total', 0)} / 1000")
        m2.metric("Average Response Latency", f"{telemetry.get('avg_latency_ms', 0.0):.1f} ms")
        m3.metric("Active Celery Worker Status", telemetry.get("celery", {}).get("status", "unknown").upper())
        m4.metric("Active Worker Tasks", telemetry.get("celery", {}).get("active_tasks", 0))
        
        st.markdown("---")
        
        # Requests breakdown
        paths = telemetry.get("path_distribution", {})
        if paths:
            fig_tel = px.bar(
                x=list(paths.keys()), y=list(paths.values()),
                title="API Request Count by Route", labels={"x": "Route Path", "y": "Requests Count"},
                template="plotly_white"
            )
            st.plotly_chart(fig_tel, use_container_width=True)
    else:
        st.error("Failed to query OpenTelemetry endpoints.")