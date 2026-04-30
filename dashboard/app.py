"""
dashboard/app.py  v3.0  -  IndiaCommerce Analytics
Premium analytics platform. No auth dependencies.
Live macro data always visible. Multi-format file upload.
"""
import io, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import core.config as cfg
from data.loader import (
    _clean, _engineer, load_any,
    fetch_worldbank, fetch_usd_inr, fetch_google_trends,
    _WB_GDP, _WB_CPI,
)
from modules.insights        import executive_summary, generate_recommendations
from modules.export          import to_excel, to_pdf
from modules.price_optimizer import run_price_optimizer, plot_price_optimizer
from modules.at_risk         import generate_at_risk_alerts, plot_at_risk
from modules.model_drift     import compute_drift, compute_prediction_drift, plot_drift

st.set_page_config(page_title=cfg.APP_NAME, page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#f1f5f9}
[data-testid="stSidebar"]{background:#0f172a}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
[data-testid="stSidebar"] .stMarkdown h3{color:#a5b4fc!important;font-size:1rem!important}
[data-testid="stSidebar"] hr{border-color:#1e293b!important}
[data-testid="metric-container"]{background:#fff;border:1px solid #cbd5e1;border-radius:12px;
  padding:16px 20px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-size:1.4rem;font-weight:700;color:#0f172a}
[data-testid="metric-container"] [data-testid="stMetricLabel"]{color:#475569;font-weight:600}
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:10px;padding:4px;border:1px solid #cbd5e1;gap:2px}
.stTabs [data-baseweb="tab"]{border-radius:8px;font-weight:500;color:#475569;padding:6px 14px}
.stTabs [aria-selected="true"]{background:#4f46e5!important;color:#fff!important}
.stButton>button[kind="primary"]{background:#4f46e5;border:none;border-radius:8px;font-weight:600;padding:8px 20px}
.stButton>button[kind="primary"]:hover{background:#4338ca}

/* ---- Executive Summary cards ---- */
.card{
  background:#ffffff;
  border:1px solid #cbd5e1;
  border-radius:12px;
  padding:18px 22px;
  margin:8px 0;
  box-shadow:0 2px 6px rgba(0,0,0,.07);
  color:#7f1d1d;
  font-size:.95rem;
  line-height:1.6;
}
.card-blue{
  border-left:5px solid #4f46e5;
  background:#eef2ff;
  color:#1e1b4b;
}
.card-red{
  border-left:5px solid #dc2626;
  background:#fef2f2;
  color:#7f1d1d;
}
.card-green{
  border-left:5px solid #16a34a;
  background:#f0fdf4;
  color:#14532d;
}
.card-amber{
  border-left:5px solid #d97706;
  background:#fffbeb;
  color:#78350f;
}
.card strong, .card b{color:inherit;font-weight:700}
.rec-card{
  background:#ffffff;
  border:1px solid #cbd5e1;
  border-radius:12px;
  padding:18px 22px;
  margin:10px 0;
  box-shadow:0 2px 6px rgba(0,0,0,.07);
}
.rec-card .rec-title{font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:6px}
.rec-card .rec-action{font-size:.95rem;color:#1e293b;margin-bottom:6px}
.rec-card .rec-meta{font-size:.82rem;color:#475569}
.section-title{font-size:1.1rem;font-weight:700;color:#0f172a;margin:20px 0 10px;
  padding-bottom:6px;border-bottom:2px solid #e2e8f0}
.headline-card{
  background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
  color:#ffffff!important;
  border-radius:14px;
  padding:22px 28px;
  margin-bottom:20px;
  font-size:1.15rem;
  font-weight:600;
  line-height:1.5;
  box-shadow:0 4px 16px rgba(79,70,229,.3);
}
</style>""", unsafe_allow_html=True)

#  Cached live data (1 hour TTL) 
@st.cache_data(ttl=3600, show_spinner=False)
def _live_macro():
    return fetch_worldbank(_WB_GDP), fetch_worldbank(_WB_CPI), fetch_usd_inr()

@st.cache_data(ttl=3600, show_spinner=False)
def _live_trends():
    try: return fetch_google_trends(timeframe="today 3-m")
    except: return pd.DataFrame()

@st.cache_data(show_spinner="Processing file...")
def _from_upload(raw: bytes, fname: str) -> pd.DataFrame:
    return load_any(io.BytesIO(raw), fname)

#  SIDEBAR 
with st.sidebar:
    st.markdown(f"### {cfg.APP_NAME}")
    st.caption(f"v{cfg.APP_VERSION}")
    st.markdown("---")

    #  Data source 
    st.markdown("### Data Source")
    df = st.session_state.get("df")

    if df is not None:
        st.success(f"{len(df):,} rows loaded")
        fname = st.session_state.get("fname","dataset")
        st.caption(f"File: {fname}")
        if st.button("Clear & upload new file", use_container_width=True):
            del st.session_state["df"]
            st.session_state.pop("fname", None)
            st.rerun()
    else:
        up = st.file_uploader(
            "Upload dataset",
            type=["csv","tsv","xlsx","xls","json","parquet"],
            help="Supported: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet",
        )
        st.caption(f"[Download Kaggle dataset]({cfg.KAGGLE_URL})")
        if up:
            try:
                df = _from_upload(up.read(), up.name)
                st.session_state["df"]   = df
                st.session_state["fname"] = up.name
                st.success(f"{len(df):,} rows loaded")
                st.rerun()
            except Exception as e:
                st.error(f"Could not parse file: {e}")

    st.markdown("---")

    #  Live data controls 
    st.markdown("### Live Signals")
    if st.button("Refresh live data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption("Auto-refreshes every hour.")

    #  Filters (only when data loaded) 
    if df is not None:
        st.markdown("---")
        st.markdown("### Filters")
        zones  = st.multiselect("Zone",        sorted(df["zone"].unique()),       default=sorted(df["zone"].unique()))
        cats   = st.multiselect("Category",    sorted(df["category"].unique()),   default=sorted(df["category"].unique()))
        brands = st.multiselect("Brand Type",  sorted(df["brand_type"].unique()), default=sorted(df["brand_type"].unique()))
        events = st.multiselect("Sales Event", sorted(df["sales_event"].unique()),default=sorted(df["sales_event"].unique()))

#  HEADER 
st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
  <span style="font-size:2rem"></span>
  <span style="font-size:1.75rem;font-weight:800;color:#1e293b">{cfg.APP_NAME}</span>
</div>
<p style="color:#64748b;font-size:.9rem;margin-bottom:20px">
  <a href="https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/" target="_blank">Live App</a>
  &nbsp;&nbsp;
  <a href="https://github.com/SumedhPatil1507/indian-ecommerce-analytics" target="_blank">GitHub</a>
  &nbsp;&nbsp; World Bank (CC BY 4.0)  fawazahmed0/exchange-api (CC0)  pytrends (Apache 2.0)
</p>
""", unsafe_allow_html=True)

#  LIVE MACRO (always visible, no dataset needed) 
gdp_df, cpi_df, fx = _live_macro()
trends_df = _live_trends()

gdp_val   = float(gdp_df["value"].iloc[-1])  if not gdp_df.empty else None
cpi_val   = float(cpi_df["value"].iloc[-1])  if not cpi_df.empty else None
gdp_delta = round(gdp_df["value"].iloc[-1]-gdp_df["value"].iloc[-2],2) if len(gdp_df)>=2 else None
cpi_delta = round(cpi_df["value"].iloc[-1]-cpi_df["value"].iloc[-2],2) if len(cpi_df)>=2 else None
trend_val = float(trends_df.mean().mean()) if not trends_df.empty else None

c1,c2,c3,c4 = st.columns(4)
c1.metric("India GDP Growth",  f"{gdp_val:.2f}%"      if gdp_val  else "N/A",
          delta=f"{gdp_delta:+.2f}pp" if gdp_delta else None, help="World Bank Open Data (CC BY 4.0)")
c2.metric("CPI Inflation",     f"{cpi_val:.2f}%"      if cpi_val  else "N/A",
          delta=f"{cpi_delta:+.2f}pp" if cpi_delta else None, help="World Bank Open Data (CC BY 4.0)")
c3.metric("USD / INR",         f"Rs{fx:.2f}",          help="fawazahmed0/exchange-api (CC0)")
c4.metric("Search Interest",   f"{trend_val:.1f}/100" if trend_val else "N/A",
          help="Google Trends via pytrends (Apache 2.0)  India  last 3 months")

with st.expander("GDP & CPI history (World Bank)", expanded=False):
    col1,col2 = st.columns(2)
    if not gdp_df.empty:
        col1.plotly_chart(px.line(gdp_df,x="year",y="value",markers=True,
            title="India GDP Growth (%)",template="plotly_white",
            labels={"value":"GDP %","year":"Year"},
            color_discrete_sequence=["#4f46e5"]),use_container_width=True)
    if not cpi_df.empty:
        col2.plotly_chart(px.line(cpi_df,x="year",y="value",markers=True,
            title="India CPI Inflation (%)",template="plotly_white",
            labels={"value":"CPI %","year":"Year"},
            color_discrete_sequence=["#ef4444"]),use_container_width=True)
    st.caption("Source: World Bank Open Data https://data.worldbank.org/country/india (CC BY 4.0)")

if not trends_df.empty:
    with st.expander("Google Trends - E-commerce search interest (India)", expanded=False):
        tr = trends_df.copy(); tr.index.name = "date"
        tp = tr.reset_index().melt(id_vars="date",var_name="keyword",value_name="interest")
        st.plotly_chart(px.line(tp,x="date",y="interest",color="keyword",
            title="Search Interest (0-100)",template="plotly_white"),use_container_width=True)
        st.caption("Source: Google Trends via pytrends https://github.com/GeneralMills/pytrends (Apache 2.0)")

st.markdown("---")

#  NO DATA STATE 
if df is None:
    st.markdown(f"""
    <div style="text-align:center;padding:60px 20px;background:#fff;border-radius:16px;
         border:1px solid #e2e8f0;margin-top:20px">
      <div style="font-size:3rem"></div>
      <h2 style="color:#1e293b;margin:12px 0 8px">Upload your dataset to unlock full analytics</h2>
      <p style="color:#64748b;font-size:1.05rem;max-width:500px;margin:0 auto 20px">
        Upload CSV, Excel, JSON or Parquet via the sidebar.<br>
        Live macro signals above update automatically every hour.
      </p>
      <a href="{cfg.KAGGLE_URL}" target="_blank"
         style="background:#4f46e5;color:#fff;padding:10px 24px;border-radius:8px;
                text-decoration:none;font-weight:600">
        Download Kaggle Dataset
      </a>
    </div>""", unsafe_allow_html=True)
    st.stop()

#  APPLY FILTERS 
dff = df[
    df["zone"].isin(zones) & df["category"].isin(cats) &
    df["brand_type"].isin(brands) & df["sales_event"].isin(events)
].copy()

# Merge live macro into filtered df
if not gdp_df.empty:
    macro = gdp_df[["year","value"]].rename(columns={"value":"india_gdp_growth_pct"})
    macro = macro.merge(
        cpi_df[["year","value"]].rename(columns={"value":"india_cpi_inflation_pct"}),
        on="year", how="outer")
    dff = dff.merge(macro, on="year", how="left")
dff["usd_inr_rate"] = fx
dff["revenue_usd"]  = (dff["revenue"] / fx).round(2)

#  DATASET KPIs 
rev_cr = dff["revenue"].sum() / 1e7
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Revenue",     f"Rs{rev_cr:.1f} Cr")
k2.metric("Total Orders",      f"{len(dff):,}")
k3.metric("Avg Order Value",   f"Rs{dff['revenue'].mean():,.0f}")
k4.metric("Avg Discount",      f"{dff['discount_percent'].mean():.1f}%")
k5.metric("Avg Units / Order", f"{dff['units_sold'].mean():.1f}")
st.caption(f"USD equivalent: **${dff['revenue_usd'].sum():,.0f}**  (@ Rs{fx:.2f}/USD)  |  {len(dff):,} of {len(df):,} orders shown")
st.markdown("---")
#  TABS 
tabs = st.tabs([
    "Executive Summary", "Price Optimizer", "At-Risk Customers", "Model Drift",
    "Revenue Trends", "Categories", "Regional",
    "Inventory", "CLV", "Anomalies", "Cohort", "Pareto",
])

#  TAB 0: Executive Summary 
with tabs[0]:
    summary = executive_summary(dff, fx)
    recs    = generate_recommendations(dff)

    # Headline gradient card
    st.markdown(
        f'<div class="headline-card">{summary["headline"]}</div>',
        unsafe_allow_html=True
    )

    ex1, ex2, _ = st.columns([1, 1, 5])
    with ex1:
        st.download_button(
            "Export Excel", to_excel(dff, summary),
            file_name=f"report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, type="primary",
        )
    with ex2:
        st.download_button(
            "Export PDF", to_pdf(summary, recs),
            file_name=f"report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf", use_container_width=True,
        )

    st.markdown("---")
    cl, cr = st.columns(2)

    with cl:
        st.markdown('<p class="section-title">Key Performance Indicators</p>', unsafe_allow_html=True)
        kpi_df = pd.DataFrame(list(summary["kpis"].items()), columns=["Metric", "Value"])
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

        st.markdown('<p class="section-title">Top Insights</p>', unsafe_allow_html=True)
        for ins in summary["top_insights"]:
            st.markdown(f'<div class="card card-blue">{ins}</div>', unsafe_allow_html=True)

    with cr:
        st.markdown('<p class="section-title">Risks</p>', unsafe_allow_html=True)
        for r in summary["risks"]:
            cls = "card-red" if any(w in r for w in ["Warning","above","declined","High","risk"]) else "card-green"
            st.markdown(f'<div class="card {cls}">{r}</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-title">Opportunities</p>', unsafe_allow_html=True)
        for o in summary["opportunities"]:
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
with tabs[1]:
    st.subheader("Dynamic Price Optimizer")
    st.caption("Computes revenue-maximising discount per category using price elasticity (Lerner index).")
    with st.spinner("Running price optimisation..."):
        price_recs = run_price_optimizer(dff)
    if not price_recs.empty:
        fig_p = plot_price_optimizer(price_recs)
        if fig_p: st.plotly_chart(fig_p, use_container_width=True)
        total_impact = price_recs["revenue_impact_pct"].sum()
        if total_impact > 0:
            st.success(f"Applying all recommendations could improve revenue by **{total_impact:+.1f}%**")
        elif total_impact < 0:
            st.warning(f"Current discounts are above optimal  reducing them could recover **{abs(total_impact):.1f}%** revenue")
        show = ["category","current_discount","optimal_discount","change","direction","elasticity","revenue_impact_pct","rationale"]
        show = [c for c in show if c in price_recs.columns]
        st.dataframe(price_recs[show], use_container_width=True, hide_index=True)
        st.download_button("Export Price Recommendations",
            price_recs.to_csv(index=False).encode(),
            file_name="price_recommendations.csv", mime="text/csv")
    else:
        st.warning("Not enough data to compute elasticity. Upload a larger dataset (500+ rows recommended).")

#  TAB 2: At-Risk Customers 
with tabs[2]:
    st.subheader("At-Risk Customer Automation")
    st.caption("Churn risk scoring using recency, frequency, and monetary signals. Identifies high-value customers at risk of churning.")
    top_n = st.slider("Customers to analyse", 20, 200, 50, 10)
    with st.spinner("Scoring churn risk..."):
        at_risk_df = generate_at_risk_alerts(dff, top_n=top_n)
    if not at_risk_df.empty:
        fig_r1, fig_r2 = plot_at_risk(at_risk_df)
        c1,c2 = st.columns(2)
        if fig_r1: c1.plotly_chart(fig_r1, use_container_width=True)
        if fig_r2: c2.plotly_chart(fig_r2, use_container_width=True)
        critical = at_risk_df[at_risk_df["risk_label"]=="Critical"]
        if not critical.empty:
            st.error(f"{len(critical)} critical high-value customers need immediate outreach")
        show = ["customer_id","churn_risk_score","risk_label","value_tier","days_since_last_order","total_revenue","recommended_action"]
        show = [c for c in show if c in at_risk_df.columns]
        st.dataframe(at_risk_df[show].head(50), use_container_width=True, hide_index=True)
        st.download_button("Export At-Risk List",
            at_risk_df[show].to_csv(index=False).encode(),
            file_name="at_risk_customers.csv", mime="text/csv")
    else:
        st.warning("Not enough customer data for churn scoring.")

#  TAB 3: Model Drift 
with tabs[3]:
    st.subheader("Model Drift Monitoring")
    st.caption("Detects feature distribution shift (PSI) and prediction performance degradation between reference and current windows.")
    c1,c2 = st.columns(2)
    ref_m = c1.slider("Reference window (months)", 3, 12, 6)
    cur_m = c2.slider("Current window (months)",   1, 6,  3)
    with st.spinner("Computing drift..."):
        drift_df   = compute_drift(dff, reference_months=ref_m, current_months=cur_m)
        pred_drift = compute_prediction_drift(dff, reference_months=ref_m, current_months=cur_m)
    if not drift_df.empty:
        fig_d = plot_drift(drift_df)
        if fig_d: st.plotly_chart(fig_d, use_container_width=True)
        drifted = drift_df[drift_df["drift_detected"]]
        if not drifted.empty:
            st.warning(f"{len(drifted)} features show significant drift  model retraining recommended.")
        else:
            st.success("No significant feature drift detected in current window.")
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
    if pred_drift:
        st.markdown("**Prediction Performance**")
        pm1,pm2,pm3,pm4 = st.columns(4)
        pm1.metric("Reference R2",  str(pred_drift.get("ref_r2","N/A")))
        pm2.metric("Current R2",    str(pred_drift.get("cur_r2","N/A")),
                   delta=f"{-pred_drift.get('r2_drop',0):+.3f}" if pred_drift.get("r2_drop") else None,
                   delta_color="inverse")
        pm3.metric("Reference MAE", f"Rs{pred_drift.get('ref_mae',0):,.0f}")
        pm4.metric("Current MAE",   f"Rs{pred_drift.get('cur_mae',0):,.0f}")
        if pred_drift.get("drift_alert"):
            st.error("Model performance has degraded significantly. Retrain with recent data.")
    elif drift_df.empty:
        st.info("Not enough data for drift analysis. Need at least 9 months of data.")

#  TAB 4: Revenue Trends 
with tabs[4]:
    st.subheader("Revenue Trends")
    m = dff.groupby("year_month")["revenue"].sum().reset_index()
    st.plotly_chart(px.line(m,x="year_month",y="revenue",markers=True,
        title="Total Monthly Revenue",labels={"revenue":"Revenue (Rs)","year_month":"Month"},
        template="plotly_white",color_discrete_sequence=["#4f46e5"]),use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.line(dff.groupby("year_month")["revenue"].mean().reset_index(),
        x="year_month",y="revenue",markers=True,title="Avg Order Value",
        labels={"revenue":"AOV (Rs)"},template="plotly_white",color_discrete_sequence=["#22c55e"]),use_container_width=True)
    c2.plotly_chart(px.line(dff.groupby("year_month")["discount_percent"].mean().reset_index(),
        x="year_month",y="discount_percent",markers=True,title="Avg Discount %",
        template="plotly_white",color_discrete_sequence=["#ef4444"]),use_container_width=True)
    st.plotly_chart(px.line(dff.groupby(["year_month","zone"])["revenue"].sum().reset_index(),
        x="year_month",y="revenue",color="zone",markers=True,title="Revenue by Zone",
        template="plotly_white"),use_container_width=True)
    st.plotly_chart(px.line(dff.groupby(["year_month","brand_type"])["revenue"].sum().reset_index(),
        x="year_month",y="revenue",color="brand_type",markers=True,title="Revenue: Mass vs Premium",
        template="plotly_white"),use_container_width=True)

#  TAB 5: Categories 
with tabs[5]:
    st.subheader("Category & Brand Analysis")
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.pie(dff.groupby("category")["revenue"].sum().reset_index(),
        names="category",values="revenue",title="Revenue by Category",hole=0.45,
        template="plotly_white"),use_container_width=True)
    c2.plotly_chart(px.pie(dff.groupby("brand_type")["revenue"].sum().reset_index(),
        names="brand_type",values="revenue",title="Mass vs Premium",hole=0.45,
        template="plotly_white"),use_container_width=True)
    metric = st.selectbox("Metric",["revenue","final_price","units_sold","discount_percent"])
    st.plotly_chart(px.bar(dff.groupby("category")[metric].mean().reset_index().sort_values(metric,ascending=False),
        x="category",y=metric,color="category",title=f"Avg {metric} by Category",
        template="plotly_white",color_discrete_sequence=px.colors.qualitative.Set2),use_container_width=True)
    st.plotly_chart(px.bar(dff.groupby(["year_month","sales_event"])["revenue"].sum().reset_index(),
        x="year_month",y="revenue",color="sales_event",title="Festival vs Normal Revenue",
        template="plotly_white",barmode="group"),use_container_width=True)

#  TAB 6: Regional 
with tabs[6]:
    st.subheader("Regional Analysis")
    st.plotly_chart(px.bar(dff.groupby("state")["revenue"].sum().nlargest(15).reset_index(),
        x="revenue",y="state",orientation="h",title="Top 15 States by Revenue",
        template="plotly_white",color="revenue",color_continuous_scale="Blues"),use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.pie(dff.groupby("zone")["revenue"].sum().reset_index(),
        names="zone",values="revenue",title="Revenue by Zone",hole=0.45,
        template="plotly_white"),use_container_width=True)
    c2.plotly_chart(px.bar(dff.groupby("zone")["units_sold"].mean().reset_index(),
        x="zone",y="units_sold",color="zone",title="Avg Units by Zone",
        template="plotly_white"),use_container_width=True)

#  TAB 7: Inventory 
with tabs[7]:
    st.subheader("Inventory Alert System")
    from modules.inventory_alerts import compute_alerts
    alerts = compute_alerts(dff)
    cmap = {" CRITICAL  Reorder Now":"#ef4444"," HIGH  Monitor Closely":"#f97316",
            " CLEARANCE  Excess Stock":"#eab308"," SLOW MOVER  Review Listing":"#3b82f6"," HEALTHY":"#22c55e"}
    st.plotly_chart(px.scatter(alerts,x="avg_discount",y="avg_units_sold",color="alert_level",
        size="high_pressure_pct",hover_data=["category","zone","recommendation"],
        color_discrete_map=cmap,title="Inventory Alert Dashboard",template="plotly_white"),use_container_width=True)
    af = st.multiselect("Filter alerts",list(cmap.keys()),default=list(cmap.keys()))
    st.dataframe(alerts[alerts["alert_level"].isin(af)]
        [["category","zone","avg_units_sold","avg_discount","alert_level","recommendation"]],
        use_container_width=True,hide_index=True)

#  TAB 8: CLV 
with tabs[8]:
    st.subheader("Customer Lifetime Value")
    from modules.clv import compute_clv
    with st.spinner("Computing CLV..."):
        clv_df = compute_clv(dff)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.histogram(clv_df,x="clv",color="clv_tier",nbins=50,
        title="CLV Distribution",template="plotly_white",marginal="box"),use_container_width=True)
    c2.plotly_chart(px.pie(clv_df.groupby("clv_tier")["clv"].sum().reset_index(),
        names="clv_tier",values="clv",title="CLV Share by Tier",hole=0.45,
        template="plotly_white"),use_container_width=True)
    st.plotly_chart(px.scatter(clv_df,x="frequency",y="clv",color="clv_tier",
        opacity=0.6,size="monetary",title="Frequency vs CLV",template="plotly_white"),use_container_width=True)
    st.dataframe(clv_df.groupby("clv_tier")["clv"].agg(count="count",mean_clv="mean",total_clv="sum").round(2),
        use_container_width=True)

#  TAB 9: Anomalies 
with tabs[9]:
    st.subheader("Anomaly Detection")
    from modules.anomaly import anomaly_report
    with st.spinner("Detecting anomalies..."):
        anom = anomaly_report(dff)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.scatter(anom,x="log_units_sold",y="log_revenue",color="confirmed_anomaly",
        color_discrete_map={True:"#ef4444",False:"#94a3b8"},opacity=0.5,
        title="log(Units) vs log(Revenue)",hover_data=["category","zone","discount_percent"],
        template="plotly_white"),use_container_width=True)
    c2.plotly_chart(px.scatter(anom,x="discount_percent",y="log_final_price",color="confirmed_anomaly",
        color_discrete_map={True:"#ef4444",False:"#e2e8f0"},opacity=0.55,
        title="Discount % vs log(Final Price)",hover_data=["category","zone"],
        template="plotly_white"),use_container_width=True)
    n = anom["confirmed_anomaly"].sum()
    st.info(f"Confirmed anomalies (>=2 detectors): **{n:,}** ({n/len(anom):.2%})")
    st.plotly_chart(px.bar(
        anom[anom["confirmed_anomaly"]].groupby("category").size().reset_index(name="count").sort_values("count",ascending=False),
        x="category",y="count",color="category",title="Anomalies by Category",template="plotly_white"),
        use_container_width=True)

#  TAB 10: Cohort 
with tabs[10]:
    st.subheader("Cohort Analysis")
    from modules.cohort import build_cohort_table
    c1,c2 = st.columns(2)
    pivot = build_cohort_table(dff,metric="count")
    ret = (pivot.div(pivot[0],axis=0)*100).round(1)
    c1.plotly_chart(px.imshow(ret,color_continuous_scale="Blues",title="Retention Rate (%) - Orders",
        labels={"x":"Months since first purchase","y":"Cohort","color":"Retention %"},
        text_auto=".0f",template="plotly_white"),use_container_width=True)
    pivot_r = build_cohort_table(dff,metric="revenue")
    ret_r = (pivot_r.div(pivot_r[0],axis=0)*100).round(1)
    c2.plotly_chart(px.imshow(ret_r,color_continuous_scale="Greens",title="Revenue Retention (%)",
        labels={"x":"Months since first purchase","y":"Cohort","color":"Retention %"},
        text_auto=".0f",template="plotly_white"),use_container_width=True)

#  TAB 11: Pareto 
with tabs[11]:
    st.subheader("Pareto & Concentration")
    agg = dff.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    agg["cum_pct"] = agg["revenue"].cumsum()/agg["revenue"].sum()*100
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=agg["category"],y=agg["revenue"],name="Revenue",marker_color="#4f46e5"),secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["category"],y=agg["cum_pct"],mode="lines+markers",
        name="Cumulative %",line=dict(color="#ef4444",width=2.5)),secondary_y=True)
    fig.add_hline(y=80,line_dash="dash",line_color="#94a3b8",annotation_text="80%",secondary_y=True)
    fig.update_layout(title="Pareto - Revenue by Category (80/20 Rule)",template="plotly_white")
    fig.update_yaxes(title_text="Revenue (Rs)",secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %",secondary_y=True)
    st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.sunburst(
        dff.groupby(["category","zone","brand_type"])["revenue"].sum().reset_index(),
        path=["category","zone","brand_type"],values="revenue",
        title="Revenue Sunburst - Category > Zone > Brand",template="plotly_white",
        color="revenue",color_continuous_scale="Blues"),use_container_width=True)
    vals = np.sort(dff["revenue"].dropna().values)[::-1]
    cum  = np.cumsum(vals)/vals.sum()
    x    = np.linspace(0,1,len(cum))
    gini = round(1-2*np.trapezoid(cum,x),3)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash",color="#94a3b8"),name="Perfect equality"))
    fig2.add_trace(go.Scatter(x=x,y=cum,mode="lines",fill="tozeroy",fillcolor="rgba(79,70,229,.10)",
        line=dict(color="#4f46e5",width=2.5),name=f"Lorenz (Gini={gini})"))
    fig2.update_layout(title=f"Lorenz Curve - Revenue Concentration (Gini={gini})",
        xaxis_title="Cumulative share of orders",yaxis_title="Cumulative share of revenue",
        template="plotly_white")
    st.plotly_chart(fig2,use_container_width=True)

#  FOOTER 
st.markdown("---")
st.caption(
    f"{cfg.APP_NAME} v{cfg.APP_VERSION}  |  "
    "World Bank Open Data (CC BY 4.0)  |  "
    "fawazahmed0/exchange-api (CC0)  |  "
    "pytrends/Google Trends (Apache 2.0)  |  "
    f"Kaggle: {cfg.KAGGLE_URL}"
)
