"""
dashboard/app.py  v3.0  -  IndiaCommerce Analytics
Production-grade: Supabase auth, Price Optimizer, At-Risk Automation, Model Drift.
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

from core.config   import settings
from core.auth     import render_auth_wall, current_user, logout
from core.database import save_dataset, load_dataset, log_event
from data.loader   import (
    _clean, _engineer, load_any,
    fetch_worldbank, fetch_usd_inr, fetch_google_trends,
    _WB_GDP, _WB_CPI,
)
from modules.insights       import executive_summary, generate_recommendations
from modules.export         import to_excel, to_pdf
from modules.price_optimizer import run_price_optimizer, plot_price_optimizer
from modules.at_risk         import generate_at_risk_alerts, plot_at_risk
from modules.model_drift     import compute_drift, compute_prediction_drift, save_drift_report, plot_drift

st.set_page_config(page_title=settings.app_name, page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#f8f9fc}
[data-testid="stSidebar"]{background:#1a1f36}
[data-testid="stSidebar"] *{color:#e2e8f0!important}
[data-testid="stSidebar"] hr{border-color:#2d3561!important}
[data-testid="metric-container"]{background:#fff;border:1px solid #e8ecf4;border-radius:12px;padding:16px 20px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
.stTabs [data-baseweb="tab-list"]{background:#fff;border-radius:10px;padding:4px;border:1px solid #e8ecf4}
.stTabs [data-baseweb="tab"]{border-radius:8px;font-weight:500;color:#64748b}
.stTabs [aria-selected="true"]{background:#4f46e5!important;color:#fff!important}
.stButton>button[kind="primary"]{background:#4f46e5;border:none;border-radius:8px;font-weight:600}
.insight-card{background:#fff;border-left:4px solid #4f46e5;border-radius:8px;padding:14px 18px;margin:8px 0;box-shadow:0 1px 4px rgba(0,0,0,.06);font-size:.95rem}
.risk-card{background:#fff8f8;border-left:4px solid #ef4444;border-radius:8px;padding:14px 18px;margin:8px 0}
.opp-card{background:#f0fdf4;border-left:4px solid #22c55e;border-radius:8px;padding:14px 18px;margin:8px 0}
.rec-card{background:#fff;border:1px solid #e8ecf4;border-radius:10px;padding:16px 20px;margin:10px 0;box-shadow:0 1px 4px rgba(0,0,0,.05)}
</style>""", unsafe_allow_html=True)

if not render_auth_wall():
    st.stop()
user = current_user()

@st.cache_data(ttl=3600, show_spinner=False)
def _live_macro():
    return fetch_worldbank(_WB_GDP), fetch_worldbank(_WB_CPI), fetch_usd_inr()

@st.cache_data(ttl=3600, show_spinner=False)
def _live_trends():
    try: return fetch_google_trends(timeframe="today 3-m")
    except: return pd.DataFrame()

@st.cache_data(show_spinner="Processing file...")
def _from_upload(raw: bytes, fname: str):
    return load_any(io.BytesIO(raw), fname)

@st.cache_data(show_spinner="Loading...")
def _from_path(p: str):
    return _engineer(_clean(pd.read_csv(p)))

with st.sidebar:
    st.markdown(f"### {settings.app_name}")
    st.markdown(f"<small>v{settings.app_version}</small>", unsafe_allow_html=True)
    if user.get("demo"):
        st.info("Demo mode - configure Supabase for full auth.")
    else:
        st.markdown(f"Signed in: **{user.get('email','')}**")
    if st.button("Sign Out", use_container_width=True):
        logout()
    st.markdown("---")
    st.subheader("Data Source")
    df = None
    if not user.get("demo") and "df" not in st.session_state:
        saved = load_dataset(user.get("id","demo"))
        if saved is not None:
            st.session_state["df"] = _engineer(_clean(saved))
            st.success("Loaded your saved dataset.")
    if "df" in st.session_state:
        df = st.session_state["df"]
        st.success(f"{len(df):,} rows loaded")
        if st.button("Clear dataset", use_container_width=True):
            del st.session_state["df"]; st.rerun()
    else:
        up = st.file_uploader("Upload dataset",
             type=["csv","tsv","xlsx","xls","json","parquet"],
             help="CSV, TSV, Excel, JSON, Parquet")
        st.caption(f"[Download from Kaggle]({settings.kaggle_url})")
        if up:
            try:
                df = _from_upload(up.read(), up.name)
                st.session_state["df"] = df
                save_dataset(user.get("id","demo"), df)
                log_event(user.get("id","demo"), "upload", up.name)
                st.success(f"{len(df):,} rows loaded")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    st.markdown("---")
    st.subheader("Live Signals")
    if st.button("Refresh live data", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption("Auto-refreshes hourly.")
    if df is not None:
        st.markdown("---")
        st.subheader("Filters")
        zones  = st.multiselect("Zone",        sorted(df["zone"].unique()),       default=sorted(df["zone"].unique()))
        cats   = st.multiselect("Category",    sorted(df["category"].unique()),   default=sorted(df["category"].unique()))
        brands = st.multiselect("Brand Type",  sorted(df["brand_type"].unique()), default=sorted(df["brand_type"].unique()))
        events = st.multiselect("Sales Event", sorted(df["sales_event"].unique()),default=sorted(df["sales_event"].unique()))

st.markdown('<p style="font-size:1.8rem;font-weight:700;color:#1a1f36;margin-bottom:2px">📊 IndiaCommerce Analytics</p>', unsafe_allow_html=True)
st.caption("Live: https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/ | GitHub: https://github.com/SumedhPatil1507/indian-ecommerce-analytics")

gdp_df, cpi_df, fx = _live_macro()
trends_df = _live_trends()
gdp_val   = float(gdp_df["value"].iloc[-1])  if not gdp_df.empty else None
cpi_val   = float(cpi_df["value"].iloc[-1])  if not cpi_df.empty else None
gdp_delta = round(gdp_df["value"].iloc[-1]-gdp_df["value"].iloc[-2],2) if len(gdp_df)>=2 else None
cpi_delta = round(cpi_df["value"].iloc[-1]-cpi_df["value"].iloc[-2],2) if len(cpi_df)>=2 else None
trend_val = float(trends_df.mean().mean()) if not trends_df.empty else None

c1,c2,c3,c4 = st.columns(4)
c1.metric("India GDP Growth",  f"{gdp_val:.2f}%"      if gdp_val  else "N/A", delta=f"{gdp_delta:+.2f}pp" if gdp_delta else None, help="World Bank CC BY 4.0")
c2.metric("CPI Inflation",     f"{cpi_val:.2f}%"      if cpi_val  else "N/A", delta=f"{cpi_delta:+.2f}pp" if cpi_delta else None, help="World Bank CC BY 4.0")
c3.metric("USD / INR",         f"Rs{fx:.2f}",          help="fawazahmed0/exchange-api CC0")
c4.metric("Search Interest",   f"{trend_val:.1f}/100" if trend_val else "N/A", help="Google Trends via pytrends Apache 2.0")
st.markdown("---")

if df is None:
    st.markdown("""<div style='text-align:center;padding:60px 20px'>
    <h2>Upload your dataset to unlock full analytics</h2>
    <p style='color:#64748b;font-size:1.1rem'>Upload CSV, Excel, JSON or Parquet via the sidebar.<br>
    Your data is saved securely and reloads automatically on next visit.</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

dff = df[df["zone"].isin(zones)&df["category"].isin(cats)&df["brand_type"].isin(brands)&df["sales_event"].isin(events)].copy()
if not gdp_df.empty:
    macro = gdp_df[["year","value"]].rename(columns={"value":"india_gdp_growth_pct"})
    macro = macro.merge(cpi_df[["year","value"]].rename(columns={"value":"india_cpi_inflation_pct"}),on="year",how="outer")
    dff = dff.merge(macro, on="year", how="left")
dff["usd_inr_rate"] = fx
dff["revenue_usd"]  = (dff["revenue"]/fx).round(2)

rev_cr = dff["revenue"].sum()/1e7
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Revenue",     f"Rs{rev_cr:.1f} Cr")
k2.metric("Total Orders",      f"{len(dff):,}")
k3.metric("Avg Order Value",   f"Rs{dff['revenue'].mean():,.0f}")
k4.metric("Avg Discount",      f"{dff['discount_percent'].mean():.1f}%")
k5.metric("Avg Units / Order", f"{dff['units_sold'].mean():.1f}")
st.caption(f"USD equivalent: ${dff['revenue_usd'].sum():,.0f}  (@ Rs{fx:.2f}/USD)  |  {len(dff):,} of {len(df):,} orders")
st.markdown("---")
tabs = st.tabs(["Executive Summary","Revenue","Categories","Regional",
                "Price Optimizer","At-Risk Customers","Model Drift",
                "Inventory","CLV","Anomalies","Cohort","Pareto"])

with tabs[0]:
    summary = executive_summary(dff, fx)
    recs    = generate_recommendations(dff)
    st.markdown(f'<div class="insight-card" style="font-size:1.1rem;font-weight:600">{summary["headline"]}</div>', unsafe_allow_html=True)
    col_ex1,col_ex2,_ = st.columns([1,1,4])
    with col_ex1:
        st.download_button("Export Excel", to_excel(dff,summary),
            file_name=f"report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_ex2:
        st.download_button("Export PDF", to_pdf(summary,recs),
            file_name=f"report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf", use_container_width=True)
    st.markdown("---")
    cl,cr = st.columns(2)
    with cl:
        st.subheader("KPIs")
        st.dataframe(pd.DataFrame(list(summary["kpis"].items()),columns=["Metric","Value"]),use_container_width=True,hide_index=True)
        st.subheader("Top Insights")
        for ins in summary["top_insights"]:
            st.markdown(f'<div class="insight-card">{ins}</div>',unsafe_allow_html=True)
    with cr:
        st.subheader("Risks")
        for r in summary["risks"]:
            st.markdown(f'<div class="{"risk-card" if "!" in r else "opp-card"}">{r}</div>',unsafe_allow_html=True)
        st.subheader("Opportunities")
        for o in summary["opportunities"]:
            st.markdown(f'<div class="opp-card">{o}</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Prioritised Recommendations")
    for rec in recs:
        border = "#ef4444" if "High" in rec["priority"] else "#f97316" if "Medium" in rec["priority"] else "#eab308"
        st.markdown(f'<div class="rec-card" style="border-left:4px solid {border}"><strong>{rec["priority"]} | {rec["category"]}</strong><br><span style="font-size:.95rem">{rec["action"]}</span><br><small style="color:#64748b">Impact: {rec["impact"]} · Effort: {rec["effort"]} · Metric: {rec["metric"]}</small></div>',unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Revenue Trends")
    m = dff.groupby("year_month")["revenue"].sum().reset_index()
    st.plotly_chart(px.line(m,x="year_month",y="revenue",markers=True,title="Total Monthly Revenue",labels={"revenue":"Revenue (Rs)","year_month":"Month"},template="plotly_white",color_discrete_sequence=["#4f46e5"]),use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.line(dff.groupby("year_month")["revenue"].mean().reset_index(),x="year_month",y="revenue",markers=True,title="Avg Order Value",labels={"revenue":"AOV (Rs)"},template="plotly_white",color_discrete_sequence=["#22c55e"]),use_container_width=True)
    c2.plotly_chart(px.line(dff.groupby("year_month")["discount_percent"].mean().reset_index(),x="year_month",y="discount_percent",markers=True,title="Avg Discount %",template="plotly_white",color_discrete_sequence=["#ef4444"]),use_container_width=True)
    st.plotly_chart(px.line(dff.groupby(["year_month","zone"])["revenue"].sum().reset_index(),x="year_month",y="revenue",color="zone",markers=True,title="Revenue by Zone",template="plotly_white"),use_container_width=True)
    st.plotly_chart(px.line(dff.groupby(["year_month","brand_type"])["revenue"].sum().reset_index(),x="year_month",y="revenue",color="brand_type",markers=True,title="Revenue: Mass vs Premium",template="plotly_white"),use_container_width=True)

with tabs[2]:
    st.subheader("Category & Brand Analysis")
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.pie(dff.groupby("category")["revenue"].sum().reset_index(),names="category",values="revenue",title="Revenue by Category",hole=0.45,template="plotly_white"),use_container_width=True)
    c2.plotly_chart(px.pie(dff.groupby("brand_type")["revenue"].sum().reset_index(),names="brand_type",values="revenue",title="Mass vs Premium",hole=0.45,template="plotly_white"),use_container_width=True)
    metric = st.selectbox("Metric",["revenue","final_price","units_sold","discount_percent"])
    st.plotly_chart(px.bar(dff.groupby("category")[metric].mean().reset_index().sort_values(metric,ascending=False),x="category",y=metric,color="category",title=f"Avg {metric} by Category",template="plotly_white"),use_container_width=True)
    st.plotly_chart(px.bar(dff.groupby(["year_month","sales_event"])["revenue"].sum().reset_index(),x="year_month",y="revenue",color="sales_event",title="Festival vs Normal Revenue",template="plotly_white",barmode="group"),use_container_width=True)

with tabs[3]:
    st.subheader("Regional Analysis")
    st.plotly_chart(px.bar(dff.groupby("state")["revenue"].sum().nlargest(15).reset_index(),x="revenue",y="state",orientation="h",title="Top 15 States by Revenue",template="plotly_white",color="revenue",color_continuous_scale="Blues"),use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.pie(dff.groupby("zone")["revenue"].sum().reset_index(),names="zone",values="revenue",title="Revenue by Zone",hole=0.45,template="plotly_white"),use_container_width=True)
    c2.plotly_chart(px.bar(dff.groupby("zone")["units_sold"].mean().reset_index(),x="zone",y="units_sold",color="zone",title="Avg Units by Zone",template="plotly_white"),use_container_width=True)

with tabs[4]:
    st.subheader("Price Optimizer")
    st.caption("Computes revenue-maximising discount per category using price elasticity (Lerner index). Saves recommendations to Supabase.")
    with st.spinner("Running price optimisation..."):
        price_recs = run_price_optimizer(dff, user_id=user.get("id",""))
    if not price_recs.empty:
        fig_price = plot_price_optimizer(price_recs)
        if fig_price:
            st.plotly_chart(fig_price, use_container_width=True)
        st.markdown("**Optimisation Recommendations**")
        display_cols = ["category","current_discount","optimal_discount","change","direction","elasticity","revenue_impact_pct","rationale"]
        display_cols = [c for c in display_cols if c in price_recs.columns]
        st.dataframe(price_recs[display_cols], use_container_width=True, hide_index=True)
        total_impact = price_recs["revenue_impact_pct"].sum()
        st.success(f"Estimated total revenue impact from applying all recommendations: **{total_impact:+.1f}%**")
    else:
        st.warning("Not enough data to compute elasticity. Upload a larger dataset.")

with tabs[5]:
    st.subheader("At-Risk Customer Automation")
    st.caption("BG/NBD-based churn risk scoring. Critical high-value customers are saved to Supabase for automated outreach.")
    top_n = st.slider("Customers to analyse", 20, 200, 50, 10)
    with st.spinner("Scoring churn risk..."):
        at_risk_df = generate_at_risk_alerts(dff, user_id=user.get("id",""), top_n=top_n)
    if not at_risk_df.empty:
        fig_r1, fig_r2 = plot_at_risk(at_risk_df)
        c1,c2 = st.columns(2)
        if fig_r1: c1.plotly_chart(fig_r1, use_container_width=True)
        if fig_r2: c2.plotly_chart(fig_r2, use_container_width=True)
        critical = at_risk_df[at_risk_df["risk_label"]=="Critical"]
        if not critical.empty:
            st.error(f"{len(critical)} critical high-value customers need immediate action")
        show_cols = ["customer_id","churn_risk_score","risk_label","value_tier","days_since_last_order","total_revenue","recommended_action"]
        show_cols = [c for c in show_cols if c in at_risk_df.columns]
        st.dataframe(at_risk_df[show_cols].head(50), use_container_width=True, hide_index=True)
        st.download_button("Export At-Risk List",
            at_risk_df[show_cols].to_csv(index=False).encode(),
            file_name="at_risk_customers.csv", mime="text/csv")
    else:
        st.warning("Not enough customer data for churn scoring.")

with tabs[6]:
    st.subheader("Model Drift Monitoring")
    st.caption("Detects feature distribution shift (PSI) and prediction performance degradation between reference and current windows.")
    c1,c2 = st.columns(2)
    ref_months = c1.slider("Reference window (months)", 3, 12, 6)
    cur_months = c2.slider("Current window (months)",   1, 6,  3)
    with st.spinner("Computing drift..."):
        drift_df   = compute_drift(dff, reference_months=ref_months, current_months=cur_months)
        pred_drift = compute_prediction_drift(dff, reference_months=ref_months, current_months=cur_months)
    if not drift_df.empty:
        fig_drift = plot_drift(drift_df)
        if fig_drift:
            st.plotly_chart(fig_drift, use_container_width=True)
        drifted = drift_df[drift_df["drift_detected"]]
        if not drifted.empty:
            st.warning(f"{len(drifted)} features show significant drift — model retraining recommended.")
        else:
            st.success("No significant feature drift detected.")
        st.dataframe(drift_df, use_container_width=True, hide_index=True)
    if pred_drift:
        st.markdown("**Prediction Performance**")
        pm1,pm2,pm3,pm4 = st.columns(4)
        pm1.metric("Reference R²",  str(pred_drift.get("ref_r2","N/A")))
        pm2.metric("Current R²",    str(pred_drift.get("cur_r2","N/A")),
                   delta=f"{-pred_drift.get('r2_drop',0):+.3f}" if pred_drift.get("r2_drop") else None,
                   delta_color="inverse")
        pm3.metric("Reference MAE", f"Rs{pred_drift.get('ref_mae',0):,.0f}")
        pm4.metric("Current MAE",   f"Rs{pred_drift.get('cur_mae',0):,.0f}")
        if pred_drift.get("drift_alert"):
            st.error("Model performance has degraded significantly. Retrain with recent data.")
        if user.get("id"):
            save_drift_report(user["id"], {
                "features_drifted": int(drift_df["drift_detected"].sum()) if not drift_df.empty else 0,
                "max_psi":          float(drift_df["psi"].max()) if not drift_df.empty else 0,
                "pred_r2_drop":     pred_drift.get("r2_drop",0),
                "drift_alert":      bool(pred_drift.get("drift_alert",False)),
                "report_json":      drift_df.to_json(orient="records") if not drift_df.empty else "{}",
            })
    else:
        st.info("Not enough data for prediction drift analysis.")

with tabs[7]:
    st.subheader("Inventory Alert System")
    from modules.inventory_alerts import compute_alerts
    alerts = compute_alerts(dff)
    cmap = {"🔴 CRITICAL – Reorder Now":"#ef4444","🟠 HIGH – Monitor Closely":"#f97316","🟡 CLEARANCE – Excess Stock":"#eab308","🔵 SLOW MOVER – Review Listing":"#3b82f6","🟢 HEALTHY":"#22c55e"}
    st.plotly_chart(px.scatter(alerts,x="avg_discount",y="avg_units_sold",color="alert_level",size="high_pressure_pct",hover_data=["category","zone","recommendation"],color_discrete_map=cmap,title="Inventory Alert Dashboard",template="plotly_white"),use_container_width=True)
    af = st.multiselect("Filter alerts",list(cmap.keys()),default=list(cmap.keys()))
    st.dataframe(alerts[alerts["alert_level"].isin(af)][["category","zone","avg_units_sold","avg_discount","alert_level","recommendation"]],use_container_width=True,hide_index=True)

with tabs[8]:
    st.subheader("Customer Lifetime Value")
    from modules.clv import compute_clv
    with st.spinner("Computing CLV..."):
        clv_df = compute_clv(dff)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.histogram(clv_df,x="clv",color="clv_tier",nbins=50,title="CLV Distribution",template="plotly_white",marginal="box"),use_container_width=True)
    c2.plotly_chart(px.pie(clv_df.groupby("clv_tier")["clv"].sum().reset_index(),names="clv_tier",values="clv",title="CLV Share by Tier",hole=0.45,template="plotly_white"),use_container_width=True)
    st.plotly_chart(px.scatter(clv_df,x="frequency",y="clv",color="clv_tier",opacity=0.6,size="monetary",title="Frequency vs CLV",template="plotly_white"),use_container_width=True)
    st.dataframe(clv_df.groupby("clv_tier")["clv"].agg(count="count",mean_clv="mean",total_clv="sum").round(2),use_container_width=True)

with tabs[9]:
    st.subheader("Anomaly Detection")
    from modules.anomaly import anomaly_report
    with st.spinner("Detecting anomalies..."):
        anom = anomaly_report(dff)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.scatter(anom,x="log_units_sold",y="log_revenue",color="confirmed_anomaly",color_discrete_map={True:"#ef4444",False:"#94a3b8"},opacity=0.5,title="log(Units) vs log(Revenue)",hover_data=["category","zone","discount_percent"],template="plotly_white"),use_container_width=True)
    c2.plotly_chart(px.scatter(anom,x="discount_percent",y="log_final_price",color="confirmed_anomaly",color_discrete_map={True:"#ef4444",False:"#e2e8f0"},opacity=0.55,title="Discount % vs log(Final Price)",hover_data=["category","zone"],template="plotly_white"),use_container_width=True)
    n = anom["confirmed_anomaly"].sum()
    st.info(f"Confirmed anomalies: **{n:,}** ({n/len(anom):.2%})")

with tabs[10]:
    st.subheader("Cohort Analysis")
    from modules.cohort import build_cohort_table
    c1,c2 = st.columns(2)
    pivot = build_cohort_table(dff,metric="count")
    ret = (pivot.div(pivot[0],axis=0)*100).round(1)
    c1.plotly_chart(px.imshow(ret,color_continuous_scale="Blues",title="Retention Rate (%) - Orders",labels={"x":"Months since first purchase","y":"Cohort","color":"Retention %"},text_auto=".0f",template="plotly_white"),use_container_width=True)
    pivot_r = build_cohort_table(dff,metric="revenue")
    ret_r = (pivot_r.div(pivot_r[0],axis=0)*100).round(1)
    c2.plotly_chart(px.imshow(ret_r,color_continuous_scale="Greens",title="Revenue Retention (%)",labels={"x":"Months since first purchase","y":"Cohort","color":"Retention %"},text_auto=".0f",template="plotly_white"),use_container_width=True)

with tabs[11]:
    st.subheader("Pareto & Concentration")
    agg = dff.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    agg["cum_pct"] = agg["revenue"].cumsum()/agg["revenue"].sum()*100
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=agg["category"],y=agg["revenue"],name="Revenue",marker_color="#4f46e5"),secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["category"],y=agg["cum_pct"],mode="lines+markers",name="Cumulative %",line=dict(color="#ef4444",width=2.5)),secondary_y=True)
    fig.add_hline(y=80,line_dash="dash",line_color="#94a3b8",annotation_text="80%",secondary_y=True)
    fig.update_layout(title="Pareto - Revenue by Category",template="plotly_white")
    fig.update_yaxes(title_text="Revenue (Rs)",secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %",secondary_y=True)
    st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.sunburst(dff.groupby(["category","zone","brand_type"])["revenue"].sum().reset_index(),path=["category","zone","brand_type"],values="revenue",title="Revenue Sunburst",template="plotly_white",color="revenue",color_continuous_scale="Blues"),use_container_width=True)
    vals = np.sort(dff["revenue"].dropna().values)[::-1]
    cum  = np.cumsum(vals)/vals.sum()
    x    = np.linspace(0,1,len(cum))
    gini = round(1-2*np.trapezoid(cum,x),3)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash",color="#94a3b8"),name="Perfect equality"))
    fig2.add_trace(go.Scatter(x=x,y=cum,mode="lines",fill="tozeroy",fillcolor="rgba(79,70,229,.10)",line=dict(color="#4f46e5",width=2.5),name=f"Lorenz (Gini={gini})"))
    fig2.update_layout(title=f"Lorenz Curve (Gini={gini})",xaxis_title="Cumulative share of orders",yaxis_title="Cumulative share of revenue",template="plotly_white")
    st.plotly_chart(fig2,use_container_width=True)

st.markdown("---")
st.caption(f"{settings.app_name} v{settings.app_version} | World Bank (CC BY 4.0) | fawazahmed0/exchange-api (CC0) | pytrends (Apache 2.0) | Kaggle: {settings.kaggle_url}")
