"""
dashboard/app.py  v2.0  —  IndiaCommerce Analytics
Production-grade. Supabase auth + DB, custom UI, executive summary,
recommendations, PDF/Excel export, live macro signals.
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
from core.auth     import render_auth_wall, is_authenticated, current_user, logout
from core.database import save_dataset, load_dataset, log_event
from data.loader   import (
    _clean, _engineer, load_any,
    fetch_worldbank, fetch_usd_inr, fetch_google_trends,
    _WB_GDP, _WB_CPI,
)
from modules.insights import executive_summary, generate_recommendations
from modules.export   import to_excel, to_pdf

st.set_page_config(
    page_title=settings.app_name,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
from dashboard.style import CUSTOM_CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Auth wall ─────────────────────────────────────────────────────────────────
if not render_auth_wall():
    st.stop()

user = current_user()

# ── Cached live data ──────────────────────────────────────────────────────────
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

@st.cache_data(show_spinner="Loading...")
def _from_path(p: str) -> pd.DataFrame:
    return _engineer(_clean(pd.read_csv(p)))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 📊 {settings.app_name}")
    st.markdown(f"<small>v{settings.app_version}</small>", unsafe_allow_html=True)
    if user.get("demo"):
        st.info("Demo mode — configure Supabase for full auth.")
    else:
        st.markdown(f"👤 **{user.get('email','')}**")
    if st.button("Sign Out", use_container_width=True):
        logout()
    st.markdown("---")

    # Data source
    st.subheader("Data Source")
    df = None

    # Try loading saved dataset from Supabase first
    if not user.get("demo") and "df" not in st.session_state:
        saved = load_dataset(user.get("id","demo"))
        if saved is not None:
            st.session_state["df"] = _engineer(_clean(saved))
            st.success("Loaded your saved dataset.")

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.success(f"✅ {len(df):,} rows loaded")
        if st.button("Clear dataset", use_container_width=True):
            del st.session_state["df"]
            st.rerun()
    else:
        up = st.file_uploader(
            "Upload dataset",
            type=["csv","tsv","xlsx","xls","json","parquet"],
            help="CSV · TSV · Excel · JSON · Parquet",
        )
        st.caption(f"[Download from Kaggle]({settings.kaggle_url})")
        if up:
            try:
                df = _from_upload(up.read(), up.name)
                st.session_state["df"] = df
                save_dataset(user.get("id","demo"), df)
                log_event(user.get("id","demo"), "upload", up.name)
                st.success(f"✅ {len(df):,} rows loaded")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.subheader("Live Signals")
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear(); st.rerun()
    st.caption("Auto-refreshes hourly.")

    if df is not None:
        st.markdown("---")
        st.subheader("Filters")
        zones  = st.multiselect("Zone",        sorted(df["zone"].unique()),       default=sorted(df["zone"].unique()))
        cats   = st.multiselect("Category",    sorted(df["category"].unique()),   default=sorted(df["category"].unique()))
        brands = st.multiselect("Brand Type",  sorted(df["brand_type"].unique()), default=sorted(df["brand_type"].unique()))
        events = st.multiselect("Sales Event", sorted(df["sales_event"].unique()),default=sorted(df["sales_event"].unique()))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="page-title">📊 IndiaCommerce Analytics</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Live: '
    '<a href="https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/" target="_blank">Streamlit App</a>'
    ' &nbsp;|&nbsp; '
    '<a href="https://github.com/SumedhPatil1507/indian-ecommerce-analytics" target="_blank">GitHub</a>'
    ' &nbsp;|&nbsp; World Bank (CC BY 4.0) · exchangerate.host · pytrends (Apache 2.0)'
    '</p>',
    unsafe_allow_html=True,
)

# ── Live macro (always visible) ───────────────────────────────────────────────
gdp_df, cpi_df, fx = _live_macro()
trends_df = _live_trends()

gdp_val   = float(gdp_df["value"].iloc[-1])  if not gdp_df.empty else None
cpi_val   = float(cpi_df["value"].iloc[-1])  if not cpi_df.empty else None
gdp_delta = round(gdp_df["value"].iloc[-1] - gdp_df["value"].iloc[-2], 2) if len(gdp_df)>=2 else None
cpi_delta = round(cpi_df["value"].iloc[-1] - cpi_df["value"].iloc[-2], 2) if len(cpi_df)>=2 else None
trend_val = float(trends_df.mean().mean()) if not trends_df.empty else None

c1,c2,c3,c4 = st.columns(4)
c1.metric("🌐 India GDP Growth",  f"{gdp_val:.2f}%"      if gdp_val  else "N/A", delta=f"{gdp_delta:+.2f}pp" if gdp_delta else None)
c2.metric("📈 CPI Inflation",     f"{cpi_val:.2f}%"      if cpi_val  else "N/A", delta=f"{cpi_delta:+.2f}pp" if cpi_delta else None)
c3.metric("💱 USD / INR",         f"₹{fx:.2f}")
c4.metric("🔍 Search Interest",   f"{trend_val:.1f}/100" if trend_val else "N/A")

st.markdown("---")

# ── No data state ─────────────────────────────────────────────────────────────
if df is None:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;'>
        <h2>Upload your dataset to unlock full analytics</h2>
        <p style='color:#64748b;font-size:1.1rem;'>
            Upload a CSV, Excel, JSON or Parquet file using the sidebar.<br>
            Your data is saved securely and reloads automatically on next visit.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Apply filters ─────────────────────────────────────────────────────────────
dff = df[
    df["zone"].isin(zones) & df["category"].isin(cats) &
    df["brand_type"].isin(brands) & df["sales_event"].isin(events)
].copy()

if not gdp_df.empty:
    macro = gdp_df[["year","value"]].rename(columns={"value":"india_gdp_growth_pct"})
    macro = macro.merge(cpi_df[["year","value"]].rename(columns={"value":"india_cpi_inflation_pct"}), on="year", how="outer")
    dff = dff.merge(macro, on="year", how="left")
dff["usd_inr_rate"] = fx
dff["revenue_usd"]  = (dff["revenue"] / fx).round(2)

# ── KPIs ──────────────────────────────────────────────────────────────────────
rev_cr = dff["revenue"].sum() / 1e7
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Revenue",     f"₹{rev_cr:.1f} Cr")
k2.metric("Total Orders",      f"{len(dff):,}")
k3.metric("Avg Order Value",   f"₹{dff['revenue'].mean():,.0f}")
k4.metric("Avg Discount",      f"{dff['discount_percent'].mean():.1f}%")
k5.metric("Avg Units / Order", f"{dff['units_sold'].mean():.1f}")
st.caption(f"💱 USD equivalent: **${dff['revenue_usd'].sum():,.0f}**  (@ ₹{fx:.2f}/USD)  ·  Filtered to {len(dff):,} of {len(df):,} orders")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🎯 Executive Summary","📈 Revenue","🗂 Categories","🗺 Regional",
                "💰 Elasticity","⚠️ Inventory","👥 CLV","🔍 Anomalies","🌀 Cohort","🏆 Pareto"])

# ── Tab 0: Executive Summary ──────────────────────────────────────────────────
with tabs[0]:
    summary = executive_summary(dff, fx)
    recs    = generate_recommendations(dff)

    # Headline
    st.markdown(f"""
    <div class="insight-card" style="border-left-color:#4f46e5;font-size:1.1rem;font-weight:600;">
        {summary['headline']}
    </div>""", unsafe_allow_html=True)

    # Export buttons
    col_ex1, col_ex2, col_ex3 = st.columns([1,1,4])
    with col_ex1:
        excel_bytes = to_excel(dff, summary)
        st.download_button("📥 Export Excel", excel_bytes,
            file_name=f"indiacommerce_report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)
    with col_ex2:
        pdf_bytes = to_pdf(summary, recs)
        st.download_button("📄 Export PDF", pdf_bytes,
            file_name=f"indiacommerce_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf", use_container_width=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📊 KPIs")
        kpi_df = pd.DataFrame(list(summary["kpis"].items()), columns=["Metric","Value"])
        st.dataframe(kpi_df, use_container_width=True, hide_index=True)

        st.subheader("💡 Top Insights")
        for ins in summary["top_insights"]:
            st.markdown(f'<div class="insight-card">{ins}</div>', unsafe_allow_html=True)

    with col_r:
        st.subheader("⚠️ Risks")
        for r in summary["risks"]:
            css = "risk-card" if "⚠️" in r else "opp-card"
            st.markdown(f'<div class="{css}">{r}</div>', unsafe_allow_html=True)

        st.subheader("🚀 Opportunities")
        for o in summary["opportunities"]:
            st.markdown(f'<div class="opp-card">{o}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Prioritised Recommendations")
    for rec in recs:
        priority_color = {"🔴 High":"#fee2e2","🟠 Medium":"#fff7ed","🟡 Low":"#fefce8"}.get(rec["priority"],"#f8f9fc")
        st.markdown(f"""
        <div class="rec-card" style="border-left:4px solid {'#ef4444' if 'High' in rec['priority'] else '#f97316' if 'Medium' in rec['priority'] else '#eab308'};">
            <strong>{rec['priority']} &nbsp;|&nbsp; {rec['category']}</strong><br>
            <span style='font-size:0.95rem'>{rec['action']}</span><br>
            <small style='color:#64748b'>Impact: {rec['impact']} &nbsp;·&nbsp; Effort: {rec['effort']} &nbsp;·&nbsp; Metric: {rec['metric']}</small>
        </div>""", unsafe_allow_html=True)

# ── Tab 1: Revenue ────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Revenue Trends")
    m = dff.groupby("year_month")["revenue"].sum().reset_index()
    st.plotly_chart(px.line(m, x="year_month", y="revenue", markers=True,
        title="Total Monthly Revenue", labels={"revenue":"Revenue (Rs)","year_month":"Month"},
        template="plotly_white", color_discrete_sequence=["#4f46e5"]), use_container_width=True)
    c1,c2 = st.columns(2)
    aov = dff.groupby("year_month")["revenue"].mean().reset_index()
    c1.plotly_chart(px.line(aov, x="year_month", y="revenue", markers=True,
        title="Avg Order Value", labels={"revenue":"AOV (Rs)"}, template="plotly_white",
        color_discrete_sequence=["#22c55e"]), use_container_width=True)
    disc = dff.groupby("year_month")["discount_percent"].mean().reset_index()
    c2.plotly_chart(px.line(disc, x="year_month", y="discount_percent", markers=True,
        title="Avg Discount %", template="plotly_white",
        color_discrete_sequence=["#ef4444"]), use_container_width=True)
    zm = dff.groupby(["year_month","zone"])["revenue"].sum().reset_index()
    st.plotly_chart(px.line(zm, x="year_month", y="revenue", color="zone", markers=True,
        title="Revenue by Zone", template="plotly_white"), use_container_width=True)
    bm = dff.groupby(["year_month","brand_type"])["revenue"].sum().reset_index()
    st.plotly_chart(px.line(bm, x="year_month", y="revenue", color="brand_type", markers=True,
        title="Revenue: Mass vs Premium", template="plotly_white"), use_container_width=True)

# ── Tab 2: Categories ─────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Category & Brand Analysis")
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.pie(dff.groupby("category")["revenue"].sum().reset_index(),
        names="category", values="revenue", title="Revenue by Category", hole=0.45,
        template="plotly_white"), use_container_width=True)
    c2.plotly_chart(px.pie(dff.groupby("brand_type")["revenue"].sum().reset_index(),
        names="brand_type", values="revenue", title="Mass vs Premium", hole=0.45,
        template="plotly_white"), use_container_width=True)
    metric = st.selectbox("Metric", ["revenue","final_price","units_sold","discount_percent"])
    agg = dff.groupby("category")[metric].mean().reset_index().sort_values(metric, ascending=False)
    st.plotly_chart(px.bar(agg, x="category", y=metric, color="category",
        title=f"Avg {metric} by Category", template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)
    em = dff.groupby(["year_month","sales_event"])["revenue"].sum().reset_index()
    st.plotly_chart(px.bar(em, x="year_month", y="revenue", color="sales_event",
        title="Festival vs Normal Revenue", template="plotly_white", barmode="group"), use_container_width=True)

# ── Tab 3: Regional ───────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Regional Analysis")
    sr = dff.groupby("state")["revenue"].sum().nlargest(15).reset_index()
    st.plotly_chart(px.bar(sr, x="revenue", y="state", orientation="h",
        title="Top 15 States by Revenue", template="plotly_white",
        color="revenue", color_continuous_scale="Blues"), use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.pie(dff.groupby("zone")["revenue"].sum().reset_index(),
        names="zone", values="revenue", title="Revenue by Zone", hole=0.45,
        template="plotly_white"), use_container_width=True)
    c2.plotly_chart(px.bar(dff.groupby("zone")["units_sold"].mean().reset_index(),
        x="zone", y="units_sold", color="zone", title="Avg Units by Zone",
        template="plotly_white"), use_container_width=True)

# ── Tab 4: Elasticity ─────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Price Elasticity Engine")
    from modules.price_elasticity import compute_elasticity
    with st.spinner("Computing elasticity..."):
        elast = compute_elasticity(dff, group_cols=["category"])
    if not elast.empty:
        elast["type"] = elast["elasticity"].apply(
            lambda e: "Elastic (<-1)" if e<-1 else ("Inelastic" if e<0 else "Giffen/Luxury"))
        fig = px.bar(elast.sort_values("elasticity"), x="elasticity", y="category",
            orientation="h", color="type",
            color_discrete_map={"Elastic (<-1)":"#ef4444","Inelastic":"#4f46e5","Giffen/Luxury":"#eab308"},
            title="Price Elasticity by Category", template="plotly_white", text="elasticity")
        fig.add_vline(x=-1, line_dash="dash", line_color="#94a3b8", annotation_text="Elastic threshold")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(elast, use_container_width=True, hide_index=True)
    else:
        st.warning("Not enough data for elasticity with current filters.")

# ── Tab 5: Inventory ──────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Inventory Alert System")
    from modules.inventory_alerts import compute_alerts
    alerts = compute_alerts(dff)
    cmap = {"🔴 CRITICAL – Reorder Now":"#ef4444","🟠 HIGH – Monitor Closely":"#f97316",
            "🟡 CLEARANCE – Excess Stock":"#eab308","🔵 SLOW MOVER – Review Listing":"#3b82f6","🟢 HEALTHY":"#22c55e"}
    st.plotly_chart(px.scatter(alerts, x="avg_discount", y="avg_units_sold",
        color="alert_level", size="high_pressure_pct",
        hover_data=["category","zone","recommendation"],
        color_discrete_map=cmap, title="Inventory Alert Dashboard", template="plotly_white"),
        use_container_width=True)
    af = st.multiselect("Filter alerts", list(cmap.keys()), default=list(cmap.keys()))
    st.dataframe(alerts[alerts["alert_level"].isin(af)]
        [["category","zone","avg_units_sold","avg_discount","alert_level","recommendation"]],
        use_container_width=True, hide_index=True)

# ── Tab 6: CLV ────────────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Customer Lifetime Value")
    from modules.clv import compute_clv
    with st.spinner("Computing CLV..."):
        clv_df = compute_clv(dff)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.histogram(clv_df, x="clv", color="clv_tier", nbins=50,
        title="CLV Distribution", template="plotly_white", marginal="box"), use_container_width=True)
    c2.plotly_chart(px.pie(clv_df.groupby("clv_tier")["clv"].sum().reset_index(),
        names="clv_tier", values="clv", title="CLV Share by Tier", hole=0.45,
        template="plotly_white"), use_container_width=True)
    st.plotly_chart(px.scatter(clv_df, x="frequency", y="clv", color="clv_tier",
        opacity=0.6, size="monetary", title="Frequency vs CLV", template="plotly_white"),
        use_container_width=True)
    st.dataframe(clv_df.groupby("clv_tier")["clv"].agg(count="count",mean_clv="mean",total_clv="sum").round(2),
        use_container_width=True)

# ── Tab 7: Anomalies ──────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Anomaly Detection")
    from modules.anomaly import anomaly_report
    with st.spinner("Detecting anomalies..."):
        anom = anomaly_report(dff)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.scatter(anom, x="log_units_sold", y="log_revenue",
        color="confirmed_anomaly", color_discrete_map={True:"#ef4444",False:"#94a3b8"},
        opacity=0.5, title="log(Units) vs log(Revenue)",
        hover_data=["category","zone","discount_percent"], template="plotly_white"), use_container_width=True)
    c2.plotly_chart(px.scatter(anom, x="discount_percent", y="log_final_price",
        color="confirmed_anomaly", color_discrete_map={True:"#ef4444",False:"#e2e8f0"},
        opacity=0.55, title="Discount % vs log(Final Price)",
        hover_data=["category","zone"], template="plotly_white"), use_container_width=True)
    n = anom["confirmed_anomaly"].sum()
    st.info(f"Confirmed anomalies (>=2 detectors): **{n:,}** ({n/len(anom):.2%})")

# ── Tab 8: Cohort ─────────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Cohort Analysis")
    from modules.cohort import build_cohort_table
    c1,c2 = st.columns(2)
    pivot = build_cohort_table(dff, metric="count")
    ret = (pivot.div(pivot[0], axis=0)*100).round(1)
    c1.plotly_chart(px.imshow(ret, color_continuous_scale="Blues",
        title="Retention Rate (%) - Orders",
        labels={"x":"Months since first purchase","y":"Cohort","color":"Retention %"},
        text_auto=".0f", template="plotly_white"), use_container_width=True)
    pivot_r = build_cohort_table(dff, metric="revenue")
    ret_r = (pivot_r.div(pivot_r[0], axis=0)*100).round(1)
    c2.plotly_chart(px.imshow(ret_r, color_continuous_scale="Greens",
        title="Revenue Retention (%)",
        labels={"x":"Months since first purchase","y":"Cohort","color":"Retention %"},
        text_auto=".0f", template="plotly_white"), use_container_width=True)

# ── Tab 9: Pareto ─────────────────────────────────────────────────────────────
with tabs[9]:
    st.subheader("Pareto & Concentration")
    agg = dff.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    agg["cum_pct"] = agg["revenue"].cumsum() / agg["revenue"].sum() * 100
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=agg["category"], y=agg["revenue"], name="Revenue",
        marker_color="#4f46e5"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["category"], y=agg["cum_pct"], mode="lines+markers",
        name="Cumulative %", line=dict(color="#ef4444",width=2.5)), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="#94a3b8",
        annotation_text="80% threshold", secondary_y=True)
    fig.update_layout(title="Pareto - Revenue by Category (80/20 Rule)", template="plotly_white")
    fig.update_yaxes(title_text="Revenue (Rs)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    sun = dff.groupby(["category","zone","brand_type"])["revenue"].sum().reset_index()
    st.plotly_chart(px.sunburst(sun, path=["category","zone","brand_type"], values="revenue",
        title="Revenue Sunburst - Category > Zone > Brand", template="plotly_white",
        color="revenue", color_continuous_scale="Blues"), use_container_width=True)

    vals = np.sort(dff["revenue"].dropna().values)[::-1]
    cum  = np.cumsum(vals)/vals.sum()
    x    = np.linspace(0,1,len(cum))
    gini = round(1 - 2*np.trapezoid(cum,x), 3)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
        line=dict(dash="dash",color="#94a3b8"),name="Perfect equality"))
    fig2.add_trace(go.Scatter(x=x,y=cum,mode="lines",fill="tozeroy",
        fillcolor="rgba(79,70,229,0.10)",line=dict(color="#4f46e5",width=2.5),
        name=f"Lorenz curve (Gini={gini})"))
    fig2.update_layout(title=f"Lorenz Curve - Revenue Concentration (Gini={gini})",
        xaxis_title="Cumulative share of orders",
        yaxis_title="Cumulative share of revenue", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"{settings.app_name} v{settings.app_version}  ·  "
    "World Bank Open Data (CC BY 4.0)  ·  exchangerate.host  ·  "
    "pytrends/Google Trends (Apache 2.0)  ·  "
    "Kaggle: https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth"
)
