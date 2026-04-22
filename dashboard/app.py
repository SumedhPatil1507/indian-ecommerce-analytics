"""
dashboard/app.py
──────────────────────────────────────────────────────────────────────────────
Streamlit Interactive Dashboard
All charts are Plotly (fully interactive).

Run:
    streamlit run dashboard/app.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.loader import (
    load, fetch_worldbank, fetch_usd_inr,
    fetch_google_trends, _WB_GDP, _WB_CPI,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 Indian E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")
data_path = st.sidebar.text_input(
    "CSV path",
    value=(
        "/kaggle/input/indian-e-commerce-pricing-revenue-growth/"
        "indian_ecommerce_pricing_revenue_growth_36_months.csv"
    ),
)
news_key = st.sidebar.text_input("NewsAPI key (optional)", type="password")
enrich   = st.sidebar.checkbox("Enrich with live macro data", value=True)

# ── load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def get_data(path, enrich_live, nkey):
    return load(path, enrich_live=enrich_live,
                news_api_key=nkey if nkey else None)

try:
    df = get_data(data_path, enrich, news_key)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

# ── sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
zones      = st.sidebar.multiselect("Zone",       df["zone"].unique(),      default=list(df["zone"].unique()))
categories = st.sidebar.multiselect("Category",   df["category"].unique(),  default=list(df["category"].unique()))
brands     = st.sidebar.multiselect("Brand Type", df["brand_type"].unique(),default=list(df["brand_type"].unique()))
events     = st.sidebar.multiselect("Sales Event",df["sales_event"].unique(),default=list(df["sales_event"].unique()))

mask = (
    df["zone"].isin(zones) &
    df["category"].isin(categories) &
    df["brand_type"].isin(brands) &
    df["sales_event"].isin(events)
)
dff = df[mask]

# ── header ────────────────────────────────────────────────────────────────────
st.title("🛒 Indian E-Commerce Analytics Dashboard")
st.caption(
    "Data: [Kaggle – Indian E-Commerce Pricing & Revenue Growth (36 months)]"
    "(https://www.kaggle.com/datasets)  |  "
    "Live macro: [World Bank Open Data](https://data.worldbank.org/) (CC BY 4.0)  |  "
    "FX: [exchangerate.host](https://exchangerate.host)  |  "
    "Trends: [Google Trends via pytrends](https://github.com/GeneralMills/pytrends) (Apache 2.0)"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue",    f"₹{dff['revenue'].sum()/1e7:.1f} Cr")
k2.metric("Total Orders",     f"{len(dff):,}")
k3.metric("Avg Order Value",  f"₹{dff['revenue'].mean():,.0f}")
k4.metric("Avg Discount",     f"{dff['discount_percent'].mean():.1f}%")
k5.metric("Avg Units / Order",f"{dff['units_sold'].mean():.1f}")

# ── live macro row ────────────────────────────────────────────────────────────
if enrich and "india_gdp_growth_pct" in dff.columns:
    st.markdown("---")
    st.subheader("📡 Live Macro Signals")
    m1, m2, m3, m4 = st.columns(4)

    gdp_val = dff["india_gdp_growth_pct"].dropna().iloc[-1] if not dff["india_gdp_growth_pct"].dropna().empty else None
    cpi_val = dff["india_cpi_inflation_pct"].dropna().iloc[-1] if not dff["india_cpi_inflation_pct"].dropna().empty else None
    fx_val  = dff["usd_inr_rate"].iloc[0] if "usd_inr_rate" in dff.columns else None
    trend_val = dff["ecommerce_search_interest"].mean() if "ecommerce_search_interest" in dff.columns else None

    m1.metric("🌐 India GDP Growth",    f"{gdp_val:.2f}%" if gdp_val else "N/A",
              help="Source: World Bank Open Data (CC BY 4.0)")
    m2.metric("📈 CPI Inflation",       f"{cpi_val:.2f}%" if cpi_val else "N/A",
              help="Source: World Bank Open Data (CC BY 4.0)")
    m3.metric("💱 USD / INR",           f"₹{fx_val:.2f}" if fx_val else "N/A",
              help="Source: exchangerate.host")
    m4.metric("🔍 Search Interest",     f"{trend_val:.1f}/100" if trend_val else "N/A",
              help="Source: Google Trends via pytrends (Apache 2.0)")

# ── tabs ──────────────────────────────────────────────────────────────────────
st.markdown("---")
tabs = st.tabs([
    "📈 Revenue Trends",
    "🗂 Categories",
    "🗺 Regional",
    "💰 Price Elasticity",
    "⚠️ Inventory Alerts",
    "👥 CLV",
    "🔍 Anomalies",
    "🌀 Cohort",
    "🏆 Pareto / Sunburst",
])

# ── Tab 0: Revenue Trends ─────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Monthly Revenue Trend")
    monthly = dff.groupby("year_month")["revenue"].sum().reset_index()
    fig = px.line(monthly, x="year_month", y="revenue", markers=True,
                  title="Total Monthly Revenue",
                  labels={"revenue": "Revenue (₹)", "year_month": "Month"},
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        aov = dff.groupby("year_month")["revenue"].mean().reset_index()
        fig2 = px.line(aov, x="year_month", y="revenue", markers=True,
                       title="Average Order Value (AOV)",
                       labels={"revenue": "AOV (₹)"}, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        disc = dff.groupby("year_month")["discount_percent"].mean().reset_index()
        fig3 = px.line(disc, x="year_month", y="discount_percent", markers=True,
                       title="Avg Discount % Trend",
                       labels={"discount_percent": "Discount %"}, template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)

    # zone comparison
    zone_m = dff.groupby(["year_month","zone"])["revenue"].sum().reset_index()
    fig4 = px.line(zone_m, x="year_month", y="revenue", color="zone", markers=True,
                   title="Revenue by Zone", template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

# ── Tab 1: Categories ─────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Category & Brand Analysis")
    col1, col2 = st.columns(2)
    with col1:
        cat_rev = dff.groupby("category")["revenue"].sum().reset_index()
        fig = px.pie(cat_rev, names="category", values="revenue",
                     title="Revenue Share by Category", hole=0.4, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        brand_rev = dff.groupby("brand_type")["revenue"].sum().reset_index()
        fig = px.pie(brand_rev, names="brand_type", values="revenue",
                     title="Revenue – Mass vs Premium", hole=0.4, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    metric = st.selectbox("Metric", ["revenue","final_price","units_sold","discount_percent"])
    agg = dff.groupby("category")[metric].mean().reset_index().sort_values(metric, ascending=False)
    fig = px.bar(agg, x="category", y=metric, color="category",
                 title=f"Avg {metric.replace('_',' ').title()} by Category",
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Regional ───────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Regional Analysis")
    state_rev = dff.groupby("state")["revenue"].sum().nlargest(15).reset_index()
    fig = px.bar(state_rev, x="revenue", y="state", orientation="h",
                 title="Top 15 States by Revenue", template="plotly_white", color="revenue",
                 color_continuous_scale="YlOrRd")
    st.plotly_chart(fig, use_container_width=True)

    zone_rev = dff.groupby("zone")["revenue"].sum().reset_index()
    fig2 = px.pie(zone_rev, names="zone", values="revenue",
                  title="Revenue Share by Zone", hole=0.4, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# ── Tab 3: Price Elasticity ───────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Price Elasticity Engine")
    from modules.price_elasticity import compute_elasticity
    elast = compute_elasticity(dff, group_cols=["category"])
    if not elast.empty:
        elast["type"] = elast["elasticity"].apply(
            lambda e: "Elastic" if e < -1 else ("Inelastic" if e < 0 else "Giffen/Luxury"))
        fig = px.bar(elast.sort_values("elasticity"), x="elasticity", y="category",
                     orientation="h", color="type",
                     color_discrete_map={"Elastic":"crimson","Inelastic":"steelblue","Giffen/Luxury":"gold"},
                     title="Price Elasticity by Category",
                     labels={"elasticity": "Elasticity (β)"}, template="plotly_white")
        fig.add_vline(x=-1, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(elast, use_container_width=True)

# ── Tab 4: Inventory Alerts ───────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Dynamic Inventory Alert System")
    from modules.inventory_alerts import compute_alerts
    alerts = compute_alerts(dff)
    colour_map = {
        "🔴 CRITICAL – Reorder Now":    "red",
        "🟠 HIGH – Monitor Closely":    "orange",
        "🟡 CLEARANCE – Excess Stock":  "gold",
        "🔵 SLOW MOVER – Review Listing":"steelblue",
        "🟢 HEALTHY":                   "green",
    }
    fig = px.scatter(alerts, x="avg_discount", y="avg_units_sold",
                     color="alert_level", size="high_pressure_pct",
                     hover_data=["category","zone","recommendation"],
                     color_discrete_map=colour_map,
                     title="Inventory Alert Dashboard",
                     template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(alerts[["category","zone","alert_level","recommendation"]],
                 use_container_width=True)

# ── Tab 5: CLV ────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Customer Lifetime Value")
    from modules.clv import compute_clv
    with st.spinner("Computing CLV…"):
        clv_df = compute_clv(dff)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(clv_df, x="clv", color="clv_tier", nbins=50,
                           title="CLV Distribution", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        tier_rev = clv_df.groupby("clv_tier")["clv"].sum().reset_index()
        fig = px.pie(tier_rev, names="clv_tier", values="clv",
                     title="CLV Share by Tier", hole=0.4, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 6: Anomalies ──────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Automated Anomaly Detection")
    from modules.anomaly import anomaly_report
    with st.spinner("Running anomaly detection…"):
        anom_df = anomaly_report(dff)
    fig = px.scatter(anom_df, x="log_units_sold", y="log_revenue",
                     color="confirmed_anomaly",
                     color_discrete_map={True:"red", False:"steelblue"},
                     opacity=0.5, title="Anomalies – log(Units) vs log(Revenue)",
                     hover_data=["category","zone","discount_percent"],
                     template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    n = anom_df["confirmed_anomaly"].sum()
    st.info(f"Confirmed anomalies (≥2 detectors): **{n:,}** ({n/len(anom_df):.2%})")

# ── Tab 7: Cohort ─────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Cohort Analysis")
    from modules.cohort import build_cohort_table
    pivot = build_cohort_table(dff, metric="count")
    retention = (pivot.div(pivot[0], axis=0) * 100).round(1)
    fig = px.imshow(retention, color_continuous_scale="Blues",
                    title="Cohort Retention Rate (%)",
                    labels={"x":"Cohort Period (months)","y":"Cohort Month","color":"Retention %"},
                    text_auto=".0f", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 8: Pareto / Sunburst ──────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Pareto & Sunburst")
    from modules.pareto import plot_lorenz, plot_ecdf

    # Pareto
    agg = dff.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    agg["cum_pct"] = agg["revenue"].cumsum() / agg["revenue"].sum() * 100
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=agg["category"], y=agg["revenue"], name="Revenue",
                         marker_color="steelblue"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["category"], y=agg["cum_pct"],
                             mode="lines+markers", name="Cumulative %",
                             line=dict(color="crimson")), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="orange", secondary_y=True)
    fig.update_layout(title="Pareto Chart – Revenue by Category", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Sunburst
    sun = dff.groupby(["category","zone","brand_type"])["revenue"].sum().reset_index()
    fig2 = px.sunburst(sun, path=["category","zone","brand_type"], values="revenue",
                       title="Revenue Sunburst", template="plotly_white",
                       color="revenue", color_continuous_scale="RdBu")
    st.plotly_chart(fig2, use_container_width=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Data citations:**  "
    "World Bank Open Data (CC BY 4.0) · "
    "exchangerate.host · "
    "Google Trends via pytrends (Apache 2.0) · "
    "NewsAPI.org"
)
