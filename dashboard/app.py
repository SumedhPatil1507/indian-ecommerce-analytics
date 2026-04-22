"""
dashboard/app.py
Streamlit Interactive Dashboard – all charts are Plotly (interactive).

Run locally:  streamlit run dashboard/app.py
"""
import io
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data.loader import _clean, _engineer, fetch_worldbank, fetch_usd_inr, fetch_google_trends
from data.loader import _WB_GDP, _WB_CPI

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🛒 Indian E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE DATA  (always shown, independent of dataset)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)   # refresh every hour
def _live_macro():
    gdp = fetch_worldbank(_WB_GDP)
    cpi = fetch_worldbank(_WB_CPI)
    fx  = fetch_usd_inr()
    return gdp, cpi, fx

@st.cache_data(ttl=3600, show_spinner=False)
def _live_trends():
    return fetch_google_trends(timeframe="today 3-m")

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_secret_path() -> str | None:
    try:
        p = st.secrets.get("DATA_PATH", "")
        if p and os.path.exists(p):
            return p
    except Exception:
        pass
    env = os.getenv("DATA_PATH", "")
    if env and os.path.exists(env):
        return env
    return None

@st.cache_data(show_spinner="Processing dataset…")
def _load_bytes(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    return _engineer(_clean(df))

@st.cache_data(show_spinner="Loading dataset…")
def _load_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _engineer(_clean(df))

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🛒 E-Commerce Analytics")
    st.markdown("---")

    # ── Dataset section ───────────────────────────────────────────────────────
    st.subheader("📂 Dataset")

    secret_path = _resolve_secret_path()
    df = None

    if secret_path:
        try:
            df = _load_path(secret_path)
            st.success(f"✅ Dataset loaded from config")
        except Exception as e:
            st.warning(f"Config path failed: {e}")

    if df is None:
        uploaded = st.file_uploader(
            "Upload Kaggle CSV",
            type=["csv"],
            help="Download from Kaggle → upload here",
        )
        st.caption(
            "📥 [Download dataset from Kaggle](https://www.kaggle.com/datasets/"
            "sahilislam007/indian-e-commerce-pricing-revenue-growth-36-months)"
        )
        if uploaded:
            try:
                df = _load_bytes(uploaded.read())
                st.success(f"✅ {len(df):,} rows loaded")
            except Exception as e:
                st.error(f"Parse error: {e}")

    st.markdown("---")

    # ── Live data controls ────────────────────────────────────────────────────
    st.subheader("📡 Live Data")
    enrich_live = st.checkbox("Merge live macro into dataset", value=True,
                              disabled=(df is None))
    if st.button("🔄 Refresh live data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ── Filters (only when dataset is loaded) ─────────────────────────────────
    if df is not None:
        st.subheader("🔍 Filters")
        zones      = st.multiselect("Zone",        df["zone"].unique(),       default=list(df["zone"].unique()))
        categories = st.multiselect("Category",    df["category"].unique(),   default=list(df["category"].unique()))
        brands     = st.multiselect("Brand Type",  df["brand_type"].unique(), default=list(df["brand_type"].unique()))
        events     = st.multiselect("Sales Event", df["sales_event"].unique(),default=list(df["sales_event"].unique()))

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("🛒 Indian E-Commerce Analytics")
st.caption(
    "Live macro: [World Bank Open Data](https://data.worldbank.org/) (CC BY 4.0) · "
    "FX: [exchangerate.host](https://exchangerate.host) · "
    "Trends: [pytrends](https://github.com/GeneralMills/pytrends) (Apache 2.0)"
)

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE MACRO ROW  (always visible)
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📡 Live Macro Signals")
with st.spinner("Fetching live data…"):
    gdp_df, cpi_df, fx_rate = _live_macro()

m1, m2, m3, m4 = st.columns(4)

gdp_val = gdp_df["value"].iloc[-1] if not gdp_df.empty else None
cpi_val = cpi_df["value"].iloc[-1] if not cpi_df.empty else None

gdp_delta = round(gdp_df["value"].iloc[-1] - gdp_df["value"].iloc[-2], 2) if len(gdp_df) >= 2 else None
cpi_delta = round(cpi_df["value"].iloc[-1] - cpi_df["value"].iloc[-2], 2) if len(cpi_df) >= 2 else None

m1.metric(
    "🌐 India GDP Growth",
    f"{gdp_val:.2f}%" if gdp_val else "N/A",
    delta=f"{gdp_delta:+.2f}pp" if gdp_delta else None,
    help="Source: World Bank Open Data (CC BY 4.0) · refreshes hourly",
)
m2.metric(
    "📈 CPI Inflation",
    f"{cpi_val:.2f}%" if cpi_val else "N/A",
    delta=f"{cpi_delta:+.2f}pp" if cpi_delta else None,
    help="Source: World Bank Open Data (CC BY 4.0) · refreshes hourly",
)
m3.metric(
    "💱 USD / INR",
    f"₹{fx_rate:.2f}",
    help="Source: exchangerate.host · refreshes hourly",
)

# Google Trends (non-blocking)
try:
    trends_df = _live_trends()
    trend_val = trends_df.mean().mean() if not trends_df.empty else None
except Exception:
    trend_val = None

m4.metric(
    "🔍 E-Commerce Search Interest",
    f"{trend_val:.1f} / 100" if trend_val else "N/A",
    help="Source: Google Trends via pytrends (Apache 2.0) · India · last 3 months",
)

# GDP trend sparkline
if not gdp_df.empty:
    with st.expander("📊 India GDP & CPI trend (World Bank)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(gdp_df, x="year", y="value", markers=True,
                          title="India GDP Growth (annual %)",
                          labels={"value": "GDP Growth %", "year": "Year"},
                          template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.line(cpi_df, x="year", y="value", markers=True,
                          title="India CPI Inflation (annual %)",
                          labels={"value": "CPI %", "year": "Year"},
                          template="plotly_white", color_discrete_sequence=["coral"])
            st.plotly_chart(fig, use_container_width=True)
        st.caption("Source: [World Bank Open Data](https://data.worldbank.org/country/india) · License: CC BY 4.0")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  DATASET SECTION  (gated behind upload)
# ─────────────────────────────────────────────────────────────────────────────

if df is None:
    st.info(
        "### 👈 Upload your dataset to unlock full analytics\n\n"
        "The live macro signals above update automatically.\n\n"
        "To explore pricing, revenue, elasticity, CLV, anomalies and more — "
        "upload the Kaggle CSV using the sidebar.\n\n"
        "📥 **[Download dataset from Kaggle](https://www.kaggle.com/datasets/"
        "sahilislam007/indian-e-commerce-pricing-revenue-growth-36-months)**"
    )
    st.stop()

# ── apply filters ─────────────────────────────────────────────────────────────
mask = (
    df["zone"].isin(zones) &
    df["category"].isin(categories) &
    df["brand_type"].isin(brands) &
    df["sales_event"].isin(events)
)
dff = df[mask]

# ── optionally merge live macro into filtered df ──────────────────────────────
if enrich_live and not gdp_df.empty:
    macro = gdp_df[["year","value"]].rename(columns={"value":"india_gdp_growth_pct"})
    macro = macro.merge(
        cpi_df[["year","value"]].rename(columns={"value":"india_cpi_inflation_pct"}),
        on="year", how="outer"
    )
    dff = dff.merge(macro, on="year", how="left")
    dff["usd_inr_rate"] = fx_rate
    dff["revenue_usd"]  = (dff["revenue"] / fx_rate).round(2)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.subheader("📊 Dataset KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Revenue",     f"₹{dff['revenue'].sum()/1e7:.1f} Cr")
k2.metric("Total Orders",      f"{len(dff):,}")
k3.metric("Avg Order Value",   f"₹{dff['revenue'].mean():,.0f}")
k4.metric("Avg Discount",      f"{dff['discount_percent'].mean():.1f}%")
k5.metric("Avg Units / Order", f"{dff['units_sold'].mean():.1f}")

if "revenue_usd" in dff.columns:
    st.caption(f"💱 Total revenue in USD: **${dff['revenue_usd'].sum():,.0f}**  (@ ₹{fx_rate:.2f}/USD)")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYTICS TABS
# ─────────────────────────────────────────────────────────────────────────────

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
        fig = px.line(aov, x="year_month", y="revenue", markers=True,
                      title="Average Order Value (AOV)",
                      labels={"revenue": "AOV (₹)"}, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        disc = dff.groupby("year_month")["discount_percent"].mean().reset_index()
        fig = px.line(disc, x="year_month", y="discount_percent", markers=True,
                      title="Avg Discount % Trend",
                      labels={"discount_percent": "Discount %"}, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    zone_m = dff.groupby(["year_month", "zone"])["revenue"].sum().reset_index()
    fig = px.line(zone_m, x="year_month", y="revenue", color="zone", markers=True,
                  title="Revenue by Zone", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    brand_m = dff.groupby(["year_month", "brand_type"])["revenue"].sum().reset_index()
    fig = px.line(brand_m, x="year_month", y="revenue", color="brand_type", markers=True,
                  title="Revenue – Mass vs Premium", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

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

    metric = st.selectbox("Metric", ["revenue", "final_price", "units_sold", "discount_percent"])
    agg = dff.groupby("category")[metric].mean().reset_index().sort_values(metric, ascending=False)
    fig = px.bar(agg, x="category", y=metric, color="category",
                 title=f"Avg {metric.replace('_',' ').title()} by Category",
                 template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # festival vs normal
    event_m = dff.groupby(["year_month", "sales_event"])["revenue"].sum().reset_index()
    fig = px.bar(event_m, x="year_month", y="revenue", color="sales_event",
                 title="Festival vs Normal Revenue by Month",
                 template="plotly_white", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Regional ───────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Regional Analysis")
    state_rev = dff.groupby("state")["revenue"].sum().nlargest(15).reset_index()
    fig = px.bar(state_rev, x="revenue", y="state", orientation="h",
                 title="Top 15 States by Revenue", template="plotly_white",
                 color="revenue", color_continuous_scale="YlOrRd")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        zone_rev = dff.groupby("zone")["revenue"].sum().reset_index()
        fig = px.pie(zone_rev, names="zone", values="revenue",
                     title="Revenue Share by Zone", hole=0.4, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        zone_units = dff.groupby("zone")["units_sold"].mean().reset_index()
        fig = px.bar(zone_units, x="zone", y="units_sold", color="zone",
                     title="Avg Units Sold by Zone", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Price Elasticity ───────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Price Elasticity Engine")
    from modules.price_elasticity import compute_elasticity
    with st.spinner("Computing elasticity…"):
        elast = compute_elasticity(dff, group_cols=["category"])
    if not elast.empty:
        elast["type"] = elast["elasticity"].apply(
            lambda e: "Elastic (< -1)" if e < -1 else ("Inelastic" if e < 0 else "Giffen/Luxury"))
        fig = px.bar(
            elast.sort_values("elasticity"), x="elasticity", y="category",
            orientation="h", color="type",
            color_discrete_map={"Elastic (< -1)": "crimson", "Inelastic": "steelblue", "Giffen/Luxury": "gold"},
            title="Price Elasticity by Category  (β < -1 = elastic demand)",
            labels={"elasticity": "Elasticity Coefficient (β)"}, template="plotly_white",
            text="elasticity",
        )
        fig.add_vline(x=-1, line_dash="dash", line_color="gray",
                      annotation_text="Elastic threshold")
        fig.add_vline(x=0, line_dash="dot", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(elast, use_container_width=True)
    else:
        st.warning("Not enough data to compute elasticity with current filters.")

# ── Tab 4: Inventory Alerts ───────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Dynamic Inventory Alert System")
    from modules.inventory_alerts import compute_alerts
    alerts = compute_alerts(dff)
    colour_map = {
        "🔴 CRITICAL – Reorder Now":     "red",
        "🟠 HIGH – Monitor Closely":     "orange",
        "🟡 CLEARANCE – Excess Stock":   "gold",
        "🔵 SLOW MOVER – Review Listing":"steelblue",
        "🟢 HEALTHY":                    "green",
    }
    fig = px.scatter(
        alerts, x="avg_discount", y="avg_units_sold",
        color="alert_level", size="high_pressure_pct",
        hover_data=["category", "zone", "recommendation"],
        color_discrete_map=colour_map,
        title="Inventory Alert Dashboard – Discount vs Velocity",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    alert_filter = st.multiselect(
        "Filter by alert level", options=list(colour_map.keys()),
        default=list(colour_map.keys()),
    )
    st.dataframe(
        alerts[alerts["alert_level"].isin(alert_filter)]
        [["category", "zone", "avg_units_sold", "avg_discount", "alert_level", "recommendation"]],
        use_container_width=True,
    )

# ── Tab 5: CLV ────────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Customer Lifetime Value")
    from modules.clv import compute_clv
    with st.spinner("Computing CLV…"):
        clv_df = compute_clv(dff)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(clv_df, x="clv", color="clv_tier", nbins=50,
                           title="CLV Distribution by Tier", template="plotly_white",
                           marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        tier_rev = clv_df.groupby("clv_tier")["clv"].sum().reset_index()
        fig = px.pie(tier_rev, names="clv_tier", values="clv",
                     title="CLV Share by Tier", hole=0.4, template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(clv_df, x="frequency", y="clv", color="clv_tier",
                     opacity=0.6, size="monetary",
                     title="Purchase Frequency vs CLV",
                     labels={"frequency": "Purchase Frequency", "clv": "CLV (₹)"},
                     template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**CLV Tier Summary**")
    st.dataframe(
        clv_df.groupby("clv_tier")["clv"].agg(count="count", mean_clv="mean", total_clv="sum").round(2),
        use_container_width=True,
    )

# ── Tab 6: Anomalies ──────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Automated Anomaly Detection")
    from modules.anomaly import anomaly_report
    with st.spinner("Running anomaly detection…"):
        anom_df = anomaly_report(dff)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(anom_df, x="log_units_sold", y="log_revenue",
                         color="confirmed_anomaly",
                         color_discrete_map={True: "red", False: "steelblue"},
                         opacity=0.5, title="log(Units) vs log(Revenue)",
                         hover_data=["category", "zone", "discount_percent"],
                         template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(anom_df, x="discount_percent", y="log_final_price",
                         color="confirmed_anomaly",
                         color_discrete_map={True: "red", False: "lightgray"},
                         opacity=0.55, title="Discount % vs log(Final Price)",
                         hover_data=["category", "zone"],
                         template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    n = anom_df["confirmed_anomaly"].sum()
    st.info(f"Confirmed anomalies (≥2 detectors): **{n:,}** ({n/len(anom_df):.2%} of orders)")

    cat_anom = (anom_df[anom_df["confirmed_anomaly"]]
                .groupby("category").size().reset_index(name="count")
                .sort_values("count", ascending=False))
    fig = px.bar(cat_anom, x="category", y="count", color="category",
                 title="Anomaly Count by Category", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 7: Cohort ─────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Cohort Analysis")
    from modules.cohort import build_cohort_table
    col1, col2 = st.columns(2)
    with col1:
        pivot = build_cohort_table(dff, metric="count")
        retention = (pivot.div(pivot[0], axis=0) * 100).round(1)
        fig = px.imshow(retention, color_continuous_scale="Blues",
                        title="Cohort Retention Rate (%) – Order Count",
                        labels={"x": "Months since first purchase", "y": "Cohort Month", "color": "Retention %"},
                        text_auto=".0f", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        pivot_r = build_cohort_table(dff, metric="revenue")
        ret_r = (pivot_r.div(pivot_r[0], axis=0) * 100).round(1)
        fig = px.imshow(ret_r, color_continuous_scale="Greens",
                        title="Cohort Revenue Retention (%)",
                        labels={"x": "Months since first purchase", "y": "Cohort Month", "color": "Retention %"},
                        text_auto=".0f", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 8: Pareto / Sunburst ──────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Pareto & Sunburst")

    # Pareto
    agg = dff.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    agg["cum_pct"] = agg["revenue"].cumsum() / agg["revenue"].sum() * 100
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=agg["category"], y=agg["revenue"], name="Revenue",
                         marker_color="steelblue"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["category"], y=agg["cum_pct"],
                             mode="lines+markers", name="Cumulative %",
                             line=dict(color="crimson", width=2.5)), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="orange",
                  annotation_text="80% threshold", secondary_y=True)
    fig.update_layout(title="Pareto Chart – Revenue by Category (80/20 Rule)",
                      template="plotly_white")
    fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Sunburst
    sun = dff.groupby(["category", "zone", "brand_type"])["revenue"].sum().reset_index()
    fig = px.sunburst(sun, path=["category", "zone", "brand_type"], values="revenue",
                      title="Revenue Sunburst – Category → Zone → Brand Type",
                      template="plotly_white", color="revenue",
                      color_continuous_scale="RdBu")
    fig.update_traces(textinfo="label+percent parent")
    st.plotly_chart(fig, use_container_width=True)

    # Lorenz curve
    vals = np.sort(dff["revenue"].dropna().values)[::-1]
    cum  = np.cumsum(vals) / vals.sum()
    x    = np.linspace(0, 1, len(cum))
    gini = round(1 - 2 * np.trapz(cum, x), 3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Perfect equality"))
    fig.add_trace(go.Scatter(x=x, y=cum, mode="lines", fill="tozeroy",
                             fillcolor="rgba(220,20,60,0.12)",
                             line=dict(color="crimson", width=2.5),
                             name=f"Lorenz curve (Gini={gini})"))
    fig.update_layout(title=f"Lorenz Curve – Revenue Concentration  (Gini = {gini})",
                      xaxis_title="Cumulative share of orders",
                      yaxis_title="Cumulative share of revenue",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Citations:** "
    "[World Bank Open Data](https://data.worldbank.org/) (CC BY 4.0) · "
    "[exchangerate.host](https://exchangerate.host) · "
    "[pytrends / Google Trends](https://github.com/GeneralMills/pytrends) (Apache 2.0) · "
    "[NewsAPI.org](https://newsapi.org)"
)
