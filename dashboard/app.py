"""
dashboard/app.py
Live URL : https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/
GitHub   : https://github.com/SumedhPatil1507/indian-ecommerce-analytics
"""
import io, os, sys

# Resolve project root – works on local, Docker, and Streamlit Cloud
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
from data.loader import (
    _clean, _engineer, load_any, generate_live_dataset,
    fetch_worldbank, fetch_usd_inr, fetch_google_trends,
    _WB_GDP, _WB_CPI, KAGGLE_URL,
)
st.set_page_config(page_title="Indian E-Commerce Analytics", page_icon="🛒", layout="wide", initial_sidebar_state="expanded")
@st.cache_data(ttl=3600, show_spinner=False)
def _live_macro():
    return fetch_worldbank(_WB_GDP), fetch_worldbank(_WB_CPI), fetch_usd_inr()
@st.cache_data(ttl=3600, show_spinner=False)
def _live_trends():
    try: return fetch_google_trends(timeframe="today 3-m")
    except: return pd.DataFrame()
@st.cache_data(ttl=3600, show_spinner="Generating live dataset...")
def _live_dataset(n: int):
    return generate_live_dataset(n_rows=int(n), months=36)
@st.cache_data(show_spinner="Processing file...")
def _from_upload(raw: bytes, fname: str):
    # load_any expects a file-like object + filename for format detection
    return load_any(io.BytesIO(raw), fname)
@st.cache_data(show_spinner="Loading...")
def _from_path(p):
    return _engineer(_clean(pd.read_csv(p)))
with st.sidebar:
    st.title("E-Commerce Analytics")
    st.markdown("---")
    st.subheader("Data Source")
    mode = st.radio("Choose source", ["Live data (no upload needed)", "Upload dataset", "Config / env path"], index=0)
    df = None
    if mode == "Live data (no upload needed)":
        n_rows = st.slider("Synthetic rows", 1000, 10000, 5000, 500)
        if st.button("Regenerate with latest macro"):
            st.cache_data.clear()
        df = _live_dataset(n_rows)
        st.success(f"{len(df):,} rows — live macro calibrated")
        st.caption("Prices <- CPI  |  Revenue trend <- GDP  |  Festivals <- Trends")
    elif mode == "Upload dataset":
        up = st.file_uploader("Upload file", type=["csv","tsv","xlsx","xls","json","parquet"], help="CSV, TSV, Excel, JSON, Parquet")
        st.caption(f"[Download from Kaggle]({KAGGLE_URL})")
        if up:
            try:
                df = _from_upload(up.read(), up.name)
                st.success(f"{len(df):,} rows from {up.name}")
            except Exception as e:
                st.error(str(e))
        else:
            st.info("Upload a file, or switch to Live data mode.")
    else:
        p = ""
        try: p = st.secrets.get("DATA_PATH", "")
        except: pass
        p = p or os.getenv("DATA_PATH", "")
        if p and os.path.exists(p):
            try:
                df = _from_path(p)
                st.success(f"{len(df):,} rows from config")
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("Set DATA_PATH in Streamlit secrets or env.")
    st.markdown("---")
    st.subheader("Live Signals")
    if st.button("Refresh live data"):
        st.cache_data.clear(); st.rerun()
    st.caption("Auto-refreshes every hour.")
    if df is not None:
        st.markdown("---")
        st.subheader("Filters")
        zones  = st.multiselect("Zone",        sorted(df["zone"].unique()),       default=sorted(df["zone"].unique()))
        cats   = st.multiselect("Category",    sorted(df["category"].unique()),   default=sorted(df["category"].unique()))
        brands = st.multiselect("Brand Type",  sorted(df["brand_type"].unique()), default=sorted(df["brand_type"].unique()))
        events = st.multiselect("Sales Event", sorted(df["sales_event"].unique()),default=sorted(df["sales_event"].unique()))
st.title("Indian E-Commerce Analytics")
st.caption("Live app: https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/ | GitHub: https://github.com/SumedhPatil1507/indian-ecommerce-analytics | World Bank (CC BY 4.0) · exchangerate.host · pytrends (Apache 2.0)")
st.subheader("Live Macro Signals")
gdp_df, cpi_df, fx = _live_macro()
trends_df = _live_trends()
gdp_val   = float(gdp_df["value"].iloc[-1])  if not gdp_df.empty else None
cpi_val   = float(cpi_df["value"].iloc[-1])  if not cpi_df.empty else None
gdp_delta = round(gdp_df["value"].iloc[-1] - gdp_df["value"].iloc[-2], 2) if len(gdp_df)>=2 else None
cpi_delta = round(cpi_df["value"].iloc[-1] - cpi_df["value"].iloc[-2], 2) if len(cpi_df)>=2 else None
trend_val = float(trends_df.mean().mean()) if not trends_df.empty else None
c1,c2,c3,c4 = st.columns(4)
c1.metric("India GDP Growth",  f"{gdp_val:.2f}%"      if gdp_val  else "N/A", delta=f"{gdp_delta:+.2f}pp" if gdp_delta else None, help="World Bank CC BY 4.0")
c2.metric("CPI Inflation",     f"{cpi_val:.2f}%"      if cpi_val  else "N/A", delta=f"{cpi_delta:+.2f}pp" if cpi_delta else None, help="World Bank CC BY 4.0")
c3.metric("USD / INR",         f"Rs{fx:.2f}",          help="exchangerate.host")
c4.metric("Search Interest",   f"{trend_val:.1f}/100" if trend_val else "N/A", help="Google Trends via pytrends (Apache 2.0)")
with st.expander("GDP & CPI history (World Bank)", expanded=False):
    col1, col2 = st.columns(2)
    if not gdp_df.empty:
        col1.plotly_chart(px.line(gdp_df, x="year", y="value", markers=True, title="India GDP Growth (%)", template="plotly_white", labels={"value":"GDP %","year":"Year"}), use_container_width=True)
    if not cpi_df.empty:
        col2.plotly_chart(px.line(cpi_df, x="year", y="value", markers=True, title="India CPI Inflation (%)", template="plotly_white", labels={"value":"CPI %","year":"Year"}, color_discrete_sequence=["coral"]), use_container_width=True)
    st.caption("Source: World Bank Open Data https://data.worldbank.org/country/india CC BY 4.0")
if not trends_df.empty:
    with st.expander("Google Trends - E-commerce search interest (India)", expanded=False):
        # reset_index safely handles both DatetimeIndex and named index
        tr = trends_df.copy()
        tr.index.name = "date"
        tp = tr.reset_index().melt(id_vars="date", var_name="keyword", value_name="interest")
        st.plotly_chart(px.line(tp, x="date", y="interest", color="keyword",
            title="Search Interest (0-100)", template="plotly_white"), use_container_width=True)
        st.caption("Source: Google Trends via pytrends https://github.com/GeneralMills/pytrends Apache 2.0")
st.markdown("---")
if df is None:
    st.info(f"### Select a data source in the sidebar\n\nLive data mode works instantly — no file needed.\n\nOr upload the Kaggle CSV / Excel / JSON / Parquet.\n\nDownload dataset: {KAGGLE_URL}")
    st.stop()
dff = df[df["zone"].isin(zones) & df["category"].isin(cats) & df["brand_type"].isin(brands) & df["sales_event"].isin(events)].copy()
if not gdp_df.empty:
    macro = gdp_df[["year","value"]].rename(columns={"value":"india_gdp_growth_pct"})
    macro = macro.merge(cpi_df[["year","value"]].rename(columns={"value":"india_cpi_inflation_pct"}), on="year", how="outer")
    dff = dff.merge(macro, on="year", how="left")
dff["usd_inr_rate"] = fx
dff["revenue_usd"]  = (dff["revenue"] / fx).round(2)
st.subheader("Dataset KPIs")
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Total Revenue",     f"Rs{dff['revenue'].sum()/1e7:.1f} Cr")
k2.metric("Total Orders",      f"{len(dff):,}")
k3.metric("Avg Order Value",   f"Rs{dff['revenue'].mean():,.0f}")
k4.metric("Avg Discount",      f"{dff['discount_percent'].mean():.1f}%")
k5.metric("Avg Units / Order", f"{dff['units_sold'].mean():.1f}")
st.caption(f"Total revenue in USD: ${dff['revenue_usd'].sum():,.0f}  (@ Rs{fx:.2f}/USD)")
if "data_source" in dff.columns and (dff["data_source"] == "live_synthetic").any():
    st.info("Showing live synthetic data calibrated to real GDP, CPI and FX. Upload the Kaggle CSV for historical analysis.")
st.markdown("---")
tabs = st.tabs(["Revenue","Categories","Regional","Elasticity","Inventory","CLV","Anomalies","Cohort","Pareto/Sunburst"])
with tabs[0]:
    st.subheader("Revenue Trends")
    m = dff.groupby("year_month")["revenue"].sum().reset_index()
    st.plotly_chart(px.line(m, x="year_month", y="revenue", markers=True, title="Total Monthly Revenue", labels={"revenue":"Revenue (Rs)","year_month":"Month"}, template="plotly_white"), use_container_width=True)
    c1,c2 = st.columns(2)
    aov = dff.groupby("year_month")["revenue"].mean().reset_index()
    c1.plotly_chart(px.line(aov, x="year_month", y="revenue", markers=True, title="Avg Order Value", labels={"revenue":"AOV (Rs)"}, template="plotly_white"), use_container_width=True)
    disc = dff.groupby("year_month")["discount_percent"].mean().reset_index()
    c2.plotly_chart(px.line(disc, x="year_month", y="discount_percent", markers=True, title="Avg Discount %", template="plotly_white"), use_container_width=True)
    zm = dff.groupby(["year_month","zone"])["revenue"].sum().reset_index()
    st.plotly_chart(px.line(zm, x="year_month", y="revenue", color="zone", markers=True, title="Revenue by Zone", template="plotly_white"), use_container_width=True)
    bm = dff.groupby(["year_month","brand_type"])["revenue"].sum().reset_index()
    st.plotly_chart(px.line(bm, x="year_month", y="revenue", color="brand_type", markers=True, title="Revenue: Mass vs Premium", template="plotly_white"), use_container_width=True)
with tabs[1]:
    st.subheader("Category & Brand Analysis")
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.pie(dff.groupby("category")["revenue"].sum().reset_index(), names="category", values="revenue", title="Revenue by Category", hole=0.4, template="plotly_white"), use_container_width=True)
    c2.plotly_chart(px.pie(dff.groupby("brand_type")["revenue"].sum().reset_index(), names="brand_type", values="revenue", title="Mass vs Premium", hole=0.4, template="plotly_white"), use_container_width=True)
    metric = st.selectbox("Metric", ["revenue","final_price","units_sold","discount_percent"])
    agg = dff.groupby("category")[metric].mean().reset_index().sort_values(metric, ascending=False)
    st.plotly_chart(px.bar(agg, x="category", y=metric, color="category", title=f"Avg {metric} by Category", template="plotly_white"), use_container_width=True)
    em = dff.groupby(["year_month","sales_event"])["revenue"].sum().reset_index()
    st.plotly_chart(px.bar(em, x="year_month", y="revenue", color="sales_event", title="Festival vs Normal Revenue", template="plotly_white", barmode="group"), use_container_width=True)
with tabs[2]:
    st.subheader("Regional Analysis")
    sr = dff.groupby("state")["revenue"].sum().nlargest(15).reset_index()
    st.plotly_chart(px.bar(sr, x="revenue", y="state", orientation="h", title="Top 15 States by Revenue", template="plotly_white", color="revenue", color_continuous_scale="YlOrRd"), use_container_width=True)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.pie(dff.groupby("zone")["revenue"].sum().reset_index(), names="zone", values="revenue", title="Revenue by Zone", hole=0.4, template="plotly_white"), use_container_width=True)
    c2.plotly_chart(px.bar(dff.groupby("zone")["units_sold"].mean().reset_index(), x="zone", y="units_sold", color="zone", title="Avg Units by Zone", template="plotly_white"), use_container_width=True)
with tabs[3]:
    st.subheader("Price Elasticity Engine")
    from modules.price_elasticity import compute_elasticity
    with st.spinner("Computing..."):
        elast = compute_elasticity(dff, group_cols=["category"])
    if not elast.empty:
        elast["type"] = elast["elasticity"].apply(lambda e: "Elastic (<-1)" if e<-1 else ("Inelastic" if e<0 else "Giffen/Luxury"))
        fig = px.bar(elast.sort_values("elasticity"), x="elasticity", y="category", orientation="h", color="type", color_discrete_map={"Elastic (<-1)":"crimson","Inelastic":"steelblue","Giffen/Luxury":"gold"}, title="Price Elasticity by Category", template="plotly_white", text="elasticity")
        fig.add_vline(x=-1, line_dash="dash", line_color="gray", annotation_text="Elastic threshold")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(elast, use_container_width=True)
    else:
        st.warning("Not enough data for elasticity with current filters.")
with tabs[4]:
    st.subheader("Inventory Alert System")
    from modules.inventory_alerts import compute_alerts
    alerts = compute_alerts(dff)
    cmap = {"🔴 CRITICAL – Reorder Now":"red","🟠 HIGH – Monitor Closely":"orange","🟡 CLEARANCE – Excess Stock":"gold","🔵 SLOW MOVER – Review Listing":"steelblue","🟢 HEALTHY":"green"}
    st.plotly_chart(px.scatter(alerts, x="avg_discount", y="avg_units_sold", color="alert_level", size="high_pressure_pct", hover_data=["category","zone","recommendation"], color_discrete_map=cmap, title="Inventory Alert Dashboard", template="plotly_white"), use_container_width=True)
    af = st.multiselect("Filter alerts", list(cmap.keys()), default=list(cmap.keys()))
    st.dataframe(alerts[alerts["alert_level"].isin(af)][["category","zone","avg_units_sold","avg_discount","alert_level","recommendation"]], use_container_width=True)
with tabs[5]:
    st.subheader("Customer Lifetime Value")
    from modules.clv import compute_clv
    with st.spinner("Computing CLV..."):
        clv_df = compute_clv(dff)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.histogram(clv_df, x="clv", color="clv_tier", nbins=50, title="CLV Distribution", template="plotly_white", marginal="box"), use_container_width=True)
    c2.plotly_chart(px.pie(clv_df.groupby("clv_tier")["clv"].sum().reset_index(), names="clv_tier", values="clv", title="CLV Share by Tier", hole=0.4, template="plotly_white"), use_container_width=True)
    st.plotly_chart(px.scatter(clv_df, x="frequency", y="clv", color="clv_tier", opacity=0.6, size="monetary", title="Frequency vs CLV", template="plotly_white"), use_container_width=True)
    st.dataframe(clv_df.groupby("clv_tier")["clv"].agg(count="count",mean_clv="mean",total_clv="sum").round(2), use_container_width=True)
with tabs[6]:
    st.subheader("Anomaly Detection")
    from modules.anomaly import anomaly_report
    with st.spinner("Detecting anomalies..."):
        anom = anomaly_report(dff)
    c1,c2 = st.columns(2)
    c1.plotly_chart(px.scatter(anom, x="log_units_sold", y="log_revenue", color="confirmed_anomaly", color_discrete_map={True:"red",False:"steelblue"}, opacity=0.5, title="log(Units) vs log(Revenue)", hover_data=["category","zone","discount_percent"], template="plotly_white"), use_container_width=True)
    c2.plotly_chart(px.scatter(anom, x="discount_percent", y="log_final_price", color="confirmed_anomaly", color_discrete_map={True:"red",False:"lightgray"}, opacity=0.55, title="Discount % vs log(Final Price)", hover_data=["category","zone"], template="plotly_white"), use_container_width=True)
    n = anom["confirmed_anomaly"].sum()
    st.info(f"Confirmed anomalies (>=2 detectors): **{n:,}** ({n/len(anom):.2%})")
    st.plotly_chart(px.bar(anom[anom["confirmed_anomaly"]].groupby("category").size().reset_index(name="count").sort_values("count",ascending=False), x="category", y="count", color="category", title="Anomalies by Category", template="plotly_white"), use_container_width=True)
with tabs[7]:
    st.subheader("Cohort Analysis")
    from modules.cohort import build_cohort_table
    c1,c2 = st.columns(2)
    pivot = build_cohort_table(dff, metric="count")
    ret = (pivot.div(pivot[0], axis=0)*100).round(1)
    c1.plotly_chart(px.imshow(ret, color_continuous_scale="Blues", title="Retention Rate (%) - Orders", labels={"x":"Months since first purchase","y":"Cohort","color":"Retention %"}, text_auto=".0f", template="plotly_white"), use_container_width=True)
    pivot_r = build_cohort_table(dff, metric="revenue")
    ret_r = (pivot_r.div(pivot_r[0], axis=0)*100).round(1)
    c2.plotly_chart(px.imshow(ret_r, color_continuous_scale="Greens", title="Revenue Retention (%)", labels={"x":"Months since first purchase","y":"Cohort","color":"Retention %"}, text_auto=".0f", template="plotly_white"), use_container_width=True)
with tabs[8]:
    st.subheader("Pareto & Sunburst")
    agg = dff.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    agg["cum_pct"] = agg["revenue"].cumsum() / agg["revenue"].sum() * 100
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(x=agg["category"], y=agg["revenue"], name="Revenue", marker_color="steelblue"), secondary_y=False)
    fig.add_trace(go.Scatter(x=agg["category"], y=agg["cum_pct"], mode="lines+markers", name="Cumulative %", line=dict(color="crimson",width=2.5)), secondary_y=True)
    fig.add_hline(y=80, line_dash="dash", line_color="orange", annotation_text="80%", secondary_y=True)
    fig.update_layout(title="Pareto - Revenue by Category (80/20)", template="plotly_white")
    fig.update_yaxes(title_text="Revenue (Rs)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)
    sun = dff.groupby(["category","zone","brand_type"])["revenue"].sum().reset_index()
    st.plotly_chart(px.sunburst(sun, path=["category","zone","brand_type"], values="revenue", title="Revenue Sunburst - Category > Zone > Brand", template="plotly_white", color="revenue", color_continuous_scale="RdBu"), use_container_width=True)
    vals = np.sort(dff["revenue"].dropna().values)[::-1]
    cum  = np.cumsum(vals)/vals.sum()
    x    = np.linspace(0,1,len(cum))
    gini = round(1 - 2*np.trapezoid(cum,x), 3)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash",color="gray"),name="Perfect equality"))
    fig2.add_trace(go.Scatter(x=x,y=cum,mode="lines",fill="tozeroy",fillcolor="rgba(220,20,60,0.12)",line=dict(color="crimson",width=2.5),name=f"Lorenz (Gini={gini})"))
    fig2.update_layout(title=f"Lorenz Curve - Revenue Concentration (Gini={gini})", xaxis_title="Cumulative share of orders", yaxis_title="Cumulative share of revenue", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
st.markdown("---")
st.caption("Citations: World Bank Open Data (CC BY 4.0) | exchangerate.host | pytrends/Google Trends (Apache 2.0) | NewsAPI.org | Kaggle: https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth")


