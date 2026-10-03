"""
modules/pareto.py

Premium Visualisations
   Pareto Chart (80/20 Rule)
   Interactive Sunburst Chart (category  zone  brand_type)
   SHAP Summary Plot (interactive via Plotly)
   Regional Choropleth Map (India states)
   Advanced plots: Q-Q, Lorenz curve, ECDF, rolling stats
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#  Pareto Chart 

def plot_pareto(
    df: pd.DataFrame,
    group_col: str = "category",
    value_col: str = "revenue",
):
    """
    Pareto chart: bars = revenue per group, line = cumulative %.
    Highlights the 80% threshold.
    """
    agg = (
        df.groupby(group_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    agg["cumulative_pct"] = agg[value_col].cumsum() / agg[value_col].sum() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=agg[group_col], y=agg[value_col], name="Revenue",
               marker_color="steelblue"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=agg[group_col], y=agg["cumulative_pct"],
                   mode="lines+markers", name="Cumulative %",
                   line=dict(color="crimson", width=2.5)),
        secondary_y=True,
    )
    fig.add_hline(y=80, line_dash="dash", line_color="orange",
                  annotation_text="80% threshold", secondary_y=True)
    fig.update_layout(
        title=f"Pareto Chart  {value_col.title()} by {group_col.title()} (80/20 Rule)",
        template="plotly_white",
    )
    fig.update_yaxes(title_text=f"Total {value_col.title()} ()", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    return fig


#  Sunburst Chart 

def plot_sunburst(df: pd.DataFrame):
    """
    Interactive sunburst: category  zone  brand_type, sized by revenue.
    """
    agg = (
        df.groupby(["category", "zone", "brand_type"])["revenue"]
        .sum()
        .reset_index()
    )
    fig = px.sunburst(
        agg,
        path=["category", "zone", "brand_type"],
        values="revenue",
        title="Revenue Sunburst  Category  Zone  Brand Type",
        template="plotly_white",
        color="revenue",
        color_continuous_scale="RdBu",
    )
    fig.update_traces(textinfo="label+percent parent")
    return fig


#  SHAP Summary (interactive Plotly) 

def plot_shap_summary(shap_values: np.ndarray, feature_names: list, top_n: int = 15):
    """
    Interactive SHAP beeswarm-style bar chart using Plotly.

    Parameters
    ----------
    shap_values   : 2-D array (n_samples  n_features)
    feature_names : list of feature name strings
    top_n         : number of top features to show
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx      = np.argsort(mean_abs)[-top_n:]
    names    = [feature_names[i] for i in idx]
    vals     = mean_abs[idx]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(
            color=vals,
            colorscale="RdBu",
            showscale=True,
            colorbar=dict(title="Mean |SHAP|"),
        ),
    ))
    fig.update_layout(
        title=f"SHAP Feature Importance  Top {top_n} Features",
        xaxis_title="Mean |SHAP value|",
        template="plotly_white",
    )
    return fig


#  Regional Choropleth Map 

# Mapping of dataset state names  ISO 3166-2:IN codes
_STATE_ISO = {
    "Andhra Pradesh":       "IN-AP",
    "Arunachal Pradesh":    "IN-AR",
    "Assam":                "IN-AS",
    "Bihar":                "IN-BR",
    "Chhattisgarh":         "IN-CT",
    "Goa":                  "IN-GA",
    "Gujarat":              "IN-GJ",
    "Haryana":              "IN-HR",
    "Himachal Pradesh":     "IN-HP",
    "Jharkhand":            "IN-JH",
    "Karnataka":            "IN-KA",
    "Kerala":               "IN-KL",
    "Madhya Pradesh":       "IN-MP",
    "Maharashtra":          "IN-MH",
    "Manipur":              "IN-MN",
    "Meghalaya":            "IN-ML",
    "Mizoram":              "IN-MZ",
    "Nagaland":             "IN-NL",
    "Odisha":               "IN-OR",
    "Punjab":               "IN-PB",
    "Rajasthan":            "IN-RJ",
    "Sikkim":               "IN-SK",
    "Tamil Nadu":           "IN-TN",
    "Telangana":            "IN-TG",
    "Tripura":              "IN-TR",
    "Uttar Pradesh":        "IN-UP",
    "Uttarakhand":          "IN-UT",
    "West Bengal":          "IN-WB",
    "Delhi":                "IN-DL",
    "Delhi NCR":            "IN-DL",
    "Jammu and Kashmir":    "IN-JK",
    "Ladakh":               "IN-LA",
}


def plot_choropleth(df: pd.DataFrame, metric: str = "revenue"):
    """
    Interactive choropleth map of India coloured by revenue / order count.

    Uses a locally cached GeoJSON (downloaded on first call).
    Falls back to a bar chart if GeoJSON is unavailable.
    """
    import os, json, requests

    GEOJSON_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'india_states.geojson')
    GEOJSON_CACHE = os.path.normpath(GEOJSON_CACHE)
    GEOJSON_URL   = 'https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson'
    os.makedirs(os.path.dirname(GEOJSON_CACHE), exist_ok=True)
    if os.path.exists(GEOJSON_CACHE):
        with open(GEOJSON_CACHE) as f:
            geojson_data = json.load(f)
    else:
        try:
            resp = requests.get(GEOJSON_URL, timeout=10)
            resp.raise_for_status()
            geojson_data = resp.json()
            with open(GEOJSON_CACHE, 'w') as f:
                json.dump(geojson_data, f)
        except Exception:
            geojson_data = None  # fallback to bar chart

    agg = df.groupby("state")[metric].sum().reset_index()
    agg["iso"] = agg["state"].map(_STATE_ISO)
    agg = agg.dropna(subset=["iso"])

    if agg.empty or geojson_data is None:
        if agg.empty:
            print("  No state  ISO mapping found. Showing bar chart instead.")
        fig = px.bar(
            df.groupby("state")[metric].sum().nlargest(20).reset_index(),
            x=metric, y="state", orientation="h",
            title=f"Top 20 States by {metric.title()}",
            template="plotly_white",
        )
        return fig

    fig = px.choropleth(
        agg,
        geojson=geojson_data,
        locations="iso",
        featureidkey="properties.ST_NM",
        color=metric,
        hover_name="state",
        color_continuous_scale="YlOrRd",
        title=f"India  {metric.replace('_',' ').title()} by State",
        template="plotly_white",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig


#  Advanced statistical plots 

def plot_lorenz(df: pd.DataFrame, col: str = "revenue"):
    vals = np.sort(df[col].dropna().values)[::-1]
    cum  = np.cumsum(vals) / vals.sum()
    x    = np.linspace(0, 1, len(cum))
    gini = 1 - 2 * np.trapz(cum, x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(dash="dash", color="gray"),
                             name="Perfect equality"))
    fig.add_trace(go.Scatter(x=x, y=cum, mode="lines", fill="tozeroy",
                             fillcolor="rgba(220,20,60,0.12)",
                             line=dict(color="crimson", width=2.5),
                             name=f"Lorenz curve (Gini={gini:.3f})"))
    fig.update_layout(
        title=f"Lorenz Curve  {col.title()} Concentration  (Gini = {gini:.3f})",
        xaxis_title="Cumulative share of orders",
        yaxis_title=f"Cumulative share of {col}",
        template="plotly_white",
    )
    return fig


def plot_ecdf(df: pd.DataFrame):
    fig = go.Figure()
    for col, colour in [("revenue","navy"), ("final_price","forestgreen"), ("units_sold","maroon")]:
        s = np.sort(df[col].dropna().values)
        y = np.arange(1, len(s)+1) / len(s)
        fig.add_trace(go.Scatter(x=s, y=y, mode="lines", name=col.replace("_"," ").title(),
                                 line=dict(color=colour, width=2)))
    fig.update_layout(
        title="Empirical CDF  Revenue, Final Price, Units Sold",
        xaxis_title="Value (log scale)", yaxis_title="Cumulative Proportion",
        xaxis_type="log", template="plotly_white",
    )
    return fig


def plot_rolling_stats(df: pd.DataFrame):
    monthly = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum()
    monthly.index = monthly.index.to_timestamp()
    roll_mean = monthly.rolling(3, center=True).mean()
    roll_std  = monthly.rolling(3, center=True).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values,
                             mode="markers+lines", name="Monthly Revenue",
                             line=dict(color="navy")))
    fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean.values,
                             mode="lines", name="3-month rolling mean",
                             line=dict(color="darkred", width=2.5)))
    fig.add_trace(go.Scatter(
        x=list(roll_mean.index) + list(roll_mean.index[::-1]),
        y=list((roll_mean + roll_std).values) + list((roll_mean - roll_std).values[::-1]),
        fill="toself", fillcolor="rgba(139,0,0,0.12)",
        line=dict(color="rgba(255,255,255,0)"), name="1 std",
    ))
    fig.update_layout(
        title="Revenue Trend + 3-month Rolling Statistics",
        xaxis_title="Month", yaxis_title="Revenue ()",
        template="plotly_white",
    )
    return fig


def run_premium_visuals(df: pd.DataFrame) -> None:
    """Standalone runner — displays all premium visualisation figures."""
    print("=" * 60)
    print("  PREMIUM VISUALISATIONS")
    print("=" * 60)
    plot_pareto(df, "category", "revenue").show()
    plot_pareto(df, "state",    "revenue").show()
    plot_sunburst(df).show()
    plot_choropleth(df, "revenue").show()
    plot_lorenz(df).show()
    plot_ecdf(df).show()
    plot_rolling_stats(df).show()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 500
    categories = ["Electronics", "Clothing", "Home", "Sports"]
    zones = ["North", "South", "East", "West"]
    brand_types = ["Mass", "Premium"]
    states = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
              "Rajasthan", "West Bengal", "Uttar Pradesh", "Telangana", "Kerala",
              "Punjab", "Haryana"]
    dates = pd.date_range("2021-01-01", periods=n, freq="D")

    df_sample = pd.DataFrame({
        "order_date":  rng.choice(dates, n),
        "revenue":     rng.uniform(500, 20000, n),
        "final_price": rng.uniform(150, 4800, n),
        "units_sold":  rng.integers(1, 10, n),
        "category":    rng.choice(categories, n),
        "zone":        rng.choice(zones, n),
        "brand_type":  rng.choice(brand_types, n),
        "state":       rng.choice(states, n),
    })
    df_sample["order_date"] = pd.to_datetime(df_sample["order_date"])

    print("Testing plot_pareto ...")
    plot_pareto(df_sample, "category", "revenue").show()

    print("Testing plot_sunburst ...")
    plot_sunburst(df_sample).show()

    print("Testing plot_shap_summary ...")
    shap_vals = rng.uniform(-1, 1, (100, 5))
    feat_names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    plot_shap_summary(shap_vals, feat_names, top_n=5).show()

    print("Testing plot_choropleth ...")
    plot_choropleth(df_sample, "revenue").show()

    print("Testing plot_lorenz ...")
    plot_lorenz(df_sample).show()

    print("Testing plot_ecdf ...")
    plot_ecdf(df_sample).show()

    print("Testing plot_rolling_stats ...")
    plot_rolling_stats(df_sample).show()

    print("All pareto/premium plots rendered.")
