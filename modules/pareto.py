"""
modules/pareto.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Premium Visualisations
  â€¢ Pareto Chart (80/20 Rule)
  â€¢ Interactive Sunburst Chart (category â†’ zone â†’ brand_type)
  â€¢ SHAP Summary Plot (interactive via Plotly)
  â€¢ Regional Choropleth Map (India states)
  â€¢ Advanced plots: Q-Q, Lorenz curve, ECDF, rolling stats
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# â”€â”€ Pareto Chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_pareto(
    df: pd.DataFrame,
    group_col: str = "category",
    value_col: str = "revenue",
) -> None:
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
        title=f"Pareto Chart â€“ {value_col.title()} by {group_col.title()} (80/20 Rule)",
        template="plotly_white",
    )
    fig.update_yaxes(title_text=f"Total {value_col.title()} (â‚¹)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    fig.show()


# â”€â”€ Sunburst Chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_sunburst(df: pd.DataFrame) -> None:
    """
    Interactive sunburst: category â†’ zone â†’ brand_type, sized by revenue.
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
        title="Revenue Sunburst â€“ Category â†’ Zone â†’ Brand Type",
        template="plotly_white",
        color="revenue",
        color_continuous_scale="RdBu",
    )
    fig.update_traces(textinfo="label+percent parent")
    fig.show()


# â”€â”€ SHAP Summary (interactive Plotly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_shap_summary(shap_values: np.ndarray, feature_names: list, top_n: int = 15) -> None:
    """
    Interactive SHAP beeswarm-style bar chart using Plotly.

    Parameters
    ----------
    shap_values   : 2-D array (n_samples Ã— n_features)
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
        title=f"SHAP Feature Importance â€“ Top {top_n} Features",
        xaxis_title="Mean |SHAP value|",
        template="plotly_white",
    )
    fig.show()


# â”€â”€ Regional Choropleth Map â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Mapping of dataset state names â†’ ISO 3166-2:IN codes
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


def plot_choropleth(df: pd.DataFrame, metric: str = "revenue") -> None:
    """
    Interactive choropleth map of India coloured by revenue / order count.

    Uses Plotly's built-in India GeoJSON via location_mode='geojson-id'.
    Falls back to a bar chart if GeoJSON is unavailable.
    """
    agg = df.groupby("state")[metric].sum().reset_index()
    agg["iso"] = agg["state"].map(_STATE_ISO)
    agg = agg.dropna(subset=["iso"])

    if agg.empty:
        print("âš   No state â†’ ISO mapping found. Showing bar chart instead.")
        fig = px.bar(
            df.groupby("state")[metric].sum().nlargest(20).reset_index(),
            x=metric, y="state", orientation="h",
            title=f"Top 20 States by {metric.title()}",
            template="plotly_white",
        )
        fig.show()
        return

    fig = px.choropleth(
        agg,
        geojson=(
            "https://raw.githubusercontent.com/geohacker/india/master/"
            "state/india_telengana.geojson"
        ),
        locations="iso",
        featureidkey="properties.ST_NM",
        color=metric,
        hover_name="state",
        color_continuous_scale="YlOrRd",
        title=f"India â€“ {metric.replace('_',' ').title()} by State",
        template="plotly_white",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()


# â”€â”€ Advanced statistical plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def plot_lorenz(df: pd.DataFrame, col: str = "revenue") -> None:
    vals = np.sort(df[col].dropna().values)[::-1]
    cum  = np.cumsum(vals) / vals.sum()
    x    = np.linspace(0, 1, len(cum))
    gini = 1 - 2 * np.trapezoid(cum, x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(dash="dash", color="gray"),
                             name="Perfect equality"))
    fig.add_trace(go.Scatter(x=x, y=cum, mode="lines", fill="tozeroy",
                             fillcolor="rgba(220,20,60,0.12)",
                             line=dict(color="crimson", width=2.5),
                             name=f"Lorenz curve (Gini={gini:.3f})"))
    fig.update_layout(
        title=f"Lorenz Curve â€“ {col.title()} Concentration  (Gini = {gini:.3f})",
        xaxis_title="Cumulative share of orders",
        yaxis_title=f"Cumulative share of {col}",
        template="plotly_white",
    )
    fig.show()


def plot_ecdf(df: pd.DataFrame) -> None:
    fig = go.Figure()
    for col, colour in [("revenue","navy"), ("final_price","forestgreen"), ("units_sold","maroon")]:
        s = np.sort(df[col].dropna().values)
        y = np.arange(1, len(s)+1) / len(s)
        fig.add_trace(go.Scatter(x=s, y=y, mode="lines", name=col.replace("_"," ").title(),
                                 line=dict(color=colour, width=2)))
    fig.update_layout(
        title="Empirical CDF â€“ Revenue, Final Price, Units Sold",
        xaxis_title="Value (log scale)", yaxis_title="Cumulative Proportion",
        xaxis_type="log", template="plotly_white",
    )
    fig.show()


def plot_rolling_stats(df: pd.DataFrame) -> None:
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
        line=dict(color="rgba(255,255,255,0)"), name="Â±1 std",
    ))
    fig.update_layout(
        title="Revenue Trend + 3-month Rolling Statistics",
        xaxis_title="Month", yaxis_title="Revenue (â‚¹)",
        template="plotly_white",
    )
    fig.show()


def run_premium_visuals(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("  PREMIUM VISUALISATIONS")
    print("=" * 60)
    plot_pareto(df, "category", "revenue")
    plot_pareto(df, "state",    "revenue")
    plot_sunburst(df)
    plot_choropleth(df, "revenue")
    plot_lorenz(df)
    plot_ecdf(df)
    plot_rolling_stats(df)

