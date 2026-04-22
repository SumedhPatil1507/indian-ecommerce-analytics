"""
modules/price_elasticity.py
──────────────────────────────────────────────────────────────────────────────
Price Elasticity Engine
  • Computes own-price elasticity of demand per category / zone / brand_type
  • Log-log OLS regression: ln(units) = α + β·ln(price) + controls
  • β is the price elasticity coefficient
  • Interactive Plotly heatmap + waterfall chart
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from itertools import product


# ── core elasticity computation ───────────────────────────────────────────────

def compute_elasticity(
    df: pd.DataFrame,
    group_cols: list[str] = ["category"],
    price_col: str = "final_price",
    qty_col: str = "units_sold",
    min_obs: int = 50,
) -> pd.DataFrame:
    """
    Estimate price elasticity for each group via log-log OLS.

    Returns a DataFrame with columns: [*group_cols, elasticity, r2, n_obs, p_value]

    Interpretation
    ──────────────
    elasticity < -1  → elastic demand  (price-sensitive)
    -1 < elasticity < 0 → inelastic demand
    elasticity > 0   → Giffen / luxury good signal
    """
    records = []
    for keys, grp in df.groupby(group_cols):
        if len(grp) < min_obs:
            continue
        grp = grp.copy()
        grp = grp[(grp[price_col] > 0) & (grp[qty_col] > 0)]
        if len(grp) < min_obs:
            continue

        log_p = np.log(grp[price_col])
        log_q = np.log(grp[qty_col])
        X = sm.add_constant(log_p)
        try:
            res = sm.OLS(log_q, X).fit()
            coef    = res.params.get(price_col, res.params.iloc[-1])
            pval    = res.pvalues.iloc[-1]
            r2      = res.rsquared
        except Exception:
            continue

        row = dict(zip(group_cols, [keys] if isinstance(keys, str) else keys))
        row.update({"elasticity": round(coef, 4), "r2": round(r2, 4),
                    "n_obs": len(grp), "p_value": round(pval, 4)})
        records.append(row)

    return pd.DataFrame(records).sort_values("elasticity")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_elasticity_heatmap(df: pd.DataFrame) -> None:
    """
    Heatmap of price elasticity: category × zone.
    """
    elast = compute_elasticity(df, group_cols=["category", "zone"])
    if elast.empty:
        print("Not enough data for category × zone elasticity.")
        return

    pivot = elast.pivot(index="category", columns="zone", values="elasticity")
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Price Elasticity Heatmap – Category × Zone",
        labels={"color": "Elasticity (β)"},
        text_auto=".2f",
        template="plotly_white",
    )
    fig.update_layout(coloraxis_colorbar_title="Elasticity")
    fig.show()


def plot_elasticity_waterfall(df: pd.DataFrame) -> None:
    """
    Waterfall / bar chart of elasticity by category, coloured by elastic vs inelastic.
    """
    elast = compute_elasticity(df, group_cols=["category"])
    if elast.empty:
        return

    elast["type"] = elast["elasticity"].apply(
        lambda e: "Elastic (< -1)" if e < -1
        else ("Inelastic" if e < 0 else "Giffen / Luxury")
    )
    fig = px.bar(
        elast.sort_values("elasticity"),
        x="elasticity", y="category", orientation="h",
        color="type",
        color_discrete_map={
            "Elastic (< -1)": "crimson",
            "Inelastic":      "steelblue",
            "Giffen / Luxury":"gold",
        },
        title="Price Elasticity by Category",
        labels={"elasticity": "Elasticity Coefficient (β)", "category": "Category"},
        template="plotly_white",
        text="elasticity",
    )
    fig.add_vline(x=-1, line_dash="dash", line_color="gray",
                  annotation_text="Elastic threshold (β = -1)")
    fig.add_vline(x=0,  line_dash="dot",  line_color="black")
    fig.show()


def plot_elasticity_by_brand(df: pd.DataFrame) -> None:
    elast = compute_elasticity(df, group_cols=["brand_type", "category"])
    if elast.empty:
        return
    fig = px.bar(
        elast, x="category", y="elasticity", color="brand_type",
        barmode="group",
        title="Price Elasticity – Mass vs Premium by Category",
        labels={"elasticity": "Elasticity (β)"},
        template="plotly_white",
    )
    fig.add_hline(y=-1, line_dash="dash", line_color="red",
                  annotation_text="Elastic threshold")
    fig.show()


def run_elasticity_engine(df: pd.DataFrame) -> None:
    """Entry point – run all elasticity plots."""
    print("=" * 60)
    print("  PRICE ELASTICITY ENGINE")
    print("=" * 60)

    # summary table
    summary = compute_elasticity(df, group_cols=["category"])
    print("\nElasticity by Category:")
    print(summary.to_string(index=False))

    plot_elasticity_waterfall(df)
    plot_elasticity_heatmap(df)
    plot_elasticity_by_brand(df)

    # optimal discount recommendation
    print("\n--- Optimal Discount Recommendation ---")
    for _, row in summary.iterrows():
        e = row["elasticity"]
        if e < -1:
            rec = "Aggressive discounting justified – demand is elastic"
        elif -1 <= e < 0:
            rec = "Moderate discounts – demand is inelastic, protect margins"
        else:
            rec = "Luxury / Giffen signal – discounting may reduce perceived value"
        print(f"  {row['category']:<30} β={e:+.3f}  → {rec}")
