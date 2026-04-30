"""
modules/price_optimizer.py
Dynamic Pricing & Elasticity Engine  Price Optimizer
Computes optimal discount per category/zone using elasticity + margin targets.
Persists recommendations to Supabase.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def compute_elasticity(
    df: pd.DataFrame,
    group_cols: list[str] = ["category"],
    price_col: str = "final_price",
    qty_col: str = "units_sold",
    min_obs: int = 30,
) -> pd.DataFrame:
    records = []
    for keys, grp in df.groupby(group_cols):
        grp = grp[(grp[price_col] > 0) & (grp[qty_col] > 0)]
        if len(grp) < min_obs:
            continue
        try:
            X = sm.add_constant(np.log(grp[price_col]))
            res = sm.OLS(np.log(grp[qty_col]), X).fit()
            coef = float(res.params.iloc[-1])
            row = dict(zip(group_cols, [keys] if isinstance(keys, str) else keys))
            row.update({
                "elasticity": round(coef, 4),
                "r2": round(res.rsquared, 4),
                "n_obs": len(grp),
                "p_value": round(float(res.pvalues.iloc[-1]), 4),
            })
            records.append(row)
        except Exception:
            continue
    return pd.DataFrame(records).sort_values("elasticity") if records else pd.DataFrame()


def optimal_discount(
    elasticity: float,
    current_discount: float,
    margin_target: float = 0.25,
    max_discount: float = 60.0,
) -> dict:
    """
    Compute the revenue-maximising discount given price elasticity.
    Uses the Lerner index: optimal markup = 1 / |elasticity|
    """
    if elasticity >= 0:
        return {
            "optimal_discount": current_discount,
            "direction": "hold",
            "rationale": "Giffen/luxury good  discounting may reduce demand",
            "revenue_impact_pct": 0.0,
        }
    abs_e = abs(elasticity)
    # Lerner: optimal price-cost margin = 1/|e|
    # Translate to discount: if |e| > 1 (elastic), lower price increases revenue
    if abs_e > 1:
        # Revenue-maximising: reduce price until |e| = 1 at margin
        optimal = min(current_discount * (abs_e / (abs_e - 0.5)), max_discount)
        direction = "increase" if optimal > current_discount else "hold"
        rev_impact = (abs_e - 1) * (optimal - current_discount) / 100 * 100
    else:
        # Inelastic: reduce discount to protect margin
        optimal = max(current_discount * 0.8, 0)
        direction = "decrease"
        rev_impact = (1 - abs_e) * (current_discount - optimal) / 100 * 100

    return {
        "optimal_discount": round(float(np.clip(optimal, 0, max_discount)), 1),
        "direction": direction,
        "rationale": (
            f"Elasticity {elasticity:.2f}  "
            f"{'elastic: increase discount to grow volume' if abs_e > 1 else 'inelastic: reduce discount to protect margin'}"
        ),
        "revenue_impact_pct": round(float(rev_impact), 1),
    }


def run_price_optimizer(df: pd.DataFrame, user_id: str = "") -> pd.DataFrame:
    """
    Full price optimisation pipeline.
    Returns a DataFrame of recommendations, optionally saved to Supabase.
    """
    elast = compute_elasticity(df, group_cols=["category"])
    if elast.empty:
        return pd.DataFrame()

    avg_disc = df.groupby("category")["discount_percent"].mean()
    results = []
    for _, row in elast.iterrows():
        cat = row["category"]
        cur_disc = float(avg_disc.get(cat, 30.0))
        opt = optimal_discount(row["elasticity"], cur_disc)
        results.append({
            "category":          cat,
            "current_discount":  round(cur_disc, 1),
            "optimal_discount":  opt["optimal_discount"],
            "change":            round(opt["optimal_discount"] - cur_disc, 1),
            "direction":         opt["direction"],
            "elasticity":        row["elasticity"],
            "revenue_impact_pct":opt["revenue_impact_pct"],
            "rationale":         opt["rationale"],
        })

    result_df = pd.DataFrame(results).sort_values("revenue_impact_pct", ascending=False)

    # Persist to Supabase
    if user_id:
        try:
            from core.database import _client
            client = _client()
            if client:
                for _, r in result_df.iterrows():
                    client.table("price_recommendations").upsert({
                        "user_id":           user_id,
                        "category":          r["category"],
                        "current_discount":  r["current_discount"],
                        "optimal_discount":  r["optimal_discount"],
                        "direction":         r["direction"],
                        "revenue_impact_pct":r["revenue_impact_pct"],
                    }).execute()
        except Exception as e:
            logger.warning("Price rec save failed: %s", e)

    return result_df


def plot_price_optimizer(result_df: pd.DataFrame):
    """Return a Plotly figure for the price optimizer results."""
    if result_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Current Discount",
        x=result_df["category"],
        y=result_df["current_discount"],
        marker_color="#94a3b8",
    ))
    fig.add_trace(go.Bar(
        name="Optimal Discount",
        x=result_df["category"],
        y=result_df["optimal_discount"],
        marker_color=["#22c55e" if d == "decrease" else "#ef4444" if d == "increase" else "#4f46e5"
                      for d in result_df["direction"]],
    ))
    fig.update_layout(
        barmode="group",
        title="Current vs Optimal Discount by Category",
        xaxis_title="Category",
        yaxis_title="Discount %",
        template="plotly_white",
        legend=dict(orientation="h", y=1.1),
    )
    return fig
