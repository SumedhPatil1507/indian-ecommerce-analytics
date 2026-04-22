"""
modules/inventory_alerts.py
──────────────────────────────────────────────────────────────────────────────
Dynamic Inventory Alert System
  • Detects high-velocity SKU groups (category × zone) approaching stock-out
  • Flags slow-movers with high inventory pressure
  • Generates a colour-coded alert dashboard (Plotly)
  • Outputs a prioritised reorder recommendation table
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── alert thresholds (tunable) ────────────────────────────────────────────────
VELOCITY_HIGH_PCTILE  = 0.80   # top 20% units/day = high velocity
DISCOUNT_SPIKE_THRESH = 45.0   # avg discount > 45% = clearance signal
LOW_STOCK_PRESSURE    = "High" # inventory_pressure value meaning low stock


# ── core alert computation ────────────────────────────────────────────────────

def compute_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-group (category × zone) inventory health metrics and alerts.

    Returns
    -------
    pd.DataFrame with columns:
        category, zone, avg_units_sold, avg_discount, inventory_pressure_pct,
        velocity_score, alert_level, recommendation
    """
    grp = (
        df.groupby(["category", "zone"])
        .agg(
            avg_units_sold    = ("units_sold",       "mean"),
            total_units       = ("units_sold",       "sum"),
            avg_discount      = ("discount_percent", "mean"),
            n_orders          = ("revenue",          "count"),
            high_pressure_pct = ("inventory_pressure",
                                 lambda x: (x == LOW_STOCK_PRESSURE).mean() * 100),
            avg_revenue       = ("revenue",          "mean"),
        )
        .reset_index()
    )

    # velocity score: normalised units sold
    vel_thresh = grp["avg_units_sold"].quantile(VELOCITY_HIGH_PCTILE)
    grp["velocity_score"] = (grp["avg_units_sold"] / grp["avg_units_sold"].max() * 100).round(1)

    # alert logic
    def _alert(row):
        if row["avg_units_sold"] >= vel_thresh and row["high_pressure_pct"] >= 60:
            return "🔴 CRITICAL – Reorder Now"
        if row["avg_units_sold"] >= vel_thresh and row["high_pressure_pct"] >= 30:
            return "🟠 HIGH – Monitor Closely"
        if row["avg_discount"] >= DISCOUNT_SPIKE_THRESH and row["high_pressure_pct"] >= 50:
            return "🟡 CLEARANCE – Excess Stock"
        if row["avg_units_sold"] < grp["avg_units_sold"].quantile(0.20):
            return "🔵 SLOW MOVER – Review Listing"
        return "🟢 HEALTHY"

    grp["alert_level"] = grp.apply(_alert, axis=1)

    def _rec(row):
        lvl = row["alert_level"]
        if "CRITICAL"  in lvl: return f"Reorder {row['category']} in {row['zone']} immediately"
        if "HIGH"      in lvl: return f"Increase safety stock for {row['category']} – {row['zone']}"
        if "CLEARANCE" in lvl: return f"Run targeted promotion to clear {row['category']} – {row['zone']}"
        if "SLOW"      in lvl: return f"Review pricing / listing quality for {row['category']} – {row['zone']}"
        return "No action required"

    grp["recommendation"] = grp.apply(_rec, axis=1)
    return grp.sort_values(["alert_level", "velocity_score"], ascending=[True, False])


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_inventory_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: avg units sold per category × zone."""
    pivot = (
        df.groupby(["category", "zone"])["units_sold"]
        .mean()
        .unstack()
        .round(1)
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="YlOrRd",
        title="Avg Units Sold – Category × Zone (Inventory Velocity)",
        labels={"color": "Avg Units Sold"},
        text_auto=True,
        template="plotly_white",
    )
    fig.show()


def plot_alert_dashboard(df: pd.DataFrame) -> None:
    alerts = compute_alerts(df)

    # colour map
    colour_map = {
        "🔴 CRITICAL – Reorder Now":    "red",
        "🟠 HIGH – Monitor Closely":    "orange",
        "🟡 CLEARANCE – Excess Stock":  "gold",
        "🔵 SLOW MOVER – Review Listing":"steelblue",
        "🟢 HEALTHY":                   "green",
    }

    fig = px.scatter(
        alerts,
        x="avg_discount",
        y="avg_units_sold",
        color="alert_level",
        size="high_pressure_pct",
        hover_data=["category", "zone", "recommendation"],
        color_discrete_map=colour_map,
        title="Inventory Alert Dashboard – Discount vs Velocity",
        labels={
            "avg_discount":   "Avg Discount %",
            "avg_units_sold": "Avg Units Sold",
            "alert_level":    "Alert Level",
        },
        template="plotly_white",
    )
    fig.add_vline(x=DISCOUNT_SPIKE_THRESH, line_dash="dash", line_color="gray",
                  annotation_text="Clearance threshold")
    fig.show()

    # bar chart – top 15 critical groups
    top = alerts[alerts["alert_level"].str.contains("CRITICAL|HIGH")].head(15)
    if not top.empty:
        top["group"] = top["category"] + " / " + top["zone"]
        fig2 = px.bar(
            top, x="velocity_score", y="group", orientation="h",
            color="alert_level", color_discrete_map=colour_map,
            title="Top Critical / High-Alert Inventory Groups",
            labels={"velocity_score": "Velocity Score (0–100)", "group": "Category / Zone"},
            template="plotly_white",
        )
        fig2.show()


def run_inventory_alerts(df: pd.DataFrame) -> None:
    """Entry point – print alert table and show plots."""
    print("=" * 70)
    print("  DYNAMIC INVENTORY ALERT SYSTEM")
    print("=" * 70)

    alerts = compute_alerts(df)
    critical = alerts[alerts["alert_level"].str.contains("CRITICAL|HIGH")]

    print(f"\n⚠  {len(critical)} groups require immediate attention:\n")
    print(critical[["category", "zone", "avg_units_sold", "avg_discount",
                     "high_pressure_pct", "alert_level", "recommendation"]]
          .to_string(index=False))

    plot_inventory_heatmap(df)
    plot_alert_dashboard(df)

    print("\n✅ Full alert table:")
    print(alerts[["category", "zone", "alert_level", "recommendation"]]
          .to_string(index=False))
