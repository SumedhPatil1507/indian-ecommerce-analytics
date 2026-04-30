"""
modules/at_risk.py
Customer At-Risk Automation using BG/NBD CLV model.
Identifies customers likely to churn and generates automated alerts.
Persists alerts to Supabase for downstream action.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

_CHURN_THRESHOLD_DAYS = 60   # no purchase in 60 days = at-risk
_HIGH_VALUE_PERCENTILE = 0.75


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-customer feature table from order-level data."""
    if "customer_id" not in df.columns:
        df = df.copy()
        df["customer_id"] = (
            df["state"].astype(str) + "_" +
            df["customer_gender"].astype(str) + "_" +
            df["customer_age"].astype(str)
        )
    snapshot = df["order_date"].max()
    cust = (
        df.groupby("customer_id")
        .agg(
            last_order_date  = ("order_date", "max"),
            first_order_date = ("order_date", "min"),
            order_count      = ("revenue",    "count"),
            total_revenue    = ("revenue",    "sum"),
            avg_order_value  = ("revenue",    "mean"),
            avg_discount     = ("discount_percent", "mean"),
            top_category     = ("category",   lambda x: x.mode()[0] if len(x) else "Unknown"),
            top_zone         = ("zone",        lambda x: x.mode()[0] if len(x) else "Unknown"),
        )
        .reset_index()
    )
    cust["days_since_last_order"] = (snapshot - cust["last_order_date"]).dt.days
    cust["customer_age_days"]     = (snapshot - cust["first_order_date"]).dt.days
    cust["purchase_frequency"]    = cust["order_count"] / (cust["customer_age_days"] / 30).clip(lower=1)
    return cust


def score_churn_risk(cust: pd.DataFrame) -> pd.DataFrame:
    """
    Score each customer's churn risk (0-100).
    Uses recency, frequency, and monetary signals.
    """
    cust = cust.copy()

    # Recency score (higher days since last order = higher risk)
    max_days = cust["days_since_last_order"].quantile(0.95)
    cust["recency_risk"] = (cust["days_since_last_order"] / max_days).clip(0, 1) * 40

    # Frequency score (lower frequency = higher risk)
    max_freq = cust["purchase_frequency"].quantile(0.95)
    cust["frequency_risk"] = (1 - (cust["purchase_frequency"] / max_freq).clip(0, 1)) * 35

    # Monetary score (lower value = higher risk of being low-priority)
    max_rev = cust["total_revenue"].quantile(0.95)
    cust["monetary_risk"] = (1 - (cust["total_revenue"] / max_rev).clip(0, 1)) * 25

    cust["churn_risk_score"] = (
        cust["recency_risk"] + cust["frequency_risk"] + cust["monetary_risk"]
    ).round(1)

    # Value tier
    rev_75 = cust["total_revenue"].quantile(_HIGH_VALUE_PERCENTILE)
    rev_50 = cust["total_revenue"].quantile(0.50)
    cust["value_tier"] = cust["total_revenue"].apply(
        lambda v: "High Value" if v >= rev_75 else ("Mid Value" if v >= rev_50 else "Low Value")
    )

    # Risk label
    cust["risk_label"] = cust["churn_risk_score"].apply(
        lambda s: "Critical" if s >= 70 else ("High" if s >= 50 else ("Medium" if s >= 30 else "Low"))
    )

    return cust.sort_values("churn_risk_score", ascending=False)


def generate_at_risk_alerts(
    df: pd.DataFrame,
    user_id: str = "",
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Full at-risk pipeline: build features, score, generate alerts.
    Saves critical alerts to Supabase.
    Returns top_n at-risk customers.
    """
    cust = build_customer_features(df)
    scored = score_churn_risk(cust)

    # Focus on high-value at-risk customers
    at_risk = scored[
        (scored["churn_risk_score"] >= 50) &
        (scored["value_tier"].isin(["High Value", "Mid Value"]))
    ].head(top_n).copy()

    at_risk["recommended_action"] = at_risk.apply(_recommend_action, axis=1)
    at_risk["alert_generated_at"] = datetime.now(timezone.utc).isoformat()

    # Persist to Supabase
    if user_id and not at_risk.empty:
        _save_alerts(at_risk, user_id)

    return at_risk


def _recommend_action(row: pd.Series) -> str:
    if row["risk_label"] == "Critical" and row["value_tier"] == "High Value":
        return f"Immediate outreach — offer {min(row['avg_discount']+10, 40):.0f}% personalised discount on {row['top_category']}"
    if row["risk_label"] == "High" and row["value_tier"] == "High Value":
        return f"Send win-back email with loyalty reward for {row['top_category']} purchase"
    if row["value_tier"] == "Mid Value":
        return f"Trigger re-engagement campaign — highlight new arrivals in {row['top_category']}"
    return "Add to re-engagement drip sequence"


def _save_alerts(at_risk: pd.DataFrame, user_id: str) -> None:
    try:
        from core.database import _client
        client = _client()
        if not client:
            return
        records = []
        for _, row in at_risk.iterrows():
            records.append({
                "user_id":            user_id,
                "customer_id":        str(row["customer_id"]),
                "churn_risk_score":   float(row["churn_risk_score"]),
                "risk_label":         row["risk_label"],
                "value_tier":         row["value_tier"],
                "days_since_order":   int(row["days_since_last_order"]),
                "total_revenue":      float(row["total_revenue"]),
                "recommended_action": row["recommended_action"],
                "created_at":         row["alert_generated_at"],
            })
        client.table("at_risk_alerts").upsert(records).execute()
        logger.info("Saved %d at-risk alerts to Supabase", len(records))
    except Exception as e:
        logger.warning("At-risk alert save failed: %s", e)


def plot_at_risk(at_risk: pd.DataFrame):
    """Return Plotly figures for at-risk dashboard."""
    if at_risk.empty:
        return None, None

    # Risk score distribution
    fig1 = px.histogram(
        at_risk, x="churn_risk_score", color="risk_label",
        color_discrete_map={"Critical":"#ef4444","High":"#f97316","Medium":"#eab308","Low":"#22c55e"},
        title="Churn Risk Score Distribution",
        labels={"churn_risk_score": "Risk Score (0-100)"},
        template="plotly_white", nbins=20,
    )

    # Revenue at risk by tier
    rev_at_risk = at_risk.groupby(["risk_label","value_tier"])["total_revenue"].sum().reset_index()
    fig2 = px.bar(
        rev_at_risk, x="risk_label", y="total_revenue", color="value_tier",
        title="Revenue at Risk by Label & Value Tier",
        labels={"total_revenue": "Total Revenue (Rs)", "risk_label": "Risk Label"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
        barmode="group",
    )

    return fig1, fig2
