"""
modules/at_risk.py
Customer At-Risk Automation using RFM churn risk scoring.
Identifies high-value customers likely to churn and generates automated alerts.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

_CHURN_THRESHOLD_DAYS  = 60
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
    """Score each customer churn risk 0-100 using RFM signals."""
    cust = cust.copy()
    max_days = cust["days_since_last_order"].quantile(0.95)
    cust["recency_risk"]   = (cust["days_since_last_order"] / max(max_days, 1)).clip(0, 1) * 40
    max_freq = cust["purchase_frequency"].quantile(0.95)
    cust["frequency_risk"] = (1 - (cust["purchase_frequency"] / max(max_freq, 1e-9)).clip(0, 1)) * 35
    max_rev  = cust["total_revenue"].quantile(0.95)
    cust["monetary_risk"]  = (1 - (cust["total_revenue"] / max(max_rev, 1e-9)).clip(0, 1)) * 25
    cust["churn_risk_score"] = (
        cust["recency_risk"] + cust["frequency_risk"] + cust["monetary_risk"]
    ).round(1)
    rev_75 = cust["total_revenue"].quantile(_HIGH_VALUE_PERCENTILE)
    rev_50 = cust["total_revenue"].quantile(0.50)
    cust["value_tier"] = cust["total_revenue"].apply(
        lambda v: "High Value" if v >= rev_75 else ("Mid Value" if v >= rev_50 else "Low Value")
    )
    cust["risk_label"] = cust["churn_risk_score"].apply(
        lambda s: "Critical" if s >= 70 else ("High" if s >= 50 else ("Medium" if s >= 30 else "Low"))
    )
    return cust.sort_values("churn_risk_score", ascending=False)


def generate_at_risk_alerts(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Full at-risk pipeline: build features, score, generate alerts."""
    cust   = build_customer_features(df)
    scored = score_churn_risk(cust)
    at_risk = scored[
        (scored["churn_risk_score"] >= 50) &
        (scored["value_tier"].isin(["High Value", "Mid Value"]))
    ].head(top_n).copy()
    at_risk["recommended_action"] = at_risk.apply(_recommend_action, axis=1)
    at_risk["alert_generated_at"] = datetime.now(timezone.utc).isoformat()
    return at_risk


def _recommend_action(row: pd.Series) -> str:
    if row["risk_label"] == "Critical" and row["value_tier"] == "High Value":
        disc = min(row["avg_discount"] + 10, 40)
        return f"Immediate outreach - offer {disc:.0f}% personalised discount on {row['top_category']}"
    if row["risk_label"] == "High" and row["value_tier"] == "High Value":
        return f"Send win-back email with loyalty reward for {row['top_category']} purchase"
    if row["value_tier"] == "Mid Value":
        return f"Trigger re-engagement campaign - highlight new arrivals in {row['top_category']}"
    return "Add to re-engagement drip sequence"


def plot_at_risk(at_risk: pd.DataFrame):
    """Return two Plotly figures for the at-risk dashboard."""
    if at_risk.empty:
        return None, None
    color_map = {
        "Critical": "#ef4444",
        "High":     "#f97316",
        "Medium":   "#eab308",
        "Low":      "#22c55e",
    }
    fig1 = px.histogram(
        at_risk, x="churn_risk_score", color="risk_label",
        color_discrete_map=color_map,
        title="Churn Risk Score Distribution",
        labels={"churn_risk_score": "Risk Score (0-100)"},
        template="plotly_white", nbins=20,
    )
    rev_at_risk = at_risk.groupby(["risk_label", "value_tier"])["total_revenue"].sum().reset_index()
    fig2 = px.bar(
        rev_at_risk, x="risk_label", y="total_revenue", color="value_tier",
        title="Revenue at Risk by Label and Value Tier",
        labels={"total_revenue": "Total Revenue (Rs)", "risk_label": "Risk Label"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2,
        barmode="group",
    )
    return fig1, fig2
