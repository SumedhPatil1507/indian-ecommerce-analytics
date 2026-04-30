"""
modules/cohort.py

Cohort Analysis Heatmap
   Groups customers by first-purchase month (cohort)
   Tracks retention & revenue over subsequent months
   Interactive Plotly heatmap showing cohort retention rates
"""
import numpy as np
import pandas as pd
import plotly.express as px


def build_cohort_table(df: pd.DataFrame, metric: str = "revenue") -> pd.DataFrame:
    """
    Build a cohort retention / revenue table.

    Parameters
    ----------
    df     : order-level DataFrame with order_date
    metric : "revenue" or "count" (order count)

    Returns
    -------
    pd.DataFrame pivot: cohort_month  cohort_period
    """
    # synthetic customer_id if missing
    if "customer_id" not in df.columns:
        df = df.copy()
        df["customer_id"] = (
            df["state"] + "_" +
            df["customer_gender"] + "_" +
            df["customer_age"].astype(str)
        )

    df = df.copy()
    df["order_month"] = df["order_date"].dt.to_period("M")

    # first purchase month per customer
    cohort_map = (
        df.groupby("customer_id")["order_month"]
        .min()
        .to_frame("cohort_month")
    )
    df = df.merge(cohort_map, left_on="customer_id", right_index=True)

    # cohort period (months since first purchase)
    df["cohort_period"] = (
        (df["order_month"] - df["cohort_month"]).apply(lambda x: x.n)
    )

    # aggregate
    if metric == "revenue":
        cohort = (
            df.groupby(["cohort_month", "cohort_period"])["revenue"]
            .sum()
            .reset_index()
        )
    else:
        cohort = (
            df.groupby(["cohort_month", "cohort_period"])
            .size()
            .reset_index(name="count")
        )
        metric = "count"

    pivot = cohort.pivot(index="cohort_month", columns="cohort_period", values=metric)
    pivot.index = pivot.index.astype(str)
    return pivot


def plot_cohort_heatmap(df: pd.DataFrame, metric: str = "revenue") -> None:
    """
    Plot an interactive cohort heatmap.

    metric : "revenue" or "count"
    """
    pivot = build_cohort_table(df, metric=metric)

    # retention rate (normalise by cohort size at period 0)
    if metric == "count":
        retention = pivot.div(pivot[0], axis=0) * 100
        title = "Cohort Retention Rate (%)  Order Count"
        fmt = ".1f"
        cmap = "Blues"
    else:
        retention = pivot.div(pivot[0], axis=0) * 100
        title = "Cohort Revenue Retention (%)  Normalised by Month 0"
        fmt = ".0f"
        cmap = "Greens"

    fig = px.imshow(
        retention,
        color_continuous_scale=cmap,
        title=title,
        labels={"x": "Cohort Period (months since first purchase)",
                "y": "Cohort Month", "color": "Retention %"},
        text_auto=fmt,
        template="plotly_white",
    )
    fig.update_xaxes(side="top")
    fig.show()


def run_cohort_analysis(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("  COHORT ANALYSIS")
    print("=" * 60)
    plot_cohort_heatmap(df, metric="count")
    plot_cohort_heatmap(df, metric="revenue")
