"""
modules/clv.py
──────────────────────────────────────────────────────────────────────────────
Customer Lifetime Value (CLV) Predictor
  • BG/NBD model for purchase frequency (via lifetimes library)
  • Gamma-Gamma model for monetary value
  • CLV = predicted_purchases × predicted_avg_order_value × margin
  • Segments customers into CLV tiers (Champions, Loyalists, At-Risk, Lost)
  • Interactive Plotly visualisations
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter  # type: ignore
    from lifetimes.utils import summary_data_from_transaction_dataframe  # type: ignore
    _LIFETIMES_OK = True
except ImportError:
    _LIFETIMES_OK = False


# ── RFM helper ────────────────────────────────────────────────────────────────

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an RFM table from order-level data.
    Uses `order_id` as customer proxy when no customer_id exists.
    """
    # Use state+gender+age as a synthetic customer key if no customer_id
    if "customer_id" not in df.columns:
        df = df.copy()
        df["customer_id"] = (
            df["state"].astype(str) + "_" +
            df["customer_gender"].astype(str) + "_" +
            df["customer_age"].astype(str)
        )

    snapshot = df["order_date"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("customer_id")
        .agg(
            recency   = ("order_date", lambda x: (snapshot - x.max()).days),
            frequency = ("order_date", "count"),
            monetary  = ("revenue",    "mean"),
            first_purchase = ("order_date", "min"),
        )
        .reset_index()
    )
    rfm["T"] = (snapshot - rfm["first_purchase"]).dt.days  # customer age in days
    return rfm


# ── BG/NBD + Gamma-Gamma CLV ──────────────────────────────────────────────────

def compute_clv(
    df: pd.DataFrame,
    time_horizon: int = 12,   # months
    discount_rate: float = 0.01,
    margin: float = 0.20,
) -> pd.DataFrame:
    """
    Compute CLV using BG/NBD (frequency) + Gamma-Gamma (monetary) models.

    Falls back to a simple RFM-based CLV if lifetimes is not installed.

    Parameters
    ----------
    time_horizon  : forecast horizon in months
    discount_rate : monthly discount rate for NPV
    margin        : gross margin assumption

    Returns
    -------
    pd.DataFrame with CLV estimates and tier labels
    """
    rfm = build_rfm(df)

    if not _LIFETIMES_OK:
        print("⚠  lifetimes not installed – using simple RFM CLV.")
        print("   Install with: pip install lifetimes")
        return _simple_clv(rfm, time_horizon, margin)

    # lifetimes expects: frequency, recency (days), T (days), monetary_value
    rfm_lt = rfm[rfm["frequency"] > 0].copy()
    rfm_lt["recency_weeks"] = rfm_lt["recency"] / 7
    rfm_lt["T_weeks"]       = rfm_lt["T"] / 7

    # BG/NBD
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(rfm_lt["frequency"], rfm_lt["recency_weeks"], rfm_lt["T_weeks"])
    rfm_lt["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        time_horizon * 4.33,   # weeks
        rfm_lt["frequency"], rfm_lt["recency_weeks"], rfm_lt["T_weeks"],
    )

    # Gamma-Gamma (only customers with > 1 purchase)
    gg_data = rfm_lt[rfm_lt["frequency"] > 1].copy()
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(gg_data["frequency"], gg_data["monetary"])
    rfm_lt.loc[rfm_lt["frequency"] > 1, "predicted_avg_value"] = (
        ggf.conditional_expected_average_profit(
            gg_data["frequency"], gg_data["monetary"]
        )
    )
    rfm_lt["predicted_avg_value"] = rfm_lt["predicted_avg_value"].fillna(rfm_lt["monetary"])

    rfm_lt["clv"] = (
        rfm_lt["predicted_purchases"] *
        rfm_lt["predicted_avg_value"] *
        margin
    ).round(2)

    return _add_tiers(rfm_lt)


def _simple_clv(rfm: pd.DataFrame, horizon: int, margin: float) -> pd.DataFrame:
    """Fallback: CLV = (frequency/T_months) * monetary * horizon * margin."""
    rfm = rfm.copy()
    rfm["T_months"] = rfm["T"] / 30
    rfm["purchase_rate"] = rfm["frequency"] / rfm["T_months"].clip(lower=1)
    rfm["clv"] = (rfm["purchase_rate"] * rfm["monetary"] * horizon * margin).round(2)
    return _add_tiers(rfm)


def _add_tiers(df: pd.DataFrame) -> pd.DataFrame:
    q1, q2, q3 = df["clv"].quantile([0.25, 0.50, 0.75]).values
    def _tier(v):
        if v >= q3:   return "Champions"
        if v >= q2:   return "Loyalists"
        if v >= q1:   return "At-Risk"
        return "Lost / Dormant"
    df["clv_tier"] = df["clv"].apply(_tier)
    return df


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_clv(df: pd.DataFrame) -> None:
    clv_df = compute_clv(df)

    # distribution
    fig = px.histogram(clv_df, x="clv", nbins=60, color="clv_tier",
                       title="CLV Distribution by Tier",
                       labels={"clv": "Customer Lifetime Value (₹)"},
                       template="plotly_white", marginal="box")
    fig.show()

    # tier pie
    tier_rev = clv_df.groupby("clv_tier")["clv"].sum().reset_index()
    fig = px.pie(tier_rev, names="clv_tier", values="clv",
                 title="CLV Share by Customer Tier", hole=0.4,
                 template="plotly_white",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.show()

    # frequency vs CLV scatter
    fig = px.scatter(clv_df, x="frequency", y="clv", color="clv_tier",
                     opacity=0.6, size="monetary",
                     title="Purchase Frequency vs CLV",
                     labels={"frequency": "Purchase Frequency", "clv": "CLV (₹)"},
                     template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.show()

    print("\nCLV Tier Summary:")
    print(clv_df.groupby("clv_tier")["clv"].agg(["count", "mean", "sum"]).round(2))


def run_clv(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("  CUSTOMER LIFETIME VALUE PREDICTOR")
    print("=" * 60)
    plot_clv(df)
