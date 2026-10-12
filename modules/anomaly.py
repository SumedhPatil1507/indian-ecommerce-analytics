"""
modules/anomaly.py

Automated Anomaly Detection
   Isolation Forest  (global outliers)
   Z-score / IQR     (univariate, per column)
   DBSCAN            (density-based spatial outliers)
   Interactive Plotly scatter + heatmap of anomaly scores
   Exportable anomaly report DataFrame
"""
import numpy as np
import pandas as pd
import logging
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


#  feature prep 

_COLS = ["customer_age", "log_base_price", "discount_percent",
         "log_final_price", "log_units_sold", "log_revenue"]


def _prepare(df: pd.DataFrame) -> np.ndarray:
    X = df[_COLS].copy()
    return RobustScaler().fit_transform(X)


#  Isolation Forest 

def detect_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.008,
) -> pd.DataFrame:
    """
    Detect anomalies using Isolation Forest.

    Returns df with added columns:
        iso_score    raw anomaly score (lower = more anomalous)
        is_anomaly   boolean flag
    """
    X = _prepare(df)
    iso = IsolationForest(n_estimators=150, contamination=contamination,
                          random_state=42, n_jobs=-1)
    df = df.copy()
    df["iso_score"]  = iso.fit_predict(X)
    df["is_anomaly"] = df["iso_score"] == -1
    return df


#  Z-score univariate 

def detect_zscore(
    df: pd.DataFrame,
    cols: list[str] = ["revenue", "units_sold", "discount_percent"],
    threshold: float = 3.5,
) -> pd.DataFrame:
    df = df.copy()
    df["zscore_anomaly"] = False
    for col in cols:
        if col not in df.columns:
            continue
        z = np.abs((df[col] - df[col].mean()) / df[col].std())
        df["zscore_anomaly"] |= z > threshold
    return df


#  DBSCAN 

def detect_dbscan(
    df: pd.DataFrame,
    eps: float = 0.5,
    min_samples: int = 10,
) -> pd.DataFrame:
    X = _prepare(df)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X)
    df = df.copy()
    df["dbscan_label"]   = labels
    df["dbscan_anomaly"] = labels == -1
    return df


#  combined report 

def anomaly_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all detectors and return a combined anomaly report.
    A row is flagged if  2 detectors agree.
    """
    df = detect_isolation_forest(df)
    df = detect_zscore(df)
    df = detect_dbscan(df)

    # Optional trained graph model contributes RTO and promotional-fraud risk.
    # A missing checkpoint or optional PyG install leaves existing analytics usable.
    try:
        from modules.rto_graph import predict_risk_scores
        graph_scores = predict_risk_scores(df)
    except Exception as exc:
        logger.warning("R-GCN risk scoring unavailable: %s", exc)
        graph_scores = None
    if graph_scores is not None:
        df["rto_probability"] = graph_scores["rto_probability"]
        df["promo_fraud_risk"] = graph_scores["promo_fraud_risk"]
        df["gnn_risk_score"] = df[["rto_probability", "promo_fraud_risk"]].max(axis=1)
        df["gnn_risk_flag"] = df["gnn_risk_score"] >= 0.5
    else:
        df["rto_probability"] = np.nan
        df["promo_fraud_risk"] = np.nan
        df["gnn_risk_score"] = np.nan
        df["gnn_risk_flag"] = False

    df["anomaly_votes"] = (
        df["is_anomaly"].astype(int) +
        df["zscore_anomaly"].astype(int) +
        df["dbscan_anomaly"].astype(int) +
        df["gnn_risk_flag"].astype(int)
    )
    # Graph risk is the strongest single attribution, blended with the
    # Isolation Forest decision when a trained GNN is available.
    df["risk_attribution_score"] = np.where(
        df["gnn_risk_score"].notna(),
        0.6 * df["gnn_risk_score"].fillna(0) + 0.4 * df["is_anomaly"].astype(float),
        df["is_anomaly"].astype(float),
    )
    df["confirmed_anomaly"] = df["anomaly_votes"] >= 2
    return df


#  plots 

def plot_anomalies(df: pd.DataFrame) -> list:
    """
    Generate anomaly detection figures.

    Returns a list of three interactive Plotly figures:
      [0] log(Units) vs log(Revenue) scatter
      [1] Discount % vs log(Final Price) scatter
      [2] Confirmed anomalies by category bar chart
    """
    df = anomaly_report(df)

    figs = []

    # scatter: units vs revenue
    fig1 = px.scatter(
        df, x="log_units_sold", y="log_revenue",
        color="confirmed_anomaly",
        color_discrete_map={True: "red", False: "steelblue"},
        opacity=0.6, size_max=8,
        title="Anomaly Detection — log(Units) vs log(Revenue)",
        labels={"log_units_sold": "log(Units Sold)", "log_revenue": "log(Revenue)",
                "confirmed_anomaly": "Anomaly"},
        template="plotly_white",
        hover_data=["category", "zone", "discount_percent", "anomaly_votes"],
    )
    figs.append(fig1)

    # scatter: discount vs price
    fig2 = px.scatter(
        df, x="discount_percent", y="log_final_price",
        color="confirmed_anomaly",
        color_discrete_map={True: "red", False: "lightgray"},
        opacity=0.55,
        title="Anomaly Detection — Discount % vs log(Final Price)",
        labels={"discount_percent": "Discount %", "log_final_price": "log(Final Price)",
                "confirmed_anomaly": "Anomaly"},
        template="plotly_white",
        hover_data=["category", "zone", "anomaly_votes"],
    )
    figs.append(fig2)

    # anomaly count by category
    cat_anom = (
        df[df["confirmed_anomaly"]]
        .groupby("category")
        .size()
        .reset_index(name="anomaly_count")
        .sort_values("anomaly_count", ascending=False)
    )
    fig3 = px.bar(cat_anom, x="category", y="anomaly_count",
                  title="Confirmed Anomalies by Category",
                  labels={"anomaly_count": "Anomaly Count"},
                  template="plotly_white", color="category")
    figs.append(fig3)

    return figs


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("  AUTOMATED ANOMALY DETECTION")
    print("=" * 60)
    figs = plot_anomalies(df)
    for fig in figs:
        fig.show()
    df = anomaly_report(df)
    n = df["confirmed_anomaly"].sum()
    print(f"\n✓ Anomaly detection complete.")
    print(f"  Confirmed anomalies (≥2 detectors): {n:,} ({n/len(df):.2%} of orders)")
    return df
