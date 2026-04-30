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
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN


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

    df["anomaly_votes"] = (
        df["is_anomaly"].astype(int) +
        df["zscore_anomaly"].astype(int) +
        df["dbscan_anomaly"].astype(int)
    )
    df["confirmed_anomaly"] = df["anomaly_votes"] >= 2
    return df


#  plots 

def plot_anomalies(df: pd.DataFrame) -> None:
    df = anomaly_report(df)

    # scatter: units vs revenue
    fig = px.scatter(
        df, x="log_units_sold", y="log_revenue",
        color="confirmed_anomaly",
        color_discrete_map={True: "red", False: "steelblue"},
        opacity=0.6, size_max=8,
        title="Anomaly Detection  log(Units) vs log(Revenue)",
        labels={"log_units_sold": "log(Units Sold)", "log_revenue": "log(Revenue)",
                "confirmed_anomaly": "Anomaly"},
        template="plotly_white",
        hover_data=["category", "zone", "discount_percent", "anomaly_votes"],
    )
    fig.show()

    # scatter: discount vs price
    fig = px.scatter(
        df, x="discount_percent", y="log_final_price",
        color="confirmed_anomaly",
        color_discrete_map={True: "red", False: "lightgray"},
        opacity=0.55,
        title="Anomaly Detection  Discount % vs log(Final Price)",
        labels={"discount_percent": "Discount %", "log_final_price": "log(Final Price)",
                "confirmed_anomaly": "Anomaly"},
        template="plotly_white",
        hover_data=["category", "zone", "anomaly_votes"],
    )
    fig.show()

    # anomaly count by category
    cat_anom = (
        df[df["confirmed_anomaly"]]
        .groupby("category")
        .size()
        .reset_index(name="anomaly_count")
        .sort_values("anomaly_count", ascending=False)
    )
    fig = px.bar(cat_anom, x="category", y="anomaly_count",
                 title="Confirmed Anomalies by Category",
                 labels={"anomaly_count": "Anomaly Count"},
                 template="plotly_white", color="category")
    fig.show()

    n = df["confirmed_anomaly"].sum()
    print(f"\n Anomaly detection complete.")
    print(f"   Confirmed anomalies (2 detectors): {n:,} ({n/len(df):.2%} of orders)")
    return df


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("  AUTOMATED ANOMALY DETECTION")
    print("=" * 60)
    return plot_anomalies(df)
