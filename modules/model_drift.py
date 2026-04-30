"""
modules/model_drift.py
MLOps & Model Integrity — Model Drift Monitoring.
Detects data drift (feature distribution shift) and prediction drift
between a reference window and the current window.
Persists drift reports to Supabase.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

logger = logging.getLogger(__name__)

_DRIFT_THRESHOLD_PSI  = 0.2   # PSI > 0.2 = significant drift
_DRIFT_THRESHOLD_KS   = 0.05  # KS p-value < 0.05 = significant drift


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index — standard drift metric."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_drift(
    df: pd.DataFrame,
    reference_months: int = 6,
    current_months: int = 3,
) -> pd.DataFrame:
    """
    Compare feature distributions between reference and current windows.

    Returns a DataFrame with drift metrics per feature.
    """
    df = df.copy()
    df = df.sort_values("order_date")

    cutoff_ref = df["order_date"].max() - pd.DateOffset(months=current_months)
    cutoff_start = cutoff_ref - pd.DateOffset(months=reference_months)

    ref = df[(df["order_date"] >= cutoff_start) & (df["order_date"] < cutoff_ref)]
    cur = df[df["order_date"] >= cutoff_ref]

    if len(ref) < 50 or len(cur) < 50:
        logger.warning("Not enough data for drift analysis (ref=%d, cur=%d)", len(ref), len(cur))
        return pd.DataFrame()

    num_cols = ["base_price", "discount_percent", "final_price", "units_sold", "revenue"]
    records = []

    for col in num_cols:
        if col not in df.columns:
            continue
        ref_vals = ref[col].dropna().values
        cur_vals = cur[col].dropna().values
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            continue

        psi_val = _psi(ref_vals, cur_vals)
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)

        drift_detected = psi_val > _DRIFT_THRESHOLD_PSI or ks_pval < _DRIFT_THRESHOLD_KS
        severity = (
            "High"   if psi_val > 0.25 else
            "Medium" if psi_val > _DRIFT_THRESHOLD_PSI else
            "Low"
        )

        records.append({
            "feature":        col,
            "psi":            round(psi_val, 4),
            "ks_statistic":   round(float(ks_stat), 4),
            "ks_p_value":     round(float(ks_pval), 4),
            "drift_detected": drift_detected,
            "severity":       severity if drift_detected else "None",
            "ref_mean":       round(float(ref_vals.mean()), 2),
            "cur_mean":       round(float(cur_vals.mean()), 2),
            "mean_shift_pct": round((cur_vals.mean() - ref_vals.mean()) / (ref_vals.mean() + 1e-9) * 100, 1),
        })

    return pd.DataFrame(records).sort_values("psi", ascending=False)


def compute_prediction_drift(
    df: pd.DataFrame,
    reference_months: int = 6,
    current_months: int = 3,
) -> dict:
    """
    Train a simple XGBoost model on reference data, evaluate on current data.
    Returns performance metrics for both windows to detect prediction drift.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        return {}

    df = df.copy().sort_values("order_date")
    cutoff_ref   = df["order_date"].max() - pd.DateOffset(months=current_months)
    cutoff_start = cutoff_ref - pd.DateOffset(months=reference_months)

    ref = df[(df["order_date"] >= cutoff_start) & (df["order_date"] < cutoff_ref)].copy()
    cur = df[df["order_date"] >= cutoff_ref].copy()

    if len(ref) < 100 or len(cur) < 50:
        return {}

    features = ["base_price", "discount_percent", "units_sold", "month", "weekday"]
    cat_features = ["category", "zone", "brand_type", "sales_event"]

    for col in cat_features:
        le = LabelEncoder()
        combined = pd.concat([ref[col], cur[col]])
        le.fit(combined)
        ref[col + "_enc"] = le.transform(ref[col])
        cur[col + "_enc"] = le.transform(cur[col])

    feat_cols = features + [c + "_enc" for c in cat_features]
    feat_cols = [c for c in feat_cols if c in ref.columns]

    model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(ref[feat_cols], ref["revenue"])

    ref_pred = model.predict(ref[feat_cols])
    cur_pred = model.predict(cur[feat_cols])

    return {
        "ref_r2":  round(float(r2_score(ref["revenue"], ref_pred)), 3),
        "cur_r2":  round(float(r2_score(cur["revenue"], cur_pred)), 3),
        "ref_mae": round(float(mean_absolute_error(ref["revenue"], ref_pred)), 0),
        "cur_mae": round(float(mean_absolute_error(cur["revenue"], cur_pred)), 0),
        "r2_drop": round(float(r2_score(ref["revenue"], ref_pred) - r2_score(cur["revenue"], cur_pred)), 3),
        "drift_alert": r2_score(ref["revenue"], ref_pred) - r2_score(cur["revenue"], cur_pred) > 0.05,
    }


def save_drift_report(drift_df: pd.DataFrame, pred_drift: dict, user_id: str) -> None:
    if not user_id or drift_df.empty:
        return
    try:
        from core.database import _client
        client = _client()
        if not client:
            return
        client.table("drift_reports").insert({
            "user_id":          user_id,
            "features_drifted": int(drift_df["drift_detected"].sum()),
            "max_psi":          float(drift_df["psi"].max()),
            "pred_r2_drop":     pred_drift.get("r2_drop", 0),
            "drift_alert":      bool(pred_drift.get("drift_alert", False)),
            "report_json":      drift_df.to_json(orient="records"),
            "created_at":       datetime.now(timezone.utc).isoformat(),
        }).execute()
        logger.info("Drift report saved to Supabase")
    except Exception as e:
        logger.warning("Drift report save failed: %s", e)


def plot_drift(drift_df: pd.DataFrame):
    """Return Plotly figure for drift dashboard."""
    if drift_df.empty:
        return None

    color_map = {"High": "#ef4444", "Medium": "#f97316", "Low": "#eab308", "None": "#22c55e"}
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=drift_df["feature"],
        y=drift_df["psi"],
        marker_color=[color_map.get(s, "#94a3b8") for s in drift_df["severity"]],
        name="PSI Score",
        text=drift_df["severity"],
        textposition="outside",
    ))
    fig.add_hline(y=_DRIFT_THRESHOLD_PSI, line_dash="dash", line_color="#ef4444",
                  annotation_text=f"Drift threshold (PSI={_DRIFT_THRESHOLD_PSI})")
    fig.update_layout(
        title="Feature Drift — Population Stability Index (PSI)",
        xaxis_title="Feature",
        yaxis_title="PSI Score",
        template="plotly_white",
    )
    return fig
