"""
core/database.py
Supabase Postgres client + operational persistence layer.
All functions fail silently when Supabase is not configured.

Tables used:
  operational_actions  - approved price changes, at-risk alerts, drift reports
  clv_cache            - cached CLV tier results per dataset hash
  anomaly_cache        - cached weekly anomaly scores
  model_results        - cached heavy model outputs (Prophet, SARIMA, MLP)
"""
from __future__ import annotations
import hashlib
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

import core.config as cfg

logger = logging.getLogger(__name__)


def _client():
    """Return a Supabase service-role client or None."""
    if not cfg.SUPABASE_READY:
        return None
    try:
        from supabase import create_client
        key = cfg.SUPABASE_SERVICE_KEY or cfg.SUPABASE_ANON_KEY
        return create_client(cfg.SUPABASE_URL, key)
    except Exception as e:
        logger.warning("Supabase client init failed: %s", e)
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _df_hash(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame for cache keying."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


#  Operational Actions 

def log_action(
    action_type: str,
    payload: dict,
    status: str = "pending",
    source: str = "dashboard",
) -> bool:
    """
    Persist an operational action to the operational_actions table.

    action_type: 'price_approval' | 'at_risk_export' | 'drift_alert' | 'inventory_alert'
    status:      'pending' | 'approved' | 'exported' | 'dismissed'
    """
    client = _client()
    if not client:
        return False
    try:
        client.table("operational_actions").insert({
            "action_type": action_type,
            "payload":     json.dumps(payload),
            "status":      status,
            "source":      source,
            "created_at":  _now(),
        }).execute()
        logger.info("Action logged: %s (%s)", action_type, status)
        return True
    except Exception as e:
        logger.warning("log_action failed: %s", e)
        return False


def update_action_status(action_id: int, status: str) -> bool:
    client = _client()
    if not client:
        return False
    try:
        client.table("operational_actions").update({
            "status": status, "updated_at": _now()
        }).eq("id", action_id).execute()
        return True
    except Exception as e:
        logger.warning("update_action_status failed: %s", e)
        return False


def get_pending_actions(action_type: str | None = None) -> list[dict]:
    client = _client()
    if not client:
        return []
    try:
        q = client.table("operational_actions").select("*").eq("status", "pending")
        if action_type:
            q = q.eq("action_type", action_type)
        res = q.order("created_at", desc=True).limit(50).execute()
        return res.data or []
    except Exception as e:
        logger.warning("get_pending_actions failed: %s", e)
        return []


#  CLV Cache 

def cache_clv(df: pd.DataFrame, clv_df: pd.DataFrame) -> bool:
    """Store CLV tier results keyed by dataset hash."""
    client = _client()
    if not client:
        return False
    try:
        key = _df_hash(df)
        client.table("clv_cache").upsert({
            "dataset_hash": key,
            "result_json":  clv_df.to_json(orient="records"),
            "row_count":    len(clv_df),
            "computed_at":  _now(),
        }).execute()
        return True
    except Exception as e:
        logger.warning("cache_clv failed: %s", e)
        return False


def load_clv_cache(df: pd.DataFrame) -> pd.DataFrame | None:
    """Load cached CLV results if available and fresh (< 24h)."""
    client = _client()
    if not client:
        return None
    try:
        key = _df_hash(df)
        res = client.table("clv_cache").select("*").eq("dataset_hash", key).execute()
        if not res.data:
            return None
        row = res.data[0]
        computed = datetime.fromisoformat(row["computed_at"])
        age_hours = (datetime.now(timezone.utc) - computed).total_seconds() / 3600
        if age_hours > 24:
            return None
        return pd.read_json(io.StringIO(row["result_json"]))
    except Exception as e:
        logger.warning("load_clv_cache failed: %s", e)
        return None


#  Anomaly Cache 

def cache_anomaly_scores(df: pd.DataFrame, anom_df: pd.DataFrame) -> bool:
    """Store weekly anomaly scores."""
    client = _client()
    if not client:
        return False
    try:
        key = _df_hash(df)
        client.table("anomaly_cache").upsert({
            "dataset_hash":    key,
            "anomaly_count":   int(anom_df["confirmed_anomaly"].sum()),
            "anomaly_pct":     float(anom_df["confirmed_anomaly"].mean()),
            "result_json":     anom_df[anom_df["confirmed_anomaly"]].head(200).to_json(orient="records"),
            "computed_at":     _now(),
        }).execute()
        return True
    except Exception as e:
        logger.warning("cache_anomaly_scores failed: %s", e)
        return False


def load_anomaly_cache(df: pd.DataFrame) -> pd.DataFrame | None:
    client = _client()
    if not client:
        return None
    try:
        key = _df_hash(df)
        res = client.table("anomaly_cache").select("*").eq("dataset_hash", key).execute()
        if not res.data:
            return None
        row = res.data[0]
        computed = datetime.fromisoformat(row["computed_at"])
        age_hours = (datetime.now(timezone.utc) - computed).total_seconds() / 3600
        if age_hours > 168:  # 1 week
            return None
        return pd.read_json(io.StringIO(row["result_json"]))
    except Exception as e:
        logger.warning("load_anomaly_cache failed: %s", e)
        return None


#  Model Results Cache 

def cache_model_result(model_name: str, df: pd.DataFrame, result: Any) -> bool:
    """Cache heavy model output (Prophet, SARIMA, MLP) keyed by model+data hash."""
    client = _client()
    if not client:
        return False
    try:
        key = f"{model_name}_{_df_hash(df)}"
        payload = result if isinstance(result, str) else json.dumps(result, default=str)
        client.table("model_results").upsert({
            "cache_key":   key,
            "model_name":  model_name,
            "result_json": payload,
            "computed_at": _now(),
        }).execute()
        return True
    except Exception as e:
        logger.warning("cache_model_result failed: %s", e)
        return False


def load_model_result(model_name: str, df: pd.DataFrame, max_age_hours: int = 24) -> Any:
    client = _client()
    if not client:
        return None
    try:
        key = f"{model_name}_{_df_hash(df)}"
        res = client.table("model_results").select("*").eq("cache_key", key).execute()
        if not res.data:
            return None
        row = res.data[0]
        computed = datetime.fromisoformat(row["computed_at"])
        age_hours = (datetime.now(timezone.utc) - computed).total_seconds() / 3600
        if age_hours > max_age_hours:
            return None
        return json.loads(row["result_json"])
    except Exception as e:
        logger.warning("load_model_result failed: %s", e)
        return None


#  Dataset Storage 

def save_dataset(df: pd.DataFrame, name: str = "dataset") -> bool:
    client = _client()
    if not client:
        return False
    try:
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        client.storage.from_("datasets").upload(
            path=f"shared/{name}.csv",
            file=buf.read(),
            file_options={"content-type": "text/csv", "upsert": "true"},
        )
        return True
    except Exception as e:
        logger.warning("save_dataset failed: %s", e)
        return False


def load_dataset(name: str = "dataset") -> pd.DataFrame | None:
    client = _client()
    if not client:
        return None
    try:
        raw = client.storage.from_("datasets").download(f"shared/{name}.csv")
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        logger.warning("load_dataset failed: %s", e)
        return None
