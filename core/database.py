"""
core/database.py
Multi-tenant Supabase Postgres client + operational persistence layer.
Propagates tenant JWT for Row-Level Security (RLS) validation.
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


def get_client(token: str | None = None):
    """
    Return a Supabase client.
    If token is provided, initializes client with the user's JWT to enforce RLS.
    Otherwise, falls back to the service-role client (useful for background workers).
    """
    if not cfg.SUPABASE_READY:
        return None
    try:
        from supabase import create_client
        try:
            from supabase.client import ClientOptions
            options = ClientOptions(headers={"Authorization": f"Bearer {token}"}) if token else ClientOptions()
        except ImportError:
            options = None

        # If token is provided, use anon key (JWT enforces permissions). Otherwise, use service key.
        key = cfg.SUPABASE_SERVICE_KEY if (not token and cfg.SUPABASE_SERVICE_KEY) else cfg.SUPABASE_ANON_KEY
        if options:
            return create_client(cfg.SUPABASE_URL, key, options=options)
        else:
            return create_client(cfg.SUPABASE_URL, key)
    except Exception as e:
        logger.warning("Supabase client init failed: %s", e)
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _df_hash(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame for cache keying."""
    if df.empty:
        return "empty"
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


#  Profiles & Webhook Secrets 

def get_profile(tenant_id: str) -> dict | None:
    client = get_client()
    if not client:
        return None
    try:
        res = client.table("profiles").select("*").eq("id", tenant_id).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        logger.warning("get_profile failed: %s", e)
        return None


def update_webhook_secret(tenant_id: str, secret: str) -> bool:
    client = get_client()
    if not client:
        return False
    try:
        client.table("profiles").upsert({
            "id": tenant_id,
            "webhook_secret": secret,
            "updated_at": _now()
        }).execute()
        return True
    except Exception as e:
        logger.warning("update_webhook_secret failed: %s", e)
        return False


#  Orders relational storage 

def save_orders(tenant_id: str, df: pd.DataFrame) -> bool:
    """Bulk insert order records into PostgreSQL."""
    client = get_client()
    if not client or df.empty:
        return False
    try:
        # Convert dataframe to records
        records = []
        for _, row in df.iterrows():
            # Parse order_date
            o_date = row.get("order_date")
            if isinstance(o_date, pd.Timestamp):
                o_date = o_date.isoformat()
            elif isinstance(o_date, str):
                try:
                    o_date = pd.to_datetime(o_date).isoformat()
                except Exception:
                    o_date = _now()
            else:
                o_date = _now()

            records.append({
                "tenant_id":             tenant_id,
                "order_id":              str(row.get("order_id", "")),
                "order_date":            o_date,
                "state":                 str(row.get("state", "Unknown")),
                "zone":                  str(row.get("zone", "Central")),
                "category":              str(row.get("category", "Fashion")),
                "brand_type":            str(row.get("brand_type", "Mass")),
                "customer_gender":       str(row.get("customer_gender", "Unknown")),
                "customer_age":          int(row.get("customer_age", 30)),
                "base_price":            float(row.get("base_price", 0.0)),
                "discount_percent":      float(row.get("discount_percent", 0.0)),
                "final_price":           float(row.get("final_price", 0.0)),
                "units_sold":            int(row.get("units_sold", 1)),
                "revenue":               float(row.get("revenue", 0.0)),
                "sales_event":           str(row.get("sales_event", "Normal")),
                "competition_intensity": str(row.get("competition_intensity", "Medium")),
                "inventory_pressure":    str(row.get("inventory_pressure", "Low")),
                "source":                str(row.get("source", "upload")),
                "customer_id":           str(row.get("customer_id", "")),
            })

        # Insert in chunks of 500 to avoid Supabase request size limits
        chunk_size = 500
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            client.table("orders").insert(chunk).execute()
        
        # Log action
        log_action(tenant_id, "ingest_orders", {"row_count": len(df)}, status="completed")
        return True
    except Exception as e:
        logger.warning("save_orders failed: %s", e)
        return False


def load_orders(tenant_id: str, token: str | None = None) -> pd.DataFrame:
    """Load all orders for a tenant, automatically paginating."""
    client = get_client(token)
    if not client:
        return pd.DataFrame()
    try:
        q = client.table("orders").select("*")
        if not token:
            q = q.eq("tenant_id", tenant_id)

        all_data = []
        offset = 0
        limit = 1000
        while True:
            res = q.range(offset, offset + limit - 1).execute()
            data = res.data or []
            all_data.extend(data)
            if len(data) < limit:
                break
            offset += limit

        df = pd.DataFrame(all_data)
        if not df.empty:
            # Reconstruct datetime and index if necessary
            df["order_date"] = pd.to_datetime(df["order_date"])
        return df
    except Exception as e:
        logger.warning("load_orders failed: %s", e)
        return pd.DataFrame()


#  Operational Actions 

def log_action(
    tenant_id: str,
    action_type: str,
    payload: dict,
    status: str = "pending",
    source: str = "dashboard",
    token: str | None = None,
) -> bool:
    client = get_client(token)
    if not client:
        return False
    try:
        client.table("operational_actions").insert({
            "tenant_id":   tenant_id,
            "action_type": action_type,
            "payload":     json.dumps(payload),
            "status":      status,
            "source":      source,
            "created_at":  _now(),
        }).execute()
        return True
    except Exception as e:
        logger.warning("log_action failed: %s", e)
        return False


def update_action_status(action_id: int, status: str, token: str | None = None) -> bool:
    client = get_client(token)
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


def get_pending_actions(tenant_id: str, action_type: str | None = None, token: str | None = None) -> list[dict]:
    client = get_client(token)
    if not client:
        return []
    try:
        q = client.table("operational_actions").select("*").eq("status", "pending")
        if not token:
            q = q.eq("tenant_id", tenant_id)
        if action_type:
            q = q.eq("action_type", action_type)
        res = q.order("created_at", desc=True).limit(50).execute()
        return res.data or []
    except Exception as e:
        logger.warning("get_pending_actions failed: %s", e)
        return []


#  CLV Cache 

def cache_clv(tenant_id: str, df: pd.DataFrame, clv_df: pd.DataFrame) -> bool:
    client = get_client()
    if not client:
        return False
    try:
        key = _df_hash(df)
        client.table("clv_cache").upsert({
            "tenant_id":    tenant_id,
            "dataset_hash": key,
            "result_json":  clv_df.to_json(orient="records"),
            "row_count":    len(clv_df),
            "computed_at":  _now(),
        }).execute()
        return True
    except Exception as e:
        logger.warning("cache_clv failed: %s", e)
        return False


def load_clv_cache(tenant_id: str, df: pd.DataFrame, token: str | None = None) -> pd.DataFrame | None:
    client = get_client(token)
    if not client:
        return None
    try:
        key = _df_hash(df)
        q = client.table("clv_cache").select("*").eq("dataset_hash", key)
        if not token:
            q = q.eq("tenant_id", tenant_id)
        res = q.execute()
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

def cache_anomaly_scores(tenant_id: str, df: pd.DataFrame, anom_df: pd.DataFrame) -> bool:
    client = get_client()
    if not client:
        return False
    try:
        key = _df_hash(df)
        client.table("anomaly_cache").upsert({
            "tenant_id":       tenant_id,
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


def load_anomaly_cache(tenant_id: str, df: pd.DataFrame, token: str | None = None) -> pd.DataFrame | None:
    client = get_client(token)
    if not client:
        return None
    try:
        key = _df_hash(df)
        q = client.table("anomaly_cache").select("*").eq("dataset_hash", key)
        if not token:
            q = q.eq("tenant_id", tenant_id)
        res = q.execute()
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

def cache_model_result(tenant_id: str, model_name: str, df: pd.DataFrame, result: Any) -> bool:
    client = get_client()
    if not client:
        return False
    try:
        key = f"{model_name}_{_df_hash(df)}"
        payload = result if isinstance(result, str) else json.dumps(result, default=str)
        client.table("model_results").upsert({
            "tenant_id":   tenant_id,
            "cache_key":   key,
            "model_name":  model_name,
            "result_json": payload,
            "computed_at": _now(),
        }).execute()
        return True
    except Exception as e:
        logger.warning("cache_model_result failed: %s", e)
        return False


def load_model_result(tenant_id: str, model_name: str, df: pd.DataFrame, max_age_hours: int = 24, token: str | None = None) -> Any:
    client = get_client(token)
    if not client:
        return None
    try:
        key = f"{model_name}_{_df_hash(df)}"
        q = client.table("model_results").select("*").eq("cache_key", key)
        if not token:
            q = q.eq("tenant_id", tenant_id)
        res = q.execute()
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


#  Price Recommendations 

def cache_price_recommendations(tenant_id: str, recs_df: pd.DataFrame) -> bool:
    client = get_client()
    if not client or recs_df.empty:
        return False
    try:
        # Delete old recommendations
        client.table("price_recommendations").delete().eq("tenant_id", tenant_id).execute()
        
        # Bulk insert new ones
        records = []
        for _, r in recs_df.iterrows():
            records.append({
                "tenant_id":          tenant_id,
                "category":           r["category"],
                "current_discount":   float(r["current_discount"]),
                "optimal_discount":   float(r["optimal_discount"]),
                "direction":          r["direction"],
                "revenue_impact_pct": float(r["revenue_impact_pct"]),
            })
        client.table("price_recommendations").insert(records).execute()
        return True
    except Exception as e:
        logger.warning("cache_price_recommendations failed: %s", e)
        return False


def load_price_recommendations(tenant_id: str, token: str | None = None) -> pd.DataFrame:
    client = get_client(token)
    if not client:
        return pd.DataFrame()
    try:
        q = client.table("price_recommendations").select("*")
        if not token:
            q = q.eq("tenant_id", tenant_id)
        res = q.order("created_at", desc=True).execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        logger.warning("load_price_recommendations failed: %s", e)
        return pd.DataFrame()


#  At-Risk Alerts 

def cache_at_risk_alerts(tenant_id: str, alerts_df: pd.DataFrame) -> bool:
    client = get_client()
    if not client or alerts_df.empty:
        return False
    try:
        # Delete old alerts
        client.table("at_risk_alerts").delete().eq("tenant_id", tenant_id).execute()
        
        # Bulk insert new ones
        records = []
        for _, r in alerts_df.iterrows():
            records.append({
                "tenant_id":          tenant_id,
                "customer_id":        str(r["customer_id"]),
                "churn_risk_score":   float(r["churn_risk_score"]),
                "risk_label":         r["risk_label"],
                "value_tier":         r["value_tier"],
                "days_since_order":   int(r.get("days_since_last_order", r.get("days_since_order", 0))),
                "total_revenue":      float(r["total_revenue"]),
                "recommended_action": r["recommended_action"],
            })
        client.table("at_risk_alerts").insert(records).execute()
        return True
    except Exception as e:
        logger.warning("cache_at_risk_alerts failed: %s", e)
        return False


def load_at_risk_alerts(tenant_id: str, token: str | None = None) -> pd.DataFrame:
    client = get_client(token)
    if not client:
        return pd.DataFrame()
    try:
        q = client.table("at_risk_alerts").select("*")
        if not token:
            q = q.eq("tenant_id", tenant_id)
        res = q.order("churn_risk_score", desc=True).execute()
        return pd.DataFrame(res.data or [])
    except Exception as e:
        logger.warning("load_at_risk_alerts failed: %s", e)
        return pd.DataFrame()


#  Drift Reports 

def cache_drift_report(tenant_id: str, report: dict) -> bool:
    client = get_client()
    if not client:
        return False
    try:
        # Delete old drift reports
        client.table("drift_reports").delete().eq("tenant_id", tenant_id).execute()
        
        # Insert new report
        client.table("drift_reports").insert({
            "tenant_id":        tenant_id,
            "features_drifted": int(report.get("features_drifted", 0)),
            "max_psi":          float(report.get("max_psi", 0.0)),
            "pred_r2_drop":     float(report.get("pred_r2_drop", 0.0)),
            "drift_alert":      bool(report.get("drift_alert", False)),
            "report_json":      json.dumps(report),
        }).execute()
        return True
    except Exception as e:
        logger.warning("cache_drift_report failed: %s", e)
        return False


def load_drift_report(tenant_id: str, token: str | None = None) -> dict | None:
    client = get_client(token)
    if not client:
        return None
    try:
        q = client.table("drift_reports").select("*")
        if not token:
            q = q.eq("tenant_id", tenant_id)
        res = q.order("created_at", desc=True).limit(1).execute()
        if res.data:
            return json.loads(res.data[0]["report_json"])
        return None
    except Exception as e:
        logger.warning("load_drift_report failed: %s", e)
        return None


#  Datasets Listing 

def save_dataset_meta(tenant_id: str, name: str, row_count: int, source: str = "upload") -> bool:
    client = get_client()
    if not client:
        return False
    try:
        client.table("datasets").insert({
            "tenant_id": tenant_id,
            "name": name,
            "row_count": row_count,
            "source": source,
            "uploaded_at": _now()
        }).execute()
        return True
    except Exception as e:
        logger.warning("save_dataset_meta failed: %s", e)
        return False


def load_dataset_meta(tenant_id: str, token: str | None = None) -> list[dict]:
    client = get_client(token)
    if not client:
        return []
    try:
        q = client.table("datasets").select("*")
        if not token:
            q = q.eq("tenant_id", tenant_id)
        res = q.order("uploaded_at", desc=True).execute()
        return res.data or []
    except Exception as e:
        logger.warning("load_dataset_meta failed: %s", e)
        return []
