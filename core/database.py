"""
core/database.py - Supabase Postgres + Storage helpers.
Falls back silently when Supabase is not configured.
"""
from __future__ import annotations
import io
import logging
from datetime import datetime, timezone
import pandas as pd
from core.config import settings

logger = logging.getLogger(__name__)


def _client():
    if not settings.supabase_ready:
        return None
    try:
        from supabase import create_client
        key = settings.supabase_service_key or settings.supabase_anon_key
        return create_client(settings.supabase_url, key)
    except Exception as e:
        logger.warning("Supabase client failed: %s", e)
        return None


def save_dataset(user_id: str, df: pd.DataFrame, name: str = "dataset") -> bool:
    client = _client()
    if not client:
        return False
    try:
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        path = f"{user_id}/{name}.csv"
        client.storage.from_("datasets").upload(
            path=path, file=buf.read(),
            file_options={"content-type": "text/csv", "upsert": "true"},
        )
        client.table("datasets").upsert({
            "user_id": user_id, "name": name,
            "row_count": len(df),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return True
    except Exception as e:
        logger.warning("Dataset save failed: %s", e)
        return False


def load_dataset(user_id: str, name: str = "dataset") -> pd.DataFrame | None:
    client = _client()
    if not client:
        return None
    try:
        raw = client.storage.from_("datasets").download(f"{user_id}/{name}.csv")
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        logger.warning("Dataset load failed: %s", e)
        return None


def log_event(user_id: str, action: str, detail: str = "") -> None:
    client = _client()
    if not client:
        return
    try:
        client.table("audit_log").insert({
            "user_id": user_id, "action": action, "detail": detail,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.debug("Audit log skipped: %s", e)


def save_price_recommendations(user_id: str, recs: list[dict]) -> None:
    client = _client()
    if not client or not recs:
        return
    try:
        for r in recs:
            r["user_id"] = user_id
        client.table("price_recommendations").upsert(recs).execute()
    except Exception as e:
        logger.warning("Price rec save failed: %s", e)


def save_at_risk_alerts(user_id: str, alerts: list[dict]) -> None:
    client = _client()
    if not client or not alerts:
        return
    try:
        for a in alerts:
            a["user_id"] = user_id
        client.table("at_risk_alerts").upsert(alerts).execute()
    except Exception as e:
        logger.warning("At-risk save failed: %s", e)


def save_drift_report(user_id: str, report: dict) -> None:
    client = _client()
    if not client:
        return
    try:
        report["user_id"] = user_id
        report["created_at"] = datetime.now(timezone.utc).isoformat()
        client.table("drift_reports").insert(report).execute()
    except Exception as e:
        logger.warning("Drift report save failed: %s", e)
