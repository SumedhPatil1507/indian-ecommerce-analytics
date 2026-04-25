"""
core/database.py — Supabase Postgres helpers.
Stores uploaded datasets and audit logs per user.
Falls back to in-memory when Supabase is not configured.
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
        from supabase import create_client  # type: ignore
        return create_client(settings.supabase_url, settings.supabase_service_key or settings.supabase_anon_key)
    except ImportError:
        return None


# ── Dataset persistence ───────────────────────────────────────────────────────

def save_dataset(user_id: str, df: pd.DataFrame, name: str = "dataset") -> bool:
    """Save a DataFrame as CSV to Supabase Storage."""
    client = _client()
    if not client:
        return False
    try:
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        path = f"{user_id}/{name}.csv"
        client.storage.from_("datasets").upload(
            path, buf.read(), {"content-type": "text/csv", "upsert": "true"}
        )
        logger.info("Dataset saved to Supabase: %s", path)
        return True
    except Exception as e:
        logger.warning("Dataset save failed: %s", e)
        return False


def load_dataset(user_id: str, name: str = "dataset") -> pd.DataFrame | None:
    """Load a previously saved DataFrame from Supabase Storage."""
    client = _client()
    if not client:
        return None
    try:
        path = f"{user_id}/{name}.csv"
        raw  = client.storage.from_("datasets").download(path)
        df   = pd.read_csv(io.BytesIO(raw))
        logger.info("Dataset loaded from Supabase: %s", path)
        return df
    except Exception as e:
        logger.warning("Dataset load failed: %s", e)
        return None


# ── Audit log ─────────────────────────────────────────────────────────────────

def log_event(user_id: str, action: str, detail: str = "") -> None:
    """Write an audit log entry to Supabase."""
    client = _client()
    if not client:
        return
    try:
        client.table("audit_log").insert({
            "user_id":    user_id,
            "action":     action,
            "detail":     detail,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.debug("Audit log failed (non-critical): %s", e)
