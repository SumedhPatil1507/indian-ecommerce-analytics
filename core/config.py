"""
core/config.py — centralised environment-based configuration.
All secrets come from .env (local) or Streamlit secrets (cloud).
Never hardcoded.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

# Load .env file automatically if python-dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass  # dotenv not required in production (env vars set directly)


def _env(key: str, default: str = "") -> str:
    """Read from env, then Streamlit secrets, then default."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


@dataclass
class Settings:
    # Supabase
    supabase_url:        str = field(default_factory=lambda: _env("SUPABASE_URL"))
    supabase_anon_key:   str = field(default_factory=lambda: _env("SUPABASE_ANON_KEY"))
    supabase_service_key:str = field(default_factory=lambda: _env("SUPABASE_SERVICE_KEY"))

    # App
    app_name:    str = "IndiaCommerce Analytics"
    app_version: str = "2.0.0"
    debug:       bool = field(default_factory=lambda: _env("DEBUG", "false").lower() == "true")

    # Data
    kaggle_url: str = (
        "https://www.kaggle.com/datasets/shukla922/"
        "indian-e-commerce-pricing-revenue-growth"
    )

    @property
    def supabase_ready(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)


settings = Settings()
