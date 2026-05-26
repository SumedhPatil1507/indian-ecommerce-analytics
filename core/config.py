"""core/config.py - Centralised app configuration."""
from __future__ import annotations
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

APP_NAME    = "IndiaCommerce Analytics"
APP_VERSION = "4.0.0"
KAGGLE_URL  = "https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth"
DEBUG       = os.getenv("DEBUG", "false").lower() == "true"

# Supabase (optional - graceful fallback when not configured)
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY    = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_READY       = bool(SUPABASE_URL and SUPABASE_ANON_KEY)

# SendGrid (optional - for at-risk cohort export)
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
SENDGRID_FROM    = os.getenv("SENDGRID_FROM_EMAIL", "analytics@indiacommerce.app")
