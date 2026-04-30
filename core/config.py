"""core/config.py - App configuration. No external auth dependencies."""
from __future__ import annotations
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

APP_NAME    = "IndiaCommerce Analytics"
APP_VERSION = "3.0.0"
KAGGLE_URL  = "https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth"
DEBUG       = os.getenv("DEBUG","false").lower() == "true"
