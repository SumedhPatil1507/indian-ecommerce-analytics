"""
data/loader.py
Centralised data loading, cleaning, feature engineering, and live data enrichment.

Live data sources:
  1. World Bank Open Data API - India GDP growth & CPI inflation
     https://data.worldbank.org/ (CC BY 4.0)
  2. fawazahmed0/exchange-api - live USD/INR rate (CC0)
     https://github.com/fawazahmed0/exchange-api
  3. pytrends - Google Trends search interest (Apache 2.0)
     https://github.com/GeneralMills/pytrends
"""
from __future__ import annotations
import io
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/shukla922/"
    "indian-e-commerce-pricing-revenue-growth"
)

# World Bank
_WB_BASE    = "https://api.worldbank.org/v2"
_WB_GDP     = "NY.GDP.MKTP.KD.ZG"
_WB_CPI     = "FP.CPI.TOTL.ZG"
_WB_COUNTRY = "IN"

# FX sources (tried in order, first valid rate wins)
_FX_SOURCES = [
    {
        "url":   "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json",
        "parse": lambda d: d.get("usd", {}).get("inr"),
        "name":  "fawazahmed0/exchange-api (CC0)",
    },
    {
        "url":   "https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR",
        "parse": lambda d: d.get("rates", {}).get("INR"),
        "name":  "frankfurter.dev (ECB)",
    },
    {
        "url":   "https://open.er-api.com/v6/latest/USD",
        "parse": lambda d: d.get("rates", {}).get("INR"),
        "name":  "open.er-api.com",
    },
]
_FX_FALLBACK = 84.0

_TREND_KEYWORDS = [
    "online shopping India",
    "Flipkart sale",
    "Amazon India deals",
    "fashion ecommerce India",
    "electronics discount India",
]


# ── Public API ────────────────────────────────────────────────────────────────

def load(path: str, enrich_live: bool = True) -> pd.DataFrame:
    """Load CSV, clean, engineer features, optionally enrich with live macro."""
    df = pd.read_csv(path)
    df = _clean(df)
    df = _engineer(df)
    if enrich_live:
        df = _enrich_worldbank(df)
        df = _enrich_fx(df)
        df = _enrich_trends(df)
    return df


def load_any(file_obj, filename: str) -> pd.DataFrame:
    """
    Load from any supported format: CSV, TSV, Excel, JSON, Parquet.
    file_obj can be a BytesIO, bytes, or file-like object.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if isinstance(file_obj, (bytes, bytearray)):
        raw = file_obj
    elif hasattr(file_obj, "read"):
        raw = file_obj.read()
    else:
        raw = bytes(file_obj)

    buf = io.BytesIO(raw)
    if ext == "csv":
        df = pd.read_csv(buf)
    elif ext == "tsv":
        df = pd.read_csv(buf, sep="\t")
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(buf)
    elif ext == "json":
        df = pd.read_json(buf)
    elif ext == "parquet":
        df = pd.read_parquet(buf)
    else:
        raise ValueError(
            f"Unsupported format: .{ext}. "
            "Supported: csv, tsv, xlsx, xls, json, parquet"
        )
    return _engineer(_clean(df))


# ── Cleaning ──────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["discount_percent"] = df["discount_percent"].clip(0, 100)
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]           = df["order_date"].dt.year
    df["month"]          = df["order_date"].dt.month
    df["month_name"]     = df["order_date"].dt.month_name()
    df["weekday"]        = df["order_date"].dt.weekday
    df["is_weekend"]     = df["weekday"] >= 5
    df["fiscal_quarter"] = ((df["month"] - 4) % 12 // 3 + 1).astype("Int8")
    df["year_month"]     = df["order_date"].dt.to_period("M").astype(str)
    df["discount_amount"]  = df["base_price"] - df["final_price"]
    df["discount_ratio"]   = df["discount_amount"] / df["base_price"].replace(0, np.nan)
    df["revenue_per_unit"] = df["final_price"]
    for col in ["base_price", "final_price", "units_sold", "revenue"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    return df


# ── World Bank ────────────────────────────────────────────────────────────────

def fetch_worldbank(indicator: str, country: str = _WB_COUNTRY) -> pd.DataFrame:
    """
    Fetch annual time-series from World Bank Open Data API.
    Source: https://data.worldbank.org/ (CC BY 4.0)
    """
    url = f"{_WB_BASE}/country/{country}/indicator/{indicator}"
    try:
        resp = requests.get(url, params={"format": "json", "per_page": 100, "mrv": 10}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2 or not data[1]:
            return pd.DataFrame(columns=["year", "value", "indicator"])
        records = [
            {"year": int(r["date"]), "value": r["value"], "indicator": indicator}
            for r in data[1] if r["value"] is not None
        ]
        return pd.DataFrame(records).sort_values("year")
    except Exception as e:
        logger.warning("World Bank fetch failed (%s): %s", indicator, e)
        return pd.DataFrame(columns=["year", "value", "indicator"])


def _enrich_worldbank(df: pd.DataFrame) -> pd.DataFrame:
    gdp = fetch_worldbank(_WB_GDP).rename(columns={"value": "india_gdp_growth_pct"})
    cpi = fetch_worldbank(_WB_CPI).rename(columns={"value": "india_cpi_inflation_pct"})
    macro = gdp[["year", "india_gdp_growth_pct"]].merge(
        cpi[["year", "india_cpi_inflation_pct"]], on="year", how="outer"
    )
    if macro.empty:
        df["india_gdp_growth_pct"]    = np.nan
        df["india_cpi_inflation_pct"] = np.nan
        return df
    return df.merge(macro, on="year", how="left")


# ── FX rate ───────────────────────────────────────────────────────────────────

def fetch_usd_inr() -> float:
    """
    Live USD/INR rate. Tries 3 free sources, falls back to hardcoded value.
    Sources: fawazahmed0/exchange-api (CC0), frankfurter.dev, open.er-api.com
    """
    for source in _FX_SOURCES:
        try:
            resp = requests.get(source["url"], timeout=8)
            resp.raise_for_status()
            rate = source["parse"](resp.json())
            if rate and float(rate) > 50:
                logger.info("USD/INR %.4f from %s", rate, source["name"])
                return float(rate)
        except Exception as e:
            logger.warning("FX source %s failed: %s", source["name"], e)
    return _FX_FALLBACK


def _enrich_fx(df: pd.DataFrame) -> pd.DataFrame:
    rate = fetch_usd_inr()
    df["usd_inr_rate"]    = rate
    df["revenue_usd"]     = (df["revenue"] / rate).round(2)
    df["final_price_usd"] = (df["final_price"] / rate).round(4)
    return df


# ── Google Trends ─────────────────────────────────────────────────────────────

def fetch_google_trends(
    keywords: list[str] = _TREND_KEYWORDS,
    timeframe: str = "today 12-m",
    geo: str = "IN",
) -> pd.DataFrame:
    """
    Fetch search interest via pytrends (Apache 2.0).
    Source: https://github.com/GeneralMills/pytrends
    """
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.warning("pytrends not installed")
        return pd.DataFrame()
    try:
        pt = TrendReq(hl="en-IN", tz=330, timeout=(10, 25))
        frames = []
        for batch in [keywords[i:i+5] for i in range(0, len(keywords), 5)]:
            pt.build_payload(batch, timeframe=timeframe, geo=geo)
            frames.append(pt.interest_over_time())
            time.sleep(1.5)
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        if "isPartial" in result.columns:
            result = result.drop(columns=["isPartial"])
        return result
    except Exception as e:
        logger.warning("Google Trends failed: %s", e)
        return pd.DataFrame()


def _enrich_trends(df: pd.DataFrame) -> pd.DataFrame:
    trends = fetch_google_trends()
    if trends.empty:
        df["ecommerce_search_interest"] = np.nan
        return df
    trends["ecommerce_search_interest"] = trends.mean(axis=1)
    trends.index = pd.to_datetime(trends.index)
    trends["year_month"] = trends.index.to_period("M").astype(str)
    return df.merge(
        trends[["year_month", "ecommerce_search_interest"]].reset_index(drop=True),
        on="year_month", how="left",
    )
