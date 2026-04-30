"""
data/loader.py
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Centralised data loading, cleaning, feature engineering, and live data
enrichment for the Indian E-Commerce Analytics project.

Live data sources (all free / no-key-required unless noted):
  1. World Bank Open Data API  â€“ India GDP growth & CPI inflation
     https://data.worldbank.org/  (CC BY 4.0)
  2. exchangerate.host          â€“ live USD â†’ INR FX rate
     https://exchangerate.host/  (free tier, no key)
  3. pytrends (Google Trends)   â€“ search-interest proxy for e-commerce
     https://pypi.org/project/pytrends/  (unofficial Google Trends API)
  4. NewsAPI.org                â€“ Indian e-commerce headlines (key required,
     free tier: 100 req/day)    https://newsapi.org/

Citations
â”€â”€â”€â”€â”€â”€â”€â”€â”€
â€¢ World Bank (2024). *World Development Indicators â€“ India*.
  Retrieved from https://data.worldbank.org/country/india
  License: Creative Commons Attribution 4.0 (CC BY 4.0)

â€¢ exchangerate.host (2024). *Open Exchange Rates API*.
  Retrieved from https://exchangerate.host

â€¢ Google Trends via pytrends (GeneralMills, 2023).
  https://github.com/GeneralMills/pytrends  License: Apache 2.0

â€¢ NewsAPI.org (2024). *News API â€“ Everything endpoint*.
  https://newsapi.org/docs/endpoints/everything
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# â”€â”€ dataset path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DEFAULT_PATH = (
    "/kaggle/input/indian-e-commerce-pricing-revenue-growth/"
    "indian_ecommerce_pricing_revenue_growth_36_months.csv"
)

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/shukla922/"
    "indian-e-commerce-pricing-revenue-growth"
)

# â”€â”€ World Bank indicator codes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_WB_BASE   = "https://api.worldbank.org/v2"
_WB_GDP    = "NY.GDP.MKTP.KD.ZG"   # GDP growth (annual %)
_WB_CPI    = "FP.CPI.TOTL.ZG"      # Inflation, consumer prices (annual %)
_WB_COUNTRY = "IN"

# â”€â”€ FX endpoints (tried in order, first success wins) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. fawazahmed0/exchange-api â€“ GitHub CDN, no key, CC0 license
#    https://github.com/fawazahmed0/exchange-api
# 2. Frankfurter â€“ ECB data, no key
#    https://frankfurter.dev/
# 3. exchangerate-api.com open access â€“ no key
#    https://www.exchangerate-api.com/docs/free-exchange-rate-api
_FX_SOURCES = [
    {
        "url": "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json",
        "parse": lambda d: d.get("usd", {}).get("inr"),
        "name": "fawazahmed0/exchange-api (CC0)",
    },
    {
        "url": "https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR",
        "parse": lambda d: d.get("rates", {}).get("INR"),
        "name": "frankfurter.dev (ECB data)",
    },
    {
        "url": "https://open.er-api.com/v6/latest/USD",
        "parse": lambda d: d.get("rates", {}).get("INR"),
        "name": "exchangerate-api.com (open access)",
    },
]
_FX_FALLBACK = 84.0   # updated periodically as a last resort

# â”€â”€ Google Trends keywords â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_TREND_KEYWORDS = [
    "online shopping India",
    "Flipkart sale",
    "Amazon India deals",
    "fashion ecommerce India",
    "electronics discount India",
]

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  PUBLIC API
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load(
    path: str = DEFAULT_PATH,
    enrich_live: bool = True,
) -> pd.DataFrame:
    """
    Load the Kaggle CSV, clean it, engineer features, and optionally enrich
    with live macro / trend signals.

    Parameters
    ----------
    path         : path to the CSV file
    enrich_live  : if True, fetch live World Bank, FX, and Google Trends data
                   and merge them into the DataFrame
    news_api_key : optional NewsAPI key; if supplied, top headlines are fetched
                   and a sentiment stub column is added

    Returns
    -------
    pd.DataFrame  â€“ cleaned, feature-rich DataFrame
    """
    df = pd.read_csv(path)
    df = _clean(df)
    df = _engineer(df)

    if enrich_live:
        df = _enrich_worldbank(df)
        df = _enrich_fx(df)
        df = _enrich_trends(df)

    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  CLEANING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _clean(df: pd.DataFrame) -> pd.DataFrame:  # also importable directly
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["discount_percent"] = df["discount_percent"].clip(0, 100)
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  FEATURE ENGINEERING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _engineer(df: pd.DataFrame) -> pd.DataFrame:  # also importable directly
    df = df.copy()

    # â”€â”€ time â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df["year"]           = df["order_date"].dt.year
    df["month"]          = df["order_date"].dt.month
    df["month_name"]     = df["order_date"].dt.month_name()
    df["weekday"]        = df["order_date"].dt.weekday
    df["is_weekend"]     = df["weekday"] >= 5
    df["fiscal_quarter"] = ((df["month"] - 4) % 12 // 3 + 1).astype("Int8")
    df["year_month"]     = df["order_date"].dt.to_period("M").astype(str)

    # â”€â”€ pricing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    df["discount_amount"]  = df["base_price"] - df["final_price"]
    df["discount_ratio"]   = df["discount_amount"] / df["base_price"].replace(0, np.nan)
    df["revenue_per_unit"] = df["final_price"]

    # â”€â”€ log transforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    for col in ["base_price", "final_price", "units_sold", "revenue"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  LIVE DATA â€“ WORLD BANK
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_worldbank(indicator: str, country: str = _WB_COUNTRY) -> pd.DataFrame:
    """
    Fetch annual time-series data from the World Bank Open Data API.

    Source  : World Bank (2024). World Development Indicators.
              https://data.worldbank.org/  (CC BY 4.0)
    Endpoint: https://api.worldbank.org/v2/country/{country}/indicator/{indicator}

    Returns
    -------
    pd.DataFrame with columns [year, value, indicator]
    """
    url = f"{_WB_BASE}/country/{country}/indicator/{indicator}"
    params = {"format": "json", "per_page": 100, "mrv": 10}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if len(data) < 2 or not data[1]:
            logger.warning("World Bank returned empty data for %s", indicator)
            return pd.DataFrame(columns=["year", "value", "indicator"])
        records = [
            {"year": int(r["date"]), "value": r["value"], "indicator": indicator}
            for r in data[1]
            if r["value"] is not None
        ]
        return pd.DataFrame(records).sort_values("year")
    except Exception as exc:
        logger.warning("World Bank fetch failed (%s): %s", indicator, exc)
        return pd.DataFrame(columns=["year", "value", "indicator"])


def _enrich_worldbank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge India GDP growth and CPI inflation (annual) into the order-level
    DataFrame on the `year` column.

    Citation: World Bank Open Data â€“ https://data.worldbank.org/ (CC BY 4.0)
    """
    gdp = fetch_worldbank(_WB_GDP).rename(columns={"value": "india_gdp_growth_pct"})
    cpi = fetch_worldbank(_WB_CPI).rename(columns={"value": "india_cpi_inflation_pct"})

    macro = gdp[["year", "india_gdp_growth_pct"]].merge(
        cpi[["year", "india_cpi_inflation_pct"]], on="year", how="outer"
    )

    if macro.empty:
        logger.warning("Macro enrichment skipped â€“ World Bank data unavailable.")
        df["india_gdp_growth_pct"]   = np.nan
        df["india_cpi_inflation_pct"] = np.nan
        return df

    df = df.merge(macro, on="year", how="left")
    logger.info("World Bank macro data merged (GDP + CPI) for years: %s",
                sorted(macro["year"].tolist()))
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  LIVE DATA â€“ FX RATE (USD â†’ INR)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_usd_inr() -> float:
    """
    Fetch the live USD â†’ INR exchange rate.
    Tries three free, no-key sources in order â€” first valid rate wins.

    Sources (in priority order)
    â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    1. fawazahmed0/exchange-api (GitHub CDN) â€“ CC0 license
       https://github.com/fawazahmed0/exchange-api
    2. Frankfurter (ECB data, no key)
       https://frankfurter.dev/
    3. exchangerate-api.com open access
       https://www.exchangerate-api.com/docs/free-exchange-rate-api
    4. Hardcoded fallback: {_FX_FALLBACK}
    """
    for source in _FX_SOURCES:
        try:
            resp = requests.get(source["url"], timeout=8)
            resp.raise_for_status()
            rate = source["parse"](resp.json())
            if rate and float(rate) > 50:   # sanity check
                logger.info("USD/INR %.4f from %s", rate, source["name"])
                return float(rate)
        except Exception as exc:
            logger.warning("FX source %s failed: %s", source["name"], exc)
    logger.warning("All FX sources failed â€“ using fallback %.2f", _FX_FALLBACK)
    return _FX_FALLBACK


def _enrich_fx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add usd_inr_rate, revenue_usd, final_price_usd columns.
    Rate sourced from fawazahmed0/exchange-api â†’ frankfurter â†’ exchangerate-api.com
    """
    rate = fetch_usd_inr()
    df["usd_inr_rate"]  = rate
    df["revenue_usd"]   = (df["revenue"] / rate).round(2)
    df["final_price_usd"] = (df["final_price"] / rate).round(4)
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  LIVE DATA â€“ GOOGLE TRENDS (pytrends)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def fetch_google_trends(
    keywords: list[str] = _TREND_KEYWORDS,
    timeframe: str = "today 12-m",
    geo: str = "IN",
) -> pd.DataFrame:
    """
    Fetch monthly Google Trends interest-over-time for Indian e-commerce
    keywords using pytrends (unofficial Google Trends API).

    Source  : Google Trends via pytrends (GeneralMills, 2023)
              https://github.com/GeneralMills/pytrends  (Apache 2.0)
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError:
        logger.warning("pytrends not installed â€“ Google Trends unavailable.")
        return pd.DataFrame()

    try:
        pt = TrendReq(hl="en-IN", tz=330, timeout=(10, 25))
        # Google allows max 5 keywords per batch
        batches = [keywords[i:i+5] for i in range(0, len(keywords), 5)]
        frames = []
        for batch in batches:
            pt.build_payload(batch, timeframe=timeframe, geo=geo)
            frames.append(pt.interest_over_time())
            time.sleep(1.5)   # be polite to Google's servers
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        if "isPartial" in result.columns:
            result = result.drop(columns=["isPartial"])
        logger.info("Google Trends fetched for %d keywords", len(result.columns))
        return result
    except Exception as exc:
        logger.warning("Google Trends fetch failed: %s", exc)
        return pd.DataFrame()


def _enrich_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a composite 'ecommerce_search_interest' score (mean of all keyword
    trends) into the DataFrame on year-month.

    Citation: Google Trends via pytrends â€“ https://github.com/GeneralMills/pytrends
    """
    trends = fetch_google_trends()
    if trends.empty:
        df["ecommerce_search_interest"] = np.nan
        return df

    trends["ecommerce_search_interest"] = trends.mean(axis=1)
    trends = trends[["ecommerce_search_interest"]].copy()
    trends.index = pd.to_datetime(trends.index)
    trends["year_month"] = trends.index.to_period("M").astype(str)
    trends = trends.reset_index(drop=True)

    df = df.merge(
        trends[["year_month", "ecommerce_search_interest"]],
        on="year_month",
        how="left",
    )
    logger.info("Google Trends search interest merged.")
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#  LIVE DATA â€“ NEWS HEADLINES (NewsAPI)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


    news["year_month"] = news["publishedAt"].dt.to_period("M").astype(str)
    monthly_count = (
        news.groupby("year_month").size().reset_index(name="news_article_count")
    )
    df = df.merge(monthly_count, on="year_month", how="left")
    df["news_article_count"] = df["news_article_count"].fillna(0).astype(int)
    logger.info("NewsAPI article counts merged.")
    return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
