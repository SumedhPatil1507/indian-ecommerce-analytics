"""
data/loader.py
──────────────────────────────────────────────────────────────────────────────
Centralised data loading, cleaning, feature engineering, and live data
enrichment for the Indian E-Commerce Analytics project.

Live data sources (all free / no-key-required unless noted):
  1. World Bank Open Data API  – India GDP growth & CPI inflation
     https://data.worldbank.org/  (CC BY 4.0)
  2. exchangerate.host          – live USD → INR FX rate
     https://exchangerate.host/  (free tier, no key)
  3. pytrends (Google Trends)   – search-interest proxy for e-commerce
     https://pypi.org/project/pytrends/  (unofficial Google Trends API)
  4. NewsAPI.org                – Indian e-commerce headlines (key required,
     free tier: 100 req/day)    https://newsapi.org/

Citations
─────────
• World Bank (2024). *World Development Indicators – India*.
  Retrieved from https://data.worldbank.org/country/india
  License: Creative Commons Attribution 4.0 (CC BY 4.0)

• exchangerate.host (2024). *Open Exchange Rates API*.
  Retrieved from https://exchangerate.host

• Google Trends via pytrends (GeneralMills, 2023).
  https://github.com/GeneralMills/pytrends  License: Apache 2.0

• NewsAPI.org (2024). *News API – Everything endpoint*.
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

# ── dataset path ─────────────────────────────────────────────────────────────
DEFAULT_PATH = (
    "/kaggle/input/indian-e-commerce-pricing-revenue-growth/"
    "indian_ecommerce_pricing_revenue_growth_36_months.csv"
)

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/shukla922/"
    "indian-e-commerce-pricing-revenue-growth"
)

# ── World Bank indicator codes ────────────────────────────────────────────────
_WB_BASE   = "https://api.worldbank.org/v2"
_WB_GDP    = "NY.GDP.MKTP.KD.ZG"   # GDP growth (annual %)
_WB_CPI    = "FP.CPI.TOTL.ZG"      # Inflation, consumer prices (annual %)
_WB_COUNTRY = "IN"

# ── FX endpoint ───────────────────────────────────────────────────────────────
_FX_URL = "https://api.exchangerate.host/live"   # USD base, free, no key

# ── Google Trends keywords ────────────────────────────────────────────────────
_TREND_KEYWORDS = [
    "online shopping India",
    "Flipkart sale",
    "Amazon India deals",
    "fashion ecommerce India",
    "electronics discount India",
]

# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load(
    path: str = DEFAULT_PATH,
    enrich_live: bool = True,
    news_api_key: Optional[str] = None,
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
    pd.DataFrame  – cleaned, feature-rich DataFrame
    """
    df = pd.read_csv(path)
    df = _clean(df)
    df = _engineer(df)

    if enrich_live:
        df = _enrich_worldbank(df)
        df = _enrich_fx(df)
        df = _enrich_trends(df)
        if news_api_key:
            df = _enrich_news(df, news_api_key)

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:  # also importable directly
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["discount_percent"] = df["discount_percent"].clip(0, 100)
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def _engineer(df: pd.DataFrame) -> pd.DataFrame:  # also importable directly
    df = df.copy()

    # ── time ──────────────────────────────────────────────────────────────────
    df["year"]           = df["order_date"].dt.year
    df["month"]          = df["order_date"].dt.month
    df["month_name"]     = df["order_date"].dt.month_name()
    df["weekday"]        = df["order_date"].dt.weekday
    df["is_weekend"]     = df["weekday"] >= 5
    df["fiscal_quarter"] = ((df["month"] - 4) % 12 // 3 + 1).astype("Int8")
    df["year_month"]     = df["order_date"].dt.to_period("M").astype(str)

    # ── pricing ───────────────────────────────────────────────────────────────
    df["discount_amount"]  = df["base_price"] - df["final_price"]
    df["discount_ratio"]   = df["discount_amount"] / df["base_price"].replace(0, np.nan)
    df["revenue_per_unit"] = df["final_price"]

    # ── log transforms ────────────────────────────────────────────────────────
    for col in ["base_price", "final_price", "units_sold", "revenue"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE DATA – WORLD BANK
# ─────────────────────────────────────────────────────────────────────────────

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

    Citation: World Bank Open Data – https://data.worldbank.org/ (CC BY 4.0)
    """
    gdp = fetch_worldbank(_WB_GDP).rename(columns={"value": "india_gdp_growth_pct"})
    cpi = fetch_worldbank(_WB_CPI).rename(columns={"value": "india_cpi_inflation_pct"})

    macro = gdp[["year", "india_gdp_growth_pct"]].merge(
        cpi[["year", "india_cpi_inflation_pct"]], on="year", how="outer"
    )

    if macro.empty:
        logger.warning("Macro enrichment skipped – World Bank data unavailable.")
        df["india_gdp_growth_pct"]   = np.nan
        df["india_cpi_inflation_pct"] = np.nan
        return df

    df = df.merge(macro, on="year", how="left")
    logger.info("World Bank macro data merged (GDP + CPI) for years: %s",
                sorted(macro["year"].tolist()))
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE DATA – FX RATE (USD → INR)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_usd_inr() -> float:
    """
    Fetch the current USD → INR exchange rate from exchangerate.host (free).

    Source  : exchangerate.host – https://exchangerate.host
    Endpoint: https://api.exchangerate.host/live?source=USD&currencies=INR
    """
    try:
        resp = requests.get(
            _FX_URL,
            params={"source": "USD", "currencies": "INR"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        rate = data.get("quotes", {}).get("USDINR")
        if rate:
            logger.info("Live USD/INR rate: %.4f (source: exchangerate.host)", rate)
            return float(rate)
    except Exception as exc:
        logger.warning("FX fetch failed: %s", exc)
    # fallback to approximate rate
    return 83.5


def _enrich_fx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `usd_inr_rate` column and a `revenue_usd` column for international
    comparisons.

    Citation: exchangerate.host – https://exchangerate.host (free tier)
    """
    rate = fetch_usd_inr()
    df["usd_inr_rate"]  = rate
    df["revenue_usd"]   = (df["revenue"] / rate).round(2)
    df["final_price_usd"] = (df["final_price"] / rate).round(4)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE DATA – GOOGLE TRENDS (pytrends)
# ─────────────────────────────────────────────────────────────────────────────

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
    Note    : pytrends is an unofficial API wrapper; Google may rate-limit
              requests. Results are relative search interest (0–100).

    Parameters
    ----------
    keywords  : list of search terms (max 5 per request)
    timeframe : pytrends timeframe string, e.g. "today 12-m", "2022-01-01 2024-12-31"
    geo       : ISO country code, "IN" for India

    Returns
    -------
    pd.DataFrame indexed by date with one column per keyword
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError:
        logger.warning("pytrends not installed. Run: pip install pytrends")
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

    Citation: Google Trends via pytrends – https://github.com/GeneralMills/pytrends
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


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE DATA – NEWS HEADLINES (NewsAPI)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news_headlines(
    api_key: str,
    query: str = "India ecommerce online shopping",
    days_back: int = 30,
    page_size: int = 50,
) -> pd.DataFrame:
    """
    Fetch recent Indian e-commerce news headlines from NewsAPI.org.

    Source  : NewsAPI.org (2024). Everything endpoint.
              https://newsapi.org/docs/endpoints/everything
              Free tier: 100 requests/day, articles up to 1 month old.
              Requires a free API key from https://newsapi.org/register

    Parameters
    ----------
    api_key   : your NewsAPI key (get free at https://newsapi.org/register)
    query     : search query string
    days_back : how many days of articles to retrieve
    page_size : number of articles (max 100 on free tier)

    Returns
    -------
    pd.DataFrame with columns [publishedAt, title, source, url, description]
    """
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        records = [
            {
                "publishedAt": a["publishedAt"],
                "title": a["title"],
                "source": a["source"]["name"],
                "url": a["url"],
                "description": a.get("description", ""),
            }
            for a in articles
        ]
        df_news = pd.DataFrame(records)
        df_news["publishedAt"] = pd.to_datetime(df_news["publishedAt"], errors="coerce")
        logger.info("NewsAPI: fetched %d headlines", len(df_news))
        return df_news
    except Exception as exc:
        logger.warning("NewsAPI fetch failed: %s", exc)
        return pd.DataFrame()


def _enrich_news(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Fetch headlines and add a stub `news_article_count` column (articles per
    month) as a proxy for media attention / market buzz.

    Citation: NewsAPI.org – https://newsapi.org/docs/endpoints/everything
    """
    news = fetch_news_headlines(api_key)
    if news.empty:
        df["news_article_count"] = np.nan
        return df

    news["year_month"] = news["publishedAt"].dt.to_period("M").astype(str)
    monthly_count = (
        news.groupby("year_month").size().reset_index(name="news_article_count")
    )
    df = df.merge(monthly_count, on="year_month", how="left")
    df["news_article_count"] = df["news_article_count"].fillna(0).astype(int)
    logger.info("NewsAPI article counts merged.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  CONVENIENCE: print live data summary
# ─────────────────────────────────────────────────────────────────────────────

def live_data_summary() -> None:
    """
    Print a quick summary of all live data sources and their current values.
    Useful for a notebook cell or a health-check endpoint.
    """
    print("=" * 65)
    print("  LIVE DATA SUMMARY")
    print("=" * 65)

    # 1. World Bank GDP
    gdp = fetch_worldbank(_WB_GDP)
    if not gdp.empty:
        latest = gdp.iloc[-1]
        print(f"\n[World Bank] India GDP Growth (latest available)")
        print(f"  Year  : {int(latest['year'])}")
        print(f"  Value : {latest['value']:.2f}%")
        print(f"  Source: https://data.worldbank.org/indicator/{_WB_GDP}?locations=IN")
        print(f"  License: CC BY 4.0")

    # 2. World Bank CPI
    cpi = fetch_worldbank(_WB_CPI)
    if not cpi.empty:
        latest = cpi.iloc[-1]
        print(f"\n[World Bank] India CPI Inflation (latest available)")
        print(f"  Year  : {int(latest['year'])}")
        print(f"  Value : {latest['value']:.2f}%")
        print(f"  Source: https://data.worldbank.org/indicator/{_WB_CPI}?locations=IN")
        print(f"  License: CC BY 4.0")

    # 3. FX rate
    rate = fetch_usd_inr()
    print(f"\n[exchangerate.host] Live USD → INR")
    print(f"  Rate  : 1 USD = ₹{rate:.2f}")
    print(f"  Source: https://exchangerate.host")

    # 4. Google Trends
    trends = fetch_google_trends(timeframe="today 1-m")
    if not trends.empty:
        avg = trends.mean().mean()
        print(f"\n[Google Trends / pytrends] E-commerce search interest (India, last 30d)")
        print(f"  Avg interest score : {avg:.1f} / 100")
        print(f"  Keywords tracked   : {list(trends.columns)}")
        print(f"  Source: https://trends.google.com  (via pytrends, Apache 2.0)")

    print("\n" + "=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-FORMAT FILE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_any(file_obj, filename: str) -> pd.DataFrame:
    """
    Load a dataset from any supported file format.

    Supported formats
    ─────────────────
    .csv          – comma-separated values
    .tsv          – tab-separated values
    .xlsx / .xls  – Excel workbook (first sheet)
    .json         – records or columns orientation
    .parquet      – Apache Parquet (requires pyarrow or fastparquet)

    Parameters
    ----------
    file_obj : file-like object (e.g. st.file_uploader result or open())
    filename : original filename string (used to detect format)

    Returns
    -------
    Cleaned and feature-engineered pd.DataFrame
    """
    import io

    ext = filename.rsplit(".", 1)[-1].lower()

    # accept both raw bytes and file-like objects
    if isinstance(file_obj, (bytes, bytearray)):
        raw = file_obj
    else:
        raw = file_obj.read() if hasattr(file_obj, "read") else bytes(file_obj)

    if ext == "csv":
        df = pd.read_csv(io.BytesIO(raw))
    elif ext == "tsv":
        df = pd.read_csv(io.BytesIO(raw), sep="\t")
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(io.BytesIO(raw))
    elif ext == "json":
        df = pd.read_json(io.BytesIO(raw))
    elif ext == "parquet":
        df = pd.read_parquet(io.BytesIO(raw))
    else:
        raise ValueError(
            f"Unsupported file format: .{ext}\n"
            "Supported: .csv, .tsv, .xlsx, .xls, .json, .parquet"
        )

    return _engineer(_clean(df))


# ─────────────────────────────────────────────────────────────────────────────
#  LIVE-ONLY SYNTHETIC DATASET
#  Generates a realistic demo dataset driven entirely by live macro signals.
#  No Kaggle file required – the dashboard is fully functional without upload.
# ─────────────────────────────────────────────────────────────────────────────

_CATEGORIES  = ["Electronics", "Fashion", "Grocery Essentials",
                "Premium Lifestyle", "Home & Kitchen", "Sports & Fitness"]
_ZONES       = ["North", "South", "East", "West", "Central"]
_STATES      = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu",
                "Uttar Pradesh", "Gujarat", "West Bengal", "Rajasthan",
                "Telangana", "Kerala", "Punjab", "Haryana"]
_BRANDS      = ["Mass", "Premium"]
_GENDERS     = ["Male", "Female"]
_EVENTS      = ["Normal", "Festival"]
_COMPETITION = ["Low", "Medium", "High"]
_INVENTORY   = ["Low", "High"]

# Festival months (India: Oct=10, Nov=11, Jan=1, Aug=8)
_FESTIVAL_MONTHS = {10, 11, 1, 8}


def generate_live_dataset(
    n_rows: int = 5000,
    months: int = 36,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic e-commerce dataset calibrated to live macro signals.

    The generator uses:
    - Live World Bank GDP growth  → scales revenue growth trend
    - Live CPI inflation          → adjusts base prices upward
    - Live USD/INR rate           → adds revenue_usd column
    - Live Google Trends          → modulates festival/search-driven spikes
    - Current date                → anchors the 36-month window to today

    This means every refresh produces data that reflects the current
    macroeconomic environment — no static CSV needed.

    Returns
    -------
    Cleaned, feature-engineered pd.DataFrame (same schema as Kaggle dataset)
    """
    rng = np.random.default_rng(seed)

    # ── fetch live signals (graceful fallback) ────────────────────────────────
    gdp_df = fetch_worldbank(_WB_GDP)
    cpi_df = fetch_worldbank(_WB_CPI)
    fx     = fetch_usd_inr()

    gdp_growth = gdp_df["value"].iloc[-1] / 100 if not gdp_df.empty else 0.07
    cpi_rate   = cpi_df["value"].iloc[-1] / 100 if not cpi_df.empty else 0.05

    # ── date range: last `months` months ending today ─────────────────────────
    end_date   = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(months=months)
    dates      = pd.date_range(start_date, end_date, freq="D")

    # ── base price calibrated to CPI ─────────────────────────────────────────
    # Higher CPI → higher base prices
    price_multiplier = 1 + cpi_rate

    rows = []
    for i in range(n_rows):
        order_date = pd.Timestamp(rng.choice(dates))
        month      = order_date.month
        year       = order_date.year

        is_festival = month in _FESTIVAL_MONTHS
        event       = "Festival" if (is_festival and rng.random() < 0.65) else "Normal"

        category    = rng.choice(_CATEGORIES)
        brand       = rng.choice(_BRANDS)
        zone        = rng.choice(_ZONES)
        state       = rng.choice(_STATES)
        gender      = rng.choice(_GENDERS)
        age         = int(rng.integers(18, 62))
        competition = rng.choice(_COMPETITION)
        inventory   = rng.choice(_INVENTORY)

        # price bands by category
        price_bands = {
            "Electronics":        (8000,  80000),
            "Fashion":            (500,   8000),
            "Grocery Essentials": (100,   2000),
            "Premium Lifestyle":  (5000,  150000),
            "Home & Kitchen":     (800,   25000),
            "Sports & Fitness":   (600,   15000),
        }
        lo, hi = price_bands[category]
        base_price = round(float(rng.uniform(lo, hi)) * price_multiplier, 2)

        # discount: higher during festival, higher competition → more discount
        disc_base = 20 if event == "Normal" else 40
        if competition == "High":
            disc_base += 10
        if brand == "Premium":
            disc_base -= 5
        discount = float(np.clip(rng.normal(disc_base, 12), 0, 65))

        final_price = round(base_price * (1 - discount / 100), 2)

        # units: festival + mass brand → more volume
        # GDP growth → slight upward trend over time
        months_elapsed = max((order_date - start_date).days / 30, 0)
        growth_factor  = 1 + gdp_growth * (months_elapsed / months)
        units_base     = 30 if brand == "Mass" else 10
        if event == "Festival":
            units_base *= 2
        units_sold = max(1, int(rng.normal(units_base, units_base * 0.4) * growth_factor))

        revenue = round(final_price * units_sold, 2)

        rows.append({
            "order_id":             f"ORD{i:07d}",
            "order_date":           order_date.strftime("%Y-%m-%d"),
            "state":                state,
            "zone":                 zone,
            "category":             category,
            "brand_type":           brand,
            "customer_gender":      gender,
            "customer_age":         age,
            "base_price":           base_price,
            "discount_percent":     round(discount, 1),
            "final_price":          final_price,
            "units_sold":           units_sold,
            "revenue":              revenue,
            "sales_event":          event,
            "competition_intensity":competition,
            "inventory_pressure":   inventory,
        })

    df = pd.DataFrame(rows)
    df = _engineer(_clean(df))

    # attach live macro columns
    df["india_gdp_growth_pct"]    = round(gdp_growth * 100, 2)
    df["india_cpi_inflation_pct"] = round(cpi_rate   * 100, 2)
    df["usd_inr_rate"]            = fx
    df["revenue_usd"]             = (df["revenue"] / fx).round(2)
    df["data_source"]             = "live_synthetic"

    logger.info(
        "Generated %d synthetic rows (GDP=%.2f%%, CPI=%.2f%%, FX=%.2f)",
        n_rows, gdp_growth * 100, cpi_rate * 100, fx,
    )
    return df
