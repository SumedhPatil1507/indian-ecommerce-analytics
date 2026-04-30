"""
api/main.py

FastAPI wrapper  exposes key analytics as REST endpoints.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints

GET  /health                   liveness check
GET  /live/macro               World Bank GDP + CPI (live)
GET  /live/fx                  USD/INR rate (live)
GET  /live/trends              Google Trends interest (live)
GET  /analytics/revenue        monthly revenue summary
GET  /analytics/elasticity     price elasticity by category
GET  /analytics/alerts         inventory alert table
GET  /analytics/clv            CLV tier summary
POST /predict/revenue          predict revenue for a single order
"""
import os
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

#  app 
app = FastAPI(
    title="Indian E-Commerce Analytics API",
    description=(
        "Production-grade analytics API for Indian e-commerce pricing, "
        "revenue, and demand intelligence.\n\n"
        "**Live data sources:**\n"
        "- [World Bank Open Data](https://data.worldbank.org/) (CC BY 4.0)\n"
        "- [exchangerate.host](https://exchangerate.host) (free tier)\n"
        "- [Google Trends via pytrends](https://github.com/GeneralMills/pytrends) (Apache 2.0)\n"
    ),
    version="1.0.0",
    contact={"name": "Analytics Team"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

#  lazy data loading 

DATA_PATH = os.getenv(
    "DATA_PATH",
    "/kaggle/input/indian-e-commerce-pricing-revenue-growth/"
    "indian_ecommerce_pricing_revenue_growth_36_months.csv",
)


@lru_cache(maxsize=1)
def _get_df() -> pd.DataFrame:
    from data.loader import load
    return load(DATA_PATH, enrich_live=False)   # skip live enrichment for speed


@lru_cache(maxsize=1)
def _get_model():
    from modules.models import train_all
    return train_all(_get_df())


#  health 

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "rows": len(_get_df())}


#  live data endpoints 

@app.get("/live/macro", tags=["Live Data"],
         summary="India GDP growth & CPI inflation (World Bank)")
def live_macro():
    """
    Returns the latest available India GDP growth and CPI inflation from the
    World Bank Open Data API.

    **Source:** World Bank (2024). World Development Indicators.
    https://data.worldbank.org/  License: CC BY 4.0
    """
    from data.loader import fetch_worldbank, _WB_GDP, _WB_CPI
    gdp = fetch_worldbank(_WB_GDP)
    cpi = fetch_worldbank(_WB_CPI)
    return {
        "gdp_growth": gdp.tail(3).to_dict(orient="records"),
        "cpi_inflation": cpi.tail(3).to_dict(orient="records"),
        "source": "World Bank Open Data  https://data.worldbank.org/",
        "license": "CC BY 4.0",
    }


@app.get("/live/fx", tags=["Live Data"],
         summary="Live USD  INR exchange rate")
def live_fx():
    """
    Returns the current USD  INR exchange rate.

    **Source:** exchangerate.host  https://exchangerate.host (free tier)
    """
    from data.loader import fetch_usd_inr
    rate = fetch_usd_inr()
    return {
        "usd_inr": rate,
        "source": "exchangerate.host  https://exchangerate.host",
    }


@app.get("/live/trends", tags=["Live Data"],
         summary="Google Trends search interest for Indian e-commerce")
def live_trends(
    timeframe: str = Query("today 3-m", description="pytrends timeframe string"),
):
    """
    Returns Google Trends relative search interest (0100) for key Indian
    e-commerce keywords.

    **Source:** Google Trends via pytrends (GeneralMills, 2023).
    https://github.com/GeneralMills/pytrends  License: Apache 2.0
    """
    from data.loader import fetch_google_trends
    trends = fetch_google_trends(timeframe=timeframe)
    if trends.empty:
        raise HTTPException(503, "Google Trends unavailable  pytrends may be rate-limited.")
    latest = trends.tail(1).reset_index().to_dict(orient="records")
    return {
        "latest": latest,
        "source": "Google Trends via pytrends  https://github.com/GeneralMills/pytrends",
        "license": "Apache 2.0",
    }


#  analytics endpoints 

@app.get("/analytics/revenue", tags=["Analytics"],
         summary="Monthly revenue summary")
def analytics_revenue():
    df = _get_df()
    monthly = (
        df.groupby("year_month")["revenue"]
        .agg(total="sum", mean="mean", orders="count")
        .reset_index()
    )
    return monthly.to_dict(orient="records")


@app.get("/analytics/elasticity", tags=["Analytics"],
         summary="Price elasticity by category")
def analytics_elasticity():
    from modules.price_elasticity import compute_elasticity
    df    = _get_df()
    elast = compute_elasticity(df, group_cols=["category"])
    return elast.to_dict(orient="records")


@app.get("/analytics/alerts", tags=["Analytics"],
         summary="Inventory alert table")
def analytics_alerts():
    from modules.inventory_alerts import compute_alerts
    df     = _get_df()
    alerts = compute_alerts(df)
    return alerts.to_dict(orient="records")


@app.get("/analytics/clv", tags=["Analytics"],
         summary="CLV tier summary")
def analytics_clv():
    from modules.clv import compute_clv
    df  = _get_df()
    clv = compute_clv(df)
    summary = (
        clv.groupby("clv_tier")["clv"]
        .agg(count="count", mean_clv="mean", total_clv="sum")
        .reset_index()
    )
    return summary.to_dict(orient="records")


#  prediction endpoint 

class OrderInput(BaseModel):
    state:                str   = Field(..., example="Maharashtra")
    zone:                 str   = Field(..., example="West")
    category:             str   = Field(..., example="Electronics")
    brand_type:           str   = Field(..., example="Premium")
    customer_gender:      str   = Field(..., example="Male")
    customer_age:         int   = Field(..., example=28)
    base_price:           float = Field(..., example=15000.0)
    discount_percent:     float = Field(..., example=20.0)
    sales_event:          str   = Field(..., example="Normal")
    competition_intensity:str   = Field(..., example="High")
    inventory_pressure:   str   = Field(..., example="Low")
    year:                 int   = Field(..., example=2024)
    month:                int   = Field(..., example=10)
    weekday:              int   = Field(..., example=2)


@app.post("/predict/revenue", tags=["Prediction"],
          summary="Predict revenue for a single order")
def predict_revenue(order: OrderInput):
    """
    Predict the expected revenue for a given order using the best trained model
    (XGBoost by default).
    """
    output = _get_model()
    pipe   = output["pipes"].get("XGBoost")
    if pipe is None:
        raise HTTPException(500, "Model not available.")

    row = pd.DataFrame([order.model_dump()])
    pred = float(pipe.predict(row)[0])
    return {
        "predicted_revenue_inr": round(pred, 2),
        "model": "XGBoost",
        "note": "Prediction is based on historical training data.",
    }
