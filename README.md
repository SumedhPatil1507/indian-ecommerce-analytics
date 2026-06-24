# IndiaCommerce Analytics v4.0

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/)

**Live Dashboard:** https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/
**GitHub:** https://github.com/SumedhPatil1507/indian-ecommerce-analytics
**Dataset:** https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth

Production-grade e-commerce analytics platform. Runs entirely on Streamlit Cloud — no external servers or login required. Upload your data and get instant insights powered by live macro signals.

> **Architecture:** Pure Streamlit + in-process Python analytics. No FastAPI server or Celery workers needed on Streamlit Cloud. Supabase is fully optional (enables result caching and operational logging when configured).

---

## What's Inside

### Data Connector Matrix (Tab 0)
Connect any e-commerce platform — all connectors normalise to the same internal schema:

| Connector | Source | Function |
|---|---|---|
| Shopify | order/create webhook | `from_shopify_webhook(payload)` |
| Amazon Seller Central | SP-API Orders v0 | `from_amazon_orders(payload)` |
| WooCommerce | REST API / DB dump | `from_woocommerce(payload)` |
| Generic File | CSV, TSV, Excel, JSON, Parquet | `load_any(file, filename)` |
| Simulation Sandbox | Live macro-calibrated synthetic data | `generate_simulation(...)` |

### 14 Analytics Tabs

| Tab | What it does |
|---|---|
| Data Connector Matrix | Connect Shopify/Amazon/WooCommerce, validate schema, run Simulation Sandbox |
| Executive Summary | Auto-written narrative, KPIs, risks, opportunities + PDF/Excel export |
| Price Optimizer | Lerner-index optimal discount per category, approve with Supabase logging |
| At-Risk Customers | RFM churn scoring, export cohort CSV for Klaviyo/SendGrid |
| Model Drift | PSI feature drift + R2 prediction degradation monitoring |
| Revenue Trends | Monthly revenue, AOV, discount trend, zone + brand breakdown |
| Categories | Revenue mix, festival vs normal, metric selector |
| Regional | Top 15 states, zone pie, units by zone |
| Inventory | Alert system with scatter dashboard + filterable table |
| CLV | BG/NBD CLV tiers, distribution, frequency scatter (Supabase cached) |
| Anomalies | Isolation Forest + DBSCAN + Z-score (Supabase cached, 7-day TTL) |
| Cohort | Retention rate + revenue retention heatmaps |
| Pareto | 80/20 chart, sunburst, Lorenz curve + Gini coefficient |
| Operational Actions | Approve price changes, export at-risk cohort, view Supabase action log |

### Live Data Sources

| Source | Data | License |
|---|---|---|
| [World Bank Open Data](https://data.worldbank.org/) | India GDP growth + CPI inflation | CC BY 4.0 |
| [fawazahmed0/exchange-api](https://github.com/fawazahmed0/exchange-api) | Live USD/INR rate (3-source waterfall) | CC0 |
| [Google Trends via pytrends](https://github.com/GeneralMills/pytrends) | E-commerce search interest India | Apache 2.0 |

### Supabase Operational Persistence (optional)

When `SUPABASE_URL` + `SUPABASE_ANON_KEY` are configured, the platform persists:

| Table | Purpose | TTL |
|---|---|---|
| `operational_actions` | Approved price changes, at-risk exports, drift alerts | Permanent |
| `clv_cache` | CLV tier computation results | 24 hours |
| `anomaly_cache` | Weekly anomaly scores | 7 days |
| `model_results` | Heavy model outputs (Prophet, SARIMA) | 24 hours |

---

## Quick Start

```bash
git clone https://github.com/SumedhPatil1507/indian-ecommerce-analytics
cd ecommerce-analytics
bash setup_env.sh
source .venv/bin/activate

# Run dashboard (opens at http://localhost:8501)
streamlit run dashboard/app.py

# Run API (optional, docs at http://localhost:8000/docs)
uvicorn api.main:app --reload --port 8000
```

## Supabase Setup (optional)

1. Create project at [supabase.com](https://supabase.com) — name: `indiacommerce-analytics`, region: `ap-south-1`
2. Run `supabase_schema.sql` in Supabase SQL Editor
3. Create Storage bucket named `datasets` (set to private)
4. Add to Streamlit Cloud **App Settings → Secrets**:

```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_ANON_KEY = "eyJ..."
SUPABASE_SERVICE_KEY = "eyJ..."
```

## GitHub Update Commands

```bash
cd C:\Users\Sumedh\projects\Indian-ecommerce-project\ecommerce-analytics
git add .
git commit -m "your message"
git push
```

## Project Structure

```
ecommerce-analytics/
├── data/
│   ├── loader.py          # Multi-format loader + live macro enrichment
│   └── connectors.py      # Shopify, Amazon, WooCommerce, Simulation Sandbox
├── modules/
│   ├── insights.py        # Executive summary + recommendations engine
│   ├── price_optimizer.py # Lerner-index dynamic pricing
│   ├── at_risk.py         # RFM churn risk scoring
│   ├── model_drift.py     # PSI + prediction drift monitoring
│   ├── clv.py             # BG/NBD + Gamma-Gamma CLV
│   ├── anomaly.py         # Isolation Forest + DBSCAN + Z-score
│   ├── cohort.py          # Cohort retention heatmaps
│   ├── inventory_alerts.py# Velocity-based inventory alerts
│   └── export.py          # PDF + Excel export
├── core/
│   ├── config.py          # App + Supabase configuration
│   └── database.py        # Supabase persistence + model result caching
├── dashboard/
│   └── app.py             # Streamlit dashboard (14 tabs, no login required)
├── api/
│   └── main.py            # FastAPI REST endpoints (optional)
├── supabase_schema.sql    # Complete Supabase schema
├── .streamlit/config.toml # Streamlit theme (dark text, indigo accent)
├── .env.example           # Environment variable template
└── requirements.txt

```

## Citations

- World Bank (2024). World Development Indicators - India. https://data.worldbank.org/country/india. License: CC BY 4.0
- fawazahmed0 (2024). exchange-api. https://github.com/fawazahmed0/exchange-api. License: CC0
- GeneralMills (2023). pytrends. https://github.com/GeneralMills/pytrends. License: Apache 2.0
- Kaggle dataset: https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth
