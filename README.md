# IndiaCommerce Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/)

**Live Dashboard:** https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/
**GitHub:** https://github.com/SumedhPatil1507/indian-ecommerce-analytics
**Dataset:** https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth

Production-grade e-commerce analytics platform with live macro data, multi-source connectors, ML-powered insights, and operational action workflows.

---

## What's Inside

### Data Connector Matrix
Connect any e-commerce platform to the analytics engine:

| Connector | Source | Schema |
|---|---|---|
| Shopify | order/create webhook | `from_shopify_webhook(payload)` |
| Amazon Seller Central | SP-API Orders v0 | `from_amazon_orders(payload)` |
| WooCommerce | REST API / DB dump | `from_woocommerce(payload)` |
| Generic File | CSV, TSV, Excel, JSON, Parquet | `load_any(file, filename)` |
| Simulation Sandbox | Live macro-calibrated synthetic data | `generate_simulation(...)` |

### Analytics Tabs (14 total)

| Tab | What it does |
|---|---|
| Data Connector Matrix | Connect Shopify/Amazon/WooCommerce, validate schema, run Simulation Sandbox |
| Executive Summary | Auto-written narrative, KPIs, risks, opportunities, PDF + Excel export |
| Price Optimizer | Lerner-index optimal discount per category, approve/dismiss with Supabase logging |
| At-Risk Customers | RFM churn scoring, export cohort to CSV for Klaviyo/SendGrid |
| Model Drift | PSI feature drift + R2 prediction degradation monitoring |
| Revenue Trends | Monthly revenue, AOV, discount trend, zone + brand breakdown |
| Categories | Revenue mix, festival vs normal, metric selector |
| Regional | Top 15 states, zone pie, units by zone |
| Inventory | Alert system with scatter dashboard + filterable table |
| CLV | BG/NBD CLV tiers, distribution, frequency scatter |
| Anomalies | Isolation Forest + DBSCAN + Z-score, anomaly by category |
| Cohort | Retention rate + revenue retention heatmaps |
| Pareto | 80/20 chart, sunburst, Lorenz curve + Gini coefficient |
| Operational Actions | Approve price changes, export at-risk cohort, view Supabase action log |

### Live Data Sources

| Source | Data | License |
|---|---|---|
| [World Bank Open Data](https://data.worldbank.org/) | India GDP growth + CPI inflation | CC BY 4.0 |
| [fawazahmed0/exchange-api](https://github.com/fawazahmed0/exchange-api) | Live USD/INR rate | CC0 |
| [Google Trends via pytrends](https://github.com/GeneralMills/pytrends) | E-commerce search interest | Apache 2.0 |

### Supabase Operational Persistence

When configured, Supabase stores:
- `operational_actions` - approved price changes, at-risk exports, drift alerts
- `clv_cache` - cached CLV tier results (24h TTL)
- `anomaly_cache` - cached weekly anomaly scores (7-day TTL)
- `model_results` - cached heavy model outputs (Prophet, SARIMA)
- `price_recommendations` - price optimizer output
- `at_risk_alerts` - churn risk scores

---

## Quick Start

```bash
git clone https://github.com/SumedhPatil1507/indian-ecommerce-analytics
cd ecommerce-analytics
bash setup_env.sh
source .venv/bin/activate

# Run dashboard
streamlit run dashboard/app.py

# Run API
uvicorn api.main:app --reload --port 8000
```

## Supabase Setup (optional)

1. Create project at [supabase.com](https://supabase.com) - name: `indiacommerce-analytics`, region: `ap-south-1`
2. Run `supabase_schema.sql` in SQL Editor
3. Create Storage bucket named `datasets` (private)
4. Add to Streamlit Cloud secrets or `.env`:

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
│   ├── export.py          # PDF + Excel export
│   └── ...
├── core/
│   ├── config.py          # App + Supabase configuration
│   └── database.py        # Supabase persistence + model caching
├── dashboard/
│   └── app.py             # Streamlit dashboard (14 tabs)
├── api/
│   └── main.py            # FastAPI REST endpoints
├── supabase_schema.sql    # Complete Supabase schema v4.0
├── .env.example           # Environment variable template
└── requirements.txt
```

## Live Data Citations

- World Bank (2024). World Development Indicators - India. https://data.worldbank.org/country/india. License: CC BY 4.0
- fawazahmed0 (2024). exchange-api. https://github.com/fawazahmed0/exchange-api. License: CC0
- GeneralMills (2023). pytrends. https://github.com/GeneralMills/pytrends. License: Apache 2.0
