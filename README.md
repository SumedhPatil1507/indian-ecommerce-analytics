# IndiaCommerce Analytics (v5.0)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/)

**Live Dashboard:** https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/
**GitHub:** https://github.com/SumedhPatil1507/indian-ecommerce-analytics
**Dataset:** https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth

Production-grade, multi-tenant e-commerce analytics platform with a stateless visual frontend, cryptographic webhook protection, relational PostgreSQL storage (isolated via Row-Level Security), distributed Celery task worker, and OpenTelemetry request instrumentation.

---

## Architecture Highlights

1. **Stateless Visual Dashboard**: The Streamlit application in `dashboard/app.py` is entirely decoupled from business logic and acts as a stateless visual browser. It authenticates users and loads JSON payloads from FastAPI.
2. **Multi-Tenant Row-Level Security (RLS)**: Transaction logs are persisted in a relational `orders` table in Supabase. PostgreSQL RLS policies automatically restrict access using the tenant's auth JWT.
3. **Cryptographically Protected Webhooks**: Ingestion routes for Shopify, WooCommerce, and Amazon require signature verification (using SHA256 HMAC) before queueing requests.
4. **Distributed Task Queue**: Heavy analytics calculations (CLV, Price Optimisation OLS, Anomaly Forest, Drift PSI) and large order uploads run asynchronously on a background Celery worker backed by Redis.
5. **OpenTelemetry Telemetry**: Requests, latencies, error states, and worker tasks are instrumented using the OpenTelemetry API and rendered inside the *Observability (OpenTelemetry)* UI tab.

---

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
│   └── database.py        # Multi-tenant Supabase database client (JWT scoped)
├── dashboard/
│   └── app.py             # Stateless Streamlit dashboard
├── api/
│   └── main.py            # FastAPI REST backend (OpenTelemetry instrumented)
├── worker/
│   ├── celery_app.py      # Celery worker application
│   └── tasks.py           # Background tasks for heavy calculations
├── supabase_schema.sql    # Multi-tenant Supabase schema v5.0
├── .env.example           # Environment variable template
└── requirements.txt
```

---

## Configuration Variables (.env)

Rename `.env.example` to `.env` and configure:

```toml
# Supabase Configuration
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_ANON_KEY = "eyJ..."
SUPABASE_SERVICE_KEY = "eyJ..."

# Celery Task Queue Broker
REDIS_URL = "redis://localhost:6379/0"

# Backend REST Server
API_URL = "http://localhost:8000"
```

---

## Quick Start (Local Setup)

### 1. Setup Environment
```bash
git clone https://github.com/SumedhPatil1507/indian-ecommerce-analytics
cd ecommerce-analytics
bash setup_env.sh
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Redis Broker (via Docker)
```bash
docker run -d -p 6379:6379 redis:alpine
```

### 3. Start Celery Worker
```bash
celery -A worker.celery_app worker --loglevel=info
```

### 4. Run REST API Engine
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Run Dashboard
```bash
streamlit run dashboard/app.py
```

---

## Webhook Signature Verification

Shopify, WooCommerce, and Amazon webhooks must contain cryptographic signature headers matching the tenant's webhook secret configured in Supabase `profiles`.

* **Shopify Header**: `X-Shopify-Hmac-SHA256`
* **WooCommerce Header**: `X-WC-Webhook-Signature`
* **Amazon Header**: `X-Amazon-Webhook-Signature`

Signature check formula: `base64(hmac-sha256(body_bytes, tenant_secret))`

---

## GitHub Update Commands

```bash
cd C:\Users\Sumedh\projects\Indian-ecommerce-project\ecommerce-analytics
git add .
git commit -m "feat: implement stateless visual window, Celery worker queue, Postgres RLS, cryptographic webhooks, and OpenTelemetry"
git push
```
