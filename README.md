# IndiaCommerce Analytics v4.3

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

### 17 Analytics Tabs

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
| 🔍 Exploratory Analysis | Interactive histograms, box plots, violin plots, pie charts, count plots |
| 📈 Forecasting | Revenue trends, seasonal decomposition, Prophet + SARIMA forecasts |
| 🤖 ML Models | Linear/Tree/RF/XGBoost/Neural Net comparison, permutation importance |
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

# Create a Python 3.11 virtual environment (required — see Python version note below)
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run dashboard (opens at http://localhost:8501)
streamlit run dashboard/app.py
```

### Python Version

**Python 3.11 is required.** A `runtime.txt` file pins Streamlit Cloud to Python 3.11 automatically.

- `lifetimes` max published version is `0.11.3` — pinned to `==0.11.3` in `requirements.txt`.
- `prophet` requires `pystan` which has no Python 3.14 wheels.
- Run locally with Python 3.11 to match the Cloud environment exactly.

### NumPy Compatibility

Uses `np.trapz` (not `np.trapezoid`) — works on NumPy 1.x (1.26.4) and 2.x.

## Visual Theme

The dashboard uses a dark indigo theme for improved contrast and readability:
- **Background:** Deep navy (`#0F172A`) with slate card surfaces (`#1E293B`)
- **Tab bar:** Inactive tabs in slate, active tab highlighted in indigo (`#6366F1`) with white text
- **Charts:** All Plotly charts use a consistent dark template with slate backgrounds and light axis labels
- **Metrics:** Card-style metric tiles with visible borders on the dark background

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

## GitHub Update Commands (VS Code Terminal)

```powershell
cd "C:\Users\Sumedh\projects\Indian-ecommerce-project\ecommerce-analytics"
git add .
git commit -m "your message"
git push
```

Streamlit Cloud auto-redeploys within ~30 seconds of a push to `main`.

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
│   ├── copilot.py         # Merchant Insights Copilot (keyword planning + TF-IDF RAG)
│   ├── price_elasticity.py# Price elasticity analysis (log-log regression)
│   ├── eda.py             # Exploratory Data Analysis (distributions, categorical, boxplots)
│   ├── models.py          # ML model training + comparison (LR/DT/RF/XGBoost/MLP)
│   ├── explainability.py  # SHAP + permutation importance + LIME
│   └── time_series.py     # Time series trends + Prophet + SARIMA forecasting
├── core/
│   ├── config.py          # App + Supabase configuration
│   └── database.py        # Supabase persistence + model result caching
├── dashboard/
│   └── app.py             # Streamlit dashboard (17 tabs, no login required)
├── api/
│   └── main.py            # FastAPI REST endpoints (optional)
├── supabase_schema.sql    # Complete Supabase schema
├── .streamlit/config.toml # Streamlit theme (dark text, indigo accent)
├── .env.example           # Environment variable template
└── requirements.txt       # Streamlit Cloud–compatible deps (Python 3.11, lifetimes==0.11.3)
└── runtime.txt            # Pins Python 3.11 on Streamlit Cloud

```

## Citations

- World Bank (2024). World Development Indicators - India. https://data.worldbank.org/country/india. License: CC BY 4.0
- fawazahmed0 (2024). exchange-api. https://github.com/fawazahmed0/exchange-api. License: CC0
- GeneralMills (2023). pytrends. https://github.com/GeneralMills/pytrends. License: Apache 2.0
- Kaggle dataset: https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth

## Merchant Insights Copilot

The **Merchant Insights Copilot** is a conversational AI agent built into the dashboard that lets merchants ask natural-language questions and get synthesised, data-backed answers — without clicking through 14 tabs manually.

### How it works

1. **Keyword Planning** — The copilot maps your question keywords to 1-3 relevant analytics modules (`price_optimizer`, `at_risk`, `clv`, `anomaly`, `insights`, `inventory_alerts`, `price_elasticity`, `cohort`, `time_series`) using a configurable keyword→tool table.
2. **Tool Execution** — It calls the real analytics functions from each planned module against your live filtered data and generates structured text summaries.
3. **TF-IDF RAG** — Each tool output is chunked and stored in an in-process vector store. Relevant chunks are retrieved via cosine similarity (no external vector DB required — uses `scikit-learn` TF-IDF).
4. **LLM Synthesis** — Retrieved chunks and fresh tool outputs are fed to OpenAI `gpt-4o-mini` (falls back to `gpt-3.5-turbo`) with an instruction to cite sources inline like `[price_optimizer]`.

### Setup

Set `OPENAI_API_KEY` in your `.env` file or Streamlit secrets:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Demo Mode

If `OPENAI_API_KEY` is not set, the Copilot runs in **demo mode**: it still executes all planned analytics tools and displays the raw structured summaries, but skips LLM synthesis. A yellow banner in the UI indicates demo mode.

### Example Questions

- "Why did AOV drop in the South zone last month?"
- "Which customers are at risk of churning in Tier-1 cities?"
- "What discount should I offer on Electronics to maximise revenue?"
- "Are there any anomalous orders or fraud signals this quarter?"
- "How does customer lifetime value compare across product categories?"

### New Analytics Tabs (v4.1)

Three new tabs have been added to the dashboard:
- **🔍 Exploratory Analysis** — Interactive distributions, categorical plots, box plots, violin plots, and pie charts for deep data exploration.
- **📈 Forecasting** — Revenue trend plots, seasonal decomposition, and Prophet + SARIMA forecasting with configurable horizon.
- **🤖 ML Models** — Train and compare Linear Regression, Decision Tree, Random Forest, XGBoost, and Neural Network models with permutation feature importance.
