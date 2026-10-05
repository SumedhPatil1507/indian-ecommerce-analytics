<div align="center">

# 🛒 IndiaCommerce Analytics

### Production-grade e-commerce intelligence platform for Indian markets

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy)](https://numpy.org)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

**[🚀 Live Demo](https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/) · [📦 Dataset](https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth) · [🐙 GitHub](https://github.com/SumedhPatil1507/indian-ecommerce-analytics)**

---

*Upload your data → get instant AI-powered insights across 18 interactive analytics tabs. No login. No servers. Runs entirely on Streamlit Cloud.*

</div>

---

## ✨ What's Inside

> A full analytics stack — from raw order data to forecasts, ML models, and a conversational AI copilot — in a single Streamlit app.

<table>
<tr>
<td width="50%">

**📊 Analytics Modules**
- Revenue trends, AOV, regional breakdown
- Price optimisation (Lerner-index)
- At-risk customer RFM scoring
- Customer Lifetime Value (BG/NBD)
- Anomaly detection (3 algorithms)
- Cohort retention heatmaps
- Pareto / Lorenz / Gini analysis
- Inventory velocity alerts

</td>
<td width="50%">

**🤖 AI & ML**
- Merchant Insights Copilot (NLP + RAG)
- Time-series forecasting (Prophet + SARIMA)
- ML model comparison (5 algorithms)
- SHAP / permutation importance
- Model drift monitoring (PSI)
- Price elasticity engine (log-log OLS)
- Exploratory data analysis suite

</td>
</tr>
</table>

---

## 🗂️ 18 Analytics Tabs

| # | Tab | What it does |
|---|-----|-------------|
| 0 | **Data Connector Matrix** | Connect Shopify · Amazon · WooCommerce · CSV/Excel · Simulation Sandbox |
| 1 | **Executive Summary** | Auto-written narrative, KPIs, risks, opportunities + PDF/Excel export |
| 2 | **Price Optimizer** | Lerner-index optimal discount per category with approval logging |
| 3 | **At-Risk Customers** | RFM churn scoring, exportable cohort for Klaviyo/SendGrid |
| 4 | **Model Drift** | PSI feature drift + R² prediction degradation monitoring |
| 5 | **Revenue Trends** | Monthly revenue, AOV, discount trends, zone + brand breakdown |
| 6 | **Categories** | Revenue mix, festival vs normal, metric selector |
| 7 | **Regional** | Top 15 states, zone pie, units by zone |
| 8 | **Inventory** | Velocity-based alert scatter dashboard + filterable table |
| 9 | **CLV** | BG/NBD CLV tiers, distribution histogram, frequency scatter |
| 10 | **Anomalies** | Isolation Forest + DBSCAN + Z-score (Supabase cached, 7-day TTL) |
| 11 | **Cohort** | Retention rate + revenue retention heatmaps |
| 12 | **Pareto** | 80/20 chart, sunburst, Lorenz curve + Gini coefficient |
| 13 | **Operational Actions** | Approve price changes, export at-risk cohort, action log |
| 14 | **🤖 Merchant Insights Copilot** | Ask natural-language questions, get cited AI answers |
| 15 | **🔍 Exploratory Analysis** | Distributions, box plots, violin plots, pie charts, count plots |
| 16 | **📈 Forecasting** | Seasonal decomposition, Prophet + SARIMA with horizon slider |
| 17 | **🤖 ML Models** | Train & compare LR/DT/RF/XGBoost/MLP + permutation importance |

---

## 🤖 Merchant Insights Copilot

Ask any business question in plain English. No tab-clicking required.

```
"Why did AOV drop in the South zone last month?"
"Which customers are at risk of churning in Tier-1 cities?"
"What discount should I offer on Electronics to maximise revenue?"
"Are there anomalous orders or fraud signals this quarter?"
"How does CLV compare across product categories?"
```

### How it works

```
Your question
     │
     ▼
┌─────────────────────────────┐
│  Keyword Planner            │  Maps question → 1-3 analytics tools
│  (price, churn, anomaly…)   │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Tool Execution             │  Calls real module functions on your live data
│  (9 analytics modules)      │
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  TF-IDF RAG Store           │  Chunks outputs, retrieves top-3 relevant chunks
│  (sklearn, no vector DB)    │  via cosine similarity
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  LLM Synthesis              │  OpenAI gpt-4o-mini with inline [source] citations
│  (demo mode if no API key)  │
└─────────────────────────────┘
     │
     ▼
  Cited answer with expandable Tool Outputs + Retrieved Context
```

> **Demo mode:** Works without `OPENAI_API_KEY` — returns raw module summaries instead of LLM synthesis.

---

## 🚀 Quick Start

### Prerequisites
- Python **3.11** (required — matches Streamlit Cloud environment)
- Git

### Local setup

```powershell
# 1. Clone
git clone https://github.com/SumedhPatil1507/indian-ecommerce-analytics
cd indian-ecommerce-analytics/ecommerce-analytics

# 2. Create virtual environment (Python 3.11)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional)
copy .env.example .env
# Edit .env — add OPENAI_API_KEY and/or Supabase keys

# 5. Run
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501** 🎉

### Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | Optional | Enables AI-generated Copilot answers (demo mode without it) |
| `SUPABASE_URL` | Optional | Enables result caching + operational logging |
| `SUPABASE_ANON_KEY` | Optional | Supabase authentication |

---

## 📡 Live Data Sources

| Source | Data | Refresh | License |
|--------|------|---------|---------|
| [World Bank Open Data](https://data.worldbank.org/) | India GDP growth + CPI inflation | Daily | CC BY 4.0 |
| [fawazahmed0/exchange-api](https://github.com/fawazahmed0/exchange-api) | Live USD/INR rate (3-source waterfall) | Real-time | CC0 |
| [Google Trends via pytrends](https://github.com/GeneralMills/pytrends) | E-commerce search interest India | Daily | Apache 2.0 |

---

## 🗄️ Supabase (Optional Persistence)

When `SUPABASE_URL` + `SUPABASE_ANON_KEY` are configured:

| Table | Purpose | TTL |
|-------|---------|-----|
| `operational_actions` | Approved price changes, at-risk exports | Permanent |
| `clv_cache` | CLV tier computation results | 24 hours |
| `anomaly_cache` | Weekly anomaly scores | 7 days |
| `model_results` | Heavy model outputs (Prophet, SARIMA) | 24 hours |

**Setup:**
1. Create project at [supabase.com](https://supabase.com) — region: `ap-south-1`
2. Run `supabase_schema.sql` in the SQL Editor
3. Add to **Streamlit Cloud → App Settings → Secrets**:

```toml
SUPABASE_URL = "https://xxxx.supabase.co"
SUPABASE_ANON_KEY = "eyJ..."
SUPABASE_SERVICE_KEY = "eyJ..."
OPENAI_API_KEY = "sk-..."
```

---

## 🏗️ Project Structure

```
ecommerce-analytics/
│
├── 📊 dashboard/
│   ├── app.py              # Main Streamlit app (18 tabs)
│   ├── style.py            # Dark indigo theme CSS + Plotly dark template
│   └── copilot_tab.py      # Merchant Insights Copilot UI
│
├── 🧠 modules/
│   ├── insights.py         # Executive summary + recommendations
│   ├── price_optimizer.py  # Lerner-index dynamic pricing
│   ├── at_risk.py          # RFM churn risk scoring
│   ├── model_drift.py      # PSI + prediction drift monitoring
│   ├── clv.py              # BG/NBD + Gamma-Gamma CLV
│   ├── anomaly.py          # Isolation Forest + DBSCAN + Z-score
│   ├── cohort.py           # Cohort retention heatmaps
│   ├── pareto.py           # Pareto, Lorenz, choropleth, sunburst
│   ├── inventory_alerts.py # Velocity-based inventory alerts
│   ├── eda.py              # Exploratory data analysis
│   ├── time_series.py      # Prophet + SARIMA forecasting
│   ├── models.py           # ML model training + comparison
│   ├── explainability.py   # SHAP + permutation importance
│   ├── price_elasticity.py # Log-log OLS elasticity engine
│   ├── copilot.py          # Copilot: keyword planner + TF-IDF RAG + LLM
│   └── export.py           # PDF + Excel export
│
├── 📡 data/
│   ├── loader.py           # Multi-format loader + live macro enrichment
│   └── connectors.py       # Shopify, Amazon, WooCommerce, Sandbox
│
├── ⚙️  core/
│   ├── config.py           # App + Supabase configuration
│   └── database.py         # Supabase persistence + result caching
│
├── 🐳 api/
│   └── main.py             # FastAPI REST endpoints (optional)
│
├── runtime.txt             # Pins Python 3.11 on Streamlit Cloud
├── requirements.txt        # All dependencies (Cloud-compatible)
├── supabase_schema.sql     # Complete Supabase schema
└── .env.example            # Environment variable template
```

---

## 🎨 Visual Theme

The dashboard uses a **dark indigo** design system:

| Element | Colour | Hex |
|---------|--------|-----|
| Background | Deep navy | `#0F172A` |
| Card surfaces | Slate | `#1E293B` |
| Active tab | Indigo | `#6366F1` |
| Primary button | Indigo | `#4F46E5` |
| Body text | Light slate | `#F1F5F9` |
| Muted text | Mid slate | `#94A3B8` |

All 18 tabs use a consistent Plotly dark template (`paper_bgcolor=#1E293B`, `plot_bgcolor=#0F172A`) with slate grid lines and light axis labels.

---

## 🔧 Compatibility Notes

| Issue | Fix |
|-------|-----|
| **NumPy 1.x vs 2.x** | `np.trapz` removed in 2.0; `np.trapezoid` added in 2.0. Uses `getattr(np, "trapezoid", None) or getattr(np, "trapz", None)` — works on both. |
| **lifetimes package** | Max published version is `0.11.3`. Pinned to `==0.11.3` in `requirements.txt`. |
| **Python version** | `runtime.txt` pins Python 3.11. Run locally with 3.11 to match Cloud exactly. |
| **prophet on Cloud** | Pinned `prophet>=1.1,<2.0` — dashboard falls back gracefully if unavailable. |

---

## 📤 Update on GitHub (VS Code Terminal)

Open the VS Code integrated terminal with **Ctrl + `** then run:

```powershell
# Navigate to project
cd "C:\Users\Sumedh\projects\Indian-ecommerce-project\ecommerce-analytics"

# See what changed
git status

# Stage all changes
git add .

# Commit with a message
git commit -m "your descriptive message here"

# Push to GitHub (triggers Streamlit Cloud auto-redeploy)
git push
```

> **Tip:** Streamlit Cloud redeploys automatically within ~30 seconds of every push to `main`.

### Stage specific files only

```powershell
# Stage individual files
git add dashboard/app.py modules/copilot.py README.md

# Or stage by folder
git add modules/
git add dashboard/

# Check what's staged before committing
git diff --staged --stat
```

### Useful git commands

```powershell
git log --oneline -10          # View last 10 commits
git diff HEAD                  # See all uncommitted changes
git restore <file>             # Discard changes to a file
git stash                      # Temporarily shelve changes
git pull                       # Pull latest from GitHub
```

---

## 📖 Citations

- World Bank (2024). *World Development Indicators — India*. https://data.worldbank.org/country/india. License: CC BY 4.0
- fawazahmed0 (2024). *exchange-api*. https://github.com/fawazahmed0/exchange-api. License: CC0
- GeneralMills (2023). *pytrends*. https://github.com/GeneralMills/pytrends. License: Apache 2.0
- Kaggle dataset: https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth

---

<div align="center">

Built with ❤️ using [Streamlit](https://streamlit.io) · [Plotly](https://plotly.com) · [scikit-learn](https://scikit-learn.org) · [OpenAI](https://openai.com)

</div>
