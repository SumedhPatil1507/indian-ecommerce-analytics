# 🛒 Indian E-Commerce Analytics

Production-grade analytics platform for Indian e-commerce pricing, revenue, and demand intelligence — built from the [Kaggle Indian E-Commerce Pricing & Revenue Growth (36 months)](https://www.kaggle.com/datasets) dataset.

---

## Quick Start (One-Click)

```bash
git clone <repo>
cd ecommerce-analytics
bash setup_env.sh
source .venv/bin/activate

# Streamlit dashboard
streamlit run dashboard/app.py

# FastAPI
uvicorn api.main:app --reload --port 8000
# → docs at http://localhost:8000/docs
```

---

## Project Structure

```
ecommerce-analytics/
├── data/
│   └── loader.py            # load, clean, engineer + live data enrichment
├── modules/
│   ├── eda.py               # distributions, bar, pie, box, violin (Plotly)
│   ├── time_series.py       # decomposition, Prophet, SARIMA
│   ├── models.py            # LR, DT, RF, XGBoost, PyTorch MLP
│   ├── explainability.py    # SHAP, LIME, permutation importance
│   ├── price_elasticity.py  # log-log OLS elasticity engine
│   ├── inventory_alerts.py  # dynamic inventory alert system
│   ├── clv.py               # BG/NBD + Gamma-Gamma CLV predictor
│   ├── anomaly.py           # Isolation Forest + DBSCAN + Z-score
│   ├── cohort.py            # cohort retention heatmap
│   └── pareto.py            # Pareto, Sunburst, Choropleth, Lorenz, ECDF
├── api/
│   └── main.py              # FastAPI REST wrapper
├── dashboard/
│   └── app.py               # Streamlit interactive dashboard
├── tests/
│   ├── test_loader.py
│   └── test_modules.py
├── .github/workflows/ci.yml # GitHub Actions CI
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## Live Data Sources & Citations

| Source | Data | License | Endpoint |
|--------|------|---------|----------|
| [World Bank Open Data](https://data.worldbank.org/) | India GDP growth & CPI inflation | **CC BY 4.0** | `api.worldbank.org/v2/country/IN/indicator/...` |
| [exchangerate.host](https://exchangerate.host) | Live USD → INR FX rate | Free tier | `api.exchangerate.host/live` |
| [Google Trends via pytrends](https://github.com/GeneralMills/pytrends) | E-commerce search interest (India) | **Apache 2.0** | Unofficial Google Trends API |
| [NewsAPI.org](https://newsapi.org) | Indian e-commerce headlines | Free tier (key required) | `newsapi.org/v2/everything` |

> **World Bank citation:** World Bank (2024). *World Development Indicators – India*. Retrieved from https://data.worldbank.org/country/india. License: Creative Commons Attribution 4.0 (CC BY 4.0).

> **pytrends citation:** GeneralMills (2023). *pytrends – Pseudo API for Google Trends*. https://github.com/GeneralMills/pytrends. License: Apache 2.0.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/live/macro` | World Bank GDP + CPI |
| GET | `/live/fx` | USD/INR rate |
| GET | `/live/trends` | Google Trends interest |
| GET | `/analytics/revenue` | Monthly revenue summary |
| GET | `/analytics/elasticity` | Price elasticity by category |
| GET | `/analytics/alerts` | Inventory alert table |
| GET | `/analytics/clv` | CLV tier summary |
| POST | `/predict/revenue` | Predict revenue for an order |

Interactive docs: `http://localhost:8000/docs`

---

## Docker

```bash
docker build -t ecommerce-analytics .
docker run -p 8000:8000 -v /path/to/data:/data ecommerce-analytics

# Streamlit
docker run -p 8501:8501 -v /path/to/data:/data ecommerce-analytics \
  streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Tests

```bash
pytest tests/ -v --cov=.
```
