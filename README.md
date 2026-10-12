# IndiaCommerce Analytics

Interactive analytics and decision support for Indian ecommerce teams. Explore sales, customers, pricing, operations, and risk in a Streamlit dashboard with Plotly visualizations. The repository also includes optional streaming, feature store, graph risk, and copilot evaluation components.

<p align="center">
  <a href="https://indian-ecommerce-analytics-arxf6zhgntbmhby5vcvsgy.streamlit.app/"><strong>Open the live dashboard</strong></a>
  · <a href="https://github.com/SumedhPatil1507/indian-ecommerce-analytics">GitHub</a>
  · <a href="https://www.kaggle.com/datasets/shukla922/indian-e-commerce-pricing-revenue-growth">Example dataset</a>
</p>

## Run the dashboard

Requires Python 3.11 (3.10+ supported). From this directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit. Use the Data Connector tab to upload a CSV/XLSX order file or start with the built-in sample data. Filters and selections update the dashboard across tabs.

### Interactive charts

Charts use Plotly. Hover for row-level details, drag to zoom, pan, select ranges, toggle traces from the legend, and use the Plotly toolbar to reset or download a chart as PNG. Tables and metric controls are interactive Streamlit components. A dark indigo theme is included in `.streamlit/config.toml`.

## Dashboard at a glance

The 18 analytics tabs cover:

1. Data connectors and sample data
2. Executive summary and exports
3. Price optimizer with Lerner-index constraints
4. At-risk customer scoring
5. Model drift monitoring
6. Revenue trends
7. Category performance
8. Regional performance
9. Inventory velocity
10. Customer lifetime value
11. Anomaly detection
12. Cohort retention
13. Pareto and concentration
14. Operational actions
15. Merchant Insights Copilot
16. Exploratory analysis
17. Forecasting
18. Machine learning model comparison

Anomaly detection combines Isolation Forest, DBSCAN, and statistical signals. When a trained graph model is available, the anomaly view also surfaces RTO and promotional fraud risk scores.

## Optional production data components

The dashboard runs without external services. Install the additional dependencies only when working with the corresponding infrastructure:

```bash
pip install -r requirements-advanced.txt
```

| Component | Implementation | Purpose |
|---|---|---|
| Feast feature store | `feature_store.yaml`, `data/features.py` | Customer and merchant entities with time-aware order velocity, AOV, and discount sensitivity features. |
| Kafka ingestion | `data/kafka_consumer.py` | Async consumption from `ecommerce.orders.v1`; manual offsets are committed after audit persistence and Redis feature-cache writes succeed. |
| Audit and online store | `supabase_schema.sql` | PostgreSQL/Supabase audit schema and Redis-backed online features. |
| R-GCN graph risk | `modules/rto_graph.py`, `scripts/train_rto_gnn.py` | Heterogeneous buyer, pincode, device, and payment graph model; training writes node embeddings to Neo4j. |
| Copilot validation and tracing | `modules/copilot.py` | Guardrails validation, grounded citations, discount controls, and OpenTelemetry spans. |
| Copilot evaluation | `eval/ragas_copilot_eval.py` | RAGAS audit over 50 merchant prompts, exporting metric results as JSON. |

### Configure services

Copy `.env.example` to `.env` and set only the credentials for services you enable. Never commit `.env` or Streamlit secrets.

```dotenv
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
DATABASE_URL=postgresql://...
REDIS_URL=redis://localhost:6379/0
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=...
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
OPENAI_API_KEY=...
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

For Streamlit Community Cloud, set supported values in the app's **Settings → Secrets**. Keep Kafka, database, Redis, and Neo4j services reachable from the deployment environment; these integrations are not required by the hosted dashboard's basic analytics.

Initialize and inspect Feast definitions with the Feast CLI from the repository root:

```bash
feast -c . apply
```

Train graph risk and write embeddings to Neo4j with:

```bash
python scripts/train_rto_gnn.py --help
```

Run the copilot evaluation (requires configured model access):

```bash
python eval/ragas_copilot_eval.py
```

## Deployment

For Streamlit Community Cloud, select this repository and set **Main file path** to `dashboard/app.py`. The standard `requirements.txt` contains the dashboard and copilot runtime dependencies; graph training, Kafka, Feast, and RAGAS are isolated in `requirements-advanced.txt` to keep the dashboard install smaller. Docker remains available for the API service through the included `Dockerfile`.

## Repository map

```text
dashboard/       Streamlit app, tabs, styling, and interactive charts
modules/         Analytics, anomaly detection, pricing, copilot, and graph model
data/            Feature definitions and Kafka consumer
api/             FastAPI service
scripts/         Graph model training CLI
eval/            RAGAS copilot evaluation
tests/           Unit tests
```

## License

See [LICENSE](LICENSE).
