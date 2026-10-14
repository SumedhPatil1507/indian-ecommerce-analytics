"""
modules/copilot.py
Merchant Insights Copilot  keyword-planned tool execution, TF-IDF RAG, LLM synthesis.
Fully importable  no Streamlit dependency.
"""
from __future__ import annotations
import logging
import os
import hashlib
import re
from functools import wraps
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.trace import ProxyTracerProvider
    if isinstance(trace.get_tracer_provider(), ProxyTracerProvider):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        provider = TracerProvider(resource=Resource.create({"service.name": "indiocommerce-copilot"}))
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("indiocommerce.copilot")
except ImportError:  # retain importability in minimal installs
    trace = None
    _TRACER = None


def _span(name: str, **attributes):
    """Create a Copilot span without putting merchant text or PII in telemetry."""
    if _TRACER is None:
        return nullcontext(None)
    manager = _TRACER.start_as_current_span(name)
    return _SpanAttributes(manager, attributes)


def _traced_answer(fn):
    @wraps(fn)
    def wrapped(self, question, df, filters=None):
        with _span("copilot.answer", **{"copilot.question_length": len(question),
                                         "copilot.input_rows": len(df)}) as span:
            result = fn(self, question, df, filters)
            if span:
                span.set_attribute("copilot.response_state", result.get("state", "unknown"))
            return result
    return wrapped


class _SpanAttributes:
    def __init__(self, manager, attributes):
        self.manager, self.attributes = manager, attributes

    def __enter__(self):
        span = self.manager.__enter__()
        for key, value in self.attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        return span

    def __exit__(self, *exc):
        return self.manager.__exit__(*exc)


def _validation_failure(answer: str, metadata: dict) -> str | None:
    """Local deterministic fallback for deployments missing Guardrails AI."""
    context_ids = set(metadata.get("chunk_ids", []))
    cited_ids = set(re.findall(r"\[chunk:([A-Za-z0-9_-]+)\]", answer))
    if not context_ids or not cited_ids.intersection(context_ids):
        return "Answer omitted a valid retrieved chunk ID citation."
    # Discount recommendations must agree with a measured optimizer output.
    recommendations = re.finditer(
        r"(?is)(?P<lead>(?:recommend(?:ation)?|suggest(?:ion)?|should|offer|set|reduce|increase|adjust)"
        r"[^.!?\n]{0,100}?)(?P<pct>\d+(?:\.\d+)?)\s*(?:%|percent)", answer,
    )
    if recommendations:
        verified_by_category = metadata.get("verified_by_category", {})
        verified = set(metadata.get("verified_optimal_discounts", []))
        for recommendation in recommendations:
            proposed = float(recommendation.group("pct"))
            if proposed > 25:
                return f"Discount recommendation exceeds the hard 25% cap ({proposed:g}%)."
            lead = recommendation.group("lead").lower()
            category_matches = [category for category in verified_by_category if category.lower() in lead]
            allowed = verified
            if category_matches:
                allowed = {verified_by_category[category_matches[0]]}
            if not allowed or not any(abs(proposed - value) <= 0.05 for value in allowed):
                return "Price adjustment is not verified by the Lerner-index optimizer."
    return None


def _guardrails_validate(answer: str, metadata: dict) -> tuple[bool, str | None]:
    """Validate output through Guardrails AI custom validators plus local checks."""
    failure = _validation_failure(answer, metadata)
    try:
        from guardrails import Guard
        from guardrails.validator_base import FailResult, PassResult, Validator

        class CopilotPolicyValidator(Validator):
            def __init__(self, policy):
                super().__init__(on_fail="exception")
                self.policy = policy

            def validate(self, value, validator_metadata):
                message = _validation_failure(value, validator_metadata)
                if message:
                    return FailResult(error_message=message)
                return PassResult()

        guard = Guard().use(CopilotPolicyValidator("citation-and-pricing"))
        result = guard.validate(answer, metadata=metadata)
        if not getattr(result, "validation_passed", True):
            return False, failure or "Guardrails AI rejected the response."
    except ImportError:
        logger.warning("guardrails-ai is missing; deterministic Copilot checks are active")
    except Exception as exc:
        return False, failure or f"Guardrails AI validation failed: {exc}"
    if failure:
        return False, failure
    return True, None

# ---------------------------------------------------------------------------
# Keyword  tool mapping
# ---------------------------------------------------------------------------
_KEYWORD_MAP: dict[str, list[str]] = {
    "price_optimizer":   ["price", "discount", "elasticity", "optimal", "margin", "lerner", "revenue"],
    "at_risk":           ["churn", "at-risk", "at risk", "retention", "lost", "inactive", "win-back", "winback"],
    "clv":               ["clv", "lifetime value", "ltv", "champions", "loyalists", "tier"],
    "anomaly":           ["anomaly", "outlier", "unusual", "spike", "fraud", "irregular"],
    "insights":          ["summary", "insight", "kpi", "headline", "recommendation", "risk", "opportunity", "aov", "avg order"],
    "inventory_alerts":  ["inventory", "stock", "reorder", "stockout", "slow mover", "clearance"],
    "price_elasticity":  ["elasticity", "elastic", "inelastic", "price sensitivity"],
    "cohort":            ["cohort", "retention rate", "repeat purchase"],
    "time_series":       ["trend", "forecast", "monthly", "seasonal", "growth", "drop", "decline"],
}


class MerchantCopilot:
    """Merchant Insights Copilot with TF-IDF RAG and optional LLM synthesis."""

    def __init__(self) -> None:
        self._chunks: list[dict] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    # ------------------------------------------------------------------
    # RAG memory helpers
    # ------------------------------------------------------------------

    def _store(self, text: str, source: str, metadata: dict | None = None) -> None:
        """Chunk text by paragraphs and store in RAG memory."""
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
        for para in paragraphs:
            chunk_id = hashlib.sha256(f"{source}\0{para}".encode("utf-8")).hexdigest()[:16]
            self._chunks.append({"chunk_id": chunk_id, "chunk": para, "source": source, "metadata": metadata or {}})
        self._vectorizer = None  # invalidate cache
        self._matrix = None

    def _retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Return top_k most relevant chunks for query using TF-IDF cosine similarity."""
        with _span("copilot.retrieval.search", **{"retrieval.backend": "tfidf", "retrieval.top_k": top_k,
                   "retrieval.chunk_count": len(self._chunks)}):
            if not self._chunks:
                return []
            if self._vectorizer is None or self._matrix is None:
                texts = [c["chunk"] for c in self._chunks]
                self._vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
                self._matrix = self._vectorizer.fit_transform(texts)
            q_vec = self._vectorizer.transform([query])
            scores = cosine_similarity(q_vec, self._matrix).flatten()
            top_idx = np.argsort(scores)[::-1][:top_k]
            matches = [self._chunks[i] for i in top_idx if scores[i] > 0]
            if _TRACER:
                from opentelemetry import trace as otel_trace
                otel_trace.get_current_span().set_attribute("retrieval.result_count", len(matches))
            return matches

    # ------------------------------------------------------------------
    # Tool: Price Optimizer
    # ------------------------------------------------------------------

    def run_price_optimizer(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Run price optimizer and return structured text summary."""
        try:
            from modules.price_optimizer import run_price_optimizer
            result = run_price_optimizer(df)
            if result.empty:
                return "[price_optimizer] No elasticity data available (need 30+ observations per category)."
            lines = ["[price_optimizer] Price Optimisation Results\n"]
            lines.append(f"Analysed {len(result)} categories for optimal discount.")
            total_impact = result["revenue_impact_pct"].sum()
            lines.append(f"Total estimated revenue impact: {total_impact:+.1f}%")
            lines.append("\n| Category | Current Disc% | Optimal Disc% | Change | Direction | Impact% |")
            lines.append("|---|---|---|---|---|---|")
            for _, row in result.iterrows():
                lines.append(
                    f"| {row['category']} | {row['current_discount']:.1f}% | "
                    f"{row['optimal_discount']:.1f}% | {row['change']:+.1f}pp | "
                    f"{row['direction']} | {row['revenue_impact_pct']:+.1f}% |"
                )
            increase_cats = result[result["direction"] == "increase"]["category"].tolist()
            decrease_cats = result[result["direction"] == "decrease"]["category"].tolist()
            if increase_cats:
                lines.append(f"\nCategories where INCREASING discount grows volume: {', '.join(increase_cats)}")
            if decrease_cats:
                lines.append(f"Categories where REDUCING discount protects margin: {', '.join(decrease_cats)}")
            summary = "\n".join(lines)
            self._store(summary, "price_optimizer", {"row_count": len(result)})
            return summary
        except Exception as e:
            logger.warning("price_optimizer tool failed: %s", e)
            return f"[price_optimizer] Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool: At-Risk Customers
    # ------------------------------------------------------------------

    def run_at_risk(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Run at-risk churn scoring and return structured text summary."""
        try:
            from modules.at_risk import generate_at_risk_alerts
            top_n = (params or {}).get("top_n", 50)
            result = generate_at_risk_alerts(df, top_n=top_n)
            if result.empty:
                return "[at_risk] No at-risk customers found in current data."
            lines = ["[at_risk] At-Risk Customer Analysis\n"]
            label_counts = result["risk_label"].value_counts()
            tier_counts  = result["value_tier"].value_counts()
            lines.append(f"Total at-risk customers analysed: {len(result)}")
            for label, cnt in label_counts.items():
                lines.append(f"  {label}: {cnt} customers")
            lines.append(f"Revenue at risk: Rs{result['total_revenue'].sum():,.0f}")
            lines.append("\nValue Tier Breakdown:")
            for tier, cnt in tier_counts.items():
                rev = result[result["value_tier"] == tier]["total_revenue"].sum()
                lines.append(f"  {tier}: {cnt} customers, Rs{rev:,.0f} total revenue")
            lines.append("\nTop 5 Customers Needing Immediate Action:")
            lines.append("| Customer | Risk Score | Risk Label | Value Tier | Days Since Order | Action |")
            lines.append("|---|---|---|---|---|---|")
            for _, row in result.head(5).iterrows():
                cid = str(row.get("customer_id", "N/A"))[:20]
                lines.append(
                    f"| {cid} | {row['churn_risk_score']:.0f} | {row['risk_label']} | "
                    f"{row['value_tier']} | {row['days_since_last_order']:.0f} | {row['recommended_action'][:60]} |"
                )
            zone_risk = result.groupby("top_zone")["total_revenue"].sum().sort_values(ascending=False)
            lines.append("\nRevenue at Risk by Zone:")
            for zone, rev in zone_risk.items():
                lines.append(f"  {zone}: Rs{rev:,.0f}")
            summary = "\n".join(lines)
            self._store(summary, "at_risk", {"at_risk_count": len(result)})
            return summary
        except Exception as e:
            logger.warning("at_risk tool failed: %s", e)
            return f"[at_risk] Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool: CLV
    # ------------------------------------------------------------------

    def run_clv(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Run CLV computation and return structured text summary."""
        try:
            from modules.clv import compute_clv
            result = compute_clv(df)
            if result.empty:
                return "[clv] CLV computation returned no results."
            lines = ["[clv] Customer Lifetime Value Analysis\n"]
            tier_summary = result.groupby("clv_tier")["clv"].agg(["count", "mean", "sum"])
            lines.append(f"Total customers analysed: {len(result)}")
            lines.append(f"Total predicted CLV: Rs{result['clv'].sum():,.0f}")
            lines.append(f"Average CLV per customer: Rs{result['clv'].mean():,.0f}")
            lines.append("\nCLV Tier Summary:")
            lines.append("| Tier | Customers | Avg CLV | Total CLV |")
            lines.append("|---|---|---|---|")
            for tier, row in tier_summary.iterrows():
                lines.append(f"| {tier} | {int(row['count'])} | Rs{row['mean']:,.0f} | Rs{row['sum']:,.0f} |")
            top5 = result.nlargest(5, "clv")
            lines.append("\nTop 5 Highest-Value Customers:")
            lines.append("| Customer | CLV | Tier | Frequency |")
            lines.append("|---|---|---|---|")
            for _, row in top5.iterrows():
                cid = str(row.get("customer_id", "N/A"))[:20]
                freq = row.get("frequency", row.get("order_count", "N/A"))
                lines.append(f"| {cid} | Rs{row['clv']:,.0f} | {row['clv_tier']} | {freq} |")
            champions_pct = (result["clv_tier"] == "Champions").mean() * 100
            lines.append(f"\n{champions_pct:.1f}% of customers are Champions (top CLV tier).")
            summary = "\n".join(lines)
            self._store(summary, "clv", {"customer_count": len(result)})
            return summary
        except Exception as e:
            logger.warning("clv tool failed: %s", e)
            return f"[clv] Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool: Anomaly Detection
    # ------------------------------------------------------------------

    def run_anomaly(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Run anomaly detection and return structured text summary."""
        try:
            from modules.anomaly import anomaly_report
            result = anomaly_report(df)
            if result.empty:
                return "[anomaly] No anomaly results."
            n_total = len(result)
            n_confirmed = int(result["confirmed_anomaly"].sum())
            pct = n_confirmed / n_total * 100 if n_total else 0
            lines = ["[anomaly] Anomaly Detection Results\n"]
            lines.append(f"Total orders analysed: {n_total:,}")
            lines.append(f"Confirmed anomalies (>=2 detectors agree): {n_confirmed:,} ({pct:.2f}%)")
            iso_ct = int(result["is_anomaly"].sum())
            z_ct   = int(result["zscore_anomaly"].sum())
            db_ct  = int(result["dbscan_anomaly"].sum())
            lines.append(f"  Isolation Forest flags: {iso_ct:,}")
            lines.append(f"  Z-score flags: {z_ct:,}")
            lines.append(f"  DBSCAN flags: {db_ct:,}")
            if result["gnn_risk_score"].notna().any():
                high_risk = result["gnn_risk_score"] >= 0.5
                lines.append(f"  R-GCN high-risk orders: {int(high_risk.sum()):,}")
                lines.append(
                    f"  Mean RTO probability: {result['rto_probability'].mean():.1%}; "
                    f"mean promotional fraud risk: {result['promo_fraud_risk'].mean():.1%}"
                )
                if "buyer_id" in result:
                    buyers = (result.groupby("buyer_id", dropna=False)[["rto_probability", "promo_fraud_risk"]]
                              .max().max(axis=1).sort_values(ascending=False).head(5))
                    lines.append("\nHighest Graph Risk Buyers:")
                    for buyer, score in buyers.items():
                        lines.append(f"  {buyer}: {score:.1%}")
            cat_anom = (
                result[result["confirmed_anomaly"]]
                .groupby("category")
                .size()
                .sort_values(ascending=False)
            )
            if not cat_anom.empty:
                lines.append("\nAnomalies by Category:")
                for cat, cnt in cat_anom.items():
                    lines.append(f"  {cat}: {cnt} anomalies")
            zone_anom = (
                result[result["confirmed_anomaly"]]
                .groupby("zone")
                .size()
                .sort_values(ascending=False)
            )
            if not zone_anom.empty:
                lines.append("\nAnomalies by Zone:")
                for zone, cnt in zone_anom.items():
                    lines.append(f"  {zone}: {cnt} anomalies")
            anom_rows = result[result["confirmed_anomaly"]]
            if not anom_rows.empty:
                avg_disc = anom_rows["discount_percent"].mean()
                avg_rev  = anom_rows["revenue"].mean()
                norm_disc = result[~result["confirmed_anomaly"]]["discount_percent"].mean()
                norm_rev  = result[~result["confirmed_anomaly"]]["revenue"].mean()
                lines.append(f"\nAnomaly vs Normal Order Comparison:")
                lines.append(f"  Anomaly avg discount: {avg_disc:.1f}% vs normal {norm_disc:.1f}%")
                lines.append(f"  Anomaly avg revenue: Rs{avg_rev:,.0f} vs normal Rs{norm_rev:,.0f}")
            summary = "\n".join(lines)
            self._store(summary, "anomaly", {"anomaly_count": n_confirmed})
            return summary
        except Exception as e:
            logger.warning("anomaly tool failed: %s", e)
            return f"[anomaly] Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool: Executive Insights (stub + real)
    # ------------------------------------------------------------------

    def run_insights(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Run executive summary insights."""
        try:
            from modules.insights import executive_summary
            result = executive_summary(df)
            lines = ["[insights] Executive Summary\n"]
            lines.append(f"Headline: {result['headline']}")
            lines.append("\nKPIs:")
            for k, v in result["kpis"].items():
                lines.append(f"  {k}: {v}")
            lines.append("\nTop Insights:")
            for ins in result["top_insights"]:
                lines.append(f"  {ins}")
            lines.append("\nRisks:")
            for r in result["risks"]:
                lines.append(f"  {r}")
            lines.append("\nOpportunities:")
            for o in result["opportunities"]:
                lines.append(f"  {o}")
            summary = "\n".join(lines)
            self._store(summary, "insights", {})
            return summary
        except Exception as e:
            logger.warning("insights tool failed: %s", e)
            return f"[insights] Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool: Inventory Alerts
    # ------------------------------------------------------------------

    def run_inventory_alerts(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Run inventory alerts."""
        try:
            from modules.inventory_alerts import compute_alerts
            result = compute_alerts(df)
            if result.empty:
                return "[inventory_alerts] No inventory data."
            lines = ["[inventory_alerts] Inventory Alert System\n"]
            level_counts = result["alert_level"].value_counts()
            for level, cnt in level_counts.items():
                lines.append(f"  {level}: {cnt} category-zone groups")
            critical = result[result["alert_level"].str.contains("CRITICAL|HIGH", na=False)]
            if not critical.empty:
                lines.append(f"\n{len(critical)} groups need immediate attention:")
                lines.append("| Category | Zone | Alert | Recommendation |")
                lines.append("|---|---|---|---|")
                for _, row in critical.head(10).iterrows():
                    lines.append(f"| {row['category']} | {row['zone']} | {row['alert_level']} | {row['recommendation'][:60]} |")
            summary = "\n".join(lines)
            self._store(summary, "inventory_alerts", {})
            return summary
        except Exception as e:
            logger.warning("inventory_alerts tool failed: %s", e)
            return f"[inventory_alerts] Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool: Price Elasticity
    # ------------------------------------------------------------------

    def run_price_elasticity(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Run price elasticity analysis."""
        try:
            from modules.price_elasticity import compute_elasticity
            result = compute_elasticity(df, group_cols=["category"])
            if result.empty:
                return "[price_elasticity] Insufficient data for elasticity analysis."
            lines = ["[price_elasticity] Price Elasticity by Category\n"]
            lines.append("| Category | Elasticity | R2 | Demand Type |")
            lines.append("|---|---|---|---|")
            for _, row in result.iterrows():
                e = row["elasticity"]
                demand_type = "Elastic" if e < -1 else ("Inelastic" if e < 0 else "Giffen/Luxury")
                lines.append(f"| {row['category']} | {e:.3f} | {row['r2']:.3f} | {demand_type} |")
            elastic_cats = result[result["elasticity"] < -1]["category"].tolist()
            inelastic_cats = result[(result["elasticity"] >= -1) & (result["elasticity"] < 0)]["category"].tolist()
            if elastic_cats:
                lines.append(f"\nElastic categories (discounts drive volume): {', '.join(elastic_cats)}")
            if inelastic_cats:
                lines.append(f"Inelastic categories (protect margins): {', '.join(inelastic_cats)}")
            summary = "\n".join(lines)
            self._store(summary, "price_elasticity", {})
            return summary
        except Exception as e:
            logger.warning("price_elasticity tool failed: %s", e)
            return f"[price_elasticity] Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool: Cohort Analysis
    # ------------------------------------------------------------------

    def run_cohort(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Run cohort retention analysis."""
        try:
            from modules.cohort import build_cohort_table
            pivot = build_cohort_table(df, metric="count")
            if pivot.empty:
                return "[cohort] Not enough data for cohort analysis."
            retention = (pivot.div(pivot[0], axis=0) * 100).round(1)
            lines = ["[cohort] Cohort Retention Analysis\n"]
            lines.append(f"Cohorts analysed: {len(retention)}")
            avg_m1 = retention[1].mean() if 1 in retention.columns else None
            avg_m3 = retention[3].mean() if 3 in retention.columns else None
            avg_m6 = retention[6].mean() if 6 in retention.columns else None
            if avg_m1 is not None:
                lines.append(f"Average 1-month retention: {avg_m1:.1f}%")
            if avg_m3 is not None:
                lines.append(f"Average 3-month retention: {avg_m3:.1f}%")
            if avg_m6 is not None:
                lines.append(f"Average 6-month retention: {avg_m6:.1f}%")
            recent = retention.tail(3)
            lines.append("\nRecent cohort retention rates (%):")
            display_cols = [c for c in [0, 1, 2, 3, 6] if c in recent.columns]
            lines.append("| Cohort | " + " | ".join(f"Month {c}" for c in display_cols) + " |")
            lines.append("|---|" + "---|" * len(display_cols))
            for cohort, row in recent.iterrows():
                vals = " | ".join(f"{row.get(c, 'N/A'):.1f}%" if pd.notna(row.get(c)) else "N/A" for c in display_cols)
                lines.append(f"| {cohort} | {vals} |")
            summary = "\n".join(lines)
            self._store(summary, "cohort", {})
            return summary
        except Exception as e:
            logger.warning("cohort tool failed: %s", e)
            return f"[cohort] Tool error: {e}"

    # ------------------------------------------------------------------
    # Tool: Time Series Trends
    # ------------------------------------------------------------------

    def run_time_series(self, df: pd.DataFrame, params: dict | None = None) -> str:
        """Summarise monthly revenue trends from order data."""

        def _analyse(monthly: pd.Series, aov_monthly: pd.Series, lines: list) -> None:
            """Shared analysis logic given pre-built monthly revenue and AOV series."""
            if monthly.empty:
                lines.append("[time_series] No time series data available.")
                return
            lines.append(f"Data spans {monthly.index.min().strftime('%b %Y')} to {monthly.index.max().strftime('%b %Y')}")
            lines.append(f"Peak revenue month: {monthly.idxmax().strftime('%b %Y')} (Rs{monthly.max():,.0f})")
            lines.append(f"Lowest revenue month: {monthly.idxmin().strftime('%b %Y')} (Rs{monthly.min():,.0f})")
            if len(monthly) >= 2:
                mom = (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100
                lines.append(f"Latest MoM revenue change: {mom:+.1f}%")
            if len(monthly) >= 3:
                recent_3 = monthly.tail(3)
                trend = (recent_3.iloc[-1] - recent_3.iloc[0]) / recent_3.iloc[0] * 100
                lines.append(f"3-month revenue trend: {trend:+.1f}%")
            lines.append(f"Latest monthly AOV: Rs{aov_monthly.iloc[-1]:,.0f}")
            if len(aov_monthly) >= 2:
                aov_mom = (aov_monthly.iloc[-1] - aov_monthly.iloc[-2]) / aov_monthly.iloc[-2] * 100
                lines.append(f"AOV MoM change: {aov_mom:+.1f}%")

        def _zone_breakdown(df: pd.DataFrame, lines: list) -> None:
            """Append zone-level MoM breakdown to lines."""
            zone_monthly = df.groupby([df["order_date"].dt.to_period("M"), "zone"])["revenue"].sum().unstack()
            zone_monthly.index = zone_monthly.index.to_timestamp()
            if not zone_monthly.empty and len(zone_monthly) >= 2:
                lines.append("\nZone Revenue Trend (last 2 months):")
                for zone in zone_monthly.columns:
                    vals = zone_monthly[zone].dropna()
                    if len(vals) >= 2:
                        z_mom = (vals.iloc[-1] - vals.iloc[-2]) / vals.iloc[-2] * 100
                        lines.append(f"  {zone}: {z_mom:+.1f}% MoM (latest Rs{vals.iloc[-1]:,.0f})")

        lines = ["[time_series] Revenue Trend Analysis\n"]
        try:
            # Lazy import inside try to avoid circular imports
            from modules.time_series import _monthly  # noqa: PLC0415

            monthly = _monthly(df, col="revenue", agg="sum")
            aov_monthly = _monthly(df, col="revenue", agg="mean")
            if monthly.empty:
                return "[time_series] No time series data available."
            _analyse(monthly, aov_monthly, lines)
            _zone_breakdown(df, lines)
            summary = "\n".join(lines)
            self._store(summary, "time_series", {})
            return summary
        except Exception as e:
            logger.warning("time_series _monthly helper failed (%s); falling back to inline implementation", e)
            # Fallback: inline groupby implementation
            try:
                lines = ["[time_series] Revenue Trend Analysis\n"]
                monthly = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum()
                monthly.index = monthly.index.to_timestamp()
                aov_monthly = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].mean()
                aov_monthly.index = aov_monthly.index.to_timestamp()
                if monthly.empty:
                    return "[time_series] No time series data available."
                _analyse(monthly, aov_monthly, lines)
                _zone_breakdown(df, lines)
                summary = "\n".join(lines)
                self._store(summary, "time_series", {})
                return summary
            except Exception as e2:
                logger.warning("time_series tool failed: %s", e2)
                return f"[time_series] Tool error: {e2}"

    # ------------------------------------------------------------------
    # Planning: keyword  tools
    # ------------------------------------------------------------------

    def _plan_tools(self, question: str) -> list[str]:
        """Map question keywords to tool names. Returns 1-3 tool names."""
        q_lower = question.lower()
        scores: dict[str, int] = {}
        for tool, keywords in _KEYWORD_MAP.items():
            for kw in keywords:
                if kw in q_lower:
                    scores[tool] = scores.get(tool, 0) + 1
        if not scores:
            # Default: insights + time_series for general questions
            return ["insights", "time_series"]
        # Return top 3 by score, max 3
        ranked = sorted(scores, key=lambda t: -scores[t])
        return ranked[:3]

    # ------------------------------------------------------------------
    # Main answer method
    # ------------------------------------------------------------------

    @_traced_answer
    def answer(
        self,
        question: str,
        df: pd.DataFrame,
        filters: dict | None = None,
    ) -> dict:
        """
        Answer a natural-language merchant question.

        Returns
        -------
        dict with keys:
            answer       : str  synthesised answer (or demo-mode concatenation)
            sources      : list[str]  tool names used
            tool_outputs : dict  {tool_name: raw_summary}
            chunks_used  : list[str]  RAG chunks fed to LLM
        """
        # 1. Plan
        with _span("copilot.intent.plan") as span:
            planned_tools = self._plan_tools(question)
            if span:
                span.set_attribute("copilot.planned_tool_count", len(planned_tools))
                span.set_attribute("copilot.planned_tools", planned_tools)
        logger.info("Copilot planned tools: %s", planned_tools)

        # 2. Execute tools
        tool_outputs: dict[str, str] = {}
        tool_fn_map = {
            "price_optimizer":  self.run_price_optimizer,
            "at_risk":          self.run_at_risk,
            "clv":              self.run_clv,
            "anomaly":          self.run_anomaly,
            "insights":         self.run_insights,
            "inventory_alerts": self.run_inventory_alerts,
            "price_elasticity": self.run_price_elasticity,
            "cohort":           self.run_cohort,
            "time_series":      self.run_time_series,
        }
        for tool_name in planned_tools:
            fn = tool_fn_map.get(tool_name)
            if fn:
                span_name = "copilot.gnn.inference" if tool_name == "anomaly" else "copilot.analytics.tool"
                with _span(span_name, **{"copilot.tool_name": tool_name}):
                    tool_outputs[tool_name] = fn(df, filters or {})

        # 3. Retrieve relevant chunks from RAG
        chunks = self._retrieve(question, top_k=3)
        chunks_text = [c["chunk"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]
        sources = list(tool_outputs.keys())

        # 4. Synthesise answer
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # Demo mode: concatenate summaries with a disclaimer
            raw = "\n\n".join(tool_outputs.values())
            demo_answer = (
                "**[DEMO MODE - Set OPENAI_API_KEY for AI-generated answers]**\n\n"
                + raw
                + ("\n\nEvidence: " + " ".join(f"[chunk:{chunk_id}]" for chunk_id in chunk_ids) if chunk_ids else "")
            )
            return {
                "answer":       demo_answer,
                "sources":      sources,
                "tool_outputs": tool_outputs,
                "chunks_used":  chunks_text,
                "chunk_ids":    chunk_ids,
                "state":        "demo",
            }

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)

            context_block = "\n\n".join(
                f"[chunk:{c['chunk_id']}] [{c['source']}]\n{c['chunk']}" for c in chunks
            )
            tools_block = "\n\n".join(
                f"[{name}]\n{out}" for name, out in tool_outputs.items()
            )

            system_prompt = (
                "You are a merchant analytics assistant for an Indian e-commerce platform. "
                "Answer merchant questions precisely and concisely, citing data sources inline "
                "using [source_name] notation and cite at least one retrieved vector chunk exactly "
                "as [chunk:CHUNK_ID]. Never invent chunk IDs. "
                "Use verified price_optimizer results for every discount recommendation. "
                "Do not recommend adjustments that violate the Lerner-index result. "
                "Never suggest a discount above the hard 25% cap, even if an optimizer proposes it. "
                "If the user message includes an Approved compliance policy context, treat its exact rules "
                "as binding; do not invent thresholds or broaden their applicability. If it does not cover "
                "the requested institution or transaction type, say the configured policy does not cover it. "
                "Use Indian currency (Rs) and Indian business context. "
                "Focus on actionable insights. Keep the answer under 300 words."
            )
            user_prompt = f"""Question: {question}

Fresh analytics data from our tools:
{tools_block}

Additional context from analytics history:
{context_block}

Answer the question with specific numbers, cite sources inline, and cite retrieved chunks as [chunk:CHUNK_ID]."""

            try:
                with _span("copilot.llm.generate", **{"llm.provider": "openai", "llm.model": "gpt-4o-mini"}):
                    response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=600,
                    )
            except openai.NotFoundError:
                with _span("copilot.llm.generate", **{"llm.provider": "openai", "llm.model": "gpt-3.5-turbo"}):
                    response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=600,
                    )

            answer_text = response.choices[0].message.content
            verified = []
            verified_by_category = {}
            if "price_optimizer" in tool_outputs:
                for line in tool_outputs["price_optimizer"].splitlines():
                    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                    if len(cells) >= 3:
                        match = re.fullmatch(r"(\d+(?:\.\d+)?)%?", cells[2])
                        if match and cells[0].lower() not in {"category", "---"}:
                            optimal = float(match.group(1))
                            verified.append(optimal)
                            verified_by_category[cells[0]] = optimal
            guard_metadata = {"chunk_ids": chunk_ids, "verified_optimal_discounts": verified,
                              "verified_by_category": verified_by_category}
            with _span("copilot.output.validate", **{"guardrails.enabled": True}) as span:
                valid, failure_reason = _guardrails_validate(answer_text or "", guard_metadata)
                if span:
                    span.set_attribute("guardrails.passed", valid)
                    if failure_reason:
                        span.set_attribute("guardrails.failure_reason", failure_reason)
            if not valid:
                logger.warning("Copilot output rejected by guardrails: %s", failure_reason)
                raw = "\n\n".join(tool_outputs.values())
                fallback_answer = (f"**[SAFE FALLBACK: generated answer failed validation]**\n\n{raw}"
                                   + ("\n\nEvidence: " + " ".join(f"[chunk:{chunk_id}]" for chunk_id in chunk_ids)
                                      if chunk_ids else "\n\nNo retrieved evidence was available."))
                return {
                    "answer": fallback_answer,
                    "sources": sources, "tool_outputs": tool_outputs,
                    "chunks_used": chunks_text, "chunk_ids": chunk_ids,
                    "state": "fallback", "fallback_reason": failure_reason,
                }
            return {
                "answer":       answer_text,
                "sources":      sources,
                "tool_outputs": tool_outputs,
                "chunks_used":  chunks_text,
                "chunk_ids":    chunk_ids,
                "state":        "answered",
            }

        except Exception as e:
            logger.error("LLM call failed: %s", e)
            # Fall back to demo mode on LLM error
            raw = "\n\n".join(tool_outputs.values())
            fallback_answer = f"**[LLM error]**\n\n{raw}"
            if chunk_ids:
                fallback_answer += "\n\nEvidence: " + " ".join(f"[chunk:{chunk_id}]" for chunk_id in chunk_ids)
            return {
                "answer":       fallback_answer,
                "sources":      sources,
                "tool_outputs": tool_outputs,
                "chunks_used":  chunks_text,
                "chunk_ids":    chunk_ids,
                "state":        "fallback",
                "fallback_reason": str(e),
            }
