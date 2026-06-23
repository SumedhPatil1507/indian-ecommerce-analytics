"""
api/main.py
FastAPI backend wrapper exposing analytics, webhooks, and Celery worker operations.
"""
from __future__ import annotations
import os
import io
import time
import hmac
import base64
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, Any
from collections import deque

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Header, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from celery.result import AsyncResult

# Path resolution
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import core.config as cfg
import core.database as db
from data.loader import load_any
from data.connectors import from_shopify_webhook, from_amazon_orders, from_woocommerce
from worker.tasks import (
    ingest_orders_task,
    recalculate_clv_task,
    recalculate_anomaly_task,
    recalculate_pricing_task,
    recalculate_drift_task
)

# OpenTelemetry Tracing Setup
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("indiacommerce-api")

logger = logging.getLogger(__name__)

# FastAPI Application
app = FastAPI(
    title="IndiaCommerce Analytics REST Engine",
    description="Multi-tenant, secure, relational e-commerce analytical engine.",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument the app
FastAPIInstrumentor.instrument_app(app)

# Local In-Memory Telemetry Queue (stores last 1000 requests for internal dashboard)
TELEMETRY_LOGS = deque(maxlen=1000)

@app.middleware("http")
async def collect_telemetry(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Log HTTP details (skip telemetry endpoint itself to avoid polling noise)
    if not request.url.path.endswith("/telemetry"):
        TELEMETRY_LOGS.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(duration * 1000, 2),
        })
    return response


# --- Authentication Dependency ---

async def get_current_tenant(authorization: Optional[str] = Header(None)) -> dict:
    """Dependency to validate JWT auth header and resolve tenant details."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
    token = authorization.split(" ")[1]
    
    client = db.get_client()
    if not client:
        raise HTTPException(status_code=503, detail="Supabase not configured")
    try:
        # Get user profiles or user session details using JWT token
        res = client.auth.get_user(token)
        if not res or not res.user:
            raise HTTPException(status_code=401, detail="Invalid token session")
        return {
            "tenant_id": res.user.id,
            "email": res.user.email,
            "token": token
        }
    except Exception as e:
        logger.warning("JWT validation failed: %s", e)
        raise HTTPException(status_code=401, detail=f"JWT verification failed: {str(e)}")


# --- Public / Health Routes ---

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# --- Authentication Endpoints ---

class AuthCredentials(BaseModel):
    email: str
    password: str

@app.post("/auth/signup", tags=["Auth"])
async def auth_signup(creds: AuthCredentials):
    client = db.get_client()
    if not client:
        raise HTTPException(503, "Supabase connection unavailable")
    try:
        res = client.auth.sign_up({"email": creds.email, "password": creds.password})
        if res.user:
            # Create standard tenant profile record
            import secrets
            secret = secrets.token_hex(16)
            client.table("profiles").insert({
                "id": res.user.id,
                "email": creds.email,
                "plan": "starter",
                "webhook_secret": secret
            }).execute()
            return {"status": "success", "message": "Sign up completed. Please verify email or log in."}
        raise HTTPException(400, "Sign up failed")
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/auth/login", tags=["Auth"])
async def auth_login(creds: AuthCredentials):
    client = db.get_client()
    if not client:
        raise HTTPException(503, "Supabase connection unavailable")
    try:
        res = client.auth.sign_in_with_password({"email": creds.email, "password": creds.password})
        if res.session:
            return {
                "access_token": res.session.access_token,
                "token_type": "bearer",
                "expires_in": res.session.expires_in,
                "user_id": res.user.id,
                "email": res.user.email,
            }
        raise HTTPException(401, "Sign in failed")
    except Exception as e:
        raise HTTPException(401, str(e))


# --- Cryptographic Webhook Ingestion ---

@app.post("/ingest/shopify", tags=["Ingestion"])
async def ingest_shopify(
    request: Request,
    tenant_id: str = Query(..., description="Tenant ID"),
    signature: str = Header(..., alias="X-Shopify-Hmac-SHA256")
):
    """Cryptographically gated Shopify webhook ingestion."""
    body = await request.body()
    
    # Retrieve tenant profile
    profile = db.get_profile(tenant_id)
    if not profile:
        raise HTTPException(401, "Invalid tenant_id")
    secret = profile.get("webhook_secret") or "global_secret"
    
    # Verify signature
    computed = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    if not hmac.compare_digest(computed, signature):
        raise HTTPException(401, "Cryptographic webhook verification failed")

    try:
        payload = request.json()
        df = from_shopify_webhook(payload)
        task = ingest_orders_task.delay(tenant_id, df.to_dict(orient="records"), "shopify_webhook")
        return {"status": "queued", "task_id": task.id}
    except Exception as e:
        raise HTTPException(400, f"Failed parsing payload: {str(e)}")


@app.post("/ingest/woocommerce", tags=["Ingestion"])
async def ingest_woocommerce(
    request: Request,
    tenant_id: str = Query(..., description="Tenant ID"),
    signature: str = Header(..., alias="X-WC-Webhook-Signature")
):
    """Cryptographically gated WooCommerce webhook ingestion."""
    body = await request.body()
    
    # Retrieve tenant profile
    profile = db.get_profile(tenant_id)
    if not profile:
        raise HTTPException(401, "Invalid tenant_id")
    secret = profile.get("webhook_secret") or "global_secret"
    
    # Verify signature
    computed = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    if not hmac.compare_digest(computed, signature):
        raise HTTPException(401, "Cryptographic webhook verification failed")

    try:
        payload = request.json()
        df = from_woocommerce(payload)
        task = ingest_orders_task.delay(tenant_id, df.to_dict(orient="records"), "woocommerce_webhook")
        return {"status": "queued", "task_id": task.id}
    except Exception as e:
        raise HTTPException(400, f"Failed parsing payload: {str(e)}")


@app.post("/ingest/amazon", tags=["Ingestion"])
async def ingest_amazon(
    request: Request,
    tenant_id: str = Query(..., description="Tenant ID"),
    signature: str = Header(..., alias="X-Amazon-Webhook-Signature")
):
    """Cryptographically gated Amazon webhook ingestion."""
    body = await request.body()
    
    # Retrieve tenant profile
    profile = db.get_profile(tenant_id)
    if not profile:
        raise HTTPException(401, "Invalid tenant_id")
    secret = profile.get("webhook_secret") or "global_secret"
    
    # Verify signature
    computed = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
    if not hmac.compare_digest(computed, signature):
        raise HTTPException(401, "Cryptographic webhook verification failed")

    try:
        payload = request.json()
        df = from_amazon_orders(payload)
        task = ingest_orders_task.delay(tenant_id, df.to_dict(orient="records"), "amazon_webhook")
        return {"status": "queued", "task_id": task.id}
    except Exception as e:
        raise HTTPException(400, f"Failed parsing payload: {str(e)}")


# --- Multi-Tenant Data Ingestion & Tasks API ---

@app.post("/orders/upload", tags=["Ingestion"])
async def upload_orders(
    file: UploadFile = File(...),
    tenant: dict = Depends(get_current_tenant)
):
    """Secure multi-format order upload endpoint."""
    contents = await file.read()
    try:
        # Load and validate file contents
        df = load_any(contents, file.filename)
        
        # Dispatch background ingestion task to Celery
        task = ingest_orders_task.delay(tenant["tenant_id"], df.to_dict(orient="records"), file.filename)
        return {
            "status": "queued",
            "task_id": task.id,
            "message": f"Dataset file {file.filename} queued for ingestion."
        }
    except Exception as e:
        raise HTTPException(400, f"File parsing failed: {str(e)}")


@app.get("/tasks/status/{task_id}", tags=["Tasks"])
async def get_task_status(task_id: str):
    """Check task completion status in Celery."""
    res = AsyncResult(task_id, app=ingest_orders_task.app)
    return {
        "task_id": task_id,
        "status": res.state,
        "result": res.result if res.ready() else None
    }


# --- Profile Endpoint ---

@app.get("/profile", tags=["Tenant"])
async def get_profile(tenant: dict = Depends(get_current_tenant)):
    profile = db.get_profile(tenant["tenant_id"])
    if not profile:
        raise HTTPException(404, "Profile not found")
    return profile


# --- Relational Analytics Queries ---

@app.get("/analytics/kpis", tags=["Analytics"])
async def get_kpis(tenant: dict = Depends(get_current_tenant)):
    """Fetch aggregated KPIs directly from PostgreSQL."""
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return {"total_revenue": 0.0, "total_orders": 0, "aov": 0.0, "avg_discount": 0.0, "units_sold": 0.0}
    
    rev_sum = float(df["revenue"].sum())
    total_orders = int(len(df))
    aov = float(df["revenue"].mean()) if total_orders > 0 else 0.0
    avg_discount = float(df["discount_percent"].mean()) if total_orders > 0 else 0.0
    avg_units = float(df["units_sold"].mean()) if total_orders > 0 else 0.0
    
    return {
        "total_revenue": rev_sum,
        "total_orders": total_orders,
        "aov": aov,
        "avg_discount": avg_discount,
        "avg_units_sold": avg_units
    }


@app.get("/analytics/executive-summary", tags=["Analytics"])
async def get_executive_summary(tenant: dict = Depends(get_current_tenant)):
    """Exposes structured executive summary recommendations."""
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return {"headline": "No transaction data uploaded yet.", "kpis": {}, "top_insights": [], "risks": [], "opportunities": []}
    
    from modules.insights import executive_summary, generate_recommendations
    # Fetch exchange rate for dollar conversions
    from data.loader import fetch_usd_inr
    fx = fetch_usd_inr()
    
    summary = executive_summary(df, fx)
    recs = generate_recommendations(df)
    return {"summary": summary, "recommendations": recs}


@app.get("/analytics/price-optimizer", tags=["Analytics"])
async def get_price_optimizer(tenant: dict = Depends(get_current_tenant)):
    """Load price recommendations from Supabase cache."""
    recs_df = db.load_price_recommendations(tenant["tenant_id"], token=tenant["token"])
    return recs_df.to_dict(orient="records")


@app.post("/analytics/price-optimizer/recalculate", tags=["Analytics"])
async def trigger_price_recalculation(tenant: dict = Depends(get_current_tenant)):
    """Queues a new Celery pricing optimization worker job."""
    task = recalculate_pricing_task.delay(tenant["tenant_id"])
    return {"status": "queued", "task_id": task.id}


@app.get("/analytics/at-risk", tags=["Analytics"])
async def get_at_risk(tenant: dict = Depends(get_current_tenant)):
    """Load at-risk customer scores from Supabase cache."""
    alerts_df = db.load_at_risk_alerts(tenant["tenant_id"], token=tenant["token"])
    return alerts_df.to_dict(orient="records")


@app.post("/analytics/at-risk/recalculate", tags=["Analytics"])
async def trigger_at_risk_recalculation(tenant: dict = Depends(get_current_tenant)):
    """Queues a new Celery CLV / risk scoring job."""
    task = recalculate_clv_task.delay(tenant["tenant_id"])
    return {"status": "queued", "task_id": task.id}


@app.get("/analytics/drift", tags=["Analytics"])
async def get_drift(tenant: dict = Depends(get_current_tenant)):
    """Load data drift monitoring from Supabase cache."""
    report = db.load_drift_report(tenant["tenant_id"], token=tenant["token"])
    if not report:
        return {"status": "not_computed"}
    return report


@app.post("/analytics/drift/recalculate", tags=["Analytics"])
async def trigger_drift_recalculation(
    ref_months: int = 6,
    cur_months: int = 3,
    tenant: dict = Depends(get_current_tenant)
):
    """Queues a new Celery drift scoring task."""
    task = recalculate_drift_task.delay(tenant["tenant_id"], ref_months, cur_months)
    return {"status": "queued", "task_id": task.id}


@app.get("/analytics/clv", tags=["Analytics"])
async def get_clv(tenant: dict = Depends(get_current_tenant)):
    """Load BG/NBD CLV tiers from Supabase cache."""
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return {"status": "no_data"}
    clv_df = db.load_clv_cache(tenant["tenant_id"], df, token=tenant["token"])
    if clv_df is None:
        return {"status": "not_computed"}
    return clv_df.to_dict(orient="records")


@app.post("/analytics/clv/recalculate", tags=["Analytics"])
async def trigger_clv_recalculation(tenant: dict = Depends(get_current_tenant)):
    """Queues a new Celery CLV computation task."""
    task = recalculate_clv_task.delay(tenant["tenant_id"])
    return {"status": "queued", "task_id": task.id}


@app.get("/analytics/anomalies", tags=["Analytics"])
async def get_anomalies(tenant: dict = Depends(get_current_tenant)):
    """Load weekly anomalies from Supabase cache."""
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return {"status": "no_data"}
    anom_cached = db.load_anomaly_cache(tenant["tenant_id"], df, token=tenant["token"])
    if anom_cached is None:
        return {"status": "not_computed"}
    return anom_cached.to_dict(orient="records")


@app.post("/analytics/anomalies/recalculate", tags=["Analytics"])
async def trigger_anomaly_recalculation(tenant: dict = Depends(get_current_tenant)):
    """Queues a new Celery anomaly detection task."""
    task = recalculate_anomaly_task.delay(tenant["tenant_id"])
    return {"status": "queued", "task_id": task.id}


# --- Operational Actions Endpoints ---

class LogActionInput(BaseModel):
    action_type: str
    payload: dict
    status: str = "pending"
    source: str = "dashboard"

@app.post("/operational-actions", tags=["Operational Actions"])
async def create_operational_action(
    action: LogActionInput,
    tenant: dict = Depends(get_current_tenant)
):
    success = db.log_action(
        tenant_id=tenant["tenant_id"],
        action_type=action.action_type,
        payload=action.payload,
        status=action.status,
        source=action.source,
        token=tenant["token"]
    )
    if not success:
        raise HTTPException(500, "Failed to log operational action")
    return {"status": "success"}


@app.get("/operational-actions/pending", tags=["Operational Actions"])
async def fetch_pending_actions(
    action_type: Optional[str] = None,
    tenant: dict = Depends(get_current_tenant)
):
    actions = db.get_pending_actions(
        tenant_id=tenant["tenant_id"],
        action_type=action_type,
        token=tenant["token"]
    )
    return actions


class UpdateActionInput(BaseModel):
    action_id: int
    status: str

@app.post("/operational-actions/update", tags=["Operational Actions"])
async def modify_action_status(
    action: UpdateActionInput,
    tenant: dict = Depends(get_current_tenant)
):
    success = db.update_action_status(
        action_id=action.action_id,
        status=action.status,
        token=tenant["token"]
    )
    if not success:
        raise HTTPException(500, "Failed to update action status")
    return {"status": "success"}


# --- Export Endpoints ---

@app.get("/analytics/export/excel", tags=["Export"])
async def export_excel_report(tenant: dict = Depends(get_current_tenant)):
    """Generates Excel spreadsheet of analytics report on the fly."""
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        raise HTTPException(400, "No orders available to export")
    
    from modules.insights import executive_summary
    from modules.export import to_excel
    from data.loader import fetch_usd_inr
    fx = fetch_usd_inr()
    summary = executive_summary(df, fx)
    
    excel_bytes = to_excel(df, summary)
    
    from fastapi.responses import Response
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=report.xlsx"}
    )


@app.get("/analytics/export/pdf", tags=["Export"])
async def export_pdf_report(tenant: dict = Depends(get_current_tenant)):
    """Generates PDF analytics report on the fly."""
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        raise HTTPException(400, "No orders available to export")
    
    from modules.insights import executive_summary, generate_recommendations
    from modules.export import to_pdf
    from data.loader import fetch_usd_inr
    fx = fetch_usd_inr()
    summary = executive_summary(df, fx)
    recs = generate_recommendations(df)
    
    pdf_bytes = to_pdf(summary, recs)
    
    from fastapi.responses import Response
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=report.pdf"}
    )


# --- Fast Relational Chart Feeds ---

@app.get("/analytics/charts/revenue-trends", tags=["Analytics"])
async def get_revenue_trends(tenant: dict = Depends(get_current_tenant)):
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return []
    # Monthly aggregation
    df["year_month"] = pd.to_datetime(df["order_date"]).dt.to_period("M").astype(str)
    
    # 1. Total monthly revenue
    m = df.groupby("year_month")["revenue"].sum().reset_index()
    # 2. Average discount
    avg_disc = df.groupby("year_month")["discount_percent"].mean().reset_index()
    # 3. Zone monthly revenue
    zone_rev = df.groupby(["year_month", "zone"])["revenue"].sum().reset_index()
    # 4. Brand monthly revenue
    brand_rev = df.groupby(["year_month", "brand_type"])["revenue"].sum().reset_index()

    return {
        "monthly_revenue": m.to_dict(orient="records"),
        "monthly_discount": avg_disc.to_dict(orient="records"),
        "zone_revenue": zone_rev.to_dict(orient="records"),
        "brand_revenue": brand_rev.to_dict(orient="records")
    }


@app.get("/analytics/charts/category-analysis", tags=["Analytics"])
async def get_category_analysis(tenant: dict = Depends(get_current_tenant)):
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return {}
    cat_rev = df.groupby("category")["revenue"].sum().reset_index()
    brand_rev = df.groupby("brand_type")["revenue"].sum().reset_index()
    
    df["year_month"] = pd.to_datetime(df["order_date"]).dt.to_period("M").astype(str)
    event_rev = df.groupby(["year_month", "sales_event"])["revenue"].sum().reset_index()
    
    return {
        "category_revenue": cat_rev.to_dict(orient="records"),
        "brand_revenue": brand_rev.to_dict(orient="records"),
        "event_revenue": event_rev.to_dict(orient="records")
    }


@app.get("/analytics/charts/regional-analysis", tags=["Analytics"])
async def get_regional_analysis(tenant: dict = Depends(get_current_tenant)):
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return {}
    state_rev = df.groupby("state")["revenue"].sum().nlargest(15).reset_index()
    zone_rev = df.groupby("zone")["revenue"].sum().reset_index()
    zone_units = df.groupby("zone")["units_sold"].mean().reset_index()
    
    return {
        "state_revenue": state_rev.to_dict(orient="records"),
        "zone_revenue": zone_rev.to_dict(orient="records"),
        "zone_units": zone_units.to_dict(orient="records")
    }


@app.get("/analytics/charts/inventory-alerts", tags=["Analytics"])
async def get_inventory_alerts(tenant: dict = Depends(get_current_tenant)):
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return []
    from modules.inventory_alerts import compute_alerts
    alerts = compute_alerts(df)
    return alerts.to_dict(orient="records")


@app.get("/analytics/charts/cohort-analysis", tags=["Analytics"])
async def get_cohort_analysis(tenant: dict = Depends(get_current_tenant)):
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return {}
    from modules.cohort import build_cohort_table
    pivot_count = build_cohort_table(df, metric="count")
    pivot_revenue = build_cohort_table(df, metric="revenue")
    
    # Fill NAs for json compliance
    pivot_count = pivot_count.fillna(0.0)
    pivot_revenue = pivot_revenue.fillna(0.0)
    
    return {
        "cohort_counts": {
            "index": list(pivot_count.index.astype(str)),
            "columns": list(pivot_count.columns.astype(str)),
            "values": pivot_count.values.tolist()
        },
        "cohort_revenue": {
            "index": list(pivot_revenue.index.astype(str)),
            "columns": list(pivot_revenue.columns.astype(str)),
            "values": pivot_revenue.values.tolist()
        }
    }


@app.get("/analytics/charts/pareto-analysis", tags=["Analytics"])
async def get_pareto_analysis(tenant: dict = Depends(get_current_tenant)):
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        return {}
    
    agg = df.groupby("category")["revenue"].sum().sort_values(ascending=False).reset_index()
    agg["cum_pct"] = (agg["revenue"].cumsum() / agg["revenue"].sum() * 100).round(2)
    
    sunburst = df.groupby(["category", "zone", "brand_type"])["revenue"].sum().reset_index()
    
    # Lorenz curve stats
    vals = np.sort(df["revenue"].dropna().values)[::-1]
    cum = np.cumsum(vals) / vals.sum()
    x = np.linspace(0, 1, len(cum))
    gini = round(1 - 2 * np.trapezoid(cum, x), 3) if len(cum) > 1 else 0.0
    
    # Subsample Lorenz for fast transfer (max 500 points)
    step = max(1, len(cum) // 500)
    lorenz_points = [{"x": float(x[i]), "y": float(cum[i])} for i in range(0, len(cum), step)]

    return {
        "pareto_categories": agg.to_dict(orient="records"),
        "sunburst_revenue": sunburst.to_dict(orient="records"),
        "lorenz_gini": gini,
        "lorenz_curve": lorenz_points
    }


# --- Observability & Telemetry Endpoints ---

@app.get("/analytics/telemetry", tags=["Observability"])
async def get_telemetry():
    """Returns aggregated API performance metrics and worker status."""
    logs = list(TELEMETRY_LOGS)
    
    # Simple aggregates
    req_count = len(logs)
    avg_latency = float(np.mean([x["latency_ms"] for x in logs])) if req_count > 0 else 0.0
    err_count = sum(1 for x in logs if x["status"] >= 500)
    
    # Path distributions
    paths = {}
    for l in logs:
        p = l["path"]
        paths[p] = paths.get(p, 0) + 1
        
    # Celery tasks telemetry
    celery_status = "offline"
    try:
        from worker.celery_app import app as celery_inst
        i = celery_inst.control.inspect()
        active = i.active()
        celery_status = "active" if active is not None else "idle"
        active_count = sum(len(tasks) for tasks in active.values()) if active else 0
    except Exception:
        celery_status = "unavailable"
        active_count = 0
        
    return {
        "status": "healthy",
        "requests_total": req_count,
        "avg_latency_ms": round(avg_latency, 2),
        "error_count_5xx": err_count,
        "path_distribution": paths,
        "celery": {
            "status": celery_status,
            "active_tasks": active_count,
        },
        "system": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }


# --- Prediction API ---

class OrderInput(BaseModel):
    state: str = Field(..., example="Maharashtra")
    zone: str = Field(..., example="West")
    category: str = Field(..., example="Electronics")
    brand_type: str = Field(..., example="Premium")
    customer_gender: str = Field(..., example="Male")
    customer_age: int = Field(..., example=28)
    base_price: float = Field(..., example=15000.0)
    discount_percent: float = Field(..., example=20.0)
    sales_event: str = Field(..., example="Normal")
    competition_intensity: str = Field(..., example="High")
    inventory_pressure: str = Field(..., example="Low")
    year: int = Field(..., example=2024)
    month: int = Field(..., example=10)
    weekday: int = Field(..., example=2)


@app.post("/predict/revenue", tags=["Prediction"])
def predict_revenue(order: OrderInput, tenant: dict = Depends(get_current_tenant)):
    """Predict revenue using XGBoost model based on tenant's dataset structure."""
    df = db.load_orders(tenant["tenant_id"], token=tenant["token"])
    if df.empty:
        raise HTTPException(400, "Train dataset must be loaded before running prediction.")

    from modules.models import train_all
    output = train_all(df)
    pipe = output["pipes"].get("XGBoost")
    if pipe is None:
        raise HTTPException(500, "ML Model could not be trained.")

    row = pd.DataFrame([order.model_dump()])
    pred = float(pipe.predict(row)[0])
    return {
        "predicted_revenue_inr": round(pred, 2),
        "model": "XGBoost",
        "note": "Tenant-specific model trained dynamically on active database data.",
    }


# Start local script if run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
