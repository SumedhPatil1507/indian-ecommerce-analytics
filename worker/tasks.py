"""
worker/tasks.py
Celery background worker tasks for asynchronous execution of heavy operations.
"""
import os
import sys
import logging

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_ROOT, ".env"))
except ImportError:
    pass

import pandas as pd
from worker.celery_app import app
import core.database as db
from data.loader import _clean, _engineer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.task(name="worker.tasks.ingest_orders")
def ingest_orders_task(tenant_id: str, records: list[dict], name: str = "dataset") -> dict:
    """Task to bulk ingest orders into PostgreSQL database."""
    logger.info("Starting ingest_orders task for tenant %s with %d records", tenant_id, len(records))
    try:
        df = pd.DataFrame(records)
        if df.empty:
            return {"status": "error", "message": "Dataset is empty"}

        # Perform clean & feature engineering
        df = _engineer(_clean(df))

        # Save to database
        success = db.save_orders(tenant_id, df)
        if success:
            db.save_dataset_meta(tenant_id, name, len(df), source="upload")
            logger.info("Successfully ingested %d orders for tenant %s", len(df), tenant_id)
            return {"status": "success", "rows_ingested": len(df)}
        else:
            return {"status": "error", "message": "Failed to save orders to database"}
    except Exception as e:
        logger.exception("ingest_orders_task failed")
        return {"status": "error", "message": str(e)}


@app.task(name="worker.tasks.recalculate_clv")
def recalculate_clv_task(tenant_id: str) -> dict:
    """Compute and cache BG/NBD customer lifetime value tiers."""
    logger.info("Starting recalculate_clv for tenant %s", tenant_id)
    try:
        df = db.load_orders(tenant_id)
        if df.empty or len(df) < 10:
            return {"status": "error", "message": "Not enough data for CLV analysis"}

        from modules.clv import compute_clv
        clv_df = compute_clv(df)
        success = db.cache_clv(tenant_id, df, clv_df)
        
        # Also generate at-risk customer alerts
        from modules.at_risk import generate_at_risk_alerts
        at_risk_df = generate_at_risk_alerts(df, top_n=200)
        db.cache_at_risk_alerts(tenant_id, at_risk_df)

        db.log_action(tenant_id, "clv_computation", {"success": success}, status="completed")
        return {"status": "success", "rows": len(clv_df)}
    except Exception as e:
        logger.exception("recalculate_clv_task failed")
        return {"status": "error", "message": str(e)}


@app.task(name="worker.tasks.recalculate_anomaly")
def recalculate_anomaly_task(tenant_id: str) -> dict:
    """Detect anomalies and cache weekly results."""
    logger.info("Starting recalculate_anomaly for tenant %s", tenant_id)
    try:
        df = db.load_orders(tenant_id)
        if df.empty or len(df) < 10:
            return {"status": "error", "message": "Not enough data for anomaly detection"}

        from modules.anomaly import anomaly_report
        anom_df = anomaly_report(df)
        success = db.cache_anomaly_scores(tenant_id, df, anom_df)

        db.log_action(tenant_id, "anomaly_computation", {"success": success}, status="completed")
        return {"status": "success", "anomaly_count": int(anom_df["confirmed_anomaly"].sum())}
    except Exception as e:
        logger.exception("recalculate_anomaly_task failed")
        return {"status": "error", "message": str(e)}


@app.task(name="worker.tasks.recalculate_pricing")
def recalculate_pricing_task(tenant_id: str) -> dict:
    """Compute Lerner-index optimal discount pricing and cache recommendations."""
    logger.info("Starting recalculate_pricing for tenant %s", tenant_id)
    try:
        df = db.load_orders(tenant_id)
        if df.empty or len(df) < 30:
            return {"status": "error", "message": "Not enough data (need 30+ rows) for pricing optimizer"}

        from modules.price_optimizer import run_price_optimizer
        recs_df = run_price_optimizer(df)
        success = db.cache_price_recommendations(tenant_id, recs_df)

        db.log_action(tenant_id, "pricing_computation", {"success": success}, status="completed")
        return {"status": "success", "recommendations_count": len(recs_df)}
    except Exception as e:
        logger.exception("recalculate_pricing_task failed")
        return {"status": "error", "message": str(e)}


@app.task(name="worker.tasks.recalculate_drift")
def recalculate_drift_task(tenant_id: str, ref_months: int = 6, cur_months: int = 3) -> dict:
    """Monitor model data and prediction distribution shift (PSI)."""
    logger.info("Starting recalculate_drift for tenant %s", tenant_id)
    try:
        df = db.load_orders(tenant_id)
        if df.empty or len(df) < 100:
            return {"status": "error", "message": "Not enough data (need 100+ rows) for drift monitoring"}

        from modules.model_drift import compute_drift, compute_prediction_drift
        drift_df = compute_drift(df, reference_months=ref_months, current_months=cur_months)
        pred_drift = compute_prediction_drift(df, reference_months=ref_months, current_months=cur_months)

        report = {
            "features_drifted": int(drift_df["drift_detected"].sum()) if not drift_df.empty else 0,
            "max_psi": float(drift_df["psi"].max()) if not drift_df.empty else 0.0,
            "drift_alert": bool(drift_df["drift_detected"].any()) if not drift_df.empty else False,
        }
        if pred_drift:
            report.update({
                "pred_r2_drop": float(pred_drift.get("r2_drop", 0.0)),
                "ref_r2": pred_drift.get("ref_r2", 0.0),
                "cur_r2": pred_drift.get("cur_r2", 0.0),
                "ref_mae": pred_drift.get("ref_mae", 0.0),
                "cur_mae": pred_drift.get("cur_mae", 0.0),
                "drift_alert": report["drift_alert"] or bool(pred_drift.get("drift_alert", False)),
            })
            
        success = db.cache_drift_report(tenant_id, report)

        db.log_action(tenant_id, "drift_computation", {"success": success}, status="completed")
        return {"status": "success", "features_drifted": report["features_drifted"], "drift_alert": report["drift_alert"]}
    except Exception as e:
        logger.exception("recalculate_drift_task failed")
        return {"status": "error", "message": str(e)}
