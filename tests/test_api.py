"""
tests/test_api.py
Automated tests for FastAPI endpoints, authentication, and cryptographic webhook validation.
"""
import hmac
import base64
import hashlib
import pytest
from fastapi.testclient import TestClient
import pandas as pd

from api.main import app, get_current_tenant
from core import database as db

client = TestClient(app)


def test_health_endpoint():
    """Verify that the health check is public and operational."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "timestamp" in resp.json()


def test_fraud_assessment_endpoint_returns_canonical_schema(monkeypatch):
    """The fraud API always returns the exact explainable four-field contract."""
    app.dependency_overrides[get_current_tenant] = lambda: {
        "tenant_id": "tenant-1",
        "token": "test-token",
    }
    monkeypatch.setattr(db, "load_orders", lambda *args, **kwargs: pd.DataFrame({
        "customer_id": ["customer-1"],
    }))
    monkeypatch.setattr(db, "load_anomaly_cache", lambda *args, **kwargs: pd.DataFrame({
        "customer_id": ["customer-1"],
        "collusion_probability": [0.91],
        "rto_probability": [0.12],
        "graph_centrality_score": [0.74],
        "xgboost_shap_contributions": [{"shared_device_count": 0.31}],
    }))
    try:
        response = client.get("/fraud/assessment?customer_id=customer-1")
    finally:
        app.dependency_overrides.pop(get_current_tenant, None)

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"verdict", "reasoning", "recommendation", "next_steps"}
    assert payload["verdict"] == "HIGH_RISK_COLLUSION_RING_DETECTED"
    assert "R-GCN graph centrality score=0.7400" in payload["reasoning"]
    assert "XGBoost SHAP feature contributions=shared_device_count=+0.3100" in payload["reasoning"]
    assert isinstance(payload["recommendation"], str) and payload["recommendation"]
    assert len(payload["next_steps"]) == 3


def test_fraud_assessment_endpoint_returns_same_schema_without_data(monkeypatch):
    """Missing customer data yields a safe unavailable assessment, not a different shape."""
    app.dependency_overrides[get_current_tenant] = lambda: {
        "tenant_id": "tenant-1",
        "token": "test-token",
    }
    monkeypatch.setattr(db, "load_orders", lambda *args, **kwargs: pd.DataFrame())
    try:
        response = client.get("/fraud/assessment?customer_id=missing-customer")
    finally:
        app.dependency_overrides.pop(get_current_tenant, None)

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"verdict", "reasoning", "recommendation", "next_steps"}
    assert payload["verdict"] == "FRAUD_ASSESSMENT_UNAVAILABLE"


def test_unauthorized_endpoints():
    """Check that secure endpoints return 401 when no Bearer token is provided."""
    endpoints = [
        "/profile",
        "/analytics/kpis",
        "/analytics/executive-summary",
        "/analytics/price-optimizer",
        "/analytics/at-risk",
        "/analytics/drift",
        "/analytics/clv",
        "/analytics/anomalies",
    ]
    for route in endpoints:
        resp = client.get(route)
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Missing or invalid Bearer token"


def test_shopify_webhook_signature_failure():
    """Ensure that Shopify webhooks reject requests with invalid HMAC signatures."""
    payload = {"id": 9999, "total_price": "100.00"}
    headers = {"X-Shopify-Hmac-SHA256": "incorrect_signature_hash"}
    
    # Target tenant_id can be any UUID for signature verification check
    resp = client.post(
        "/ingest/shopify?tenant_id=00000000-0000-0000-0000-000000000000",
        json=payload,
        headers=headers
    )
    # Since profile doesn't exist, it should fail with 401
    assert resp.status_code == 401


def test_woocommerce_webhook_signature_failure():
    """Ensure that WooCommerce webhooks reject invalid signatures."""
    payload = {"id": 1234, "total": "50.00"}
    headers = {"X-WC-Webhook-Signature": "incorrect_hash"}
    
    resp = client.post(
        "/ingest/woocommerce?tenant_id=00000000-0000-0000-0000-000000000000",
        json=payload,
        headers=headers
    )
    assert resp.status_code == 401


def test_amazon_webhook_signature_failure():
    """Ensure that Amazon webhooks reject invalid signatures."""
    payload = {"AmazonOrderId": "123-456", "OrderTotal": {"Amount": "10.00"}}
    headers = {"X-Amazon-Webhook-Signature": "incorrect_hash"}
    
    resp = client.post(
        "/ingest/amazon?tenant_id=00000000-0000-0000-0000-000000000000",
        json=payload,
        headers=headers
    )
    assert resp.status_code == 401
