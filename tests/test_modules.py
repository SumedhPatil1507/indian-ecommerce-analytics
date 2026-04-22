"""tests/test_modules.py – smoke tests for analytics modules."""
import os
import pytest
import pandas as pd

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")


@pytest.fixture(scope="module")
def df():
    from data.loader import load
    return load(FIXTURE, enrich_live=False)


def test_price_elasticity(df):
    from modules.price_elasticity import compute_elasticity
    result = compute_elasticity(df, group_cols=["category"], min_obs=10)
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "elasticity" in result.columns


def test_inventory_alerts(df):
    from modules.inventory_alerts import compute_alerts
    alerts = compute_alerts(df)
    assert "alert_level" in alerts.columns
    assert "recommendation" in alerts.columns


def test_anomaly_report(df):
    from modules.anomaly import anomaly_report
    result = anomaly_report(df)
    assert "confirmed_anomaly" in result.columns
    assert result["confirmed_anomaly"].dtype == bool


def test_cohort_table(df):
    from modules.cohort import build_cohort_table
    pivot = build_cohort_table(df, metric="count")
    assert isinstance(pivot, pd.DataFrame)
    assert pivot.shape[0] > 0
