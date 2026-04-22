"""tests/test_loader.py – unit tests for data loading & live data stubs."""
import os
import pandas as pd
import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")


@pytest.fixture(scope="module")
def df():
    from data.loader import load
    return load(FIXTURE, enrich_live=False)


def test_load_shape(df):
    assert len(df) > 0
    assert "revenue" in df.columns


def test_derived_columns(df):
    for col in ["year", "month", "discount_amount", "log_revenue", "log_final_price"]:
        assert col in df.columns, f"Missing derived column: {col}"


def test_no_negative_discount(df):
    assert (df["discount_percent"] >= 0).all()
    assert (df["discount_percent"] <= 100).all()


def test_log_transforms_non_negative(df):
    assert (df["log_revenue"] >= 0).all()
    assert (df["log_final_price"] >= 0).all()


def test_worldbank_returns_dataframe():
    from data.loader import fetch_worldbank, _WB_GDP
    result = fetch_worldbank(_WB_GDP)
    # may be empty if offline, but must be a DataFrame
    assert isinstance(result, pd.DataFrame)


def test_fx_returns_float():
    from data.loader import fetch_usd_inr
    rate = fetch_usd_inr()
    assert isinstance(rate, float)
    assert rate > 0
