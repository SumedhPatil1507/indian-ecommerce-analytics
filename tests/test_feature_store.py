"""Point-in-time feature retrieval must exclude events after each label time."""
from datetime import datetime, timezone

import pandas as pd
import pytest


def test_historical_features_are_point_in_time(tmp_path):
    pytest.importorskip("feast")
    from feast import FeatureStore

    repo = tmp_path / "repo"
    (repo / "data" / "feature_repo").mkdir(parents=True)
    orders = pd.DataFrame([
        {"customer_id": "c1", "merchant_id": "m1", "event_timestamp": "2024-01-01T00:00:00Z", "created_timestamp": "2024-01-01T00:01:00Z", "order_velocity_30d": 1.0, "order_velocity_60d": 1.0, "order_velocity_90d": 1.0, "aov_30d": 100.0, "aov_60d": 100.0, "aov_90d": 100.0, "category_discount_sensitivity": 0.1, "category": "electronics"},
        {"customer_id": "c1", "merchant_id": "m1", "event_timestamp": "2024-01-10T00:00:00Z", "created_timestamp": "2024-01-10T00:01:00Z", "order_velocity_30d": 2.0, "order_velocity_60d": 2.0, "order_velocity_90d": 2.0, "aov_30d": 250.0, "aov_60d": 250.0, "aov_90d": 250.0, "category_discount_sensitivity": 0.8, "category": "electronics"},
    ])
    orders.to_parquet(repo / "data" / "feature_repo" / "order_features.parquet", index=False)
    (repo / "feature_store.yaml").write_text(
        "project: pit_test\nprovider: local\nregistry: data/feature_repo/registry.db\n"
        "offline_store:\n  type: file\nonline_store:\n  type: sqlite\n  path: data/online.db\n",
        encoding="utf-8",
    )
    # Load the production definitions but bind their source to this isolated fixture.
    from data import features
    features.orders_source.path = str(repo / "data" / "feature_repo" / "order_features.parquet")
    store = FeatureStore(repo_path=str(repo))
    store.apply([features.customer, features.merchant, features.customer_order_features, features.merchant_order_features])

    entity_df = pd.DataFrame({
        "customer_id": ["c1", "c1"],
        "event_timestamp": [
            datetime(2024, 1, 5, tzinfo=timezone.utc),
            datetime(2024, 1, 12, tzinfo=timezone.utc),
        ],
    })
    result = store.get_historical_features(
        entity_df=entity_df,
        features=["customer_order_features:aov_30d", "customer_order_features:order_velocity_30d"],
    ).to_df()
    assert result["aov_30d"].tolist() == [100.0, 250.0]
    assert result["order_velocity_30d"].tolist() == [1.0, 2.0]
    assert pd.to_datetime(result["event_timestamp"], utc=True).tolist() == pd.to_datetime(entity_df["event_timestamp"], utc=True).tolist()
