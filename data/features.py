"""Event-time Feast definitions for order-derived features.

The offline source must retain one row per entity and feature event time. Feast
uses those timestamps for point-in-time joins; ingestion time is never used as
the feature event time.
"""
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String
from feast.value_type import ValueType
from datetime import timedelta

customer = Entity(name="customer_id", join_keys=["customer_id"], value_type=ValueType.STRING)
merchant = Entity(name="merchant_id", join_keys=["merchant_id"], value_type=ValueType.STRING)

orders_source = FileSource(
    name="orders_feature_source",
    path="data/feature_repo/order_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

customer_order_features = FeatureView(
    name="customer_order_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="order_velocity_30d", dtype=Float32),
        Field(name="order_velocity_60d", dtype=Float32),
        Field(name="order_velocity_90d", dtype=Float32),
        Field(name="aov_30d", dtype=Float32),
        Field(name="aov_60d", dtype=Float32),
        Field(name="aov_90d", dtype=Float32),
    ],
    source=orders_source,
    online=True,
    tags={"domain": "orders", "time_semantics": "event_time"},
)

merchant_order_features = FeatureView(
    name="merchant_order_features",
    entities=[merchant],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="order_velocity_30d", dtype=Float32),
        Field(name="order_velocity_60d", dtype=Float32),
        Field(name="order_velocity_90d", dtype=Float32),
        Field(name="aov_30d", dtype=Float32),
        Field(name="aov_60d", dtype=Float32),
        Field(name="aov_90d", dtype=Float32),
        Field(name="category_discount_sensitivity", dtype=Float32),
        Field(name="category", dtype=String),
    ],
    source=orders_source,
    online=True,
    tags={"domain": "merchant_category", "time_semantics": "event_time"},
)
