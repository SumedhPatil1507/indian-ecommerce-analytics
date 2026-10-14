"""Dependency-light tests for Kafka event normalization and window features."""
import pytest

from src.streaming.fraud_features import aggregate_transaction_window, parse_transaction, timestamp_ms


def test_parse_transaction_normalizes_timezone_and_fields():
    parsed = parse_transaction({
        "user_id": "customer-7",
        "transaction_timestamp": "2025-01-02T03:04:05Z",
        "transaction_amount": "125.50",
        "device_id": "device-a",
    })
    assert parsed == ("customer-7", 1735787045000, 125.5, "device-a")


def test_timestamp_accepts_epoch_seconds_and_milliseconds():
    assert timestamp_ms(1_735_787_045) == 1_735_787_045_000
    assert timestamp_ms(1_735_787_045_000) == 1_735_787_045_000


@pytest.mark.parametrize("payload, message", [
    ({"customer_id": "c1", "event_timestamp": "2025-01-02T03:04:05", "amount": 2}, "timezone"),
    ({"customer_id": "c1", "event_timestamp": 1, "amount": -1}, "negative"),
    ({"customer_id": "c1", "event_timestamp": 1}, "amount"),
    ({"event_timestamp": 1, "amount": 2}, "customer_id"),
])
def test_parse_transaction_rejects_invalid_events(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_transaction(payload)


def test_sliding_window_aggregation_computes_velocity_amount_and_device_switches():
    events = [
        ("c1", 3000, 10.0, "phone-a"),
        ("c1", 1000, 20.0, "phone-a"),
        ("c1", 2000, 30.0, "phone-b"),
    ]
    assert aggregate_transaction_window("c1", events, 0, 10_000) == (
        "c1", 3, 60.0, 2, 0, 10_000, 10_000
    )


def test_empty_window_has_no_feature_snapshot():
    assert aggregate_transaction_window("c1", [], 0, 10_000) is None
