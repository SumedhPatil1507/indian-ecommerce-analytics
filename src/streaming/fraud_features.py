"""Dependency-light parsing and window feature helpers for the Flink job."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable


def timestamp_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value if value > 10_000_000_000 else value * 1000)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("transaction timestamp must include a timezone")
        return int(parsed.astimezone(timezone.utc).timestamp() * 1000)
    raise ValueError("transaction requires event_timestamp or transaction_timestamp")


def parse_transaction(raw: str | dict[str, Any]) -> tuple[str, int, float, str]:
    """Normalize JSON event text or a dict into the Flink tuple shape."""
    event = json.loads(raw) if isinstance(raw, str) else raw
    customer_id = event.get("customer_id") or event.get("user_id")
    if customer_id is None or str(customer_id).strip() == "":
        raise ValueError("transaction requires customer_id (or user_id)")
    timestamp = timestamp_ms(
        event.get("event_timestamp", event.get("transaction_timestamp", event.get("timestamp")))
    )
    raw_amount = event.get("amount", event.get("transaction_amount"))
    if raw_amount is None:
        raise ValueError("transaction requires amount (or transaction_amount)")
    amount = float(raw_amount)
    if amount < 0:
        raise ValueError("transaction amount cannot be negative")
    device_id = str(event.get("device_id") or "unknown")
    return str(customer_id), timestamp, amount, device_id


def aggregate_transaction_window(
    customer_id: str,
    events: Iterable[tuple[str, int, float, str]],
    window_start_ms: int,
    window_end_ms: int,
) -> tuple[str, int, float, int, int, int, int] | None:
    """Return count, amount and adjacent device switches for one window."""
    rows = sorted(events, key=lambda event: event[1])
    if not rows:
        return None
    switches = sum(1 for previous, current in zip(rows, rows[1:]) if previous[3] != current[3])
    return (
        customer_id,
        len(rows),
        sum(row[2] for row in rows),
        switches,
        int(window_start_ms),
        int(window_end_ms),
        int(window_end_ms),
    )
