"""At-least-once Kafka ingestion for live orders.

Offsets advance only after the audit sink and Redis feature cache both succeed.
The audit table should enforce UNIQUE(event_id) so replay after a partial
failure is safe. Dependencies are imported lazily to keep batch workflows usable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Protocol

logger = logging.getLogger(__name__)
TOPIC = "ecommerce.orders.v1"


class AuditSink(Protocol):
    async def write_order(self, payload: dict[str, Any]) -> None: ...


class FeatureCache(Protocol):
    async def update_order(self, payload: dict[str, Any]) -> None: ...


def _event_time(payload: dict[str, Any]) -> datetime:
    value = payload.get("order_date") or payload.get("event_timestamp")
    if not value:
        raise ValueError("order payload requires order_date (event time)")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError("order event time must include a timezone")
    return parsed.astimezone(timezone.utc)


class PostgresAuditSink:
    """Async PostgreSQL writer; use DATABASE_URL (Supabase pooler URL works)."""
    def __init__(self, pool: Any):
        self.pool = pool

    async def write_order(self, payload: dict[str, Any]) -> None:
        event_id = payload.get("event_id") or payload.get("order_id")
        if not event_id:
            raise ValueError("order payload requires event_id or order_id")
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO public.order_ingest_audit
                   (event_id, order_id, customer_id, merchant_id, event_timestamp, payload)
                   VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                   ON CONFLICT (event_id) DO NOTHING""",
                str(event_id), str(payload.get("order_id", event_id)),
                payload.get("customer_id"), payload.get("merchant_id"),
                _event_time(payload), json.dumps(payload, default=str),
            )


class RedisFeatureCache:
    """Idempotent per-order feature updates using Redis sorted sets."""
    def __init__(self, redis: Any):
        self.redis = redis

    async def update_order(self, payload: dict[str, Any]) -> None:
        event_id = str(payload.get("event_id") or payload.get("order_id") or "")
        if not event_id:
            raise ValueError("order payload requires event_id or order_id")
        event_time = _event_time(payload).timestamp()
        revenue = float(payload.get("revenue", payload.get("final_price", 0)))
        pipe = self.redis.pipeline(transaction=True)
        updated_keys = []
        for entity, entity_id in (("customer", payload.get("customer_id")), ("merchant", payload.get("merchant_id"))):
            if entity_id is None:
                continue
            key = f"features:orders:{entity}:{entity_id}"
            updated_keys.append((entity, key))
            pipe.zadd(key, {event_id: event_time})
            pipe.hset(f"{key}:values:{event_id}", mapping={
                "event_timestamp": event_time,
                "revenue": revenue,
                "category": str(payload.get("category", "")),
                "discount_percent": float(payload.get("discount_percent", 0)),
                "units_sold": int(payload.get("units_sold", 1)),
            })
            for days in (30, 60, 90):
                pipe.zremrangebyscore(key, "-inf", event_time - days * 86400)
        await pipe.execute()

        # Build snapshots using only events at or before this event time. This
        # also prevents a late Kafka replay from replacing a newer online value.
        for entity, key in updated_keys:
            ids = await self.redis.zrangebyscore(key, event_time - 90 * 86400, event_time)
            if not ids:
                continue
            read_pipe = self.redis.pipeline(transaction=False)
            for member in ids:
                read_pipe.hgetall(f"{key}:values:{member}")
            rows = await read_pipe.execute()
            events = [row for row in rows if row]
            if not events:
                continue
            fields: dict[str, float | str] = {"event_timestamp": event_time}
            for days in (30, 60, 90):
                window = [row for row in events if float(row["event_timestamp"]) >= event_time - days * 86400]
                fields[f"order_velocity_{days}d"] = len(window) / days
                fields[f"aov_{days}d"] = sum(float(row["revenue"]) for row in window) / len(window) if window else 0.0
            # Pearson correlation between discount and units is a transparent
            # online proxy; downstream modeling should treat it as descriptive.
            discounts = [float(row["discount_percent"]) for row in events]
            units = [float(row["units_sold"]) for row in events]
            sensitivity = 0.0
            if len(events) > 1 and len(set(discounts)) > 1 and len(set(units)) > 1:
                import statistics
                sensitivity = statistics.correlation(discounts, units)
            fields["category_discount_sensitivity"] = sensitivity
            if entity == "merchant":
                fields["category"] = str(events[-1].get("category", ""))
            # Redis Lua guard makes updates monotonic in event time.
            await self.redis.eval(
                "local old=redis.call('HGET',KEYS[1],'event_timestamp'); "
                "if (not old) or tonumber(ARGV[1]) >= tonumber(old) then "
                "redis.call('HSET',KEYS[1],unpack(ARGV,2)); return 1 end; return 0",
                1, f"{key}:latest", event_time,
                *[item for name, value in fields.items() for item in (name, value)],
            )


async def consume_orders(
    *, bootstrap_servers: str | None = None, group_id: str | None = None,
    audit_sink: AuditSink | None = None, feature_cache: FeatureCache | None = None,
    consumer: Any | None = None,
) -> None:
    """Consume continuously; dependencies can be supplied for tests/embedding."""
    owns_consumer = consumer is None
    pool = redis_client = None
    try:
        if audit_sink is None:
            import asyncpg
            dsn = os.environ["DATABASE_URL"]
            pool = await asyncpg.create_pool(dsn)
            audit_sink = PostgresAuditSink(pool)
        if feature_cache is None:
            from redis.asyncio import from_url
            redis_client = from_url(os.environ["REDIS_URL"], decode_responses=True)
            feature_cache = RedisFeatureCache(redis_client)
        if consumer is None:
            from aiokafka import AIOKafkaConsumer
            consumer = AIOKafkaConsumer(
                TOPIC,
                bootstrap_servers=bootstrap_servers or os.environ["KAFKA_BOOTSTRAP_SERVERS"],
                group_id=group_id or os.getenv("KAFKA_GROUP_ID", "indiocommerce-order-features"),
                enable_auto_commit=False,
                value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            )
            await consumer.start()
        elif not getattr(consumer, "started", False) and hasattr(consumer, "start"):
            await consumer.start()

        async for message in consumer:
            payload = message.value
            _event_time(payload)  # Reject malformed timestamps before side effects.
            # Sequential durable writes: if either fails, leave this offset uncommitted.
            await audit_sink.write_order(payload)
            await feature_cache.update_order(payload)
            from aiokafka.structs import TopicPartition
            await consumer.commit({
                TopicPartition(message.topic, message.partition): message.offset + 1
            })
    finally:
        if owns_consumer and consumer is not None:
            await consumer.stop()
        if redis_client is not None:
            await redis_client.aclose()
        if pool is not None:
            await pool.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(consume_orders())
