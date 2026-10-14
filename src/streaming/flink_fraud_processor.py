"""Event-time transaction features for FraudGuard AI.

Consumes ``financial.transactions.v1`` and writes one rolling customer feature
snapshot per one-second slide over the preceding ten seconds.  The Kafka
connector JAR must be available to the Flink runtime (see README).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from pyflink.common import Duration, Time, Types, WatermarkStrategy
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource
from pyflink.datastream.functions import (
    MapFunction,
    RuntimeContext,
    SinkFunction,
    TimestampAssigner,
    WindowFunction,
)
from pyflink.datastream.window import SlidingEventTimeWindows
from src.streaming.fraud_features import aggregate_transaction_window, parse_transaction

WINDOW_SECONDS = 10
SLIDE_SECONDS = 1


class ParseTransaction(MapFunction):
    """Validate and normalize the JSON event into a picklable tuple."""

    def map(self, raw: str) -> tuple[str, int, float, str]:
        return parse_transaction(raw)


class TransactionTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, event: tuple[str, int, float, str], record_timestamp: int) -> int:
        return event[1]


class CustomerWindowFeatures(WindowFunction):
    """Aggregate transaction velocity, amount, and adjacent device switches."""

    def apply(self, key: str, window: Any, events: Iterable[tuple[str, int, float, str]]):
        feature = aggregate_transaction_window(key, events, int(window.start), int(window.end))
        if feature is not None:
            yield feature


class RedisFeatureSink(SinkFunction):
    """Write an idempotent rolling snapshot to ``features:fraud:customer:<id>``."""

    def __init__(self, redis_url: str, key_prefix: str = "features:fraud:customer"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix.rstrip(":")
        self.redis = None

    def open(self, runtime_context: RuntimeContext) -> None:
        import redis

        self.redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
        self.redis.ping()

    def invoke(self, feature: tuple, context: SinkFunction.Context) -> None:
        if self.redis is None:
            raise RuntimeError("Redis sink was not opened")
        customer_id, count, amount, switches, start_ms, end_ms, snapshot_ms = feature
        fields = {
            "window_start_ms": start_ms,
            "window_end_ms": end_ms,
            "snapshot_timestamp_ms": snapshot_ms,
            "transaction_velocity_10s": count,
            "rolling_amount_10s": f"{amount:.6f}",
            "device_switching_frequency_10s": switches,
            "device_switches_per_second_10s": f"{switches / WINDOW_SECONDS:.6f}",
        }
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(f"{self.key_prefix}:{customer_id}", mapping=fields)
        # Keep the online snapshot useful while preventing unbounded key growth.
        pipe.expire(f"{self.key_prefix}:{customer_id}", 86400)
        pipe.execute()

    def close(self) -> None:
        if self.redis is not None:
            self.redis.close()


def build_job() -> None:
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    topic = os.environ.get("FRAUD_TRANSACTIONS_TOPIC", "financial.transactions.v1")
    group_id = os.environ.get("FRAUD_KAFKA_GROUP_ID", "fraudguard-flink-features")
    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(bootstrap_servers)
        .set_topics(topic)
        .set_group_id(group_id)
        .set_starting_offsets_from_committed_offsets()
        .set_value_only_deserializer(SimpleStringSchema())
        .build()
    )

    env = StreamExecutionEnvironment.get_execution_environment()
    connector_jar = os.environ.get("FLINK_KAFKA_CONNECTOR_JAR")
    if connector_jar:
        env.add_jars(Path(connector_jar).resolve().as_uri())
    else:
        raise RuntimeError("Set FLINK_KAFKA_CONNECTOR_JAR to a Kafka SQL/DataStream connector JAR compatible with your Flink runtime")
    env.enable_checkpointing(int(os.environ.get("FLINK_CHECKPOINT_INTERVAL_MS", "10000")))
    env.set_parallelism(int(os.environ.get("FLINK_PARALLELISM", "2")))
    watermark_delay = int(os.environ.get("FLINK_WATERMARK_DELAY_SECONDS", "2"))
    transactions = (
        env.from_source(
            source,
            WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(watermark_delay))
            .with_timestamp_assigner(TransactionTimestampAssigner()),
            "financial-transactions-kafka-source",
        )
        .map(ParseTransaction(), output_type=Types.TUPLE([Types.STRING(), Types.LONG(), Types.DOUBLE(), Types.STRING()]))
    )
    features = (
        transactions.key_by(lambda event: event[0], key_type=Types.STRING())
        .window(SlidingEventTimeWindows.of(Time.seconds(WINDOW_SECONDS), Time.seconds(SLIDE_SECONDS)))
        .apply(
            CustomerWindowFeatures(),
            output_type=Types.TUPLE([
                Types.STRING(), Types.LONG(), Types.DOUBLE(), Types.LONG(),
                Types.LONG(), Types.LONG(), Types.LONG(),
            ]),
        )
    )
    features.add_sink(RedisFeatureSink(redis_url, os.environ.get("FRAUD_REDIS_KEY_PREFIX", "features:fraud:customer")))
    env.execute("fraudguard-transaction-window-features")


if __name__ == "__main__":
    build_job()
