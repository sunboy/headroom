"""CompressionStore.store()/retrieve() report through Headroom's OTEL telemetry.

CCR (Compress-Cache-Retrieve) is a parallel subsystem to the compression
pipeline and previously reported nothing to `headroom/observability/` — no
span, no OTEL metric, no Prometheus counter. `CompressionStore` is the single
choke point every CCR entry point (proxy handlers, the streaming handler, and
the standalone MCP server) funnels through, so instrumenting it there covers
every surface without touching each call site individually.
"""

from __future__ import annotations

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from headroom.cache.compression_store import CompressionStore
from headroom.observability import HeadroomOtelMetrics, reset_otel_metrics, set_otel_metrics


def _collect_metrics(reader: InMemoryMetricReader) -> dict[str, object]:
    data = reader.get_metrics_data()
    collected: dict[str, object] = {}
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                collected[metric.name] = metric
    return collected


def _find_point(metric, **expected_attributes):
    for point in metric.data.data_points:
        if all(point.attributes.get(key) == value for key, value in expected_attributes.items()):
            return point
    raise AssertionError(f"No datapoint matched attributes: {expected_attributes}")


class TestCompressionStoreReportsCcrTelemetry:
    def setup_method(self) -> None:
        self.reader = InMemoryMetricReader()
        provider = MeterProvider(metric_readers=[self.reader])
        set_otel_metrics(HeadroomOtelMetrics(meter_provider=provider))

    def teardown_method(self) -> None:
        reset_otel_metrics()

    def test_store_records_stored_outcome(self) -> None:
        store = CompressionStore()

        store.store(
            original="[1,2,3]",
            compressed="[1]",
            compressed_item_count=1,
            tool_name="search_api",
        )

        operations = _collect_metrics(self.reader)["headroom.ccr.operations"]
        point = _find_point(operations, operation="store", outcome="stored")
        assert point.value == 1

    def test_retrieve_hit_records_hit_outcome(self) -> None:
        store = CompressionStore()
        hash_key = store.store(
            original="[1,2,3]",
            compressed="[1]",
            compressed_item_count=1,
            tool_name="search_api",
        )

        store.retrieve(hash_key)

        operations = _collect_metrics(self.reader)["headroom.ccr.operations"]
        point = _find_point(operations, operation="retrieve", outcome="hit")
        assert point.value == 1

    def test_retrieve_missing_hash_records_miss_outcome(self) -> None:
        store = CompressionStore()

        entry = store.retrieve("does-not-exist")

        assert entry is None
        operations = _collect_metrics(self.reader)["headroom.ccr.operations"]
        point = _find_point(operations, operation="retrieve", outcome="miss")
        assert point.value == 1

    def test_retrieve_expired_records_expired_not_miss(self) -> None:
        store = CompressionStore(default_ttl=1)
        hash_key = store.store(original="[1,2,3]", compressed="[1]", ttl=0)

        import time

        time.sleep(0.05)
        entry = store.retrieve(hash_key)

        assert entry is None
        operations = _collect_metrics(self.reader)["headroom.ccr.operations"]
        expired_point = _find_point(operations, operation="retrieve", outcome="expired")
        assert expired_point.value == 1
        # An expired entry must not also be counted as a plain miss.
        for point in operations.data.data_points:
            if point.attributes.get("operation") == "retrieve":
                assert point.attributes.get("outcome") != "miss"

    def test_store_rejects_bare_marker_and_records_rejected_outcome(self) -> None:
        store = CompressionStore()

        store.store(original="<<ccr:abc123,base64,2.0KB>>", compressed="[1]")

        operations = _collect_metrics(self.reader)["headroom.ccr.operations"]
        point = _find_point(operations, operation="store", outcome="rejected")
        assert point.value == 1
