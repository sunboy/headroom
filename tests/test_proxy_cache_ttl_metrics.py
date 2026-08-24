"""Tests for observed Anthropic cache TTL bucket metrics."""

from __future__ import annotations

import asyncio

import pytest

from headroom.observability import reset_headroom_tracing, reset_otel_metrics
from headroom.proxy.cost import CostTracker, build_prefix_cache_stats
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.semantic_cache import SemanticCache


def test_prometheus_metrics_tracks_observed_ttl_buckets() -> None:
    metrics = PrometheusMetrics()

    asyncio.run(
        metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=20,
            tokens_saved=5,
            latency_ms=10.0,
            cache_read_tokens=40,
            cache_write_tokens=60,
            cache_write_5m_tokens=10,
            cache_write_1h_tokens=50,
        )
    )

    stats = metrics.cache_by_provider["anthropic"]
    assert stats["cache_write_5m_tokens"] == 10
    assert stats["cache_write_1h_tokens"] == 50
    assert stats["cache_write_5m_requests"] == 1
    assert stats["cache_write_1h_requests"] == 1


def test_cost_tracker_exposes_observed_ttl_buckets_per_model() -> None:
    tracker = CostTracker()
    tracker.record_tokens(
        "claude-opus-4-6",
        tokens_saved=10,
        tokens_sent=90,
        cache_read_tokens=40,
        cache_write_tokens=60,
        cache_write_5m_tokens=10,
        cache_write_1h_tokens=50,
        uncached_tokens=20,
    )

    stats = tracker.stats()
    assert stats["cache_write_5m_tokens"] == 10
    assert stats["cache_write_1h_tokens"] == 50
    assert stats["per_model"]["claude-opus-4-6"]["cache_write_5m_tokens"] == 10
    assert stats["per_model"]["claude-opus-4-6"]["cache_write_1h_tokens"] == 50


def test_prefix_cache_stats_include_observed_ttl_mix() -> None:
    metrics = PrometheusMetrics()
    provider_stats = metrics.cache_by_provider["anthropic"]
    provider_stats["requests"] = 2
    provider_stats["hit_requests"] = 1
    provider_stats["cache_read_tokens"] = 40
    provider_stats["cache_write_tokens"] = 60
    provider_stats["cache_write_5m_tokens"] = 15
    provider_stats["cache_write_1h_tokens"] = 45
    provider_stats["cache_write_5m_requests"] = 1
    provider_stats["cache_write_1h_requests"] = 1

    stats = build_prefix_cache_stats(metrics, None)
    anthropic = stats["by_provider"]["anthropic"]

    assert anthropic["observed_ttl_buckets"]["5m"]["tokens"] == 15
    assert anthropic["observed_ttl_buckets"]["1h"]["tokens"] == 45
    assert anthropic["observed_ttl_mix"]["5m_pct"] == 25.0
    assert anthropic["observed_ttl_mix"]["1h_pct"] == 75.0
    assert stats["totals"]["observed_ttl_buckets"]["5m"]["tokens"] == 15
    assert stats["totals"]["observed_ttl_buckets"]["1h"]["tokens"] == 45


def test_prometheus_metrics_export_includes_extended_fields() -> None:
    metrics = PrometheusMetrics()

    asyncio.run(
        metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=20,
            tokens_saved=5,
            latency_ms=12.5,
            overhead_ms=3.0,
            ttfb_ms=9.0,
            pipeline_timing={"router": 4.5},
            waste_signals={"json_bloat": 7},
            cache_read_tokens=40,
            cache_write_tokens=60,
            cache_write_5m_tokens=10,
            cache_write_1h_tokens=50,
            uncached_input_tokens=20,
        )
    )
    asyncio.run(metrics.record_cache_bust(11))

    exported = asyncio.run(metrics.export())

    assert "headroom_latency_ms_count 1" in exported
    assert 'headroom_transform_timing_ms_sum{transform="router"} 4.5' in exported
    assert 'headroom_waste_signal_tokens_total{signal="json_bloat"} 7' in exported
    assert 'headroom_cache_write_ttl_tokens_total{provider="anthropic",ttl="5m"} 10' in exported
    assert 'headroom_provider_cache_hit_requests_total{provider="anthropic"} 1' in exported
    assert "headroom_cache_bust_tokens_lost_total 11" in exported


def test_streaming_parser_extracts_anthropic_ttl_bucket_usage() -> None:
    from headroom.proxy.server import HeadroomProxy, ProxyConfig

    proxy = HeadroomProxy(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    chunk = (
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":12,'
        b'"cache_read_input_tokens":3,"cache_creation_input_tokens":9,'
        b'"cache_creation":{"ephemeral_5m_input_tokens":4,"ephemeral_1h_input_tokens":5}}}}\n\n'
    )
    usage = proxy._parse_sse_usage(chunk, "anthropic")

    assert usage is not None
    assert usage["cache_creation_ephemeral_5m_input_tokens"] == 4
    assert usage["cache_creation_ephemeral_1h_input_tokens"] == 5


def test_stats_endpoint_reports_observed_ttl_buckets() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    proxy = app.state.proxy
    provider_stats = proxy.metrics.cache_by_provider["anthropic"]
    provider_stats["requests"] = 1
    provider_stats["hit_requests"] = 1
    provider_stats["cache_read_tokens"] = 30
    provider_stats["cache_write_tokens"] = 70
    provider_stats["cache_write_5m_tokens"] = 20
    provider_stats["cache_write_1h_tokens"] = 50
    provider_stats["cache_write_5m_requests"] = 1
    provider_stats["cache_write_1h_requests"] = 1

    with TestClient(app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    prefix_cache = response.json()["prefix_cache"]
    anthropic = prefix_cache["by_provider"]["anthropic"]
    assert anthropic["observed_ttl_buckets"]["5m"]["tokens"] == 20
    assert anthropic["observed_ttl_buckets"]["1h"]["tokens"] == 50
    assert prefix_cache["totals"]["observed_ttl_mix"]["active_buckets"] == ["5m", "1h"]


def test_stats_endpoint_reports_otel_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    reset_otel_metrics()
    monkeypatch.setenv("HEADROOM_OTEL_METRICS_ENABLED", "1")
    monkeypatch.setenv("HEADROOM_OTEL_METRICS_EXPORTER", "console")

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    with TestClient(app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    otel = response.json()["otel"]
    assert otel["configured"] is True
    assert otel["enabled"] is True
    assert otel["service_name"] == "headroom-proxy"
    assert otel["exporter"] == "console"


def test_stats_endpoint_reports_langfuse_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    reset_headroom_tracing()
    monkeypatch.setenv("HEADROOM_LANGFUSE_ENABLED", "1")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    with TestClient(app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    langfuse = response.json()["langfuse"]
    assert langfuse["configured"] is True
    assert langfuse["enabled"] is True
    assert langfuse["service_name"] == "headroom-proxy"
    assert langfuse["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"


# ── P1 metrics-split fix: provider cache vs. response cache ────────────


def test_provider_cache_hit_only_increments_provider_counter() -> None:
    """A request with an upstream prompt-cache read but no response-cache
    hit must increment ``requests_provider_cache_hit`` and the union
    ``requests_cached``, but leave ``requests_response_cache_hit`` at 0."""
    metrics = PrometheusMetrics()

    asyncio.run(
        metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=20,
            tokens_saved=5,
            latency_ms=10.0,
            cached=True,
            provider_cache_hit=True,
            response_cache_hit=False,
        )
    )

    assert metrics.requests_provider_cache_hit == 1
    assert metrics.requests_response_cache_hit == 0
    assert metrics.requests_cached == 1


def test_response_cache_hit_only_increments_response_counter() -> None:
    """A request served entirely from Headroom's own response cache must
    increment ``requests_response_cache_hit`` and the union
    ``requests_cached``, but leave ``requests_provider_cache_hit`` at 0."""
    metrics = PrometheusMetrics()

    asyncio.run(
        metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=2.0,
            cached=True,
            provider_cache_hit=False,
            response_cache_hit=True,
            response_cache_tokens_saved=250,
        )
    )

    assert metrics.requests_provider_cache_hit == 0
    assert metrics.requests_response_cache_hit == 1
    assert metrics.requests_cached == 1
    assert metrics.response_cache_tokens_saved_total == 250
    # Response-cache savings must NOT be folded into tokens_saved_total —
    # that total means something different (compression pipeline savings).
    assert metrics.tokens_saved_total == 0


def test_requests_cached_equals_union_of_provider_and_response_hits() -> None:
    """Across a mix of requests, the pre-existing ``requests_cached``
    union must equal the sum of the two granular counters whenever a
    single request never trips both signals at once (the normal case:
    provider-cache-only and response-cache-only requests never overlap,
    since a response-cache hit means the provider was never called)."""
    metrics = PrometheusMetrics()

    async def _run():
        # Provider-cache-only request.
        await metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=20,
            tokens_saved=0,
            latency_ms=10.0,
            cached=True,
            provider_cache_hit=True,
            response_cache_hit=False,
        )
        # Response-cache-only request.
        await metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=1.0,
            cached=True,
            provider_cache_hit=False,
            response_cache_hit=True,
        )
        # Uncached request — neither signal fires.
        await metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=20,
            tokens_saved=0,
            latency_ms=10.0,
            cached=False,
        )

    asyncio.run(_run())

    assert metrics.requests_provider_cache_hit == 1
    assert metrics.requests_response_cache_hit == 1
    assert metrics.requests_cached == 2
    assert (
        metrics.requests_provider_cache_hit + metrics.requests_response_cache_hit
        == metrics.requests_cached
    )
    assert metrics.requests_total == 3


def test_tool_schema_tokens_saved_folded_into_total_but_tracked_separately() -> None:
    """``tool_schema_tokens_saved`` is a subset of ``tokens_saved`` (folded
    into ``tokens_saved_total`` as always) but must also be visible on its
    own counter, distinct from generic compression."""
    metrics = PrometheusMetrics()

    asyncio.run(
        metrics.record_request(
            provider="openai",
            model="gpt-5",
            input_tokens=500,
            output_tokens=20,
            tokens_saved=180,
            latency_ms=10.0,
            tool_schema_tokens_saved=60,
        )
    )

    assert metrics.tokens_saved_total == 180
    assert metrics.tool_schema_tokens_saved_total == 60
    assert metrics.tool_schema_tokens_saved_total < metrics.tokens_saved_total


def test_prometheus_export_includes_granular_cache_and_savings_metrics() -> None:
    """The new counters must actually reach the Prometheus text exposition,
    as siblings of (not replacements for) the pre-existing metrics."""
    metrics = PrometheusMetrics()

    asyncio.run(
        metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=20,
            tokens_saved=50,
            latency_ms=10.0,
            cached=True,
            provider_cache_hit=True,
            response_cache_hit=False,
            tool_schema_tokens_saved=10,
        )
    )
    asyncio.run(
        metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=1.0,
            cached=True,
            provider_cache_hit=False,
            response_cache_hit=True,
            response_cache_tokens_saved=75,
        )
    )

    exported = asyncio.run(metrics.export())

    # Pre-existing metric untouched in name and semantics.
    assert "headroom_requests_cached_total 2" in exported
    # New siblings, each counting only its own kind.
    assert "headroom_requests_provider_cache_hit_total 1" in exported
    assert "headroom_requests_response_cache_hit_total 1" in exported
    assert "headroom_tokens_saved_total 50" in exported
    assert "headroom_tool_schema_tokens_saved_total 10" in exported
    assert "headroom_response_cache_tokens_saved_total 75" in exported


def test_semantic_cache_stats_aggregates_tokens_saved_per_hit() -> None:
    """Pre-fix, ``CacheEntry.tokens_saved_per_hit`` was written on every
    cache store but never summed anywhere — a dead field. ``stats()``
    must now aggregate it as ``total_tokens_saved``, accounting for
    repeat hits on the same entry (not just one hit per entry)."""
    cache = SemanticCache(max_entries=10, ttl_seconds=3600)

    async def _run():
        await cache.set(
            messages=[{"role": "user", "content": "hello"}],
            model="claude-opus-4-6",
            response_body=b"{}",
            response_headers={},
            tokens_saved=100,
        )
        await cache.set(
            messages=[{"role": "user", "content": "world"}],
            model="claude-opus-4-6",
            response_body=b"{}",
            response_headers={},
            tokens_saved=40,
        )
        # Hit the first entry twice — its savings should count each time.
        await cache.get([{"role": "user", "content": "hello"}], "claude-opus-4-6")
        await cache.get([{"role": "user", "content": "hello"}], "claude-opus-4-6")
        # Hit the second entry once.
        await cache.get([{"role": "user", "content": "world"}], "claude-opus-4-6")
        return await cache.stats()

    stats = asyncio.run(_run())

    assert stats["total_hits"] == 3
    # entry 1: 100 * 2 hits = 200; entry 2: 40 * 1 hit = 40 → 240 total.
    assert stats["total_tokens_saved"] == 240
    # Pre-existing keys stay present and correct — additive only.
    assert stats["entries"] == 2
    assert stats["max_entries"] == 10


def test_stats_endpoint_reports_granular_cache_and_savings_fields() -> None:
    """The new /stats fields must exist and sum consistently against the
    legacy blended fields they sit alongside — the additive contract this
    whole fix depends on."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
        )
    )

    proxy = app.state.proxy

    async def _seed():
        await proxy.metrics.record_request(
            provider="anthropic",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=20,
            tokens_saved=50,
            latency_ms=10.0,
            cached=True,
            provider_cache_hit=True,
            response_cache_hit=False,
        )
        await proxy.metrics.record_request(
            provider="openai",
            model="gpt-5",
            input_tokens=0,
            output_tokens=0,
            tokens_saved=0,
            latency_ms=1.0,
            cached=True,
            provider_cache_hit=False,
            response_cache_hit=True,
        )
        await proxy.metrics.record_request(
            provider="openai",
            model="gpt-5",
            input_tokens=500,
            output_tokens=20,
            tokens_saved=90,
            latency_ms=5.0,
            tool_schema_tokens_saved=30,
        )

    asyncio.run(_seed())

    with TestClient(app) as client:
        response = client.get("/stats")

    assert response.status_code == 200
    body = response.json()

    # requests: legacy union field unchanged; new fields sum to it (no
    # request in this seed trips both signals at once).
    requests = body["requests"]
    assert requests["cached"] == 2
    assert requests["provider_cache_hits"] == 1
    assert requests["response_cache_hits"] == 1
    assert requests["provider_cache_hits"] + requests["response_cache_hits"] == requests["cached"]

    # savings.by_layer: new tool_schema_deferral / response_cache siblings
    # exist next to the pre-existing compression / prefix_cache blocks,
    # and tool_schema tokens are a subset of (not additional to) the
    # compression total.
    by_layer = body["savings"]["by_layer"]
    assert "tool_schema_deferral" in by_layer
    assert "response_cache" in by_layer
    assert by_layer["tool_schema_deferral"]["tokens"] == 30
    assert by_layer["tool_schema_deferral"]["tokens"] <= by_layer["compression"]["tokens"]
    assert by_layer["prefix_cache"]["attribution"] == "provider_native_baseline"

    # prefix_cache (top-level, raw stats): honest not-yet-computed uplift
    # slot exists alongside the pre-existing attribution prose and
    # savings_usd, without changing what savings_usd means.
    prefix_cache = body["prefix_cache"]
    assert prefix_cache["totals"]["attribution"] == "provider_native_baseline"
    assert prefix_cache["cachealigner_uplift"]["available"] is False
    assert prefix_cache["cachealigner_uplift"]["method"] is None
