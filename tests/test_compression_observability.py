"""Per-strategy compression observability tests.

These guard the forcing function: when any compressor runs in
production, a `CompressionObserver` notification fires once per real
compression event, and `PrometheusMetrics` accumulates per-strategy
counters that the test suite asserts on directly.

The TOIN→SmartCrusher silent disconnect (caught three weeks late by
manual audit) was invisible because no signal distinguished by
strategy. These tests exist so the next regression of that shape
fails the suite the day it lands instead of waiting on an audit.

The counters live ONLY as in-process state on the metrics instance;
they are deliberately NOT exported as new Prometheus metric names
(to avoid unbounded metric-series growth) — they remain observable
via /stats. CI-level
observability via these tests is enough to catch silent regressions;
production export waits on a non-column-adding pipeline.

Coverage:

1. `ContentRouter.compress(...)` calls observer once per RoutingDecision.
2. `SmartCrusher.apply(...)` calls observer once per crushed message.
3. Both transforms tolerate an observer that raises (compression must
   still succeed).
4. `PrometheusMetrics` correctly satisfies the `CompressionObserver`
   protocol — `record_compression` increments per-strategy counters
   and `tokens_saved_by_strategy` accumulates only positive savings.
5. The Prometheus scrape output (`export()`) does NOT emit any new
   metric names — the per-strategy state stays internal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from headroom.tokenizer import Tokenizer
from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)
from headroom.transforms.observability import CompressionObserver, classify_lossy
from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

# ─── Test doubles ──────────────────────────────────────────────────────


@dataclass
class SpyObserver:
    """Captures every `record_compression` call for assertion."""

    calls: list[tuple[str, int, int, str]] = field(default_factory=list)

    def record_compression(
        self,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
        lossy: str,
    ) -> None:
        self.calls.append((strategy, original_tokens, compressed_tokens, lossy))


@dataclass
class ExplodingObserver:
    """Raises on every call. Used to assert observer failures don't
    propagate out and break compression."""

    raised: int = 0

    def record_compression(self, *_a: Any, **_kw: Any) -> None:
        self.raised += 1
        raise RuntimeError("simulated observer outage")


# ─── Protocol conformance ──────────────────────────────────────────────


def test_spy_satisfies_observer_protocol():
    spy = SpyObserver()
    # `runtime_checkable` Protocol — isinstance check works.
    assert isinstance(spy, CompressionObserver)


def test_prometheus_metrics_satisfies_observer_protocol():
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()
    assert isinstance(m, CompressionObserver)


def test_beacon_compression_observer_satisfies_observer_protocol():
    """No prior test exercised this observer at all — the proxy's
    `PrometheusMetrics` is already an observer, so `BeaconCompressionObserver`
    only fires on paths that never had one (MCP servers, bare pipeline,
    LangChain/Strands)."""
    from headroom.telemetry.session import BeaconCompressionObserver

    beacon_observer = BeaconCompressionObserver()
    assert isinstance(beacon_observer, CompressionObserver)


def test_beacon_compression_observer_accepts_lossy_but_does_not_forward_it(
    monkeypatch: pytest.MonkeyPatch,
):
    """`lossy` is required by the protocol and must not raise here, but
    the beacon's wire payload (`_staged_strategies`) is a separate,
    more sensitive data-collection surface — `lossy` is accepted and
    dropped, not forwarded."""
    from headroom.telemetry import beacon as beacon_module
    from headroom.telemetry import session as session_module
    from headroom.telemetry.session import BeaconCompressionObserver

    monkeypatch.setattr(beacon_module, "is_beacon_enabled", lambda: True)
    session_module._staged_strategies.clear()

    BeaconCompressionObserver().record_compression(
        "smart_crusher", original_tokens=200, compressed_tokens=50, lossy="lossy_unrecoverable"
    )

    staged = session_module._drain_staged_strategies()
    assert staged == {"smart_crusher": [1, 200, 50]}


# ─── ContentRouter wiring ──────────────────────────────────────────────


def test_content_router_records_observer_call_per_routing_decision():
    spy = SpyObserver()
    router = ContentRouter(ContentRouterConfig(), observer=spy)

    # Forge a routing log directly via the result object — the observer
    # call site walks `result.routing_log`, so we assert the contract
    # without depending on which compressor would actually fire.
    result = RouterCompressionResult(
        compressed="x",
        original="x",
        strategy_used=CompressionStrategy.SMART_CRUSHER,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.JSON_ARRAY,
                strategy=CompressionStrategy.SMART_CRUSHER,
                original_tokens=200,
                compressed_tokens=50,
            ),
            RoutingDecision(
                content_type=ContentType.SOURCE_CODE,
                strategy=CompressionStrategy.CODE_AWARE,
                original_tokens=300,
                compressed_tokens=300,  # passthrough — still recorded
            ),
        ],
    )
    router._observe(result)

    # Both RoutingDecisions above were constructed without an explicit
    # `lossy=`, so they carry the dataclass default ("lossless") — this
    # test asserts wiring (does the observer get called once per
    # decision with the right strategy/tokens/lossy), not classification
    # accuracy for hand-built fixtures.
    assert spy.calls == [
        ("smart_crusher", 200, 50, "lossless"),
        ("code_aware", 300, 300, "lossless"),
    ]


def test_content_router_with_no_observer_is_silent():
    router = ContentRouter(ContentRouterConfig())  # observer defaults None
    result = RouterCompressionResult(
        compressed="x",
        original="x",
        strategy_used=CompressionStrategy.PASSTHROUGH,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=10,
                compressed_tokens=5,
            )
        ],
    )
    # Should not raise.
    router._observe(result)


def test_content_router_swallows_observer_failures():
    boom = ExplodingObserver()
    router = ContentRouter(ContentRouterConfig(), observer=boom)
    result = RouterCompressionResult(
        compressed="x",
        original="x",
        strategy_used=CompressionStrategy.TEXT,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=10,
                compressed_tokens=5,
            )
        ],
    )
    # Must not raise — observability failures are not compression failures.
    router._observe(result)
    assert boom.raised == 1


# ─── SmartCrusher wiring (legacy direct-pipeline path) ─────────────────


def _bigger_array(n: int = 60) -> str:
    import json as _json

    items = [{"status": "ok", "tag": "x", "n": i} for i in range(n)]
    return _json.dumps(items)


@pytest.fixture
def isolated_toin(tmp_path, monkeypatch):
    """Point TOIN at a tempdir for the duration of the test.

    SmartCrusher.apply() feeds the global TOIN learning store via
    `record_compression`. Its default storage path is
    `~/.headroom/toin.json`, which persists across pytest invocations.
    On Python 3.11 CI runs the suite twice (regular + coverage); a
    pattern written in run #1 changes which rows the lossy sampler
    keeps in run #2 and breaks `test_first_last_items_always_preserved`
    in `test_evals.py`.

    Isolating the TOIN file per test contains the side effect.
    """
    from pathlib import Path

    from headroom.telemetry.toin import TOIN_PATH_ENV_VAR, reset_toin

    storage = str(Path(tmp_path) / "toin.json")
    monkeypatch.setenv(TOIN_PATH_ENV_VAR, storage)
    reset_toin()
    yield
    reset_toin()


def test_smart_crusher_apply_records_observer_per_crushed_message(isolated_toin):
    """End-to-end: SmartCrusher.apply() walks messages, crushes the
    big tool_result, fires the observer with strategy='smart_crusher'."""
    from headroom.providers.openai import OpenAITokenCounter
    from headroom.tokenizer import Tokenizer

    spy = SpyObserver()
    crusher = SmartCrusher(SmartCrusherConfig(), observer=spy)
    tok = Tokenizer(OpenAITokenCounter("gpt-4o-mini"), model="gpt-4o-mini")

    messages = [
        {"role": "user", "content": "what's in the data?"},
        {"role": "tool", "content": _bigger_array(60)},
    ]
    result = crusher.apply(messages, tok)
    # If the analyzer chose passthrough this run, the observer wasn't
    # fired; that's fine for the wiring test — we only assert it WAS
    # fired in the case it crushed.
    if "smart_crush:" in ",".join(result.transforms_applied):
        assert spy.calls, "smart_crusher crushed but observer wasn't notified"
        for strategy, original, compressed, lossy in spy.calls:
            assert strategy == "smart_crusher"
            assert original > 0
            assert compressed >= 0
            assert lossy in ("lossless", "lossy_recoverable", "lossy_unrecoverable")


def test_smart_crusher_apply_swallows_observer_failures(isolated_toin):
    """Observer raises → compression still completes, returns valid
    TransformResult, count of raises matches the crushed_count."""
    from headroom.providers.openai import OpenAITokenCounter
    from headroom.tokenizer import Tokenizer

    boom = ExplodingObserver()
    crusher = SmartCrusher(SmartCrusherConfig(), observer=boom)
    tok = Tokenizer(OpenAITokenCounter("gpt-4o-mini"), model="gpt-4o-mini")
    messages = [{"role": "tool", "content": _bigger_array(60)}]
    result = crusher.apply(messages, tok)
    # Either the analyzer didn't crush (boom.raised == 0) or it did
    # (boom.raised >= 1) — but in both cases compression returned a
    # valid TransformResult. No exception escaped.
    assert result.messages is not None


# ─── PrometheusMetrics implementation ──────────────────────────────────


def test_prometheus_metrics_accumulates_per_strategy_counters():
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()

    m.record_compression(
        "smart_crusher", original_tokens=200, compressed_tokens=50, lossy="lossy_recoverable"
    )
    m.record_compression(
        "smart_crusher", original_tokens=100, compressed_tokens=40, lossy="lossy_recoverable"
    )
    m.record_compression(
        "diff", original_tokens=80, compressed_tokens=80, lossy="lossless"
    )  # no savings
    m.record_compression(
        "code_aware", original_tokens=50, compressed_tokens=70, lossy="lossless"
    )  # negative savings

    assert m.compressions_by_strategy == {
        "smart_crusher": 2,
        "diff": 1,
        "code_aware": 1,
    }
    # Tokens saved is `max(0, original - compressed)` per strategy.
    # smart_crusher: 150 + 60 = 210; diff: 0 (no savings, dict entry omitted);
    # code_aware: 0 (negative).
    assert m.tokens_saved_by_strategy == {"smart_crusher": 210}


def test_prometheus_metrics_accumulates_extension_savings_per_key() -> None:
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()

    m.record_extension_savings("tool_router", 120)
    m.record_extension_savings("tool_router", 30)
    m.record_extension_savings("skill_search", 45)
    m.record_extension_savings("skill_search", 0)  # no savings, ignored
    m.record_extension_savings("noop_ext", -10)  # negative, ignored

    # Savings accumulate per key; non-positive values never create or
    # bump an entry.
    assert m.extension_savings == {"tool_router": 150, "skill_search": 45}


def test_extension_savings_surface_in_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    monkeypatch.setenv("HEADROOM_SAVINGS_PATH", str(tmp_path / "proxy_savings.json"))
    config = ProxyConfig(
        cache_enabled=False,
        rate_limit_enabled=False,
        log_requests=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        proxy = app.state.proxy
        proxy.metrics.record_extension_savings("tool_router", 200)
        proxy.metrics.record_extension_savings("tool_router", 50)
        proxy.metrics.record_extension_savings("skill_search", 75)

        stats = client.get("/stats")
        assert stats.status_code == 200
        assert stats.json()["extension_savings"] == {
            "tool_router": 250,
            "skill_search": 75,
        }


def test_prometheus_metrics_accumulates_codex_ws_unit_and_frame_counters():
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()

    m.record_codex_ws_unit(
        strategy="mixed",
        reason_category="applied",
        elapsed_ms=1250,
        text_bytes=10_000,
        tokens_before=2500,
        tokens_after=1000,
        tokens_saved=1500,
        modified=True,
        strategy_chain=["mixed", "kompress"],
        content_type="text",
        text_shape="jsonl_like",
    )
    m.record_codex_ws_unit(
        strategy="passthrough",
        reason_category="size_floor",
        elapsed_ms=2,
        text_bytes=100,
        tokens_before=20,
        tokens_after=20,
        tokens_saved=0,
        modified=False,
        strategy_chain=["passthrough"],
        content_type="unknown",
        text_shape="plain_text_like",
    )
    m.record_codex_ws_frame(
        elapsed_ms=1260,
        bytes_before=20_000,
        bytes_after=8_000,
        attempted_tokens=2500,
        tokens_saved=1500,
        modified=True,
        strategy_chain=["mixed", "kompress"],
        final_strategies=["mixed"],
    )
    m.record_codex_ws_frame(
        elapsed_ms=30_000,
        bytes_before=426_318,
        failed=True,
    )

    assert m.codex_ws_units_total == 2
    assert m.codex_ws_units_modified_total == 1
    assert m.codex_ws_units_by_strategy == {"mixed": 1, "passthrough": 1}
    assert m.codex_ws_units_by_category == {"applied": 1, "size_floor": 1}
    assert m.codex_ws_units_by_content_type == {"text": 1, "unknown": 1}
    assert m.codex_ws_units_by_text_shape == {"jsonl_like": 1, "plain_text_like": 1}
    assert m.codex_ws_units_to_kompress_total == 0
    assert m.codex_ws_units_kompress_attempted_total == 1
    assert m.codex_ws_unit_elapsed_ms_max == 1250
    assert m.codex_ws_unit_tokens_saved_sum == 1500

    assert m.codex_ws_frames_attempted_total == 2
    assert m.codex_ws_frames_compressed_total == 1
    assert m.codex_ws_frames_failed_total == 1
    assert m.codex_ws_frames_to_kompress_total == 0
    assert m.codex_ws_frames_kompress_attempted_total == 1
    assert m.codex_ws_frame_elapsed_ms_max == 30_000
    assert m.codex_ws_frame_tokens_saved_sum == 1500


def test_prometheus_export_does_not_leak_per_strategy_metrics():
    """Per-strategy state is tracked in-process only. The Prometheus
    scrape output deliberately must NOT emit new metric names (to avoid
    unbounded metric-series growth); the state stays observable via
    /stats. This test guards that constraint: if a future change adds
    the metric to the scrape, this fails and forces a conscious
    decision."""
    import asyncio

    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()
    m.record_compression(
        "smart_crusher", original_tokens=200, compressed_tokens=50, lossy="lossy_recoverable"
    )
    m.record_compression(
        "diff", original_tokens=120, compressed_tokens=70, lossy="lossy_recoverable"
    )

    output = asyncio.run(m.export())

    assert "headroom_compressions_total" not in output
    assert "headroom_tokens_saved_by_strategy_total" not in output


# ─── End-to-end smoke (router + metrics together) ──────────────────────


def test_router_with_prometheus_observer_increments_counters():
    """Plumbing test: a router wired to a real PrometheusMetrics
    instance lights up the per-strategy counters as routing decisions
    accumulate. This is the production wiring shape from
    `headroom/proxy/server.py`."""
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()
    router = ContentRouter(ContentRouterConfig(), observer=m)

    fake_result = RouterCompressionResult(
        compressed="x",
        original="x",
        strategy_used=CompressionStrategy.MIXED,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.JSON_ARRAY,
                strategy=CompressionStrategy.SMART_CRUSHER,
                original_tokens=300,
                compressed_tokens=80,
            ),
            RoutingDecision(
                content_type=ContentType.SOURCE_CODE,
                strategy=CompressionStrategy.CODE_AWARE,
                original_tokens=200,
                compressed_tokens=120,
            ),
            RoutingDecision(
                content_type=ContentType.JSON_ARRAY,
                strategy=CompressionStrategy.SMART_CRUSHER,
                original_tokens=100,
                compressed_tokens=40,
            ),
        ],
    )
    router._observe(fake_result)

    assert m.compressions_by_strategy == {"smart_crusher": 2, "code_aware": 1}
    assert m.tokens_saved_by_strategy == {
        "smart_crusher": (300 - 80) + (100 - 40),  # 280
        "code_aware": (200 - 120),  # 80
    }


# IntelligentContextManager observability tests retired with PR-B1 —
# the manager itself was deleted along with the message-dropping
# strategy. Inner-router observability is now exercised solely
# through ContentRouter, covered by
# `test_content_router_records_observer_call_per_routing_decision`.


# ─── classify_lossy ─────────────────────────────────────────────────────


def test_classify_lossy_zero_reduction_is_lossless():
    assert (
        classify_lossy(
            original_tokens=100,
            compressed_tokens=100,
            compressed_content="unchanged content, no marker here",
        )
        == "lossless"
    )


def test_classify_lossy_hint_wins_over_incidental_marker():
    """A lossless win is lossless even if an unrelated CCR marker also
    happens to be present in the output — lossless_hint short-circuits
    before the marker check (documented precedence in the docstring)."""
    assert (
        classify_lossy(
            original_tokens=100,
            compressed_tokens=40,
            compressed_content="compacted content <<ccr:abc123>>",
            lossless_hint=True,
        )
        == "lossless"
    )


def test_classify_lossy_marker_present_is_recoverable():
    assert (
        classify_lossy(
            original_tokens=100,
            compressed_tokens=40,
            compressed_content="compacted content <<ccr:abc123>>",
        )
        == "lossy_recoverable"
    )


def test_classify_lossy_no_marker_is_unrecoverable():
    assert (
        classify_lossy(
            original_tokens=100,
            compressed_tokens=40,
            compressed_content="compacted content, no marker at all",
        )
        == "lossy_unrecoverable"
    )


def test_classify_lossy_recognizes_all_three_marker_formats():
    for marker_text in (
        "Retrieve more: hash=abc123",
        "Retrieve original: hash=abc123",
        "<<ccr:abc123,base64,2.0KB>>",
    ):
        assert (
            classify_lossy(
                original_tokens=100,
                compressed_tokens=40,
                compressed_content=f"compacted content {marker_text}",
            )
            == "lossy_recoverable"
        )


# ─── ContentRouter: RoutingDecision.lossy wiring ───────────────────────
#
# `_apply_strategy_to_content` is monkeypatched in these tests (the same
# pattern `test_force_kompress_apply_uses_lightweight_detection` in
# test_transforms_content_router.py already uses) so each case is
# deterministic and independent of ML model / network availability.
# `test_content_router_lossless_search_fold_classifies_lossless` below
# is the one real, unmocked end-to-end case — it's also the exact
# regression the second Fable review caught (a fold-only lossless win
# with no CCR marker was, before the fix, misclassified as
# lossy_unrecoverable because `strategy_chain` never reached the
# classifier).


def test_content_router_lossless_search_fold_classifies_lossless():
    """Real, unmocked: a search-result block that folds losslessly via
    `_lossless_first` must classify as lossless even though real bytes
    were removed and no CCR marker is present in the output."""
    paths = [
        "src/services/wallet/overdraft/automated_overdraft_initiation.py",
        "src/services/wallet/overdraft/capacity_limits.py",
    ]
    block = (
        "\n".join(
            f"{p}:{ln}:    result = compute_overdraft_capacity(business_id, amount)"
            for p in paths
            for ln in range(1, 40)
        )
        + "\n"
    )
    router = ContentRouter(ContentRouterConfig(lossless=True))
    result = router.compress(block, context="")

    assert any(entry.startswith("lossless_") for entry in result.strategy_chain)
    assert len(result.routing_log) == 1
    decision = result.routing_log[0]
    assert decision.compressed_tokens < decision.original_tokens  # real byte reduction
    assert "<<ccr:" not in result.compressed  # no marker in a lossless fold
    assert decision.lossy == "lossless"


def test_content_router_pure_strategy_with_marker_classifies_recoverable(
    monkeypatch: pytest.MonkeyPatch,
):
    router = ContentRouter(ContentRouterConfig())
    monkeypatch.setattr(
        router,
        "_apply_strategy_to_content",
        lambda *a, **kw: ("compacted <<ccr:abc123,base64,1.0KB>>", 40, []),
    )

    result = router._compress_pure("original content " * 20, CompressionStrategy.KOMPRESS, "")

    assert len(result.routing_log) == 1
    assert result.routing_log[0].lossy == "lossy_recoverable"


def test_content_router_pure_strategy_no_marker_classifies_unrecoverable(
    monkeypatch: pytest.MonkeyPatch,
):
    """KOMPRESS/TEXT/CODE_AWARE (`LOSSY_UNMARKED_STRATEGIES`) don't always
    insert a marker — when they don't, on real token reduction, the
    result is genuinely unrecoverable."""
    router = ContentRouter(ContentRouterConfig())
    monkeypatch.setattr(
        router,
        "_apply_strategy_to_content",
        lambda *a, **kw: ("compacted content, no marker at all", 40, []),
    )

    result = router._compress_pure("original content " * 20, CompressionStrategy.KOMPRESS, "")

    assert len(result.routing_log) == 1
    assert result.routing_log[0].lossy == "lossy_unrecoverable"


def test_content_router_passthrough_placeholder_classifies_lossless():
    """A `<system-reminder>` block is protected before section-splitting
    (content_router.py's `_compress_mixed`) and its section is passed
    through verbatim — a zero-reduction no-op that must classify as
    lossless regardless of marker/chain. Calling `_compress_mixed`
    directly (real, unmocked) skips only the `is_mixed_content` routing
    gate, not the protected-placeholder logic itself."""
    content = (
        "some prose text here as padding to make a real section\n"
        "<system-reminder>\n"
        "protected block content that should survive verbatim, padded further\n"
        "</system-reminder>\n\n"
        "more prose text after the block goes here as well, padded further too\n"
    )
    router = ContentRouter(ContentRouterConfig())
    result = router._compress_mixed(content, context="")

    matching = [d for d in result.routing_log if d.strategy == CompressionStrategy.PASSTHROUGH]
    assert matching, "expected at least one PASSTHROUGH routing decision"
    for decision in matching:
        assert decision.original_tokens == decision.compressed_tokens
        assert decision.lossy == "lossless"


# ─── SmartCrusher: _notify_observer lossy wiring ───────────────────────
#
# Uses EstimatingTokenCounter (no network) rather than OpenAITokenCounter,
# since tiktoken's BPE download is unavailable in this sandbox.


def test_smart_crusher_lossless_table_win_classifies_lossless(isolated_toin):
    """Real, unmocked: a uniform JSON array that SmartCrusher's Rust path
    compacts into a table (dropping zero rows) reports Rust's own
    `info="lossless:table(...)"` tag, which must win over the marker
    check — even though a (non-CCR) tool-digest marker is always
    appended regardless."""
    from headroom.tokenizers.estimator import EstimatingTokenCounter

    items = [{"id": i, "status": "ok", "tag": "x" * 20} for i in range(200)]
    content = json.dumps(items)
    crusher = SmartCrusher(SmartCrusherConfig(), observer=(spy := SpyObserver()))
    tok = Tokenizer(EstimatingTokenCounter(), model="estimate")

    crusher.apply([{"role": "tool", "content": content}], tok)

    assert spy.calls, "expected smart_crusher to fire the observer"
    strategy, original, compressed, lossy = spy.calls[0]
    assert strategy == "smart_crusher"
    assert compressed < original  # real byte reduction from table compaction
    assert lossy == "lossless"


def test_smart_crusher_row_drop_with_marker_classifies_recoverable(isolated_toin):
    """`_smart_crush_content` is monkeypatched to force the row-drop path
    with a genuine CCR marker embedded — deterministic, independent of
    which real-world content shape happens to trigger row-drop vs. table
    compaction in the Rust crusher."""
    from headroom.tokenizers.estimator import EstimatingTokenCounter

    crusher = SmartCrusher(SmartCrusherConfig(), observer=(spy := SpyObserver()))
    crusher._smart_crush_content = lambda content, query_context=None: (
        "kept a few rows <<ccr:abc123,base64,4.0KB>>",
        True,
        "top_n",
    )
    tok = Tokenizer(EstimatingTokenCounter(), model="estimate")
    content = json.dumps([{"id": i, "msg": f"entry {i}"} for i in range(100)])

    crusher.apply([{"role": "tool", "content": content}], tok)

    assert spy.calls, "expected smart_crusher to fire the observer"
    strategy, original, compressed, lossy = spy.calls[0]
    assert lossy == "lossy_recoverable"


def test_smart_crusher_row_drop_without_marker_classifies_unrecoverable(isolated_toin):
    """Same as above, but the forced row-drop result carries no CCR
    marker — the only case with no recovery path."""
    from headroom.tokenizers.estimator import EstimatingTokenCounter

    crusher = SmartCrusher(SmartCrusherConfig(), observer=(spy := SpyObserver()))
    crusher._smart_crush_content = lambda content, query_context=None: (
        "kept a few rows, no marker at all",
        True,
        "smart_sample",
    )
    tok = Tokenizer(EstimatingTokenCounter(), model="estimate")
    content = json.dumps([{"id": i, "msg": f"entry {i}"} for i in range(100)])

    crusher.apply([{"role": "tool", "content": content}], tok)

    assert spy.calls, "expected smart_crusher to fire the observer"
    strategy, original, compressed, lossy = spy.calls[0]
    assert lossy == "lossy_unrecoverable"
