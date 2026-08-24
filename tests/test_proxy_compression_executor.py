"""Audit follow-up C3: bounded compression executor + cancel-aware metrics.

Replaces ``asyncio.to_thread`` for ``pipeline.apply()`` calls with a dedicated
``ThreadPoolExecutor`` that's bounded by ``ProxyConfig.compression_max_workers``.

Locks the following invariants:

1. The pool exists and respects ``compression_max_workers`` (auto and explicit).
2. ``compression_in_flight`` increments while a compression is running and
   decrements after it completes — under load, the high-water mark moves up
   as expected.
3. When a compression call exceeds its timeout, the awaiter unblocks with
   ``TimeoutError`` — but the worker thread keeps running (Python cannot
   preempt running CPython bytecode or in-flight Rust calls), and when the
   work eventually completes, ``compression_leaked_threads`` increments.
4. Jobs that time out while still queued do not leak the running gauge.
5. ``/stats runtime.compression_executor`` surfaces the gauges + counters so
   operators can see leaked-thread rate and queue pressure.
6. **Bounded capacity reclamation.** A leaked (zombie) thread keeps
   occupying its pool slot for its full real runtime — Python cannot
   preempt it. Once the number of *concurrently* leaked threads reaches
   ``compression_leak_recycle_threshold`` (default: the pool's own
   ``max_workers`` — i.e. the pool has zero healthy slots left), the
   executor is recycled: a fresh, full-capacity ``ThreadPoolExecutor``
   replaces it, so new compression calls are served immediately instead
   of queuing behind zombies that may never return. The old pool's
   zombies are never interrupted (still impossible to preempt) — they
   keep running in the background, and their eventual completion still
   updates ``compression_leaked_threads`` (the lifetime counter).
   ``compression_leaked_in_flight`` (currently-leaked gauge) and
   ``compression_recycles_total`` (how many times the watchdog fired)
   make this observable.
7. ``compression_total`` counts every attempt submitted to the executor —
   the denominator that turns ``leaked_threads_total`` /
   ``queue_timeouts_total`` from raw counts into rates.
8. The compression-executor series are exported to Prometheus ``/metrics``
   (previously only in the JSON ``/stats``/``/health`` payload, which
   cannot be scraped or alerted on).

These tests also serve as documentation: anyone reading them sees that
"timeout fired" does not mean "compression was cancelled" — it means "we
stopped waiting; the worker is still going". A bounded pool plus the
leaked-thread counter is how we make that visible — and the recycling
watchdog (invariant 6) is how we make sure a burst of stuck jobs doesn't
permanently shrink the pool's effective capacity.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS  # noqa: F401
from headroom.proxy.server import ProxyConfig, create_app


def _make_proxy(
    compression_max_workers: int | None = None,
    compression_leak_recycle_threshold: int | None = None,
):
    """Construct a HeadroomProxy with a no-op pipeline. Returns the proxy."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        compression_max_workers=compression_max_workers,
        compression_leak_recycle_threshold=compression_leak_recycle_threshold,
    )
    app = create_app(config)
    return app.state.proxy


def test_compression_executor_default_size_matches_asyncio_default() -> None:
    """When ``compression_max_workers`` is None, the resolved size should
    match asyncio's default executor sizing (``min(32, (cpu+1)*4)`` style).
    """
    import os

    proxy = _make_proxy(compression_max_workers=None)
    expected = min(32, (os.cpu_count() or 1) * 4)
    assert proxy.compression_max_workers == expected
    assert proxy._compression_executor._max_workers == expected


def test_compression_executor_explicit_override() -> None:
    """``ProxyConfig.compression_max_workers=N`` is honored verbatim."""
    proxy = _make_proxy(compression_max_workers=3)
    assert proxy.compression_max_workers == 3
    assert proxy._compression_executor._max_workers == 3


def test_compression_executor_minimum_one_worker() -> None:
    """A non-positive override clamps to 1 (zero workers would deadlock)."""
    proxy = _make_proxy(compression_max_workers=0)
    assert proxy.compression_max_workers == 1


def test_in_flight_gauge_tracks_running_compressions() -> None:
    """While a compression is running, ``_compression_in_flight`` reads ≥ 1.
    After it completes, it returns to 0. The high-water mark records the
    peak observed.
    """
    proxy = _make_proxy(compression_max_workers=4)

    enter_event = threading.Event()
    release_event = threading.Event()
    observed: dict[str, int] = {}

    def _slow_compression():
        enter_event.set()
        # Block until the test thread reads in_flight from the gauge.
        release_event.wait(timeout=5.0)
        return "done"

    async def _drive():
        task = asyncio.create_task(
            proxy._run_compression_in_executor(_slow_compression, timeout=10.0)
        )
        # Wait for the worker to actually start.
        for _ in range(50):
            if enter_event.is_set():
                break
            await asyncio.sleep(0.01)
        with proxy._compression_metrics_lock:
            observed["mid_flight"] = proxy._compression_in_flight
            observed["mid_flight_max"] = proxy._compression_in_flight_max
        release_event.set()
        result = await task
        return result

    result = asyncio.run(_drive())
    assert result == "done"
    assert observed["mid_flight"] == 1, (
        f"in_flight should be 1 mid-call, got {observed['mid_flight']}"
    )
    assert observed["mid_flight_max"] >= 1
    # Decremented after task completes.
    with proxy._compression_metrics_lock:
        assert proxy._compression_in_flight == 0


def test_high_water_mark_persists_after_completion() -> None:
    """``_compression_in_flight_max`` is monotonic — never decreases."""
    proxy = _make_proxy(compression_max_workers=8)

    enter_events = [threading.Event() for _ in range(3)]
    release_events = [threading.Event() for _ in range(3)]

    def _make_slow(idx: int):
        def _slow():
            enter_events[idx].set()
            release_events[idx].wait(timeout=5.0)
            return idx

        return _slow

    async def _drive():
        tasks = [
            asyncio.create_task(proxy._run_compression_in_executor(_make_slow(i), timeout=10.0))
            for i in range(3)
        ]
        # Wait for all 3 to enter.
        for ev in enter_events:
            for _ in range(50):
                if ev.is_set():
                    break
                await asyncio.sleep(0.01)
        peak = proxy._compression_in_flight
        for ev in release_events:
            ev.set()
        for t in tasks:
            await t
        return peak

    peak = asyncio.run(_drive())
    assert peak == 3, f"Should have observed 3 concurrent compressions, got {peak}"
    # After all complete, in_flight is back to 0 but max remains 3.
    with proxy._compression_metrics_lock:
        assert proxy._compression_in_flight == 0
        assert proxy._compression_in_flight_max >= 3


def test_timeout_fires_and_leaked_thread_is_counted() -> None:
    """When the compression exceeds ``timeout``, the awaiter sees
    ``TimeoutError`` immediately. The worker keeps running; when it finishes,
    ``_compression_leaked_threads`` increments by 1.

    NOTE on the reclamation follow-up: this test locks a single-leak
    scenario with a 2-worker pool. The default recycle threshold equals
    ``compression_max_workers`` (2 here), so ONE leaked thread does not
    cross it and the executor is NOT recycled — this still exercises the
    original "leak is counted" contract unchanged. The bounded-recycling
    contract itself (leaks crossing the threshold DO reclaim capacity) is
    a new, additional invariant covered by
    ``test_leaked_threads_are_bounded_and_capacity_is_reclaimed`` below —
    it does not replace this test, since a single leak in a larger pool is
    exactly the "background noise, not an emergency" case the pool should
    tolerate without disruption.
    """
    proxy = _make_proxy(compression_max_workers=2)
    finished_event = threading.Event()
    timeout_seconds = 0.10

    def _slow_compression():
        # Sleep well past the timeout so the asyncio side cancels first.
        time.sleep(timeout_seconds * 5)
        finished_event.set()
        return "completed-after-deadline"

    async def _drive():
        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_slow_compression, timeout=timeout_seconds)

    asyncio.run(_drive())

    # Wait for the worker to actually finish (it ran past the deadline).
    finished_event.wait(timeout=2.0)
    # Give the worker thread a moment to update the counter under the lock.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with proxy._compression_metrics_lock:
            if proxy._compression_leaked_threads >= 1:
                break
        time.sleep(0.01)

    with proxy._compression_metrics_lock:
        assert proxy._compression_leaked_threads >= 1, (
            f"leaked_threads should be ≥ 1; got {proxy._compression_leaked_threads}. "
            f"The worker either didn't finish past the deadline, or the wrapper "
            f"didn't increment the counter."
        )
        # In-flight gauge restored.
        assert proxy._compression_in_flight == 0
        # New: compression_total is the attempts denominator — exactly 1
        # attempt was made, so leaked_threads_total / compression_total
        # is a well-defined 100% leak rate for this call.
        assert proxy._compression_total == 1
        # New: a single leak in a 2-worker pool does not cross the
        # (default) recycle threshold of 2 — no disruptive recycle for
        # what is, at pool scale, background noise.
        assert proxy._compression_recycles_total == 0
        # New: the zombie finished, so it no longer counts as
        # "currently leaked and consuming a slot".
        assert proxy._compression_leaked_in_flight == 0


def test_timeout_before_worker_start_does_not_leak_in_flight() -> None:
    """If a queued job times out before a worker starts, queued accounting
    is cleaned up without touching the running gauge.
    """
    proxy = _make_proxy(compression_max_workers=1)
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    def _blocking_compression():
        first_started.set()
        release_first.wait(timeout=5.0)
        return "first"

    def _queued_compression():
        second_started.set()
        return "second"

    async def _drive():
        first_task = asyncio.create_task(
            proxy._run_compression_in_executor(_blocking_compression, timeout=10.0)
        )
        for _ in range(50):
            if first_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert first_started.is_set()

        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_queued_compression, timeout=0.05)

        with proxy._compression_metrics_lock:
            mid_queued = proxy._compression_queued
            mid_in_flight = proxy._compression_in_flight
            queue_timeouts = proxy._compression_queue_timeouts

        release_first.set()
        assert await first_task == "first"
        return mid_queued, mid_in_flight, queue_timeouts

    mid_queued, mid_in_flight, queue_timeouts = asyncio.run(_drive())

    assert not second_started.is_set()
    assert mid_queued == 0
    assert mid_in_flight == 1
    assert queue_timeouts == 1
    with proxy._compression_metrics_lock:
        assert proxy._compression_queued == 0
        assert proxy._compression_in_flight == 0
        assert proxy._compression_leaked_threads == 0
        # New: two attempts were made (the blocking first job + the
        # queued-and-cancelled second job) even though only one ever ran
        # to completion under a leak — compression_total counts attempts,
        # not just leaks.
        assert proxy._compression_total == 2
        # New: cancelling a not-yet-started job is NOT a leak — it never
        # occupied a slot, so it must never count toward the
        # concurrently-leaked gauge or trigger a recycle.
        assert proxy._compression_leaked_in_flight == 0
        assert proxy._compression_recycles_total == 0


def test_compression_executor_metrics_appear_in_runtime_payload() -> None:
    """``/stats runtime.compression_executor`` surfaces the new gauges."""
    from fastapi.testclient import TestClient

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        compression_max_workers=5,
    )
    app = create_app(config)

    with TestClient(app) as client:
        # The compression_executor metrics are published from the runtime
        # payload (also surfaced in /health). Hit /health and look there.
        r = client.get("/health")
        assert r.status_code == 200
        runtime = r.json()["runtime"]
        assert "compression_executor" in runtime
        ce = runtime["compression_executor"]
        assert ce["max_workers"] == 5
        assert ce["queued"] == 0
        assert ce["running"] == 0
        assert ce["in_flight"] == 0
        assert ce["queue_timeouts_total"] == 0
        assert ce["queue_wait_seconds_total"] == 0.0
        assert ce["run_seconds_total"] == 0.0
        assert ce["leaked_threads_total"] == 0
        assert ce["source"] == "explicit"
        # New: denominator + reclamation-watchdog fields, additive to the
        # existing payload shape.
        assert ce["compression_total"] == 0
        assert ce["leaked_in_flight"] == 0
        assert ce["recycles_total"] == 0
        assert ce["recycle_threshold"] == 5  # defaults to max_workers


def test_explicit_None_resolves_to_auto_source() -> None:
    """When max_workers is None (default), the runtime payload reports
    ``source: auto``."""
    from fastapi.testclient import TestClient

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.json()["runtime"]["compression_executor"]["source"] == "auto"


# ---------------------------------------------------------------------------
# Bounded capacity reclamation (the heart of this follow-up fix).
# ---------------------------------------------------------------------------


def test_leaked_threads_are_bounded_and_capacity_is_reclaimed() -> None:
    """The key regression test: saturate every worker slot with jobs that
    leak (their asyncio side times out while the thread keeps running),
    then submit a fast, well-behaved job and assert it completes within a
    bounded time.

    This is the test the repo lacked: the two pre-existing timeout tests
    only assert that a leak is *counted* — with ``compression_max_workers=1``
    they explicitly document that a second job cannot start until the
    already-timed-out first job's slot frees "naturally" (i.e. never, if
    the job never returns). Without the watchdog in
    ``_maybe_recycle_compression_executor_locked``, a fully-leaked pool
    stays fully leaked for the rest of the process lifetime: this test
    fails against that code (the final call to
    ``_run_compression_in_executor`` would itself time out, because the
    fast job's work item just sits queued on a pool with zero free
    workers — none of the zombies below ever finish in time to free one).
    """
    max_workers = 2
    proxy = _make_proxy(compression_max_workers=max_workers)
    # Default recycle threshold == max_workers: only recycle once the
    # pool has genuinely zero healthy slots left.
    assert proxy._compression_leak_recycle_threshold == max_workers

    per_job_timeout = 0.05
    # Zombies finish on their own eventually (bounded sleep, not an
    # unreleased Event) so no background thread blocks interpreter exit
    # forever — but long enough to still be running when the fast job's
    # assertion below executes.
    zombie_sleep_seconds = 0.5

    def _make_zombie():
        def _zombie():
            time.sleep(zombie_sleep_seconds)
            return "zombie-done"

        return _zombie

    async def _drive():
        leaked_in_flight_after: list[int] = []
        recycles_after: list[int] = []

        # Saturate every worker slot, one at a time. Each submission
        # lands on a free worker (the pool isn't full yet) and then times
        # out on the asyncio side while the thread keeps sleeping — a
        # leak — per the documented C3 behavior.
        for _ in range(max_workers):
            with pytest.raises(asyncio.TimeoutError):
                await proxy._run_compression_in_executor(_make_zombie(), timeout=per_job_timeout)
            with proxy._compression_metrics_lock:
                leaked_in_flight_after.append(proxy._compression_leaked_in_flight)
                recycles_after.append(proxy._compression_recycles_total)

        # A fast, well-behaved job submitted AFTER the pool was fully
        # starved must complete within a bounded time. On unfixed code
        # this call itself raises ``asyncio.TimeoutError`` (or hangs
        # until this generous bound), because the pool has no free
        # worker and the zombies won't finish for another
        # ``zombie_sleep_seconds``. On fixed code it should return almost
        # immediately, because the watchdog already swapped in a fresh
        # pool during the loop above.
        fast_started = time.monotonic()
        fast_result = await proxy._run_compression_in_executor(lambda: "fast-done", timeout=5.0)
        fast_elapsed = time.monotonic() - fast_started
        return leaked_in_flight_after, recycles_after, fast_result, fast_elapsed

    leaked_in_flight_after, recycles_after, fast_result, fast_elapsed = asyncio.run(_drive())

    # Bounded: the concurrently-leaked gauge never exceeds the configured
    # cap at any point — the watchdog fires exactly when it's crossed,
    # never later.
    for observed in leaked_in_flight_after:
        assert observed <= proxy._compression_leak_recycle_threshold, (
            f"leaked_in_flight={observed} exceeded the configured cap "
            f"{proxy._compression_leak_recycle_threshold} — reclamation is "
            f"not bounded."
        )

    # The watchdog must have fired by the time every slot was leaked.
    assert recycles_after[-1] >= 1, (
        "compression executor was never recycled after every worker slot "
        "leaked — capacity was permanently lost."
    )

    # The forcing function: capacity was actually reclaimed, not just
    # counted. A generous bound (well under the zombies' own
    # ``zombie_sleep_seconds`` and the fast call's own 5s timeout) proves
    # the fast job ran on a fresh pool rather than queuing behind zombies.
    assert fast_result == "fast-done"
    assert fast_elapsed < 2.0, (
        f"fast job took {fast_elapsed:.3f}s to complete after the pool was "
        f"starved — capacity does not look reclaimed (expected well under "
        f"2s on a fresh pool)."
    )

    with proxy._compression_metrics_lock:
        # compression_total counts every attempt: max_workers zombies +
        # the 1 fast job.
        assert proxy._compression_total == max_workers + 1
        # leaked_threads_total (lifetime) is unaffected by recycling —
        # it's the numerator for a leak-rate calculation, not a gauge.
        assert proxy._compression_leaked_threads <= max_workers


def test_recycle_threshold_is_configurable() -> None:
    """``compression_leak_recycle_threshold`` can be set below the pool
    size to recycle more eagerly (e.g. operators who'd rather pay for
    more frequent, smaller disruptions than risk full starvation).
    """
    proxy = _make_proxy(compression_max_workers=4, compression_leak_recycle_threshold=1)
    assert proxy._compression_leak_recycle_threshold == 1

    def _zombie():
        time.sleep(0.3)
        return "zombie-done"

    async def _drive():
        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_zombie, timeout=0.05)
        with proxy._compression_metrics_lock:
            return proxy._compression_recycles_total

    recycles = asyncio.run(_drive())
    # A single leak already meets threshold=1 — recycled immediately,
    # even though 3 of the 4 workers are still healthy.
    assert recycles == 1


def test_compression_total_counts_every_attempt() -> None:
    """``compression_total`` increments once per call regardless of
    outcome (success, timeout-while-queued, or timeout-while-leaked) —
    the denominator for leak/timeout rates.
    """
    proxy = _make_proxy(compression_max_workers=2)

    async def _drive():
        # One successful attempt.
        await proxy._run_compression_in_executor(lambda: "ok", timeout=5.0)
        # One attempt that leaks.
        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(
                lambda: time.sleep(0.3) or "late", timeout=0.05
            )

    asyncio.run(_drive())

    # compression_total is incremented synchronously by the awaiting
    # coroutine (no race), but leaked_threads_total is only incremented
    # later, from the zombie thread's own ``finally`` block once its
    # 0.3s sleep finishes — poll for it rather than asserting instantly.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with proxy._compression_metrics_lock:
            if proxy._compression_leaked_threads >= 1:
                break
        time.sleep(0.01)

    with proxy._compression_metrics_lock:
        total = proxy._compression_total
        leaked = proxy._compression_leaked_threads
    assert total == 2
    # The rate is now derivable without a separate denominator lookup.
    assert 0.0 <= leaked / total <= 1.0
    assert leaked == 1


# ---------------------------------------------------------------------------
# Prometheus /metrics export (previously JSON-only).
# ---------------------------------------------------------------------------


def test_compression_executor_metrics_exported_to_prometheus() -> None:
    """The compression-executor series must be scrapeable from
    ``/metrics``, not just present in the JSON ``/stats``/``/health``
    payload — that's what makes them alertable.
    """
    proxy = _make_proxy(compression_max_workers=3)

    async def _drive():
        await proxy._run_compression_in_executor(lambda: "ok", timeout=5.0)

    asyncio.run(_drive())

    # ``/metrics`` reads directly from ``proxy.metrics.export()`` (see
    # the ``@app.get("/metrics")`` handler in server.py) — call it
    # directly rather than through a live TestClient/app.
    text = asyncio.run(proxy.metrics.export())

    for expected_name in (
        "headroom_compression_total",
        "headroom_compression_queued",
        "headroom_compression_in_flight",
        "headroom_compression_run_seconds_total",
        "headroom_compression_queue_wait_seconds_total",
        "headroom_compression_queue_timeouts_total",
        "headroom_compression_leaked_threads_total",
        "headroom_compression_leaked_in_flight",
        "headroom_compression_recycles_total",
        "headroom_compression_max_workers",
    ):
        assert expected_name in text, f"{expected_name} missing from /metrics export"

    assert "headroom_compression_total 1" in text
    assert "headroom_compression_max_workers 3" in text


def test_compression_executor_metrics_exported_via_metrics_endpoint() -> None:
    """End-to-end: hit the real ``/metrics`` HTTP endpoint (not just
    ``proxy.metrics.export()`` directly) and confirm the series round-trip
    through ``create_app`` wiring.
    """
    from fastapi.testclient import TestClient

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        compression_max_workers=2,
    )
    app = create_app(config)
    proxy = app.state.proxy

    async def _drive():
        await proxy._run_compression_in_executor(lambda: "ok", timeout=5.0)

    asyncio.run(_drive())

    with TestClient(app) as client:
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "headroom_compression_total" in r.text
        assert "headroom_compression_leaked_threads_total" in r.text
        assert "headroom_compression_recycles_total" in r.text
