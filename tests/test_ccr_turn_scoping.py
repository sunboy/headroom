"""Tests for per-scope turn number tracking in ContextTracker.

Regression coverage for the bug where the CCR "turn number" recorded
against a compression event was driven by a single process-global
counter (``HeadroomProxyServer._turn_counter``) shared by every
conversation/workspace the proxy served. Two unrelated concurrent
conversations racing the same counter meant the recorded turn_number
reflected total proxy traffic, not either conversation's own turn.

The fix moves the counter into ``ContextTracker`` itself as a dict
keyed by scope (the CCR workspace key already resolved at the call
site for cross-project leak gating — see the module docstring in
``headroom/ccr/context_tracker.py``), with LRU eviction so the dict
stays bounded on a long-running proxy.

These tests exercise ``ContextTracker.next_turn_number`` /
``current_turn_number`` directly (the unit under test), plus a
regression check that ``track_compression`` still records exactly
what it did before once a caller supplies a turn number via
``next_turn_number``.
"""

import pytest

from headroom.ccr.context_tracker import ContextTracker, ContextTrackerConfig


class TestTurnScopingIsolation:
    """Two distinct scope keys must not share a turn sequence."""

    def test_interleaved_scopes_each_get_independent_sequence(self):
        """Interleaving requests for two workspaces must not cross-increment.

        This is the direct regression test for the bug: against the old
        process-global ``self._turn_counter`` design, interleaving calls
        for two conversations would produce a shared, ever-increasing
        counter (e.g. ws-a: 1, ws-b: 2, ws-a: 3, ws-b: 4, ...) instead of
        each conversation seeing its own 1, 2, 3, ... This test fails
        against that design and passes against the per-scope dict.
        """
        tracker = ContextTracker()

        # Interleave: a, b, a, b, a, b
        seq_a = []
        seq_b = []
        for _ in range(3):
            seq_a.append(tracker.next_turn_number("ws-a"))
            seq_b.append(tracker.next_turn_number("ws-b"))

        assert seq_a == [1, 2, 3]
        assert seq_b == [1, 2, 3]

    def test_many_interleaved_scopes_stay_independent(self):
        """N distinct scopes interleaved all produce their own 1..k sequence."""
        tracker = ContextTracker()
        scopes = [f"ws-{i}" for i in range(10)]
        expected = {scope: [] for scope in scopes}

        for turn in range(1, 6):
            for scope in scopes:
                n = tracker.next_turn_number(scope)
                expected[scope].append(n)

        for scope in scopes:
            assert expected[scope] == [1, 2, 3, 4, 5], scope


class TestTurnScopingMonotonicity:
    """A single scope key increments monotonically across its own turns."""

    def test_single_scope_increments_monotonically(self):
        tracker = ContextTracker()

        results = [tracker.next_turn_number("ws-solo") for _ in range(5)]

        assert results == [1, 2, 3, 4, 5]

    def test_current_turn_number_reflects_last_issued(self):
        tracker = ContextTracker()

        assert tracker.current_turn_number("ws-solo") == 0  # never issued

        tracker.next_turn_number("ws-solo")
        tracker.next_turn_number("ws-solo")
        n = tracker.next_turn_number("ws-solo")

        assert n == 3
        # current_turn_number is a read, not an advance.
        assert tracker.current_turn_number("ws-solo") == 3
        assert tracker.current_turn_number("ws-solo") == 3

    def test_current_turn_number_does_not_leak_across_scopes(self):
        tracker = ContextTracker()

        tracker.next_turn_number("ws-a")
        tracker.next_turn_number("ws-a")
        tracker.next_turn_number("ws-b")

        assert tracker.current_turn_number("ws-a") == 2
        assert tracker.current_turn_number("ws-b") == 1
        assert tracker.current_turn_number("ws-never-seen") == 0


class TestTurnScopingEviction:
    """The turn-counter map stays bounded under many distinct keys."""

    def test_map_stays_bounded_under_many_keys(self):
        config = ContextTrackerConfig(max_tracked_turn_scopes=5)
        tracker = ContextTracker(config)

        for i in range(50):
            tracker.next_turn_number(f"ws-{i}")

        assert len(tracker._turn_counters) == 5
        assert len(tracker._turn_counter_order) == 5

    def test_lru_evicts_oldest_untouched_scope_first(self):
        config = ContextTrackerConfig(max_tracked_turn_scopes=3)
        tracker = ContextTracker(config)

        tracker.next_turn_number("ws-0")
        tracker.next_turn_number("ws-1")
        tracker.next_turn_number("ws-2")
        # At capacity: ws-0, ws-1, ws-2 all tracked.

        tracker.next_turn_number("ws-3")
        # ws-0 (least recently touched) should be evicted.

        assert tracker.current_turn_number("ws-0") == 0
        assert tracker.current_turn_number("ws-1") == 1
        assert tracker.current_turn_number("ws-2") == 1
        assert tracker.current_turn_number("ws-3") == 1

    def test_touching_a_scope_refreshes_its_lru_position(self):
        config = ContextTrackerConfig(max_tracked_turn_scopes=3)
        tracker = ContextTracker(config)

        tracker.next_turn_number("ws-0")
        tracker.next_turn_number("ws-1")
        tracker.next_turn_number("ws-2")
        # Touch ws-0 again so it's no longer the least-recently-used.
        tracker.next_turn_number("ws-0")

        # Now ws-1 is the least-recently touched; adding ws-3 should
        # evict ws-1, not ws-0.
        tracker.next_turn_number("ws-3")

        assert tracker.current_turn_number("ws-0") == 2
        assert tracker.current_turn_number("ws-1") == 0  # evicted
        assert tracker.current_turn_number("ws-2") == 1
        assert tracker.current_turn_number("ws-3") == 1

    def test_evicted_scope_restarts_numbering_from_one(self):
        config = ContextTrackerConfig(max_tracked_turn_scopes=1)
        tracker = ContextTracker(config)

        first = tracker.next_turn_number("ws-a")
        # ws-a evicted by ws-b since capacity is 1.
        tracker.next_turn_number("ws-b")

        second = tracker.next_turn_number("ws-a")

        assert first == 1
        assert second == 1  # restarted, not 2

    def test_clear_resets_turn_counters(self):
        tracker = ContextTracker()
        tracker.next_turn_number("ws-a")
        tracker.next_turn_number("ws-a")

        tracker.clear()

        assert tracker.current_turn_number("ws-a") == 0
        assert tracker.next_turn_number("ws-a") == 1

    def test_default_config_max_turn_scopes_is_positive(self):
        """Sanity check the documented default bound is sane (not unbounded)."""
        config = ContextTrackerConfig()
        assert 0 < config.max_tracked_turn_scopes < 100_000


class TestTrackCompressionRegression:
    """track_compression's own recorded behavior is unchanged.

    track_compression's public signature (still taking an explicit
    ``turn_number``) is untouched by this fix — callers now source that
    number from ``ContextTracker.next_turn_number(scope_key)`` instead
    of a process-global counter, but what gets recorded for a given
    turn_number is identical to before.
    """

    @pytest.fixture(autouse=True)
    def _reset(self):
        from headroom.cache.compression_store import reset_compression_store

        reset_compression_store()
        yield
        reset_compression_store()

    def test_single_conversation_flow_matches_prior_behavior(self):
        tracker = ContextTracker()
        workspace = "ws-single-convo"

        # Simulate 3 request turns for one conversation, exactly as the
        # proxy handler does: resolve the next turn number for the
        # workspace once per request, then track each detected hash
        # under that turn number.
        turn_1 = tracker.next_turn_number(workspace)
        tracker.track_compression(
            hash_key="hash-a",
            turn_number=turn_1,
            tool_name="Bash",
            original_count=100,
            compressed_count=10,
            workspace_key=workspace,
            query_context="find all python files",
            sample_content="src/main.py, src/auth.py",
        )

        turn_2 = tracker.next_turn_number(workspace)
        tracker.track_compression(
            hash_key="hash-b",
            turn_number=turn_2,
            tool_name="Grep",
            original_count=50,
            compressed_count=5,
            workspace_key=workspace,
            query_context="search auth",
            sample_content="auth.py:12: def login()",
        )

        turn_3 = tracker.next_turn_number(workspace)
        tracker.track_compression(
            hash_key="hash-c",
            turn_number=turn_3,
            tool_name="Glob",
            original_count=20,
            compressed_count=2,
            workspace_key=workspace,
            query_context="list configs",
            sample_content="config.yaml, settings.json",
        )

        assert turn_1 == 1
        assert turn_2 == 2
        assert turn_3 == 3

        stats = tracker.get_stats()
        assert stats["tracked_contexts"] == 3
        by_hash = {c["hash"]: c for c in stats["contexts"]}
        assert by_hash["hash-a"]["turn"] == 1
        assert by_hash["hash-b"]["turn"] == 2
        assert by_hash["hash-c"]["turn"] == 3
        assert by_hash["hash-a"]["items"] == "10/100"
        assert by_hash["hash-b"]["items"] == "5/50"
        assert by_hash["hash-c"]["items"] == "2/20"

        assert set(tracker.get_tracked_hashes()) == {"hash-a", "hash-b", "hash-c"}

    def test_multiple_hashes_in_one_turn_share_the_turn_number(self):
        """All hashes detected within a single request share one turn number.

        This mirrors the proxy handler's loop: ``next_turn_number`` is
        called once per request, and every ``track_compression`` call
        for hashes detected in that same request/turn reuses that value
        (not incremented per-hash).
        """
        tracker = ContextTracker()
        workspace = "ws-batch"

        turn = tracker.next_turn_number(workspace)
        for i in range(4):
            tracker.track_compression(
                hash_key=f"hash-{i}",
                turn_number=turn,
                tool_name="Bash",
                original_count=100,
                compressed_count=10,
                workspace_key=workspace,
            )

        stats = tracker.get_stats()
        turns = {c["turn"] for c in stats["contexts"]}
        assert turns == {1}
