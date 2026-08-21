"""CCR proactive expansion: `headroom.ccr.expansion.items` delivered vs. discarded.

`ContextTracker.execute_expansions()` retrieves from the same `CompressionStore`
`headroom.ccr.operations` already covers, so a retrieve "hit" there says nothing
about whether the data actually reached the model. Cache mode computes the
expansion and then explicitly discards it (`is_cache_mode` gate in
`handle_anthropic_messages`) to preserve prefix-cache stability; token mode
appends it. No prior test exercised the cache-mode skip branch at all.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import headroom.proxy.handlers.anthropic as anthropic_handler_module
from headroom.ccr.context_tracker import ExpansionRecommendation
from headroom.proxy.server import ProxyConfig, create_app


class _SpyOtelMetrics:
    def __init__(self) -> None:
        self.expansion_calls: list[tuple[bool, int]] = []

    def record_ccr_expansion(self, *, delivered: bool, item_count: int) -> None:
        self.expansion_calls.append((delivered, item_count))


def _client(mode: str) -> TestClient:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        mode=mode,
        ccr_context_tracking=True,
        ccr_proactive_expansion=True,
        image_optimize=False,
    )
    return TestClient(create_app(config))


def _force_expansion(monkeypatch: pytest.MonkeyPatch, proxy) -> _SpyOtelMetrics:
    """Force `ccr_context_tracker` to recommend and return exactly one
    expansion, and capture `record_ccr_expansion` calls via a spy in place
    of the real OTEL metrics singleton."""
    monkeypatch.setattr(
        proxy.ccr_context_tracker,
        "analyze_query",
        lambda *a, **kw: [
            ExpansionRecommendation(hash_key="abc123", reason="query match", relevance_score=0.9)
        ],
    )
    monkeypatch.setattr(
        proxy.ccr_context_tracker,
        "execute_expansions",
        lambda recommendations: [
            {
                "hash": "abc123",
                "type": "full",
                "content": "expanded content",
                "item_count": 5,
                "reason": "query match",
            }
        ],
    )
    spy = _SpyOtelMetrics()
    monkeypatch.setattr(anthropic_handler_module, "get_otel_metrics", lambda: spy)
    return spy


async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    )


def _send_message(client: TestClient) -> httpx.Response:
    return client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "x-headroom-cwd": "/home/user/some-project",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "what about the auth middleware?"}],
        },
    )


def test_cache_mode_discards_expansion_and_reports_not_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(mode="cache") as client:
        proxy = client.app.state.proxy
        assert proxy.ccr_context_tracker is not None
        spy = _force_expansion(monkeypatch, proxy)
        proxy._retry_request = _fake_retry

        response = _send_message(client)

        assert response.status_code == 200, response.text
        assert spy.expansion_calls == [(False, 1)]


def test_token_mode_delivers_expansion_and_reports_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _client(mode="token") as client:
        proxy = client.app.state.proxy
        assert proxy.ccr_context_tracker is not None
        spy = _force_expansion(monkeypatch, proxy)
        proxy._retry_request = _fake_retry

        response = _send_message(client)

        assert response.status_code == 200, response.text
        assert spy.expansion_calls == [(True, 1)]
