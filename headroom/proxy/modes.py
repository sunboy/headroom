"""Proxy run mode helpers.

Canonical modes:
- token: prioritize compression (history may be rewritten for max savings)
- cache: prioritize provider prefix cache stability (freeze prior turns)

Unknown-value handling is fail-fast, matching
``headroom.agent_savings.get_agent_savings_profile``'s behavior for an
unknown ``HEADROOM_SAVINGS_PROFILE``: an unrecognized ``HEADROOM_MODE``
raises ``ValueError`` with the list of valid modes/aliases rather than
silently falling back to the default. The two config knobs used to behave
differently -- an unknown savings profile crashed the proxy with a raw
traceback while an unknown mode was silently swallowed with only a log
line -- which is confusing in exactly the situation an operator is trying
to debug. The ``headroom proxy`` CLI command (``headroom/cli/proxy.py``)
validates the resolved mode and savings profile up front -- before any
startup output -- and catches this ``ValueError`` there, turning it into
one clean, actionable line instead of a traceback from deep inside
``run_server``/``create_app``. Direct library callers of
``create_app()``/``HeadroomProxy`` still see the raised ``ValueError``
itself, which is the correct behavior for a Python API. An empty/unset
value is not an "unknown value" and still resolves to ``default`` without
raising.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("headroom.proxy")

PROXY_MODE_TOKEN = "token"
PROXY_MODE_CACHE = "cache"

_MODE_ALIASES = {
    "token": PROXY_MODE_TOKEN,
    "token_mode": PROXY_MODE_TOKEN,
    "token_savings": PROXY_MODE_TOKEN,
    "token_headroom": PROXY_MODE_TOKEN,
    "cache": PROXY_MODE_CACHE,
    "cache_mode": PROXY_MODE_CACHE,
    "cost_savings": PROXY_MODE_CACHE,
}


def normalize_proxy_mode(mode: str | None, *, default: str = PROXY_MODE_TOKEN) -> str:
    """Normalize a user-provided proxy mode to canonical token/cache values.

    Raises:
        ValueError: ``mode`` is a non-empty string that isn't a known mode
            or alias. Callers at a startup boundary (CLI, proxy init) should
            catch this and print a clean, actionable message -- see the
            module docstring.
    """
    key = (mode or "").strip().lower()
    if not key:
        return default

    normalized = _MODE_ALIASES.get(key)
    if normalized is None:
        valid = ", ".join(sorted(_MODE_ALIASES))
        raise ValueError(f"unknown HEADROOM_MODE {mode!r}; expected one of: {valid}")

    if key != normalized:
        logger.info("HEADROOM_MODE alias '%s' normalized to '%s'", mode, normalized)
    return normalized


def is_token_mode(mode: str | None) -> bool:
    """Return True when mode resolves to token mode."""
    return normalize_proxy_mode(mode) == PROXY_MODE_TOKEN


def is_cache_mode(mode: str | None) -> bool:
    """Return True when mode resolves to cache mode."""
    return normalize_proxy_mode(mode) == PROXY_MODE_CACHE
