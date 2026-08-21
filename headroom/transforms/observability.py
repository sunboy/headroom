"""Observability protocol for compression events.

A single `CompressionObserver` interface that any transform can call
after a real compression event. Concrete observers — Prometheus, OTel,
structured logs — implement this; transforms only see the protocol.

The motivating regression: `ContentRouter._record_to_toin` skipped
SmartCrusher on the assumption SmartCrusher recorded its own TOIN
events (it did when SmartCrusher was Python; it stopped when the Rust
port took over). The disconnect was invisible for three weeks because
no metric distinguished compression events by strategy. This module
exists so the next regression of that shape alerts on day 1: if
SmartCrusher events drop to zero in production, the Prometheus
counter shows it immediately.

Design choices, called out for posterity:

- **No fallback observer.** Callers pass `None` or pass a real
  observer. There is no "default no-op" instance — that would let a
  caller silently disable observability by forgetting to pass one,
  and we just spent a PR fixing exactly that class of bug. Be
  explicit.
- **No observer registry.** A single observer per transform instance.
  If you need multi-fanout, compose at the call site (one wrapper
  observer that forwards to N children) — but the trivial pattern
  doesn't need a registry baked in.
- **No batching.** Each compression event is one call. Volume is
  bounded by the number of routing decisions per request — small.
  Batching would only matter if observers had to round-trip to a
  remote system; production observers (Prometheus) are in-process
  counter increments, which are cheaper than the protocol dispatch.
- **Strategy as a string.** The router and crusher both already
  serialize their strategy as the enum's `.value` tag. Passing the
  string keeps observers from importing `CompressionStrategy` and
  lets non-router callers (e.g. SmartCrusher in legacy mode) emit
  the same shape without round-tripping through the enum.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..parser import CCR_RETRIEVAL_MARKER_RE


@runtime_checkable
class CompressionObserver(Protocol):
    """Receive one notification per real compression event.

    Implementations should be cheap — this lives on the proxy hot path,
    one call per routing decision per request. A Prometheus-counter
    increment is the right order of magnitude.

    Args:
        strategy: Lowercase tag identifying the compression strategy
            that ran. Matches `CompressionStrategy.<NAME>.value` for
            ContentRouter; SmartCrusher's legacy direct-call path
            passes the literal `"smart_crusher"`.
        original_tokens: Token count of the input the strategy
            received.
        compressed_tokens: Token count of the output the strategy
            produced. Equal to `original_tokens` for passthrough;
            less when compression saved tokens.
        lossy: One of `"lossless"`, `"lossy_recoverable"`, or
            `"lossy_unrecoverable"` — see `classify_lossy`. Required,
            not defaulted: this module's own history (see the module
            docstring) is a case study in a silent regression hiding
            behind an implicit default: be explicit here too.

    Implementations MUST NOT raise. If the observer needs to fail-
    over (Prometheus client misconfigured, OTel exporter offline)
    handle that internally — bubbling exceptions out of an observer
    would break the compression that just succeeded, which is the
    opposite of what observability should do. (See the audit
    in `RUST_DEV.md`: any silent regression is bad, but a noisy
    observer that breaks compression is worse.)
    """

    def record_compression(
        self,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
        lossy: str,
    ) -> None: ...


def classify_lossy(
    *,
    original_tokens: int,
    compressed_tokens: int,
    compressed_content: str,
    lossless_hint: bool = False,
) -> str:
    """Classify one compression event as lossless / lossy_recoverable / lossy_unrecoverable.

    Strategy-name membership (e.g. "KOMPRESS is always lossy and
    unrecoverable") is NOT reliable: KOMPRESS sometimes stores a real
    CCR-recoverable marker (`store_kompress_in_ccr`), and SmartCrusher's
    own tool-digest marker (`create_tool_digest_marker`, a `<headroom:`
    -prefixed string) is a *different* format that does not match
    `CCR_RETRIEVAL_MARKER_RE` — so SmartCrusher can be unrecoverable
    despite looking "marked." Recoverability is only knowable by
    checking the actual compressed output for a genuine CCR marker.

    Args:
        original_tokens: Token count of the input.
        compressed_tokens: Token count of the output.
        compressed_content: The actual compressed text, searched for a
            genuine CCR retrieval marker (`<<ccr:...>>`,
            `Retrieve more: hash=`, `Retrieve original: hash=`).
        lossless_hint: True when the caller already knows this was a
            structural, information-preserving win — e.g. ContentRouter's
            `lossless_<kind>` compaction labels, or SmartCrusher's Rust
            `strategy_info` starting with `"lossless:"`. Takes precedence
            over the marker check: a lossless win is lossless even if an
            unrelated marker also happens to be present in the output.

    Returns:
        `"lossless"`, `"lossy_recoverable"`, or `"lossy_unrecoverable"`.

    Known limitations:
    - `CompressionStore.store()` can reject a bare CCR marker string as
      an entry's "original" content (a producer that lost its source
      bytes upstream) — in that case the compressed output can still
      *contain* a marker-shaped string that will never resolve on
      retrieve. This function only checks the regex, so that
      rejected-store case reads as `lossy_recoverable` even though the
      hash won't resolve. Distinguishing it would require plumbing
      store-rejection status back to the classifier, out of proportion
      to what this classifier is for.
    - `ContentRouter`'s `SMART_CRUSHER`-strategy dispatch (the
      `_registry_compress("smart_crusher", ...)` branch in
      `_apply_strategy_to_content`, distinct from `SmartCrusher.apply()`'s
      own direct-pipeline path) doesn't currently surface a lossless
      signal into `strategy_chain` even when the underlying Rust crusher
      achieves a table/dedup compaction that drops nothing (verified: no
      CCR marker is inserted for that case either, since nothing needs
      retrieving) — a real reduction with neither a marker nor a hint, so
      it conservatively classifies as `lossy_unrecoverable`. Fixing this
      would mean threading a lossless signal through that dispatch
      branch, a separate change from what this classifier covers.
    """

    if compressed_tokens >= original_tokens:
        return "lossless"
    if lossless_hint:
        return "lossless"
    if CCR_RETRIEVAL_MARKER_RE.search(compressed_content):
        return "lossy_recoverable"
    return "lossy_unrecoverable"
