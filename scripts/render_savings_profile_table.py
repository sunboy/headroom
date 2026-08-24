#!/usr/bin/env python3
"""Render (and sync) the Savings Profile table embedded in the docs.

``headroom.agent_savings._PROFILES`` is the single source of truth for every
named savings profile (``agent-90``, ``balanced``, ...). Historically the
docs describing those profiles were hand-maintained prose, which is exactly
how they drifted from the real values. This script generates the table
directly from ``_PROFILES`` (via ``dataclasses.fields``, so a new profile
field shows up automatically) instead.

Usage:

    python scripts/render_savings_profile_table.py            # print the table to stdout
    python scripts/render_savings_profile_table.py --check    # exit 1 if the committed docs are stale
    python scripts/render_savings_profile_table.py --write    # rewrite the committed docs in place

``tests/test_agent_savings.py::test_docs_savings_profile_table_matches_profiles``
runs ``--check`` so CI fails the moment ``_PROFILES`` and the docs disagree,
instead of the two silently drifting the way the prose version did.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from headroom.agent_savings import _PROFILES, AgentSavingsProfile  # noqa: E402

DOC_TARGETS: tuple[Path, ...] = (
    ROOT / "docs/content/docs/configuration.mdx",
    ROOT / "wiki/configuration.md",
)

MARKER_START = "<!-- headroom:savings-profiles:start -->"
MARKER_END = "<!-- headroom:savings-profiles:end -->"

_PROFILE_FIELD_NAMES: tuple[str, ...] = tuple(
    f.name for f in fields(AgentSavingsProfile) if f.name != "name"
)


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_table() -> str:
    """Render the profile table as a GitHub-flavored Markdown table."""

    names = list(_PROFILES)
    profiles = [_PROFILES[name] for name in names]

    header = "| Field | " + " | ".join(f"`{name}`" for name in names) + " |"
    separator = "|---|" + "---|" * len(names)
    lines = [header, separator]

    # `savings_percent` is a derived property (not a dataclass field) but is
    # the number most readers actually want, so it leads the table.
    lines.append(
        "| `savings_percent` | "
        + " | ".join(f"{profile.savings_percent}%" for profile in profiles)
        + " |"
    )
    for field_name in _PROFILE_FIELD_NAMES:
        cells = [f"`{field_name}`"]
        cells.extend(_format(getattr(profile, field_name)) for profile in profiles)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def rendered_block() -> str:
    return (
        f"{MARKER_START}\n"
        "_Generated from `headroom/agent_savings.py::_PROFILES` by "
        "`scripts/render_savings_profile_table.py` -- do not edit by hand._\n\n"
        f"{render_table()}"
        f"{MARKER_END}"
    )


def _splice(doc_text: str, block: str) -> str | None:
    """Replace the marker-delimited section of ``doc_text`` with ``block``.

    Returns ``None`` if the file has no marker pair (nothing to splice).
    """

    start = doc_text.find(MARKER_START)
    end = doc_text.find(MARKER_END)
    if start == -1 or end == -1 or end < start:
        return None
    end += len(MARKER_END)
    return doc_text[:start] + block + doc_text[end:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any committed doc's table doesn't match _PROFILES.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the committed docs in place to match _PROFILES.",
    )
    args = parser.parse_args()

    block = rendered_block()

    if not args.check and not args.write:
        print(block)
        return 0

    stale: list[Path] = []
    for path in DOC_TARGETS:
        doc_text = path.read_text(encoding="utf-8")
        spliced = _splice(doc_text, block)
        if spliced is None:
            print(f"error: {path} has no {MARKER_START} / {MARKER_END} markers", file=sys.stderr)
            return 1
        if spliced == doc_text:
            continue
        if args.write:
            path.write_text(spliced, encoding="utf-8")
            print(f"updated {path}")
        else:
            stale.append(path)

    if stale:
        print("Savings-profile table docs are stale (out of sync with _PROFILES):", file=sys.stderr)
        for path in stale:
            print(f"  {path}", file=sys.stderr)
        print(
            "Run: python scripts/render_savings_profile_table.py --write",
            file=sys.stderr,
        )
        return 1

    if args.check:
        print("Savings-profile table docs are in sync with _PROFILES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
