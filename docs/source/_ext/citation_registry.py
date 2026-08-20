"""Pure registry-validation helpers for the citation docs.

These functions are deliberately free of Sphinx/docutils imports so they can be
unit-tested without the docs toolchain installed. The Sphinx extension
``merlin_citations`` imports and uses them; the standalone fetch script keeps
its own copies since it must run without the docs package too.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

REPRODUCTIONS_PREFIX = "reproduced_papers/reproductions/"


def duplicate_keys(registry: list[dict[str, Any]]) -> list[str]:
    """Return registry keys that appear more than once, sorted."""
    keys = [entry["key"] for entry in registry]
    return sorted({key for key in keys if keys.count(key) > 1})


def unregistered_reproduction_pages(
    registry: list[dict[str, Any]], found_docs: Iterable[str]
) -> list[str]:
    """Return reproduction pages that exist but are absent from the registry."""
    registered = {entry["doc"] for entry in registry}
    expected = {
        doc
        for doc in found_docs
        if doc.startswith(REPRODUCTIONS_PREFIX) and not doc.endswith("/template")
    }
    return sorted(expected - registered)
