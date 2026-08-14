"""Unit tests for the citation-docs validation helpers.

The helpers live in the Sphinx extension at ``docs/source/_ext``, which is not
an installed package, so we add that directory to ``sys.path`` before importing.
"""

import sys
from pathlib import Path

_EXT = Path(__file__).resolve().parents[2] / "docs" / "source" / "_ext"
if str(_EXT) not in sys.path:
    sys.path.insert(0, str(_EXT))

import merlin_citations as mc  # noqa: E402


def test_unregistered_reproduction_pages_flags_unlisted_page():
    """A reproduction page absent from the registry is reported."""
    registry = [{"doc": "reproduced_papers/reproductions/foo"}]
    found = {
        "reproduced_papers/reproductions/foo",
        "reproduced_papers/reproductions/bar",
        "reproduced_papers/reproductions/template",
        "index",
    }
    assert mc._unregistered_reproduction_pages(registry, found) == [
        "reproduced_papers/reproductions/bar"
    ]


def test_unregistered_reproduction_pages_ignores_template_and_other_docs():
    """The template page and non-reproduction docs are never flagged."""
    registry = [{"doc": "reproduced_papers/reproductions/foo"}]
    found = {
        "reproduced_papers/reproductions/foo",
        "reproduced_papers/reproductions/template",
        "user_guide/index",
    }
    assert mc._unregistered_reproduction_pages(registry, found) == []
