"""Unit tests for the citation-docs validation helpers.

The helpers live in the Sphinx extension at ``docs/source/_ext``, which is not
an installed package, so we add that directory to ``sys.path`` before importing.
"""

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
for _path in (_REPO / "docs" / "source" / "_ext", _REPO / "docs"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fetch_citations as fc  # noqa: E402
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


def test_duplicate_keys_reported_once_and_sorted():
    """Repeated registry keys are reported once each, sorted."""
    registry = [
        {"key": "a"},
        {"key": "b"},
        {"key": "a"},
        {"key": "b"},
        {"key": "c"},
    ]
    assert mc._duplicate_keys(registry) == ["a", "b"]


def test_duplicate_keys_empty_when_unique():
    """Unique keys yield no duplicates."""
    assert mc._duplicate_keys([{"key": "a"}, {"key": "b"}]) == []


def test_extract_counts_reads_valid_record():
    """A well-formed OpenAlex record yields the stored fields."""
    work = {"cited_by_count": 12, "id": "https://openalex.org/W123"}
    assert fc.extract_counts(work) == {"cited_by_count": 12, "openalex_id": "W123"}


def test_extract_counts_rejects_malformed_record():
    """A record missing required fields raises ValueError instead of crashing."""
    with pytest.raises(ValueError, match="unexpected OpenAlex payload"):
        fc.extract_counts({"id": "https://openalex.org/W1"})
