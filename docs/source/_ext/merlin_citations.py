"""Citation-impact components for the reproduced-papers documentation.

Renders citation counts for the papers reproduced in MerLin from two
committed data files (the Sphinx build never touches the network):

- ``_data/citations/papers.json`` — the paper registry (title, authors,
  year, venue, DOI, reproduction docname, optional ``not_indexed`` flag);
- ``_data/citations/citations.json`` — counts fetched from OpenAlex by
  ``docs/fetch_citations.py``, with the fetch date.

Three directives are provided:

- ``.. merlin-citations-summary::`` — impact banner (papers reproduced,
  total citations, data source and freshness date);
- ``.. merlin-citations-table::`` — all reproduced papers sorted by
  citation count, linking to each reproduction page and paper DOI;
- ``.. merlin-citations-badge:: <key>`` — one paper's citation count, for
  use on its reproduction page.

Registry/citation mismatches (unknown key, missing count for a paper not
marked ``not_indexed``) are hard directive errors so the build fails
loudly instead of rendering stale or partial numbers.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.errors import NoUri

_DATA_DIR = "_data/citations"
_REGISTRY_FILE = "papers.json"
_CITATIONS_FILE = "citations.json"


class MerlinCitationsSummaryNode(nodes.General, nodes.Element):
    """Docutils node carrying the aggregate citation banner data."""


class MerlinCitationsTableNode(nodes.General, nodes.Element):
    """Docutils node carrying the per-paper citation table data."""


class MerlinCitationsBadgeNode(nodes.General, nodes.Element):
    """Docutils node carrying one paper's citation badge data."""


def _load_json(directive: Directive, filename: str) -> Any:
    """Load a citation data file relative to the docs source directory.

    Parameters
    ----------
    directive : Directive
        The requesting directive; used for error reporting and to
        register the file as a build dependency.
    filename : str
        File name inside ``_data/citations``.

    Returns
    -------
    Any
        Parsed JSON content.

    Raises
    ------
    sphinx.errors.SphinxError
        Via ``directive.error`` if the file is missing or unparsable.
    """
    env = directive.state.document.settings.env
    path = Path(env.srcdir) / _DATA_DIR / filename
    if not path.exists():
        raise directive.error(f"Citation data file '{path}' was not found.")
    env.note_dependency(str(path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise directive.error(f"Invalid JSON in '{path}': {exc}") from exc


def _load_citation_data(
    directive: Directive,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and cross-validate the registry and fetched citation counts.

    Returns
    -------
    tuple[list[dict], dict]
        The registry entries and the citations payload
        (``{"source", "fetched_at", "papers"}``).

    Raises
    ------
    sphinx.errors.SphinxError
        Via ``directive.error`` if a registry entry lacks required
        fields, or an indexed paper has no fetched citation count.
    """
    registry = _load_json(directive, _REGISTRY_FILE)
    citations = _load_json(directive, _CITATIONS_FILE)

    if not isinstance(registry, list) or not registry:
        raise directive.error(f"'{_REGISTRY_FILE}' must be a non-empty JSON list.")
    counts = citations.get("papers") if isinstance(citations, dict) else None
    if not isinstance(counts, dict):
        raise directive.error(
            f"'{_CITATIONS_FILE}' must be an object with a 'papers' mapping. "
            "Run docs/fetch_citations.py to generate it."
        )

    for entry in registry:
        required = ("key", "title", "authors_short", "year", "venue", "doi", "doc")
        missing = [field for field in required if not entry.get(field)]
        if missing:
            raise directive.error(
                f"Registry entry {entry.get('key', entry)!r} is missing "
                f"required fields: {missing}."
            )
        if not entry.get("not_indexed") and entry["key"] not in counts:
            raise directive.error(
                f"Paper '{entry['key']}' has no citation count in "
                f"'{_CITATIONS_FILE}'. Run docs/fetch_citations.py, or mark "
                'the registry entry with "not_indexed": true.'
            )
    return registry, citations


def _citation_count(entry: dict[str, Any], citations: dict[str, Any]) -> int | None:
    """Return a paper's citation count, or ``None`` when not indexed."""
    if entry.get("not_indexed"):
        return None
    return int(citations["papers"][entry["key"]]["cited_by_count"])


class MerlinCitationsSummaryDirective(Directive):
    """Render the aggregate citation-impact banner."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        registry, citations = _load_citation_data(self)
        counts = [
            count
            for entry in registry
            if (count := _citation_count(entry, citations)) is not None
        ]
        node = MerlinCitationsSummaryNode()
        node["paper_count"] = len(registry)
        node["total_citations"] = sum(counts)
        node["source"] = str(citations.get("source", "OpenAlex"))
        node["fetched_at"] = str(citations.get("fetched_at", "unknown"))
        return [node]


class MerlinCitationsTableDirective(Directive):
    """Render all reproduced papers sorted by citation count."""

    has_content = False

    def run(self) -> list[nodes.Node]:
        env = self.state.document.settings.env
        registry, citations = _load_citation_data(self)

        rows: list[dict[str, Any]] = []
        for entry in registry:
            if entry["doc"] not in env.found_docs:
                raise self.error(
                    f"Paper '{entry['key']}' references missing doc '{entry['doc']}'."
                )
            rows.append({
                **{
                    field: entry[field]
                    for field in (
                        "key",
                        "title",
                        "authors_short",
                        "year",
                        "venue",
                        "doi",
                        "doc",
                    )
                },
                "citations": _citation_count(entry, citations),
            })

        # Most cited first; papers not yet indexed sink to the bottom.
        rows.sort(key=lambda row: (row["citations"] is None, -(row["citations"] or 0)))

        node = MerlinCitationsTableNode()
        node["rows"] = rows
        node["source"] = str(citations.get("source", "OpenAlex"))
        node["fetched_at"] = str(citations.get("fetched_at", "unknown"))
        return [node]


class MerlinCitationsBadgeDirective(Directive):
    """Render one paper's citation count for its reproduction page."""

    required_arguments = 1
    has_content = False

    def run(self) -> list[nodes.Node]:
        registry, citations = _load_citation_data(self)
        key = self.arguments[0].strip()
        entry = next((item for item in registry if item["key"] == key), None)
        if entry is None:
            known = ", ".join(sorted(item["key"] for item in registry))
            raise self.error(
                f"Unknown paper key '{key}' for merlin-citations-badge. "
                f"Known keys: {known}."
            )
        node = MerlinCitationsBadgeNode()
        node["citations"] = _citation_count(entry, citations)
        node["source"] = str(citations.get("source", "OpenAlex"))
        node["fetched_at"] = str(citations.get("fetched_at", "unknown"))
        return [node]


def _doc_href(translator: Any, docname: str) -> str:
    """Resolve a relative URI from the current page to ``docname``."""
    try:
        return translator.builder.get_relative_uri(
            translator.builder.current_docname, docname
        )
    except NoUri:
        return "#"


def _freshness_html(source: str, fetched_at: str) -> str:
    """Return the shared 'source, as of date' footnote markup."""
    return (
        '<p class="mq-citations-freshness">Citation data: '
        f"{escape(source)}, as of {escape(fetched_at)}.</p>"
    )


def visit_citations_summary_html(
    translator: Any, node: MerlinCitationsSummaryNode
) -> None:
    tiles = (
        (f"{node['paper_count']}", "Papers reproduced"),
        (f"{node['total_citations']:,}", "Total citations"),
    )
    translator.body.append('<div class="mq-citations-summary">')
    for value, label in tiles:
        translator.body.append(
            '<div class="mq-citations-tile">'
            f'<span class="mq-citations-tile-value">{escape(value)}</span>'
            f'<span class="mq-citations-tile-label">{escape(label)}</span>'
            "</div>"
        )
    translator.body.append("</div>")
    translator.body.append(_freshness_html(node["source"], node["fetched_at"]))
    raise nodes.SkipNode


def visit_citations_table_html(translator: Any, node: MerlinCitationsTableNode) -> None:
    translator.body.append(
        '<table class="mq-citations-table docutils align-default">'
        "<thead><tr>"
        '<th class="head">Paper</th>'
        '<th class="head">Authors</th>'
        '<th class="head">Venue</th>'
        '<th class="head">Year</th>'
        '<th class="head mq-citations-count-col">Citations</th>'
        "</tr></thead><tbody>"
    )
    for row in node["rows"]:
        doc_href = _doc_href(translator, row["doc"])
        doi_href = f"https://doi.org/{row['doi']}"
        count = row["citations"]
        count_html = (
            f'<span class="mq-citations-count">{count:,}</span>'
            if count is not None
            else '<span class="mq-citations-count mq-citations-count-na" '
            'title="Not yet indexed by OpenAlex">&mdash;</span>'
        )
        translator.body.append(
            "<tr>"
            f'<td><a href="{escape(doc_href, quote=True)}">{escape(row["title"])}</a>'
            f' <a class="mq-citations-doi" href="{escape(doi_href, quote=True)}"'
            ' target="_blank" rel="noopener noreferrer"'
            ' title="Open the paper (DOI)">&#8599;</a></td>'
            f"<td>{escape(row['authors_short'])}</td>"
            f"<td>{escape(row['venue'])}</td>"
            f"<td>{row['year']}</td>"
            f'<td class="mq-citations-count-col">{count_html}</td>'
            "</tr>"
        )
    translator.body.append("</tbody></table>")
    translator.body.append(_freshness_html(node["source"], node["fetched_at"]))
    raise nodes.SkipNode


def visit_citations_badge_html(translator: Any, node: MerlinCitationsBadgeNode) -> None:
    count = node["citations"]
    if count is not None:
        value = f"<strong>Citations:</strong> {count:,}"
    else:
        value = "<strong>Citations:</strong> not yet indexed"
    translator.body.append(
        f'<p class="mq-citations-badge">{value}'
        '<span class="mq-citations-badge-source">'
        f"({escape(node['source'])}, as of {escape(node['fetched_at'])})</span></p>"
    )
    raise nodes.SkipNode


def depart_citations_node_html(translator: Any, node: nodes.Element) -> None:
    del translator, node


def visit_citations_node_unsupported(translator: Any, node: nodes.Element) -> None:
    del translator, node
    raise nodes.SkipNode


def depart_citations_node_unsupported(translator: Any, node: nodes.Element) -> None:
    del translator, node


def setup(app: Any) -> dict[str, Any]:
    """Register the citation directives and nodes with Sphinx."""
    app.add_directive("merlin-citations-summary", MerlinCitationsSummaryDirective)
    app.add_directive("merlin-citations-table", MerlinCitationsTableDirective)
    app.add_directive("merlin-citations-badge", MerlinCitationsBadgeDirective)

    unsupported = (
        visit_citations_node_unsupported,
        depart_citations_node_unsupported,
    )
    for node_class, html_visit in (
        (MerlinCitationsSummaryNode, visit_citations_summary_html),
        (MerlinCitationsTableNode, visit_citations_table_html),
        (MerlinCitationsBadgeNode, visit_citations_badge_html),
    ):
        app.add_node(
            node_class,
            html=(html_visit, depart_citations_node_html),
            latex=unsupported,
            text=unsupported,
            man=unsupported,
            texinfo=unsupported,
        )
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
