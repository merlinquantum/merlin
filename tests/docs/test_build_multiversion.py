"""Tests for multiversion documentation output and citation overlays."""

from docs.build_multiversion import (
    _redirect_page,
    discover_docnames,
    update_exported_citation_data,
    write_legacy_page_redirects,
)


def test_redirect_page_contains_redirect_and_fallback_link():
    target_url = "../../0.4/api_reference/api/merlin.core.html"

    assert _redirect_page(target_url, "Open the moved page") == "\n".join(
        [
            "<!doctype html>",
            '<meta charset="utf-8">',
            f'<meta http-equiv="refresh" content="0; url={target_url}">',
            f'<link rel="canonical" href="{target_url}">',
            f'<a href="{target_url}">Open the moved page</a>',
            "",
        ]
    )


def test_write_legacy_page_redirects_writes_nested_relative_urls(tmp_path):
    write_legacy_page_redirects(
        tmp_path,
        "0.4",
        ["index", "guide", "api_reference/api/merlin.core"],
    )

    assert not (tmp_path / "index.html").exists()
    assert (tmp_path / "guide.html").read_text() == _redirect_page(
        "0.4/guide.html",
        "This page moved to Merlin 0.4 documentation",
    )
    assert (
        tmp_path / "api_reference" / "api" / "merlin.core.html"
    ).read_text() == _redirect_page(
        "../../0.4/api_reference/api/merlin.core.html",
        "This page moved to Merlin 0.4 documentation",
    )


def test_discover_docnames_uses_posix_paths_and_skips_hidden_components(tmp_path):
    source_path = tmp_path / "source"
    (source_path / "api_reference" / "api").mkdir(parents=True)
    (source_path / ".ipynb_checkpoints").mkdir()
    (source_path / "nested" / ".hidden").mkdir(parents=True)

    (source_path / "api_reference" / "api" / "merlin.core.rst").touch()
    (source_path / "notebook.ipynb").touch()
    (source_path / ".ipynb_checkpoints" / "notebook-checkpoint.ipynb").touch()
    (source_path / "nested" / ".hidden" / "page.rst").touch()
    (source_path / ".hidden.rst").touch()

    assert discover_docnames(source_path) == [
        "api_reference/api/merlin.core",
        "notebook",
    ]


def test_update_exported_citation_data_updates_supported_tag(tmp_path):
    # Model a release tag created after citation tracking was introduced.
    checkout_path = tmp_path / "checkout"
    citations_directory = (
        checkout_path / "docs" / "source" / "_data" / "citations"
    )
    citations_directory.mkdir(parents=True)
    (citations_directory / "papers.json").write_text("[]")
    (citations_directory / "citations.json").write_text('{"old": true}')
    refreshed_citations = tmp_path / "refreshed-citations.json"
    refreshed_citations.write_text('{"new": true}')

    assert update_exported_citation_data(checkout_path, refreshed_citations)
    assert (citations_directory / "citations.json").read_text() == '{"new": true}'


def test_update_exported_citation_data_skips_tag_without_registry(tmp_path):
    # Older release tags have no citation registry and must remain untouched.
    checkout_path = tmp_path / "checkout"
    refreshed_citations = tmp_path / "refreshed-citations.json"
    refreshed_citations.write_text('{"new": true}')

    assert not update_exported_citation_data(checkout_path, refreshed_citations)
    assert not (
        checkout_path
        / "docs"
        / "source"
        / "_data"
        / "citations"
        / "citations.json"
    ).exists()
