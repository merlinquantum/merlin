from docs.build_multiversion import (
    _redirect_page,
    discover_docnames,
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
