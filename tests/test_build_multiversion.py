from docs.build_multiversion import discover_docnames


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
