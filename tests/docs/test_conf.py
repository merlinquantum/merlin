"""Tests for the Sphinx configuration module."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def test_conf_excludes_legacy_orphaned_kernel_notebook() -> None:
    """The docs config ignores the legacy orphan notebook path seen in CI."""
    repo_root = Path(__file__).resolve().parents[2]
    conf_path = repo_root / "docs" / "source" / "conf.py"
    spec = importlib.util.spec_from_file_location("merlin_docs_conf", conf_path)
    assert spec is not None
    assert spec.loader is not None

    conf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf)

    assert "notebooks/classical_vs_quantum_kernels_iris.ipynb" in conf.exclude_patterns
