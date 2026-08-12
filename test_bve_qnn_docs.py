"""Tests for the BVE QNN docs PR (#294) — run from merlin2 repo root."""

import json
import os
import re

DOCS = os.path.join("docs", "source")
RST = os.path.join(DOCS, "reproduced_papers", "reproductions", "bve_qnn.rst")
EXT_JSON = os.path.join(
    DOCS, "_data", "galleries", "reproduced_papers", "bve_qnn_external_links.json"
)
ADV_JSON = os.path.join(
    DOCS,
    "_data",
    "galleries",
    "reproduced_papers",
    "reproduced_papers_advanced_training.json",
)
NOTEBOOK = os.path.join(DOCS, "notebooks", "reproduced_papers", "bve_qnn.ipynb")
IMAGE = os.path.join(DOCS, "_static", "reproduced_papers", "bve_qnn.png")
TOCTREE_RST = os.path.join(DOCS, "reproduced_papers", "reproduced_papers.rst")
EXAMPLES_RST = os.path.join(DOCS, "examples", "index.rst")


# ---------------------------------------------------------------------------
# RST checks
# ---------------------------------------------------------------------------

def test_rst_exists():
    assert os.path.isfile(RST), f"{RST} not found"


def test_rst_no_duplicate_reproduced_papers_target():
    """CI warning: Duplicate explicit target name 'reproduced_papers'."""
    with open(RST, encoding="utf-8") as f:
        text = f.read()
    targets = re.findall(r"`reproduced_papers\s*<", text)
    assert len(targets) <= 1, (
        f"Found {len(targets)} explicit 'reproduced_papers' targets — "
        "Sphinx emits a duplicate-target warning"
    )


def test_rst_has_paper_info_block():
    with open(RST, encoding="utf-8") as f:
        text = f.read()
    assert "Paper Information" in text
    assert "Jaderberg" in text
    assert "Phys. Rev. A" in text


def test_rst_no_dangling_qrnn_qlstm():
    """QRNN/QLSTM pages don't exist — references should be removed."""
    with open(RST, encoding="utf-8") as f:
        text = f.read()
    assert "QRNN" not in text, "Dangling QRNN reference still present"
    assert "QLSTM" not in text, "Dangling QLSTM reference still present"


def test_rst_hqpinn_is_crossref():
    """HQPINN should be a :doc: cross-ref, not plain text."""
    with open(RST, encoding="utf-8") as f:
        text = f.read()
    assert ":doc:`hqpinn`" in text, "HQPINN should use :doc: cross-reference"


def test_rst_runner_description():
    """lib/runner.py should NOT be called 'shared-runtime entrypoint'."""
    with open(RST, encoding="utf-8") as f:
        text = f.read()
    assert "shared-runtime entrypoint" not in text


def test_rst_has_mre_footnote():
    with open(RST, encoding="utf-8") as f:
        text = f.read()
    assert ".. [1]" in text, "MRE range footnote missing"


def test_rst_has_notebook_toctree():
    """Notebook should be in a hidden toctree, not just a bare :doc: link."""
    with open(RST, encoding="utf-8") as f:
        text = f.read()
    assert ".. toctree::" in text
    assert "notebooks/reproduced_papers/bve_qnn" in text


# ---------------------------------------------------------------------------
# Gallery JSON checks
# ---------------------------------------------------------------------------

def test_external_links_json_has_image():
    """CI fail: merlin-gallery skips cards without 'image'."""
    with open(EXT_JSON, encoding="utf-8") as f:
        cards = json.load(f)
    assert len(cards) >= 1
    for card in cards:
        assert "image" in card, f"Card '{card.get('title')}' missing 'image'"
        assert card["image"], "image value is empty"


def test_external_links_json_has_notebook_card():
    """Cassandre requested a gallery card linking the notebook."""
    with open(EXT_JSON, encoding="utf-8") as f:
        cards = json.load(f)
    notebook_cards = [c for c in cards if "notebook" in c.get("title", "").lower()
                      or "notebook" in c.get("doc", "")]
    assert len(notebook_cards) >= 1, "No notebook gallery card found"


def test_image_file_exists():
    assert os.path.isfile(IMAGE), f"Gallery image {IMAGE} not found on disk"


def test_advanced_training_json_has_bve_qnn():
    """Page must be reachable from the index gallery."""
    with open(ADV_JSON, encoding="utf-8") as f:
        cards = json.load(f)
    docs = [c.get("doc", "") for c in cards]
    assert any("bve_qnn" in d for d in docs), (
        "bve_qnn not found in reproduced_papers_advanced_training.json"
    )


def test_advanced_training_card_has_image():
    with open(ADV_JSON, encoding="utf-8") as f:
        cards = json.load(f)
    bve = [c for c in cards if "bve_qnn" in c.get("doc", "")]
    assert len(bve) == 1
    assert "image" in bve[0] and bve[0]["image"]


# ---------------------------------------------------------------------------
# Toctree wiring
# ---------------------------------------------------------------------------

def test_toctree_rst_has_bve_qnn():
    with open(TOCTREE_RST, encoding="utf-8") as f:
        text = f.read()
    assert "reproductions/bve_qnn" in text


def test_examples_index_has_bve_qnn():
    with open(EXAMPLES_RST, encoding="utf-8") as f:
        text = f.read()
    assert "bve_qnn" in text


# ---------------------------------------------------------------------------
# Notebook checks
# ---------------------------------------------------------------------------

def _load_notebook():
    with open(NOTEBOOK, encoding="utf-8") as f:
        return json.load(f)


def _cell_source(cell):
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(src)
    return src


def test_notebook_exists():
    assert os.path.isfile(NOTEBOOK)


def test_notebook_has_h1_title():
    """Titleless notebook breaks toctree with -W."""
    nb = _load_notebook()
    first_md = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            first_md = _cell_source(cell)
            break
    assert first_md is not None, "No markdown cell found"
    assert first_md.lstrip().startswith("# "), (
        "First markdown cell must start with an H1 heading"
    )


def test_notebook_has_prerequisites():
    nb = _load_notebook()
    first_md = _cell_source(nb["cells"][0])
    assert "Prerequisites" in first_md or "prerequisite" in first_md.lower()
    assert "sem_supervised_dataset.npz" in first_md
    assert "qnn_exp1_merlin_dualrail_depth32_step5000.pt" in first_md


def test_notebook_no_pip_noise():
    """pip install outputs should be cleared."""
    nb = _load_notebook()
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            text = out.get("text", [])
            if isinstance(text, list):
                text = "".join(text)
            assert "Requirement already satisfied" not in text, (
                "pip install output not cleared"
            )


def test_notebook_no_qnn_rebound():
    """The check cell should use qnn_check, not rebind qnn."""
    nb = _load_notebook()
    for cell in nb["cells"]:
        src = _cell_source(cell)
        if "spec_mappings" in src and "quantum_layer" in src:
            assert "qnn_check" in src, (
                "Check cell should use 'qnn_check' to avoid rebinding 'qnn'"
            )
            lines = src.strip().splitlines()
            first_line = lines[0].strip()
            assert not first_line.startswith("qnn ="), (
                "Check cell still starts with 'qnn = ...'"
            )
            break


def test_notebook_no_leaked_filename():
    """running_exp1 (4).ipynb should not appear."""
    nb = _load_notebook()
    full = "".join(_cell_source(c) for c in nb["cells"])
    assert "running_exp1" not in full, "Leaked local filename still present"


def test_notebook_no_duplicated_sentence():
    """The trainables/trainable duplication should be gone."""
    nb = _load_notebook()
    full = "".join(_cell_source(c) for c in nb["cells"])
    assert "trainables" not in full, "'trainables' typo still present"


def test_notebook_no_v3_label():
    nb = _load_notebook()
    full = "".join(_cell_source(c) for c in nb["cells"])
    assert "(v3," not in full, "Internal version label '(v3,' still present"


def test_notebook_train_from_scratch_flag():
    nb = _load_notebook()
    code = "".join(_cell_source(c) for c in nb["cells"] if c["cell_type"] == "code")
    assert "TRAIN_FROM_SCRATCH" in code, (
        "Training cell must use a TRAIN_FROM_SCRATCH flag"
    )


def test_notebook_weather_pde_has_period():
    nb = _load_notebook()
    full = "".join(_cell_source(c) for c in nb["cells"])
    assert "weather PDE." in full, "Sentence ending with 'weather PDE' needs a period"


# ---------------------------------------------------------------------------
# Git metadata
# ---------------------------------------------------------------------------

def test_no_cursor_coauthor_in_commits():
    """No Cursor co-author trailer in any commit on this branch."""
    import subprocess

    result = subprocess.run(
        ["git", "log", "--format=%B", "origin/main..HEAD"],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    assert "Co-authored-by: Cursor" not in result.stdout, (
        "Cursor co-author trailer found in commit history"
    )


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
