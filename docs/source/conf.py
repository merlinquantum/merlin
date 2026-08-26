# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
"""
conf.py used by sphinx to build docs

The repo is copied to the correct commit of the tag
Then this file is interpreted

"""

import os
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

# ``build_multiversion.py`` builds exported tag trees with this current config.
# These overrides point Sphinx at the versioned source tree while keeping this
# config file as the shared build entry point.
DOCS_SOURCE_PATH = Path(
    os.environ.get("MERLIN_DOCS_SOURCE_PATH", Path(__file__).parent)
).resolve()
CONFIG_SOURCE_PATH = Path(__file__).parent.resolve()
REPO_PATH = Path(
    os.environ.get("MERLIN_DOCS_REPO_PATH", DOCS_SOURCE_PATH.parent.parent)
).resolve()

sys.path.insert(0, str(REPO_PATH))
# Older tags may not contain newer docs extensions. Keep the current extension
# directory available, while still preferring extension files from the tag when
# they exist.
sys.path.insert(0, str(CONFIG_SOURCE_PATH / "_ext"))
sys.path.insert(0, str(DOCS_SOURCE_PATH / "_ext"))


merlin_metadata = metadata("merlinquantum")

build_directory = os.path.join(REPO_PATH, "docs", "build")
if not os.path.exists(build_directory):
    os.makedirs(build_directory)

# -- Project information -----------------------------------------------------
author = merlin_metadata["Author"].capitalize()
project = merlin_metadata["Name"]
copyright = f"{datetime.now().year}, {author}"

release = os.environ.get("MERLIN_DOCS_VERSION", merlin_metadata["Version"])

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "enum_tools.autoenum",
    "nbsphinx",
    "sphinx_multiversion",
    "merlin_gallery",
    "merlin_citations",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/2.10/", None),
    "perceval": ("https://perceval.quandela.net/docs/v1.2/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}


nitpick_ignore = [
    ("py:attr", "dst_type"),
    ("py:attr", "non_blocking"),
    ("py:class", "Dropout"),
    ("py:class", "BatchNorm"),
    ("py:class", "perceval.utils.states.BasicState"),
    ("py:class", "merlin.measurement.strategies._LegacyMeasurementStrategy"),
    ("py:class", "torch.nn.modules.loss._Loss"),
    ("py:class", "Module"),
    # Perceval's published v1.2 inventory does not expose these symbols under
    # the import paths used by Merlin's type hints.
    ("py:class", "AProcessor"),
    ("py:class", "RemoteJob"),
    ("py:class", "Sampler"),
    ("py:class", "perceval.algorithm.Sampler"),
    ("py:class", "perceval.runtime.AProcessor"),
    ("py:class", "perceval.runtime.RemoteJob"),
    ("py:class", "perceval.runtime.RemoteProcessor"),
    ("py:class", "perceval.runtime.session.ISession"),
    ("py:class", "pcvl.ACircuit"),
    ("py:class", "pcvl.SVDistribution"),
    ("py:attr", "dtype"),
    ("py:attr", "device"),
]

suppress_warnings = ["autosectionlabel.*"]
configured_bibtex_bibfiles = [
    "references.bib",
    "QML_library/QML_library_other_papers.bib",
    "QML_library/QML_library_reproduced_papers.bib",
    "QML_library/QML_library_reproduced_papers_to_do.bib",
    "QML_library/QML_library_reproduced_papers_in_progress.bib",
]
# Historical docs tags do not all contain the same bibliography files. Use only
# files present in the source tree currently being built.
bibtex_bibfiles = [
    bibtex_file
    for bibtex_file in configured_bibtex_bibfiles
    if (DOCS_SOURCE_PATH / bibtex_file).exists()
]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members": False,  # Don't document imported members to avoid duplicates
}
autodoc_typehints = "signature"

typehints_use_rtype = False
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Suppress duplicate object warnings for re-exported classes
suppress_warnings.extend(["autodoc.import_object"])

# Use absolute paths so exported tag trees load their own templates and static
# assets instead of the files next to this shared config.
templates_path = [str(DOCS_SOURCE_PATH / "_templates")]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    # ``Kernels.ipynb`` replaced this legacy notebook, but some build contexts
    # still surface the old filename and fail strict ``-W`` docs builds unless
    # it is explicitly ignored.
    "notebooks/classical_vs_quantum_kernels_iris.ipynb",
]

# nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "renku"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [str(DOCS_SOURCE_PATH / "_static")]

html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "version_selector": True,
}

html_style = "css/style.css"
html_logo = str(DOCS_SOURCE_PATH / "_static/img/Merlin logo white 160x160.png")
html_favicon = str(DOCS_SOURCE_PATH / "_static/img/Merlin icon white 32x32.ico")

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
