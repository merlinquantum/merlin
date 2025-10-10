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

sys.path.insert(0, os.path.realpath("../../"))


merlin_metadata = metadata("merlinquantum")

REPO_PATH = Path(__file__).parent.parent.parent.resolve()

build_directory = os.path.join(REPO_PATH, "docs", "build")
if not os.path.exists(build_directory):
    os.makedirs(build_directory)

# -- Project information -----------------------------------------------------
author = merlin_metadata["Author"].capitalize()
project = merlin_metadata["Name"]
copyright = f"{datetime.now().year}, {author}"

release = merlin_metadata["Version"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.bibtex",
    "enum_tools.autoenum",
    "nbsphinx",
    "sphinx_multiversion",
]

suppress_warnings = ["autosectionlabel.*"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members": False,  # Don't document imported members to avoid duplicates
}

# Suppress duplicate object warnings for re-exported classes
suppress_warnings.extend(["autodoc.import_object"])

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# nbsphinx_allow_errors = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "renku"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 2,
    "titles_only": False,
    "version_selector": True,
}

html_style = "css/style.css"
html_logo = "_static/img/Merlin logo white 160x160.png"
html_favicon = "_static/img/Merlin icon white 32x32.ico"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
