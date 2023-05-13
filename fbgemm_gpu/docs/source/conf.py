# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
import subprocess
import sys

import pytorch_sphinx_theme

for dir_i in os.listdir("../.."):
    if dir_i == "fbgemm_gpu":
        continue
    possible_dir = os.path.join("../..", dir_i)
    if os.path.isdir(possible_dir):
        sys.path.insert(0, possible_dir)


# Doxygen
subprocess.call("doxygen Doxyfile.in", shell=True)

# -- Project information -----------------------------------------------------
highlight_language = "c++"

project = "fbgemm"
copyright = "2022, FBGEMM team"
author = "FBGEMM team"

# The full version, including alpha/beta/rc tags
release = "0.1.2"

# breathe_projects_source = {"auto": ("../src/", ["auto_function.h", "auto_class.h"])}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

extensions = ["sphinx.ext.intersphinx", "breathe", "sphinx.ext.autodoc"]

intersphinx_mapping = {"pytorch": ("https://pytorch.org/docs/master", None)}

# Setup absolute paths for communicating with breathe / exhale where
# items are expected / should be trimmed by.
# This file is {repo_root}/docs/cpp/source/conf.py

breathe_projects = {"fbgemm_gpu": "../build/xml/", "codegen": "../build/xml/codegen/"}

breathe_default_project = "fbgemm_gpu"


# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]
html_theme_options = {
    "pytorch_project": "fbgemm",
    "collapse_navigation": True,
    "analytics_id": "UA-117752657-2",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
