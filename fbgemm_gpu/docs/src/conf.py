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
#
#
# Configuration is based on:
# https://github.com/pytorch/pytorch/blob/main/docs/cpp/source/conf.py

import os
import sys

import pytorch_sphinx_theme


# -- Project information -----------------------------------------------------

project = "FBGEMM"
copyright = "2023, FBGEMM Team"
author = "FBGEMM Team"

# The short X.Y version.
version = "0.6"

# The full version, including alpha/beta/rc tags
release = "0.6.0"


# -- Path setup --------------------------------------------------------------

for dir_i in os.listdir("../.."):
    if dir_i == "fbgemm_gpu":
        continue
    possible_dir = os.path.join("../..", dir_i)
    if os.path.isdir(possible_dir):
        sys.path.insert(0, possible_dir)


# Setup absolute paths for communicating with breathe / exhale where
# items are expected / should be trimmed by.
# This file is {repo_root}/fbgemm_gpu/docs/src/conf.py
this_file_dir = os.path.abspath(os.path.dirname(__file__))

doxygen_xml_dir = os.path.join(
    os.path.dirname(this_file_dir),  # {repo_root}/fbgemm_gpu/docs
    "build",  # {repo_root}/fbgemm_gpu/docs/build
    "xml",  # {repo_root}/fbgemm_gpu/docs/build/xml
)

repo_root = os.path.dirname(  # {repo_root}
    os.path.dirname(  # {repo_root}/fbgemm_gpu
        os.path.dirname(  # {repo_root}/fbgemm_gpu/docs
            this_file_dir  # {repo_root}/fbgemm_gpu/docs/src
        )
    )
)


# -- General configuration ---------------------------------------------------

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# If true, Sphinx will warn about all references where the target cannot be
# found.
nitpicky = True

# Make sure the target is unique
autosectionlabel_prefix_document = True

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pytorch": ("https://pytorch.org/docs/main", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Breathe configuration ---------------------------------------------------

# This should be a dictionary in which the keys are project names and the values
# are paths to the folder containing the doxygen output for that project.
breathe_projects = {
    "FBGEMM": doxygen_xml_dir,
    "codegen": f"{doxygen_xml_dir}/codegen",
}

# This should match one of the keys in the breathe_projects dictionary and
# indicates which project should be used when the project is not specified on
# the directive.
breathe_default_project = "FBGEMM"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pytorch_sphinx_theme"
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# NOTE: sharing python docs resources
html_static_path = ["_static"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "pytorch_project": "fbgemm",
    "collapse_navigation": True,
    "display_version": True,
    "analytics_id": "UA-117752657-2",
}


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "fbgemm.tex",
        "FBGEMM Documentation",
        "FBGEMM Team",
        "manual",
    ),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "FBGEMM", "FBGEMM Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "FBGEMM",
        "FBGEMM Documentation",
        author,
        "FBGEMM",
        "One line description of project.",
        "Miscellaneous",
    ),
]
