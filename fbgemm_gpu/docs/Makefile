#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Makefile for Doxygen and Sphinx documentation generation

# There is an issue with Sphinx and the Breathe extension, where running the
# build with linting enabled causes the HTML pages to be rendered incorrectly
# (only Doxygen pages will show up), even when all linting passes and the build
# exits normally.  On the other hand, we would like to keep linting around to
# minimize documentation errors such as broken links.
#
# To solve this problem, we run two build passes as part of the documentation
# build.  The first build is run with lint options on, while the second one is
# run without.  This gives us the best of both worlds at the cost of doubling
# the documentation build time, which isn't too large.
#
# As such, The `SPHINX_LINT` environment variable is used to enable this
# toggling.
ifdef SPHINX_LINT
SPHINXOPTS    ?= -j auto -b linkcheck -W
else
SPHINXOPTS    ?= -j auto
endif

SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = src
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build the C++ Doxygen documentation
doxygen:
	doxygen Doxyfile.in

# Clean out the build directory
clean:
	rm -rf build/*
