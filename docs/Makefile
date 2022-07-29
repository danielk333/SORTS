# Minimal makefile for Sphinx documentation
#

PYVER:=$(shell python3 -c 'from sys import version_info as v; print("{0}.{1}".format(v[0], v[1]))')
PYTHON = python$(PYVER)

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
DOXYGEN       ?= doxygen

TAG ?= $(shell git describe --tag | sed -n -e's,v\([1-9]\.[0-9]*\)\.[0-9].*,\1,p')

FILES=

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -WT --keep-going -d build/doctrees $(PAPEROPT_$(PAPER)) \
                  $(SPHINXOPTS) source

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean html web htmlhelp latex changes linkcheck \
	dist dist-build gitwash-update version-check html-build latex-build \
	merge-doc show docenv

# # Catch-all target: route all unknown targets to Sphinx using the new
# # "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

wipe: | clean
	rm source/auto_gallery -r
	rm source/_autodoc -r

mv_github:
	rm -r ../docs
	mkdir ../docs
	cp -a build/html/. ../docs

github: | wipe html mv_github

# build documentation
generate: build/generate-stamp
build/generate-stamp: $(wildcard source/reference/*.rst)
	mkdir -p build
	touch build/generate-stamp

html: html-build
html-build: generate
	mkdir -p build/html build/doctrees
	$(PYTHON) preprocess.py
ifeq (, $(shell which $(DOXYGEN)))
	@echo "Unable to find 'Doxygen:$(DOXYGEN)', skip generating C/C++ API from comment blocks."
else
	$(DOXYGEN) build/doxygen/Doxyfile
endif
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html $(FILES)
	$(PYTHON) postprocess.py html build/html/*.html
	@echo
	@echo "Build finished. The HTML pages are in build/html."