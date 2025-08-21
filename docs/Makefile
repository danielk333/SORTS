# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	sphinx-build -b html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rfv $(BUILDDIR)
	rm -rfv $(SOURCEDIR)/autogallery
	rm -rfv $(SOURCEDIR)/autosummary
	rm -rfv $(SOURCEDIR)/autopackages
	rm -rfv jupyter_execute
