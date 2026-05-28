# Documentation build for MULTIFID-TH.
#
# Quick start (from the repo root):
#   pip install -e ".[docs]"
#   make -C docs html
#   open docs/_build/html/index.html
#
# Set SPHINXOPTS on the command line to pass extra flags, e.g.:
#   make -C docs html SPHINXOPTS="-W -n"

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
PORT          ?= 8000

.PHONY: help html strict clean livehtml serve linkcheck

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo
	@echo "Project-specific targets:"
	@echo "  html      Build the HTML site to $(BUILDDIR)/html"
	@echo "  strict    Build with -W -n (warnings fatal, nitpicky refs)"
	@echo "  livehtml  Rebuild on change and serve at http://localhost:$(PORT)"
	@echo "  serve     Serve an already-built site at http://localhost:$(PORT)"
	@echo "  linkcheck Validate external links"
	@echo "  clean     Remove the $(BUILDDIR) directory"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

strict:
	@$(SPHINXBUILD) -b html -W -n "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

livehtml:
	@sphinx-autobuild --port $(PORT) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

serve:
	@python3 -m http.server -d "$(BUILDDIR)/html" $(PORT)

linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

clean:
	@rm -rf "$(BUILDDIR)"
