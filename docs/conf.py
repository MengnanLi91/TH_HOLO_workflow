"""Sphinx configuration for the TH_HOLO_workflow documentation site."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# -- Make the installed package importable for autodoc ----------------------
# The package is `pip install -e .`'d from the repo's src/ layout, so it is
# already on sys.path inside the runtime container. We still add `src/` here
# to remain robust against builds run from a clean environment.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# -- Project information ----------------------------------------------------
project = "TH_HOLO_workflow"
author = "TH_HOLO_workflow contributors"
copyright = "2026, TH_HOLO_workflow contributors"  # noqa: A001
release = "0.1.0"


# -- General configuration --------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
]
# Note: `sphinx.ext.autosummary` is intentionally NOT enabled. The API
# pages render each package as one ``.. automodule::`` block per module
# (one page per logical package), which keeps the navigation flat and
# matches how readers reason about the codebase. If a single module
# grows large enough to need per-symbol stub pages, add `autosummary`
# back here and switch that specific module's API page over.

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"
language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

templates_path = ["_templates"]


# -- MyST configuration -----------------------------------------------------
myst_enable_extensions = [
    "colon_fence",       # :::{tab-set} and similar fenced directives
    "deflist",
    "tasklist",
    "fieldlist",
    "attrs_inline",
    "substitution",
    "smartquotes",
]

myst_heading_anchors = 3
myst_dmath_double_inline = True


# -- Autodoc ----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_member_order = "bysource"
autoclass_content = "class"

# Lazy / optional deps used inside the codebase. None are required at import
# time for the modules we autodoc, but list a couple of known-optional names
# defensively so future module-level imports do not break the build.
autodoc_mock_imports: list[str] = [
    "pycaret",
    "torch_geometric",
    "torch_scatter",
    "torch_sparse",
]


# -- Napoleon (NumPy-style docstrings) --------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True


# -- Intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
    # Hydra's documentation site does not publish an objects.inv file, so it
    # is intentionally omitted from intersphinx. Reference hydra classes by
    # name in prose rather than via cross-reference.
}
# Don't fail the build if intersphinx can't reach a remote inventory (e.g.
# offline build environments). Warnings about unresolved references stay on.
intersphinx_disabled_reftypes = ["std:doc"]


# -- HTML output ------------------------------------------------------------
html_theme = "furo"
html_title = "TH_HOLO_workflow"
html_short_title = "TH_HOLO"
html_static_path: list[str] = []

html_theme_options = {
    "announcement": (
        "TH_HOLO_workflow: PhysicsNeMo-based ETL and surrogate-modeling "
        "pipeline for MOOSE thermal-hydraulics simulations."
    ),
    "navigation_with_keys": True,
    "source_repository": "https://github.com/MengnanLi91/TH_HOLO_workflow/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Mermaid ---------------------------------------------------------------
# Newer sphinxcontrib-mermaid versions default to "raw" output (inline
# <pre class="mermaid">) which is the recommended path with the bundled
# mermaid.js loader; keep the default.

# -- Suppress noisy warnings (do not silence anything load-bearing) ---------
# `docutils` is suppressed to drop a small set of docstring-formatting
# parse warnings inside existing `src/` modules (this docs build is not
# allowed to edit src/). The specific docstrings flagged today are:
#   - cases.moose_grid.etl.data_sources.csv_source.CSVProbeSource.read_all
#   - cases.moose_grid.etl.data_sources.zarr_sink.MooseZarrSink.cleanup_temp_files
#   - dataset.moose_dataset (module docstring, block-quote unindent at L35)
#   - dataset.moose_dataset.MooseDataset (class docstring, inline emphasis L4)
# Fix these in `src/` to drop the blanket and re-enable docutils warnings.
#
# We DO NOT suppress `ref.python` — that category catches typoed cross-refs
# in the prose pages and must stay loud.
suppress_warnings: list[str] = [
    "docutils",
    "misc.highlighting_failure",
]

# Single-target ambiguity in autodoc: ``probe_data`` is an attribute on
# both `MooseRawData` and `MooseProcessedData` in the same module. The
# autoresolved cross-ref matches both targets and Sphinx emits a `ref.python`
# warning that does not respect `nitpick_ignore`. We install a narrow
# logging filter (below) to drop that one message while keeping all other
# `ref.python` warnings live.
# Nitpicky mode is intentionally **off**. Turning it on (or passing
# `sphinx-build -n`) escalates every unresolved cross-reference to a
# warning, including autodoc-driven references into physicsnemo and
# physicsnemo_curator (which we do not have intersphinx inventories for).
# Run a nitpicky build manually with `sphinx-build -n` when sweeping for
# typoed `{py:class}` references in prose. The defaults stay quiet.
nitpicky = False
nitpick_ignore: list[tuple[str, str]] = []


import logging as _logging  # noqa: E402


class _DropProbeDataAmbiguity(_logging.Filter):
    """Filter out the one cross-reference ambiguity we cannot fix from src/."""

    _NEEDLE = "more than one target found for cross-reference 'probe_data'"

    def filter(self, record: _logging.LogRecord) -> bool:
        message = record.getMessage()
        return self._NEEDLE not in message


_logging.getLogger("sphinx").addFilter(_DropProbeDataAmbiguity())
_logging.getLogger("sphinx.sphinx").addFilter(_DropProbeDataAmbiguity())
_logging.getLogger("").addFilter(_DropProbeDataAmbiguity())


# -- Helpful defaults for sphinx-copybutton --------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False


def _demote_duplicate_section_headers(app, what, name, obj, options, lines):
    """Demote known duplicate section headings to inline bold paragraphs.

    The two pycaret_selection modules share a near-identical module
    docstring that contains a ``V1 contract`` RST section heading. Both
    docstrings are rendered into the API site, which produces a
    "duplicate label" warning under ``-W``. Editing the source docstrings
    is out of scope for the docs build; instead we rewrite the heading
    on the fly inside autodoc so each rendered page still shows the same
    text but no longer registers a colliding implicit label.

    Section headings whose label collides are rewritten as
    ``**Title**`` (bold paragraph), which keeps the visual emphasis
    without producing an anchor.
    """
    if what != "module":
        return
    if not name.endswith(".pycaret_selection"):
        return

    headings_to_demote = {"V1 contract", "V1 Contract", "v1 contract"}

    new_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped in headings_to_demote and i + 1 < len(lines):
            underline = lines[i + 1].strip()
            if underline and set(underline) <= {"-", "=", "~", "^"}:
                new_lines.append(f"**{stripped}**")
                new_lines.append("")
                i += 2
                continue
        new_lines.append(line)
        i += 1

    lines[:] = new_lines


def setup(app):  # noqa: D401 - sphinx hook
    """Wire build-time autodoc event handlers."""
    app.connect("autodoc-process-docstring", _demote_duplicate_section_headers)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
