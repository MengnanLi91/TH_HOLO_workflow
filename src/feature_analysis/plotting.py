"""Deprecated shim — re-exports from ``feature_selection.plotting``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). Update call sites to the new
path before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "feature_analysis.plotting is deprecated; "
    "use feature_selection.plotting instead.",
    DeprecationWarning,
    stacklevel=2,
)

from feature_selection.plotting import save_feature_analysis_plots  # noqa: E402, F401

__all__ = ["save_feature_analysis_plots"]
