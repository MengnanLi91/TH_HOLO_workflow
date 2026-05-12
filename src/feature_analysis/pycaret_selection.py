"""Deprecated shim — re-exports from ``feature_selection.pycaret_selection``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). Update call sites to the new
path before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "feature_analysis.pycaret_selection is deprecated; "
    "use feature_selection.pycaret_selection instead.",
    DeprecationWarning,
    stacklevel=2,
)

from feature_selection.pycaret_selection import (  # noqa: E402, F401
    build_dataframe,
    case_level_split,
    enforce_allowlist,
    run_pycaret_selection,
    write_selected_features,
)

__all__ = [
    "build_dataframe",
    "case_level_split",
    "enforce_allowlist",
    "run_pycaret_selection",
    "write_selected_features",
]
