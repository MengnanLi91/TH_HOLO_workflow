"""Deprecated package shim — re-exports the public API from its new homes.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``):

- Generic feature-selection methods → ``feature_selection.*``.
- Alpha-D feature-data loader → ``cases.alpha_d.feature_data``.

This shim keeps existing ``from feature_analysis import X`` imports
working with a deprecation warning; update call sites to the new homes
before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "feature_analysis is deprecated; use feature_selection.* or "
    "cases.alpha_d.feature_data instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cases.alpha_d.feature_data import (  # noqa: E402, F401
    ALLOWLIST,
    GROUPED_FEATURES,
    FeatureAnalysisData,
    load_feature_matrix,
)
from feature_selection.manifest import build_manifest, write_manifest  # noqa: E402, F401
from feature_selection.pycaret_selection import (  # noqa: E402, F401
    build_dataframe,
    case_level_split,
    enforce_allowlist,
    run_pycaret_selection,
    write_selected_features,
)

__all__ = [
    "ALLOWLIST",
    "GROUPED_FEATURES",
    "FeatureAnalysisData",
    "load_feature_matrix",
    "build_manifest",
    "write_manifest",
    "build_dataframe",
    "case_level_split",
    "enforce_allowlist",
    "run_pycaret_selection",
    "write_selected_features",
]
