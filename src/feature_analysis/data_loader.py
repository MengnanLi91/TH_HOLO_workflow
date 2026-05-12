"""Deprecated shim — re-exports from ``cases.alpha_d.feature_data``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). This file's contents are
alpha-D-specific (allowlist + dataset loader), so they moved into the
alpha-D case package; the generic feature-selection methods moved to
``src/feature_selection/`` instead. Update call sites to either home
before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "feature_analysis.data_loader is deprecated; "
    "use cases.alpha_d.feature_data instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cases.alpha_d.feature_data import (  # noqa: E402, F401
    ALLOWLIST,
    BASE_ALLOWLIST,
    ENGINEERED_FEATURES,
    GROUPED_FEATURES,
    FeatureAnalysisData,
    build_engineered_feature_map,
    load_feature_matrix,
)

__all__ = [
    "ALLOWLIST",
    "BASE_ALLOWLIST",
    "ENGINEERED_FEATURES",
    "GROUPED_FEATURES",
    "FeatureAnalysisData",
    "build_engineered_feature_map",
    "load_feature_matrix",
]
