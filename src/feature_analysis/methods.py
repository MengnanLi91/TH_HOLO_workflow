"""Deprecated shim — re-exports from ``feature_selection.methods``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). Update call sites to the new
path before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "feature_analysis.methods is deprecated; "
    "use feature_selection.methods instead.",
    DeprecationWarning,
    stacklevel=2,
)

from feature_selection.methods import (  # noqa: E402, F401
    MethodResult,
    borda_consensus,
    build_report,
    collapse_blocks_to_selection,
    run_baseline,
    run_methods,
    score_f_regression,
    score_gbr_permutation,
    score_lasso,
    score_mutual_info,
    score_rfecv_ridge,
    score_sequential_ridge,
)

__all__ = [
    "MethodResult",
    "borda_consensus",
    "build_report",
    "collapse_blocks_to_selection",
    "run_baseline",
    "run_methods",
    "score_f_regression",
    "score_gbr_permutation",
    "score_lasso",
    "score_mutual_info",
    "score_rfecv_ridge",
    "score_sequential_ridge",
]
