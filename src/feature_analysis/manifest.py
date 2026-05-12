"""Deprecated shim — re-exports from ``feature_selection.manifest``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). Update call sites to the new
path before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "feature_analysis.manifest is deprecated; "
    "use feature_selection.manifest instead.",
    DeprecationWarning,
    stacklevel=2,
)

from feature_selection.manifest import build_manifest, write_manifest  # noqa: E402, F401

__all__ = ["build_manifest", "write_manifest"]
