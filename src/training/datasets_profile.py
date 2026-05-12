"""Deprecated shim — re-exports from ``cases.alpha_d.datasets.profile``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). The shim keeps existing imports
working with a deprecation warning; update call sites to the new path
before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "training.datasets_profile is deprecated; "
    "use cases.alpha_d.datasets.profile instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cases.alpha_d.datasets.profile import AlphaDProfileDataset  # noqa: E402, F401

__all__ = ["AlphaDProfileDataset"]
