"""Deprecated shim — re-exports from ``cases.alpha_d.physics.baseline``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). The shim keeps existing imports
working with a deprecation warning; update call sites to the new path
before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "training.alpha_d_baseline is deprecated; "
    "use cases.alpha_d.physics.baseline instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cases.alpha_d.physics.baseline import (  # noqa: E402, F401
    BaselineGeometry,
    alpha_d_baseline_profile,
    integrated_baseline_delta_p,
)

__all__ = [
    "BaselineGeometry",
    "alpha_d_baseline_profile",
    "integrated_baseline_delta_p",
]
