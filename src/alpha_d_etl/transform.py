"""Deprecated shim — re-exports from ``cases.alpha_d.etl.transform``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). The shim keeps Hydra ``_target_``
lookups (e.g. ``alpha_d_etl.transform.AlphaDTransformation``) working with
a deprecation warning; update configs to the new path before Phase 4 drops
the shim.
"""

import warnings

warnings.warn(
    "alpha_d_etl.transform is deprecated; "
    "use cases.alpha_d.etl.transform instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cases.alpha_d.etl.transform import AlphaDTransformation  # noqa: E402, F401

__all__ = ["AlphaDTransformation"]
