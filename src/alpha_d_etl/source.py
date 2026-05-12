"""Deprecated shim — re-exports from ``cases.alpha_d.etl.source``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). The shim keeps Hydra ``_target_``
lookups (e.g. ``alpha_d_etl.source.AlphaDSource``) working with a
deprecation warning; update configs to the new path before Phase 4 drops
the shim.
"""

import warnings

warnings.warn(
    "alpha_d_etl.source is deprecated; "
    "use cases.alpha_d.etl.source instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cases.alpha_d.etl.source import AlphaDSource  # noqa: E402, F401

__all__ = ["AlphaDSource"]
