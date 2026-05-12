"""Deprecated shim — re-exports from ``cases.alpha_d.physics.targets``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). The shim keeps existing imports
working with a deprecation warning; update call sites to the new path
before Phase 4 drops the shim.
"""

import warnings

warnings.warn(
    "training.alpha_d_targets is deprecated; "
    "use cases.alpha_d.physics.targets instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cases.alpha_d.physics.targets import (  # noqa: E402, F401
    alpha_d_bulk_to_values,
    alpha_d_values_to_bulk,
    convert_alpha_d_values_between_bases,
    decode_alpha_d_target,
    encode_alpha_d_target,
    field_values_to_physical,
    is_alpha_d_target,
)

__all__ = [
    "alpha_d_bulk_to_values",
    "alpha_d_values_to_bulk",
    "convert_alpha_d_values_between_bases",
    "decode_alpha_d_target",
    "encode_alpha_d_target",
    "field_values_to_physical",
    "is_alpha_d_target",
]
