"""Deprecated shim — delegates to ``cases.alpha_d.run_etl``.

Moved as part of the per-case repo refactor (Phase 1 of
``docs/dev/repo_layout_refactor_plan.md``). The shim keeps the legacy
``python run_alpha_d_etl.py`` invocation working with a deprecation
warning; update docs and tooling to the new path before Phase 4 drops
the shim.
"""

import warnings

warnings.warn(
    "src/run_alpha_d_etl.py is deprecated; "
    "use src/cases/alpha_d/run_etl.py instead.",
    DeprecationWarning,
    stacklevel=2,
)

from cases.alpha_d.run_etl import main  # noqa: E402, F401

__all__ = ["main"]


if __name__ == "__main__":
    main()
