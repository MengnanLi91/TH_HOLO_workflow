"""Case-specific bridge from processed data to generic feature selection."""

from __future__ import annotations

from pathlib import Path

from feature_selection.data import FeatureAnalysisData

# TODO(case): replace these placeholders with reviewed, leakage-safe inputs.
ALLOWLIST: tuple[str, ...] = ("feature_a", "feature_b")


def load_feature_matrix(
    zarr_dir: str | Path,
    *,
    target: str,
    selected_from_allowlist: list[str] | None = None,
    exclude_cases: list[str] | None = None,
) -> FeatureAnalysisData:
    """Return rows, targets, and case groups for PyCaret.

    TODO(case): read the case's processed stores and return
    :class:`feature_selection.data.FeatureAnalysisData`. Enforce ``ALLOWLIST``
    here, exclude ``exclude_cases`` before building arrays, and assign every
    correlated row from one simulation the same integer group.

    ``selected_from_allowlist`` may restrict ``ALLOWLIST`` but must never add
    a feature. Target-derived or post-solution quantities must not enter
    ``X``.
    """
    del zarr_dir, target, selected_from_allowlist, exclude_cases
    raise NotImplementedError("TODO(case): implement the leakage-safe FeatureAnalysisData loader")
