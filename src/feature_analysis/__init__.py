"""Feature analysis and selection component for the alpha_D surrogate.

Exposes a sklearn-based pipeline for understanding the data and selecting
input features with leak-safe group cross-validation (grouped by case).

Entry point: ``src/run_feature_analysis.py``.
"""

from feature_analysis.data_loader import (
    ALLOWLIST,
    GROUPED_FEATURES,
    FeatureAnalysisData,
    load_feature_matrix,
)
from feature_analysis.manifest import build_manifest, write_manifest
from feature_analysis.pycaret_selection import (
    build_dataframe,
    case_level_split,
    enforce_allowlist,
    run_pycaret_selection,
    write_selected_features,
)

__all__ = [
    "ALLOWLIST",
    "GROUPED_FEATURES",
    "FeatureAnalysisData",
    "load_feature_matrix",
    "build_manifest",
    "write_manifest",
    "build_dataframe",
    "case_level_split",
    "enforce_allowlist",
    "run_pycaret_selection",
    "write_selected_features",
]
