"""Generic feature-analysis data container.

Case-specific feature loaders (e.g.
``cases.alpha_d.feature_data.load_feature_matrix``) materialise feature
matrices as this dataclass; the generic
``feature_selection.pycaret_selection`` consumes it. The dataclass itself
is alpha-D-agnostic — it just carries the per-row feature matrix, the
target, the case grouping, and bookkeeping for the case-level CV split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureAnalysisData:
    X: np.ndarray  # [N, D] float32
    y: np.ndarray  # [N] float32
    groups: np.ndarray  # [N] int32, case index
    feature_names: list[str]
    target_name: str
    case_ids: list[str]  # len == n_cases
    rows_per_case: list[int]
    local_velocity_normalization: bool

    @property
    def n_cases(self) -> int:
        return len(self.case_ids)
