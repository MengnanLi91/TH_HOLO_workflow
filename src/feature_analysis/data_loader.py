"""Data loader for feature analysis.

Flattens the per-case zarr stores produced by the alpha_D ETL pipeline
into dense numpy arrays ``(X, y, groups)`` suitable for sklearn.

Leakage controls
----------------
1. ``ALLOWLIST`` hard-codes the input features that are safe to consider.
   The YAML config may *restrict* this set via ``selected_from_allowlist``
   but cannot extend it. Unknown names raise ``ValueError``. Extending the
   allowlist is a code change so it appears in review.

2. ``groups`` is the per-row case index. Callers must use
   ``sklearn.model_selection.GroupKFold(groups=groups)`` for all CV;
   rows inside a case are spatially correlated.

3. Metadata attributes such as ``delta_p_case`` are *never* included as
   features here — they are target-adjacent and would leak.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from training.alpha_d_targets import (
    convert_alpha_d_values_between_bases,
    is_alpha_d_target,
)


BASE_ALLOWLIST: tuple[str, ...] = (
    "log10_Re",
    "Dr",
    "Lr",
    "z_hat",
    "d_local_over_D",
    "A_local_over_A",
    "is_upstream",
    "is_throat",
    "is_downstream",
)

ENGINEERED_FEATURES: tuple[str, ...] = (
    "log10_Re_throat",
    "log10_Re_local",
    "inv_Dr",
    "Dr_times_Lr",
    "z_hat_times_Dr",
    "z_hat_times_Lr",
    "dist_to_throat_start",
    "dist_to_throat_end",
    "dist_to_nearest_step",
)

# Full candidate set accepted by `selected_from_allowlist`.
ALLOWLIST: tuple[str, ...] = BASE_ALLOWLIST + ENGINEERED_FEATURES

GROUPED_FEATURES: dict[str, tuple[str, ...]] = {
    "region_onehot": ("is_upstream", "is_throat", "is_downstream"),
}


@dataclass
class FeatureAnalysisData:
    X: np.ndarray            # [N, D] float32
    y: np.ndarray            # [N] float32
    groups: np.ndarray       # [N] int32, case index
    feature_names: list[str]
    target_name: str
    case_ids: list[str]      # len == n_cases
    rows_per_case: list[int]
    local_velocity_normalization: bool

    @property
    def n_cases(self) -> int:
        return len(self.case_ids)

    def feature_block(self, name: str) -> list[int] | None:
        """Return column indices for a grouped block, or None if not present."""
        members = GROUPED_FEATURES.get(name)
        if members is None:
            return None
        return [self.feature_names.index(m) for m in members if m in self.feature_names]


def _parse_Dr(case_name: str) -> float:
    for part in case_name.split("__"):
        if part.startswith("Dr_"):
            return float(part[3:].replace("p", "."))
    return 0.0


def _resolve_features(selected: list[str] | None) -> list[str]:
    if selected is None:
        # Preserve historical default behaviour unless explicitly requested:
        # only base (non-engineered) features are used by default.
        return list(BASE_ALLOWLIST)
    allowed = set(ALLOWLIST)
    unknown = [c for c in selected if c not in allowed]
    if unknown:
        raise ValueError(
            f"selected_from_allowlist contains names outside ALLOWLIST: {unknown}. "
            f"To add a feature, edit ALLOWLIST in feature_analysis/data_loader.py."
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for name in ALLOWLIST:
        if name in selected and name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def build_engineered_feature_map(
    features: np.ndarray,
    raw_feature_names: list[str],
) -> dict[str, np.ndarray]:
    """Build leakage-safe engineered features from raw per-row feature columns."""
    feat_map = {n: i for i, n in enumerate(raw_feature_names)}

    log10_Re = features[:, feat_map["log10_Re"]].astype(np.float64)
    Dr = np.maximum(features[:, feat_map["Dr"]].astype(np.float64), 1e-12)
    Lr = np.maximum(features[:, feat_map["Lr"]].astype(np.float64), 1e-12)
    z_hat = features[:, feat_map["z_hat"]].astype(np.float64)
    d_local_over_D = np.maximum(
        features[:, feat_map["d_local_over_D"]].astype(np.float64),
        1e-12,
    )

    # Locate throat interfaces from one-hot region flags directly.
    throat_mask = features[:, feat_map["is_throat"]].astype(np.float64) > 0.5
    if np.any(throat_mask):
        z_throat_start = float(np.min(z_hat[throat_mask]))
        z_throat_end = float(np.max(z_hat[throat_mask]))
    else:
        # Fallback: if mask is absent/corrupted, put interfaces outside range.
        z_throat_start = 2.0
        z_throat_end = 2.0

    dist_to_throat_start = z_hat - z_throat_start
    dist_to_throat_end = z_hat - z_throat_end
    dist_to_nearest_step = np.minimum(
        np.abs(dist_to_throat_start),
        np.abs(dist_to_throat_end),
    )

    return {
        "log10_Re_throat": (log10_Re - np.log10(Dr)).astype(np.float32),
        "log10_Re_local": (log10_Re - np.log10(d_local_over_D)).astype(np.float32),
        "inv_Dr": (1.0 / Dr).astype(np.float32),
        "Dr_times_Lr": (Dr * Lr).astype(np.float32),
        "z_hat_times_Dr": (z_hat * Dr).astype(np.float32),
        "z_hat_times_Lr": (z_hat * Lr).astype(np.float32),
        "dist_to_throat_start": dist_to_throat_start.astype(np.float32),
        "dist_to_throat_end": dist_to_throat_end.astype(np.float32),
        "dist_to_nearest_step": dist_to_nearest_step.astype(np.float32),
    }


def load_feature_matrix(
    zarr_dir: str | Path,
    *,
    target: str = "log_alpha_D",
    selected_from_allowlist: list[str] | None = None,
    local_velocity_normalization: bool = True,
    min_Dr: float | None = None,
    exclude_cases: list[str] | None = None,
) -> FeatureAnalysisData:
    """Load all cases under ``zarr_dir`` into a flat ``FeatureAnalysisData``.

    Parameters
    ----------
    zarr_dir
        Directory containing ``*.zarr`` stores.
    target
        Target column name. Must exist in zarr ``target_names``.
    selected_from_allowlist
        Restrict the ALLOWLIST to this subset. Cannot add new names.
    local_velocity_normalization
        If True, rescale alpha_D-family targets to the local-velocity
        basis before returning them. Matches training behaviour when the
        MLP is trained with ``local_velocity_normalization: true``.
    min_Dr
        Drop cases with diameter ratio below this value (matches training).
    exclude_cases
        Drop cases whose stem is in this list.
    """
    zarr_dir = Path(zarr_dir)
    sim_paths = sorted(zarr_dir.glob("*.zarr"))
    if not sim_paths:
        raise FileNotFoundError(f"No .zarr stores found in {zarr_dir}")

    if exclude_cases:
        exclude_set = set(exclude_cases)
        sim_paths = [sp for sp in sim_paths if sp.stem not in exclude_set]
    if min_Dr is not None:
        sim_paths = [sp for sp in sim_paths if _parse_Dr(sp.stem) >= min_Dr]
    if not sim_paths:
        raise ValueError("All cases filtered out; check min_Dr / exclude_cases.")

    feat_cols = _resolve_features(selected_from_allowlist)

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    case_ids: list[str] = []
    rows_per_case: list[int] = []

    raw_feature_names: list[str] | None = None
    raw_target_names: list[str] | None = None
    tgt_col: int | None = None
    d_over_D_col_raw: int | None = None
    apply_local_velocity_normalization = False

    for sp in sim_paths:
        root = zarr.open(store=str(sp), mode="r")
        features = np.array(root["features"][:], dtype=np.float32)
        targets = np.array(root["targets"][:], dtype=np.float32)
        meta = root["metadata"]

        if raw_feature_names is None:
            raw_feature_names = list(json.loads(meta.attrs.get("feature_names", "[]")))
            raw_target_names = list(json.loads(meta.attrs.get("target_names", "[]")))

            raw_set = set(raw_feature_names)
            eng_set = set(ENGINEERED_FEATURES)
            missing = [c for c in feat_cols if c not in raw_set and c not in eng_set]
            if missing:
                raise ValueError(
                    "selected_from_allowlist entries not available from raw + engineered "
                    f"features: {missing}"
                )
            if target not in raw_target_names:
                raise ValueError(
                    f"target={target!r} not in zarr target_names={raw_target_names}"
                )
            tgt_col = raw_target_names.index(target)

            if local_velocity_normalization and is_alpha_d_target(target):
                if "d_local_over_D" not in raw_feature_names:
                    raise ValueError(
                        "local_velocity_normalization requires 'd_local_over_D' "
                        "in zarr feature_names."
                    )
                d_over_D_col_raw = raw_feature_names.index("d_local_over_D")
                apply_local_velocity_normalization = True

        engineered = build_engineered_feature_map(features, raw_feature_names)
        row_cols: list[np.ndarray] = []
        for name in feat_cols:
            if name in raw_feature_names:
                row_cols.append(features[:, raw_feature_names.index(name)].astype(np.float32))
            else:
                row_cols.append(engineered[name])
        x_chunks.append(np.column_stack(row_cols).astype(np.float32))

        y_case = targets[:, tgt_col].astype(np.float64)
        if d_over_D_col_raw is not None:
            d_over_D = features[:, d_over_D_col_raw].astype(np.float64)
            y_case = convert_alpha_d_values_between_bases(
                y_case,
                target_name=target,
                d_over_D=d_over_D,
                from_local_velocity_normalization=False,
                to_local_velocity_normalization=True,
            )
        y_chunks.append(y_case.astype(np.float32))

        case_id = str(meta.attrs.get("case_id", sp.stem))
        case_ids.append(case_id)
        rows_per_case.append(features.shape[0])

    X = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    groups = np.concatenate(
        [np.full(n, i, dtype=np.int32) for i, n in enumerate(rows_per_case)]
    )

    return FeatureAnalysisData(
        X=X,
        y=y,
        groups=groups,
        feature_names=feat_cols,
        target_name=target,
        case_ids=case_ids,
        rows_per_case=rows_per_case,
        local_velocity_normalization=apply_local_velocity_normalization,
    )
