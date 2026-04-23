"""Tabular dataset for pointwise/axial-profile MLP training.

Reads per-case Zarr stores produced by the alpha_D ETL pipeline.
Each store contains:

    {case_name}.zarr/
        features/   float32 [N_stations, D_in]
        targets/    float32 [N_stations, D_out]
        metadata/   attrs: case_id, feature_names, target_names, ...

All cases are loaded and concatenated row-wise.  Splitting is done at
the case level via ``subset_by_case_indices``.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from feature_analysis.data_loader import (
    ENGINEERED_FEATURES,
    build_engineered_feature_map,
)
from training.alpha_d_targets import (
    convert_alpha_d_values_between_bases,
    is_alpha_d_target,
)


class TabularPairDataset(Dataset):
    """Reads a directory of ``.zarr`` stores and produces ``(x, y)`` pairs.

    Parameters
    ----------
    zarr_dir : str or Path
        Directory containing ``*.zarr`` stores.
    input_columns : list[str] or None
        Feature column names to select.  If *None*, use all features.
    output_columns : list[str] or None
        Target column names to select.  If *None*, use all targets.
    normalize : bool
        If *True*, z-score normalize input features after loading.
        Statistics are computed from the loaded data (or from externally
        supplied ``norm_stats``).
    norm_stats : dict or None
        Externally supplied ``{"x_mean": Tensor, "x_std": Tensor}``.
        If *None* and *normalize* is True, computed from the loaded data.
    throat_weight : float or None
        Stations where ``is_throat == 1`` receive this weight; others
        receive weight 1.
    downstream_weight : float or None
        Stations where ``is_downstream == 1`` receive this weight.
        Applied multiplicatively with ``throat_weight`` when both are set.
    include_case_idx : bool
        If *True*, ``__getitem__`` returns a case-index tensor as the last
        element so that per-case losses can be computed.
    min_Dr : float or None
        If set, exclude cases whose diameter ratio Dr is below this value.
        Dr is parsed from the case name (``Re_*__Dr_XpXXX__Lr_*``).
    """

    def __init__(
        self,
        zarr_dir: str | Path,
        input_columns: list[str] | None = None,
        output_columns: list[str] | None = None,
        normalize: bool = False,
        norm_stats: dict | None = None,
        norm_from_case_indices: list[int] | None = None,
        throat_weight: float | None = None,
        downstream_weight: float | None = None,
        include_case_idx: bool = False,
        exclude_cases: list[str] | None = None,
        local_velocity_normalization: bool = False,
        min_Dr: float | None = None,
    ):
        import json

        import zarr

        self.zarr_dir = Path(zarr_dir)
        sim_paths = sorted(self.zarr_dir.glob("*.zarr"))
        if not sim_paths:
            raise FileNotFoundError(f"No .zarr stores found in {self.zarr_dir}")

        # Filter out excluded cases
        if exclude_cases:
            exclude_set = set(exclude_cases)
            sim_paths = [sp for sp in sim_paths if sp.stem not in exclude_set]

        # Filter by minimum diameter ratio
        if min_Dr is not None:
            sim_paths = [sp for sp in sim_paths if self._parse_Dr(sp.stem) >= min_Dr]

        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_w: list[np.ndarray] = []
        case_ids: list[str] = []
        rows_per_case: list[int] = []
        case_meta_list: list[dict] = []
        has_weights = False
        engineered_feature_names = list(ENGINEERED_FEATURES)

        for sp in sim_paths:
            root = zarr.open(store=str(sp), mode="r")
            features = np.array(root["features"][:], dtype=np.float32)
            targets = np.array(root["targets"][:], dtype=np.float32)
            if "sample_weight" in root:
                weights = np.array(root["sample_weight"][:], dtype=np.float32)
                has_weights = True
            else:
                weights = np.ones(features.shape[0], dtype=np.float32)

            meta = root["metadata"]
            case_id = str(meta.attrs.get("case_id", sp.stem))

            # On first store, resolve column names
            if not case_ids:
                raw_feature_names = json.loads(meta.attrs.get("feature_names", "[]"))
                raw_target_names = json.loads(meta.attrs.get("target_names", "[]"))

                self._base_feature_names = list(raw_feature_names)
                self._all_feature_names = (
                    list(raw_feature_names) + engineered_feature_names
                )
                self._all_target_names = list(raw_target_names)

                if output_columns is not None:
                    tgt_map = {n: i for i, n in enumerate(raw_target_names)}
                    missing = [c for c in output_columns if c not in tgt_map]
                    if missing:
                        raise ValueError(f"Unknown output columns: {missing}")
                    self._tgt_idx = [tgt_map[c] for c in output_columns]
                    self.output_columns = list(output_columns)
                else:
                    self._tgt_idx = list(range(targets.shape[1]))
                    self.output_columns = list(raw_target_names)

            # Per-case metadata for physics-informed losses
            case_meta_list.append({
                "delta_p_case": float(meta.attrs.get("delta_p_case", 0.0)),
                "Lr": float(meta.attrs.get("Lr", 0.0)),
                "Dr": float(meta.attrs.get("Dr", 0.0)),
            })

            # Load ALL base features (derived columns need access to
            # source columns that may not be in input_columns).
            engineered = build_engineered_feature_map(features, raw_feature_names)
            engineered_cols = [
                engineered[name].reshape(-1, 1) for name in engineered_feature_names
            ]
            all_x.append(
                np.concatenate([features] + engineered_cols, axis=1).astype(np.float32)
            )
            all_y.append(targets[:, self._tgt_idx])
            all_w.append(weights)
            case_ids.append(case_id)
            rows_per_case.append(features.shape[0])

        # ----------------------------------------------------------
        # Concatenate
        # ----------------------------------------------------------
        full_x = np.concatenate(all_x, axis=0)  # [N, D_base]
        full_y = np.concatenate(all_y, axis=0)   # [N, D_out]

        # Store per-case metadata
        self._case_meta = case_meta_list
        self.local_velocity_normalization = False
        self.exclude_cases = list(exclude_cases) if exclude_cases else []

        # Store raw geometry columns (before normalization) for delta_p loss
        z_hat_col = (
            self._all_feature_names.index("z_hat")
            if "z_hat" in self._all_feature_names else None
        )
        d_over_D_col = (
            self._all_feature_names.index("d_local_over_D")
            if "d_local_over_D" in self._all_feature_names else None
        )
        self._raw_z_hat = (
            torch.from_numpy(full_x[:, z_hat_col].copy())
            if z_hat_col is not None else None
        )
        self._raw_d_local_over_D = (
            torch.from_numpy(full_x[:, d_over_D_col].copy())
            if d_over_D_col is not None else None
        )

        # Apply local-velocity normalization to alpha_D-family targets.
        if local_velocity_normalization and d_over_D_col is not None:
            d_over_D = full_x[:, d_over_D_col].astype(np.float64)
            transformed_any = False
            for j, tgt_name in enumerate(self.output_columns):
                if is_alpha_d_target(tgt_name):
                    full_y[:, j] = convert_alpha_d_values_between_bases(
                        full_y[:, j].astype(np.float64),
                        target_name=tgt_name,
                        d_over_D=d_over_D,
                        from_local_velocity_normalization=False,
                        to_local_velocity_normalization=True,
                    ).astype(np.float32)
                    transformed_any = True
            self.local_velocity_normalization = transformed_any

        # Resolve input columns
        if input_columns is not None:
            feat_map = {n: i for i, n in enumerate(self._all_feature_names)}
            missing = [c for c in input_columns if c not in feat_map]
            if missing:
                raise ValueError(f"Unknown input columns: {missing}")
            self._feat_idx = [feat_map[c] for c in input_columns]
            self.input_columns = list(input_columns)
        else:
            self._feat_idx = list(range(len(self._base_feature_names)))
            self.input_columns = list(self._base_feature_names)

        self._x = torch.from_numpy(full_x[:, self._feat_idx].copy())
        self._y = torch.from_numpy(full_y)
        raw_w = np.concatenate(all_w, axis=0)
        self._w = torch.from_numpy(raw_w).unsqueeze(-1) if has_weights else None
        self._case_ids_unique = case_ids
        self._rows_per_case = rows_per_case

        # Build per-row case index for subsetting
        self._row_case_idx = np.concatenate(
            [np.full(n, i, dtype=np.int32) for i, n in enumerate(rows_per_case)]
        )

        # ----------------------------------------------------------
        # Region weights (throat)
        # ----------------------------------------------------------
        throat_col_full = (
            self._all_feature_names.index("is_throat")
            if "is_throat" in self._all_feature_names else None
        )

        self.throat_weight = throat_weight
        self.downstream_weight = downstream_weight
        if throat_weight is not None and throat_weight > 0 and throat_col_full is not None:
            new_w = torch.ones(len(self._x), dtype=torch.float32)
            new_w[full_x[:, throat_col_full] > 0.5] = float(throat_weight)
            self._w = new_w.unsqueeze(-1)

        # Region weights (downstream)
        downstream_col_full = (
            self._all_feature_names.index("is_downstream")
            if "is_downstream" in self._all_feature_names else None
        )
        if downstream_weight is not None and downstream_weight > 0 and downstream_col_full is not None:
            if self._w is not None:
                # Multiply with existing weights (e.g. throat weights)
                ds_mask = full_x[:, downstream_col_full] > 0.5
                self._w = self._w.clone()
                self._w[ds_mask] *= float(downstream_weight)
            else:
                new_w = torch.ones(len(self._x), dtype=torch.float32)
                new_w[full_x[:, downstream_col_full] > 0.5] = float(downstream_weight)
                self._w = new_w.unsqueeze(-1)

        # Case index tensor for per-case losses
        self.include_case_idx = include_case_idx
        self._case_idx_tensor = torch.from_numpy(self._row_case_idx).long()

        # Feature normalization
        self.normalize = normalize
        if norm_stats is not None:
            self.norm_stats = self._coerce_norm_stats(norm_stats, dtype=self._x.dtype)
        elif normalize:
            stats_source_x = self._x
            if norm_from_case_indices is not None:
                keep = sorted({int(i) for i in norm_from_case_indices})
                if not keep:
                    raise ValueError("norm_from_case_indices must not be empty.")
                invalid = [i for i in keep if i < 0 or i >= len(self._case_ids_unique)]
                if invalid:
                    raise ValueError(
                        "norm_from_case_indices contains out-of-range case index(es): "
                        f"{invalid}"
                    )
                mask = np.isin(self._row_case_idx, keep)
                if not np.any(mask):
                    raise ValueError(
                        "norm_from_case_indices selected zero rows; cannot compute normalization stats."
                    )
                stats_source_x = self._x[mask]
            self.norm_stats = self._compute_norm_stats(stats_source_x)
        else:
            self.norm_stats = None

        if self.normalize and self.norm_stats is not None:
            self._x = (self._x - self.norm_stats["x_mean"]) / self.norm_stats["x_std"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_Dr(case_name: str) -> float:
        """Extract the diameter ratio from a case name like ``Re_*__Dr_0p333__Lr_*``."""
        for part in case_name.split("__"):
            if part.startswith("Dr_"):
                return float(part[3:].replace("p", "."))
        return 0.0

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_norm_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute per-feature mean and std from input data tensor."""
        return {
            "x_mean": x.mean(dim=0),
            "x_std": x.std(dim=0).clamp(min=1e-8),
        }

    @staticmethod
    def _coerce_norm_stats(
        norm_stats: dict,
        *,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if "x_mean" not in norm_stats or "x_std" not in norm_stats:
            raise ValueError("norm_stats must contain both 'x_mean' and 'x_std'.")

        x_mean = torch.as_tensor(norm_stats["x_mean"], dtype=dtype)
        x_std = torch.as_tensor(norm_stats["x_std"], dtype=dtype).clamp(min=1e-8)
        if x_mean.ndim != 1 or x_std.ndim != 1:
            raise ValueError("norm_stats['x_mean'] and norm_stats['x_std'] must be 1D.")
        if x_mean.shape != x_std.shape:
            raise ValueError(
                "norm_stats['x_mean'] and norm_stats['x_std'] must have matching shapes."
            )
        return {"x_mean": x_mean, "x_std": x_std}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def in_features(self) -> int:
        return len(self.input_columns)

    @property
    def out_features(self) -> int:
        return len(self.output_columns)

    @property
    def sim_names(self) -> list[str]:
        """Unique case IDs in discovery order (compatible with split_indices)."""
        return self._case_ids_unique

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int):
        if self.include_case_idx:
            if self._w is not None:
                return self._x[idx], self._y[idx], self._w[idx], self._case_idx_tensor[idx]
            return self._x[idx], self._y[idx], self._case_idx_tensor[idx]
        if self._w is not None:
            return self._x[idx], self._y[idx], self._w[idx]
        return self._x[idx], self._y[idx]

    # ------------------------------------------------------------------
    # Case-level subsetting
    # ------------------------------------------------------------------

    def subset_by_case_indices(self, case_indices: list[int]) -> "TabularPairDataset":
        """Return a new dataset containing only rows for the given case indices.

        ``case_indices`` indexes into ``self.sim_names``.
        """
        keep = set(case_indices)
        mask = np.isin(self._row_case_idx, list(keep))

        new = object.__new__(TabularPairDataset)
        new.zarr_dir = self.zarr_dir
        new.input_columns = list(self.input_columns)
        new.output_columns = list(self.output_columns)
        new._base_feature_names = list(self._base_feature_names)
        new._all_feature_names = list(self._all_feature_names)
        new._all_target_names = list(self._all_target_names)
        new._feat_idx = list(self._feat_idx)
        new._tgt_idx = list(self._tgt_idx)
        new._x = self._x[mask]
        new._y = self._y[mask]
        new._w = self._w[mask] if self._w is not None else None
        new._case_ids_unique = [self._case_ids_unique[i] for i in case_indices]
        new.normalize = self.normalize
        new.norm_stats = self.norm_stats  # share parent's stats (don't recompute)
        new.throat_weight = self.throat_weight
        new.downstream_weight = self.downstream_weight
        new.include_case_idx = self.include_case_idx
        new.local_velocity_normalization = self.local_velocity_normalization
        new.exclude_cases = list(self.exclude_cases)

        # Propagate per-case metadata and raw geometry arrays
        new._case_meta = [self._case_meta[i] for i in case_indices]
        new._raw_z_hat = self._raw_z_hat[mask] if self._raw_z_hat is not None else None
        new._raw_d_local_over_D = (
            self._raw_d_local_over_D[mask]
            if self._raw_d_local_over_D is not None else None
        )

        # Rebuild rows_per_case and row_case_idx for the subset
        new._rows_per_case = [self._rows_per_case[i] for i in case_indices]
        new._row_case_idx = np.concatenate(
            [np.full(n, new_i, dtype=np.int32) for new_i, n in enumerate(new._rows_per_case)]
        )
        new._case_idx_tensor = torch.from_numpy(new._row_case_idx).long()
        return new
