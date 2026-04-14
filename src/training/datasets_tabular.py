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
    include_case_idx : bool
        If *True*, ``__getitem__`` returns a case-index tensor as the last
        element so that per-case losses can be computed.
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
        include_case_idx: bool = False,
    ):
        import json

        import zarr

        self.zarr_dir = Path(zarr_dir)
        sim_paths = sorted(self.zarr_dir.glob("*.zarr"))
        if not sim_paths:
            raise FileNotFoundError(f"No .zarr stores found in {self.zarr_dir}")

        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_w: list[np.ndarray] = []
        case_ids: list[str] = []
        rows_per_case: list[int] = []
        has_weights = False

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

                self._all_feature_names = list(raw_feature_names)
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

            # Load ALL base features (derived columns need access to
            # source columns that may not be in input_columns).
            all_x.append(features)
            all_y.append(targets[:, self._tgt_idx])
            all_w.append(weights)
            case_ids.append(case_id)
            rows_per_case.append(features.shape[0])

        # ----------------------------------------------------------
        # Concatenate
        # ----------------------------------------------------------
        full_x = np.concatenate(all_x, axis=0)  # [N, D_base]

        # Resolve input columns
        if input_columns is not None:
            feat_map = {n: i for i, n in enumerate(self._all_feature_names)}
            missing = [c for c in input_columns if c not in feat_map]
            if missing:
                raise ValueError(f"Unknown input columns: {missing}")
            self._feat_idx = [feat_map[c] for c in input_columns]
            self.input_columns = list(input_columns)
        else:
            self._feat_idx = list(range(len(self._all_feature_names)))
            self.input_columns = list(self._all_feature_names)

        self._x = torch.from_numpy(full_x[:, self._feat_idx].copy())
        self._y = torch.from_numpy(np.concatenate(all_y, axis=0))
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
        if throat_weight is not None and throat_weight > 0 and throat_col_full is not None:
            new_w = torch.ones(len(self._x), dtype=torch.float32)
            new_w[full_x[:, throat_col_full] > 0.5] = float(throat_weight)
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
        new.include_case_idx = self.include_case_idx

        # Rebuild rows_per_case and row_case_idx for the subset
        new._rows_per_case = [self._rows_per_case[i] for i in case_indices]
        new._row_case_idx = np.concatenate(
            [np.full(n, new_i, dtype=np.int32) for new_i, n in enumerate(new._rows_per_case)]
        )
        new._case_idx_tensor = torch.from_numpy(new._row_case_idx).long()
        return new
