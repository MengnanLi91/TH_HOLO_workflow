"""Per-case profile dataset for 1D-conv α_D training.

Wraps a :class:`TabularPairDataset` and exposes per-case views shaped
``(features, stations)`` so that a 1D conv along the station axis can
treat each case as a single sample.

The wrapper delegates flat row-level state to the inner tabular dataset,
which is what every existing access site in ``runner.py``,
``experiments/alpha_d.py``, and ``plotting.py`` reads. Subsets are built
by delegating to ``TabularPairDataset.subset_by_case_indices`` so that
flat state stays aligned with the train/val/test case split.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cases.alpha_d.feature_data import engineered_features_spec
from cases.alpha_d.transforms import alpha_d_residual_transform
from training.datasets import parse_field_list
from training.datasets_tabular import TabularPairDataset


class AlphaDProfileDataset(Dataset):
    """Per-case dataset producing ``(x, y, w, case_idx)`` profile tensors.

    Shapes per item:
        x          : ``[in_features, n_stations]``
        y          : ``[out_features, n_stations]``
        w          : ``[1, n_stations]`` (broadcast-compatible with y)
        case_idx   : scalar long tensor

    Stations are sorted by ``z_hat`` per case so the conv sees a monotone
    spatial sequence.
    """

    def __init__(self, **tabular_kwargs):
        if "engineered_feature_names" not in tabular_kwargs:
            names, builder = engineered_features_spec()
            tabular_kwargs["engineered_feature_names"] = names
            tabular_kwargs["engineered_feature_builder"] = builder
        tabular_kwargs.setdefault("target_transform", alpha_d_residual_transform)
        self._inner = TabularPairDataset(**tabular_kwargs)
        self._case_slices = _build_case_slices(self._inner)

    @classmethod
    def _from_inner(cls, inner: TabularPairDataset) -> "AlphaDProfileDataset":
        new = object.__new__(cls)
        new._inner = inner
        new._case_slices = _build_case_slices(inner)
        return new

    # ------------------------------------------------------------------
    # Delegated flat properties (consumed by runner / plotting / alpha_d)
    # ------------------------------------------------------------------

    @property
    def _x(self):
        return self._inner._x

    @property
    def _y(self):
        return self._inner._y

    @property
    def _w(self):
        return self._inner._w

    @property
    def _baseline_encoded(self):
        return self._inner._baseline_encoded

    @property
    def _row_case_idx(self):
        return self._inner._row_case_idx

    @property
    def _case_ids_unique(self):
        return self._inner._case_ids_unique

    @property
    def _case_meta(self):
        return self._inner._case_meta

    @property
    def _raw_z_hat(self):
        return self._inner._raw_z_hat

    @property
    def _raw_d_local_over_D(self):
        return self._inner._raw_d_local_over_D

    @property
    def norm_stats(self):
        return self._inner.norm_stats

    @property
    def normalize(self):
        return self._inner.normalize

    @property
    def has_target_baseline(self):
        return self._inner.has_target_baseline

    @property
    def local_velocity_normalization(self):
        return self._inner.local_velocity_normalization

    @property
    def exclude_cases(self):
        return self._inner.exclude_cases

    @property
    def input_columns(self):
        return self._inner.input_columns

    @property
    def output_columns(self):
        return self._inner.output_columns

    @property
    def in_features(self):
        return self._inner.in_features

    @property
    def out_features(self):
        return self._inner.out_features

    @property
    def sim_names(self):
        return self._inner.sim_names

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._case_slices)

    def __getitem__(self, ci: int):
        idx = self._case_slices[ci]
        x = self._inner._x[idx].T.contiguous()  # [C, S]
        y = self._inner._y[idx].T.contiguous()  # [O, S]
        if self._inner._w is not None:
            w = self._inner._w[idx].squeeze(-1).unsqueeze(0).contiguous()  # [1, S]
        else:
            w = torch.ones(1, len(idx), dtype=x.dtype)
        return x, y, w, torch.tensor(ci, dtype=torch.long)

    # ------------------------------------------------------------------
    # Case-level subsetting
    # ------------------------------------------------------------------

    def subset_by_case_indices(self, case_indices) -> "AlphaDProfileDataset":
        case_indices = [int(i) for i in case_indices]
        return AlphaDProfileDataset._from_inner(
            self._inner.subset_by_case_indices(case_indices)
        )


def _build_case_slices(inner: TabularPairDataset) -> list[np.ndarray]:
    """Index per-case row slices into ``inner``, sorted by z_hat."""
    slices: list[np.ndarray] = []
    z_hat = inner._raw_z_hat
    for ci in range(len(inner._case_ids_unique)):
        idx = np.where(inner._row_case_idx == ci)[0]
        if z_hat is not None:
            order = np.argsort(z_hat[idx].numpy())
            idx = idx[order]
        slices.append(idx)
    return slices


def build_dataset(data_cfg: dict) -> AlphaDProfileDataset:
    """Construct an :class:`AlphaDProfileDataset` from a Hydra data config.

    Called by :class:`training.adapters.ProfileAdapter.build_dataset` after it
    resolves ``data.dataset_entrypoint``. Reads all the alpha-D-flavoured
    kwargs the generic adapter no longer knows about; engineered features and
    target transform default-injection still happen inside
    ``AlphaDProfileDataset.__init__``.
    """
    norm_from_case_indices = data_cfg.get("norm_from_case_indices")
    if norm_from_case_indices is not None:
        norm_from_case_indices = [int(i) for i in norm_from_case_indices]

    if data_cfg.get("input_columns_file") is not None:
        cols_path = Path(str(data_cfg["input_columns_file"]))
        if not cols_path.exists():
            raise FileNotFoundError(f"input_columns_file not found: {cols_path}")
        input_columns = [
            line.strip() for line in cols_path.read_text().splitlines()
            if line.strip()
        ]
        if not input_columns:
            raise ValueError(f"input_columns_file is empty: {cols_path}")
    else:
        input_columns = parse_field_list(data_cfg.get("input_columns"))

    def _opt_float(key: str) -> float | None:
        v = data_cfg.get(key)
        return float(v) if v is not None else None

    exclude_cases = data_cfg.get("exclude_cases")
    if exclude_cases is not None:
        exclude_cases = [str(c) for c in exclude_cases]

    return AlphaDProfileDataset(
        zarr_dir=data_cfg["zarr_dir"],
        input_columns=input_columns,
        output_columns=parse_field_list(data_cfg.get("output_columns")),
        normalize=bool(data_cfg.get("normalize", False)),
        norm_stats=data_cfg.get("norm_stats"),
        norm_from_case_indices=norm_from_case_indices,
        throat_weight=_opt_float("throat_weight"),
        downstream_weight=_opt_float("downstream_weight"),
        include_case_idx=bool(data_cfg.get("include_case_idx", False)),
        exclude_cases=exclude_cases,
        local_velocity_normalization=bool(
            data_cfg.get("local_velocity_normalization", False)
        ),
        min_Dr=_opt_float("min_Dr"),
    )
