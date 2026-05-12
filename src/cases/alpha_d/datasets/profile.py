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

import numpy as np
import torch
from torch.utils.data import Dataset

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
    def target_residual_baseline(self):
        return self._inner.target_residual_baseline

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
