"""Phase 2c: AlphaDExperiment.decode_for_plotting hook.

The plotter must be alpha-D-free; the case experiment provides a callable
that re-adds the encoded baseline and converts to physical-space α_D.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cases.alpha_d.experiment import AlphaDExperiment
from cases.alpha_d.feature_data import engineered_features_spec
from cases.alpha_d.transforms import alpha_d_residual_transform
from training.datasets_tabular import TabularPairDataset


def test_decode_for_plotting_returns_physical_array_and_label(
    alpha_d_zarr_dir: Path,
) -> None:
    eng_names, eng_builder = engineered_features_spec()
    ds = TabularPairDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
        target_transform=alpha_d_residual_transform,
    )

    class _ZeroModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)

    experiment = AlphaDExperiment(
        model=_ZeroModel(),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )

    mask = ds._row_case_idx == 0
    values = ds._y[mask][:, 0].clone()
    n_stations = int(mask.sum())

    result = experiment.decode_for_plotting(
        values,
        dataset=ds,
        field_name="log_alpha_D",
        mask=mask,
    )

    assert result is not None
    decoded, label = result
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (n_stations,)
    assert label == "alpha_D"
