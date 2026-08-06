"""AlphaDExperiment.prepare_for_training lifecycle hook.

Pins the contract that the runner's previous direct attribute writes
(``experiment.alpha_d_target_name = ...``) move onto the experiment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cases.alpha_d.experiment import AlphaDExperiment
from cases.alpha_d.feature_data import engineered_features_spec
from cases.alpha_d.transforms import alpha_d_residual_transform
from training.datasets_tabular import TabularPairDataset


def _make_alpha_d_dataset(alpha_d_zarr_dir: Path) -> TabularPairDataset:
    eng_names, eng_builder = engineered_features_spec()
    return TabularPairDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
        target_transform=alpha_d_residual_transform,
        target_transform_kwargs={"include_acceleration_head": True},
    )


def _make_experiment() -> AlphaDExperiment:
    return AlphaDExperiment(
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )


def test_prepare_for_training_binds_alpha_d_target_name(
    alpha_d_zarr_dir: Path,
) -> None:
    ds = _make_alpha_d_dataset(alpha_d_zarr_dir)
    exp = _make_experiment()
    # Default is "log_alpha_D"; after prepare it should match the dataset.
    exp.alpha_d_target_name = "WRONG"
    exp.prepare_for_training(ds, None, torch.device("cpu"))
    assert exp.alpha_d_target_name == "log_alpha_D"


def test_prepare_for_training_picks_up_lv_normalization_flag(
    alpha_d_zarr_dir: Path,
) -> None:
    ds = _make_alpha_d_dataset(alpha_d_zarr_dir)
    # The fixture builds with local_velocity_normalization=False (default),
    # so prepare_for_training should mirror that onto the experiment.
    exp = _make_experiment()
    exp.prepare_for_training(ds, None, torch.device("cpu"))
    assert exp.local_velocity_normalization is False
