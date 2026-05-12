"""Phase 2d: AlphaDExperiment.prepare_for_training lifecycle hook.

Pins the contract that the runner's previous direct attribute writes
(``experiment.alpha_d_target_name = ...``, ``experiment.case_geometry = ...``)
move onto the experiment.
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
    )


def _make_experiment(delta_p_weight: float = 0.0) -> AlphaDExperiment:
    return AlphaDExperiment(
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
        delta_p_weight=delta_p_weight,
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


def test_prepare_for_training_builds_case_geometry_when_delta_p_enabled(
    alpha_d_zarr_dir: Path,
) -> None:
    ds = _make_alpha_d_dataset(alpha_d_zarr_dir)
    exp = _make_experiment(delta_p_weight=0.1)
    exp.prepare_for_training(ds, None, torch.device("cpu"))

    assert exp.case_geometry, "case_geometry should be non-empty for delta_p runs"
    first_key = next(iter(exp.case_geometry))
    entry = exp.case_geometry[first_key]
    for required in (
        "x_full", "z_hat", "d_local_over_D", "n_stations",
        "L_roi", "D_big", "delta_p_case", "rho", "V_bulk",
    ):
        assert required in entry, f"case_geometry entry missing '{required}'"


def test_prepare_for_training_skips_case_geometry_when_delta_p_disabled(
    alpha_d_zarr_dir: Path,
) -> None:
    ds = _make_alpha_d_dataset(alpha_d_zarr_dir)
    exp = _make_experiment(delta_p_weight=0.0)
    exp.prepare_for_training(ds, None, torch.device("cpu"))
    assert exp.case_geometry == {}
