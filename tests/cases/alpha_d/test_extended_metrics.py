"""Shape contract for ``AlphaDExperiment.compute_extended_metrics``.

Pins the dict shape the alpha-D experiment must return: per-field R² / MAE
for the alpha-D target, with the same key names the plotting layer and
``eval_metrics.json`` consumers read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cases.alpha_d.datasets.profile import AlphaDProfileDataset
from cases.alpha_d.experiment import AlphaDExperiment
from cases.alpha_d.feature_data import engineered_features_spec
from training.datasets_tabular import TabularPairDataset


def test_alpha_d_compute_extended_metrics_shape(alpha_d_zarr_dir: Path) -> None:
    eng_names, eng_builder = engineered_features_spec()
    ds = TabularPairDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
    )

    # Trivial model that returns zeros; just need it to be callable for the
    # delta_p forward inside compute_extended_metrics.
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

    preds_batch = torch.zeros((len(ds), 1), dtype=torch.float32)
    targets_batch = ds._y.clone()

    metrics = experiment.compute_extended_metrics(
        ds,
        [preds_batch],
        [targets_batch],
    )

    assert isinstance(metrics, dict)
    assert "per_field" in metrics
    per_field = metrics["per_field"]
    assert isinstance(per_field, list) and len(per_field) == 1
    entry = per_field[0]
    assert entry["name"] == "log_alpha_D"
    assert "r2" in entry and "mae" in entry


def test_alpha_d_compute_extended_metrics_profile_path(
    alpha_d_zarr_dir: Path,
) -> None:
    """Conv1D path: cat'd preds are [N_cases, O, S]. Experiment must scatter
    back to flat row layout before delegating to the pointwise helper so the
    same per-field keys exist as on the MLP path."""
    eng_names, eng_builder = engineered_features_spec()
    ds = AlphaDProfileDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
        target_transform=None,
    )

    # Stub Conv1D-shaped model: accepts [B, C, S], returns [B, 1, S].
    class _ZeroProfileModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (x.shape[0], 1, x.shape[2]),
                dtype=x.dtype,
                device=x.device,
            )

    experiment = AlphaDExperiment(
        model=_ZeroProfileModel(),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )

    n_cases = len(ds)
    n_stations = ds[0][1].shape[-1]
    preds = torch.zeros((n_cases, 1, n_stations), dtype=torch.float32)
    targets = torch.stack([ds[i][1] for i in range(n_cases)], dim=0)

    metrics = experiment.compute_extended_metrics(ds, [preds], [targets])

    assert isinstance(metrics, dict)
    assert "per_field" in metrics
    per_field = metrics["per_field"]
    assert isinstance(per_field, list) and len(per_field) == 1
    entry = per_field[0]
    assert entry["name"] == "log_alpha_D"
    assert "r2" in entry and "mae" in entry


def test_alpha_d_delta_p_metrics_expose_full_per_case_list(
    alpha_d_zarr_dir: Path,
) -> None:
    """Δp parity plotting needs every test case, not just top/bottom 10.

    The per-case entries must also carry Dr and Re so the plotter can color
    points by contraction ratio without reparsing the case name.
    """
    ds = AlphaDProfileDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
    )

    class _ZeroProfileModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                (x.shape[0], 1, x.shape[2]),
                dtype=x.dtype,
                device=x.device,
            )

    experiment = AlphaDExperiment(
        model=_ZeroProfileModel(),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )

    n_cases = len(ds)
    n_stations = ds[0][1].shape[-1]
    preds = torch.zeros((n_cases, 1, n_stations), dtype=torch.float32)
    targets = torch.stack([ds[i][1] for i in range(n_cases)], dim=0)

    metrics = experiment.compute_extended_metrics(ds, [preds], [targets])

    dp = metrics.get("delta_p")
    assert dp is not None, "delta_p block missing from extended metrics"
    assert "per_case" in dp, "delta_p must expose the full per-case list"
    per_case = dp["per_case"]
    assert isinstance(per_case, list)
    assert len(per_case) == n_cases
    sample = per_case[0]
    for key in ("case", "delta_p_gt", "delta_p_pred", "relative_error", "Dr", "Re"):
        assert key in sample, f"per-case entry missing '{key}': {sample}"
