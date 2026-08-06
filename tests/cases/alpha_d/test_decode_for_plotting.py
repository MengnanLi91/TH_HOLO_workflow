"""``AlphaDExperiment.decode_for_plotting`` hook and downstream plotters.

The generic plotter stays alpha-D-agnostic; the case experiment provides a
callable that re-adds the encoded baseline and converts to physical-space
α_D. These tests also exercise the per-case profile, α_D parity, and Δp
parity plotters via the same hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from cases.alpha_d.datasets.profile import AlphaDProfileDataset
from cases.alpha_d.experiment import AlphaDExperiment
from cases.alpha_d.feature_data import engineered_features_spec
from cases.alpha_d.transforms import alpha_d_residual_transform
from training.datasets_tabular import TabularPairDataset
from training.plotting import (
    save_delta_p_parity_plot,
    save_parity_plot,
    save_pointwise_profile_plots,
    save_profile_prediction_plots,
)


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
        target_transform_kwargs={"include_acceleration_head": True},
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


def test_baseline_for_plotting_returns_decoded_analytical_baseline(
    alpha_d_zarr_dir: Path,
) -> None:
    eng_names, eng_builder = engineered_features_spec()
    ds = TabularPairDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
        target_transform=alpha_d_residual_transform,
        target_transform_kwargs={"include_acceleration_head": True},
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
    n_stations = int(mask.sum())

    result = experiment.baseline_for_plotting(
        dataset=ds,
        field_name="log_alpha_D",
        mask=mask,
    )

    assert result is not None
    decoded_baseline, label = result
    assert isinstance(decoded_baseline, np.ndarray)
    assert decoded_baseline.shape == (n_stations,)
    assert label == "alpha_D"
    # The baseline is alpha_D > 0 across stations (closed-form sudden contraction
    # + throat friction). A zero or negative baseline would mean the decode
    # pipeline lost the analytical contribution.
    assert np.all(decoded_baseline > 0.0)

    # Sanity: decode_for_plotting(zero residual) and baseline_for_plotting
    # produce the same array — the latter is the former with values=0.
    zero = torch.zeros(n_stations, dtype=torch.float32)
    decoded_from_zero, _ = experiment.decode_for_plotting(
        zero,
        dataset=ds,
        field_name="log_alpha_D",
        mask=mask,
    )
    np.testing.assert_allclose(decoded_baseline, decoded_from_zero)


def test_save_pointwise_profile_plots_overlays_baseline_curve(
    alpha_d_zarr_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    eng_names, eng_builder = engineered_features_spec()
    ds = TabularPairDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
        target_transform=alpha_d_residual_transform,
        target_transform_kwargs={"include_acceleration_head": True},
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

    case_name = ds._case_ids_unique[0]
    case_entries = [{"label": "best", "case": case_name, "rmse": 0.0}]

    captured: dict[str, Any] = {}
    real_subplots = plt.subplots

    def _capturing_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _capturing_subplots)

    plot_dir = tmp_path / "plots"
    saved = save_pointwise_profile_plots(
        model=experiment.model,
        dataset=ds,
        output_fields=["log_alpha_D"],
        device=torch.device("cpu"),
        plot_dir=plot_dir,
        case_entries=case_entries,
        decode_fn=experiment.decode_for_plotting,
        baseline_fn=experiment.baseline_for_plotting,
    )

    assert len(saved) == 1
    assert Path(saved[0]).exists()

    ax = captured["ax"]
    line_labels = [line.get_label() for line in ax.get_lines()]
    assert "Ground Truth" in line_labels
    assert "Predicted" in line_labels
    assert "Baseline" in line_labels


def test_save_profile_prediction_plots_overlays_baseline_curve(
    alpha_d_zarr_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    ds = AlphaDProfileDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        target_transform_kwargs={"include_acceleration_head": True},
    )

    class _ZeroConv1d(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, C, S] -> [B, 1, S]
            return torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)

    experiment = AlphaDExperiment(
        model=_ZeroConv1d(),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )

    case_name = ds._case_ids_unique[0]
    case_entries = [{"label": "best", "case": case_name, "rmse": 0.0}]

    captured: dict[str, Any] = {}
    real_subplots = plt.subplots

    def _capturing_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["fig"] = fig
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _capturing_subplots)

    plot_dir = tmp_path / "plots"
    saved = save_profile_prediction_plots(
        model=experiment.model,
        dataset=ds,
        output_fields=["log_alpha_D"],
        device=torch.device("cpu"),
        plot_dir=plot_dir,
        case_entries=case_entries,
        decode_fn=experiment.decode_for_plotting,
        baseline_fn=experiment.baseline_for_plotting,
    )

    assert len(saved) == 1
    assert Path(saved[0]).exists()

    ax = captured["ax"]
    line_labels = [line.get_label() for line in ax.get_lines()]
    assert "Ground Truth" in line_labels
    assert "Predicted" in line_labels
    assert "Baseline" in line_labels


def test_save_profile_prediction_plots_without_baseline_fn_omits_baseline_curve(
    alpha_d_zarr_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    ds = AlphaDProfileDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        target_transform_kwargs={"include_acceleration_head": True},
    )

    class _ZeroConv1d(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)

    experiment = AlphaDExperiment(
        model=_ZeroConv1d(),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )

    captured: dict[str, Any] = {}
    real_subplots = plt.subplots

    def _capturing_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _capturing_subplots)

    saved = save_profile_prediction_plots(
        model=experiment.model,
        dataset=ds,
        output_fields=["log_alpha_D"],
        device=torch.device("cpu"),
        plot_dir=tmp_path / "plots",
        case_entries=[{"label": "best", "case": ds._case_ids_unique[0], "rmse": 0.0}],
        decode_fn=experiment.decode_for_plotting,
        baseline_fn=None,
    )

    assert len(saved) == 1
    line_labels = [line.get_label() for line in captured["ax"].get_lines()]
    assert "Baseline" not in line_labels
    assert "Ground Truth" in line_labels
    assert "Predicted" in line_labels


def test_save_parity_plot_profile_dataset_decoded_physical_space(
    alpha_d_zarr_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    ds = AlphaDProfileDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        target_transform_kwargs={"include_acceleration_head": True},
    )

    class _ZeroConv1d(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)

    experiment = AlphaDExperiment(
        model=_ZeroConv1d(),
        optimizer=None,
        loss_fn=None,
        adapter=None,
        device=torch.device("cpu"),
    )

    # Simulate runner.evaluate's collection: one batch per case for the profile
    # adapter, each tensor shaped [B, O, S].
    n_cases = len(ds)
    n_stations = ds[0][0].shape[-1]
    cat_preds = torch.zeros(n_cases, 1, n_stations)
    cat_targets = torch.stack([ds[i][1] for i in range(n_cases)])  # [N, O, S]

    captured: dict[str, Any] = {}
    real_subplots = plt.subplots

    def _capturing_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _capturing_subplots)

    plot_dir = tmp_path / "plots"
    saved = save_parity_plot(
        cat_preds=cat_preds,
        cat_targets=cat_targets,
        dataset=ds,
        output_fields=["log_alpha_D"],
        plot_dir=plot_dir,
        decode_fn=experiment.decode_for_plotting,
    )

    assert len(saved) == 1
    assert Path(saved[0]).exists()

    ax = captured["ax"]
    # Scatter creates a PathCollection; the y=x reference line is a Line2D.
    assert ax.collections, "Expected scatter points on the parity axes."
    line_labels = [line.get_label() for line in ax.get_lines()]
    assert any("y" in lbl.lower() or "1:1" in lbl or "x" in lbl.lower() for lbl in line_labels), (
        f"Expected a y=x reference line; got labels {line_labels}"
    )
    assert any("10%" in lbl for lbl in line_labels), (
        f"Expected ±10% deviation lines; got labels {line_labels}"
    )
    # Decoded alpha_D is positive, so the X/Y label should reflect the
    # physical (alpha_D) space, not the encoded (log_alpha_D) space.
    assert "alpha_D" in ax.get_xlabel()
    assert "alpha_D" in ax.get_ylabel()


def test_save_parity_plot_without_decode_fn_uses_encoded_values(
    alpha_d_zarr_dir: Path,
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")

    ds = AlphaDProfileDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        target_transform_kwargs={"include_acceleration_head": True},
    )

    n_cases = len(ds)
    n_stations = ds[0][0].shape[-1]
    cat_preds = torch.zeros(n_cases, 1, n_stations)
    cat_targets = torch.stack([ds[i][1] for i in range(n_cases)])

    saved = save_parity_plot(
        cat_preds=cat_preds,
        cat_targets=cat_targets,
        dataset=ds,
        output_fields=["log_alpha_D"],
        plot_dir=tmp_path / "plots",
        decode_fn=None,
    )
    assert len(saved) == 1
    assert Path(saved[0]).exists()


def test_save_delta_p_parity_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    per_case = [
        {
            "case": "Re_5000__Dr_0p333__Lr_0p1",
            "delta_p_gt": 10.0,
            "delta_p_pred": 12.0,
            "relative_error": 0.2,
            "Dr": 0.333,
            "Re": 5000.0,
        },
        {
            "case": "Re_5000__Dr_0p617__Lr_0p1",
            "delta_p_gt": 25.0,
            "delta_p_pred": 24.0,
            "relative_error": 0.04,
            "Dr": 0.617,
            "Re": 5000.0,
        },
        {
            "case": "Re_104807__Dr_0p9__Lr_0p1",
            "delta_p_gt": 80.0,
            "delta_p_pred": 79.5,
            "relative_error": 0.006,
            "Dr": 0.9,
            "Re": 104807.0,
        },
    ]

    captured: dict[str, Any] = {}
    real_subplots = plt.subplots

    def _capturing_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plt, "subplots", _capturing_subplots)

    out_path = save_delta_p_parity_plot(
        per_case=per_case,
        plot_dir=tmp_path / "plots",
    )

    assert out_path is not None
    assert Path(out_path).exists()

    ax = captured["ax"]
    assert ax.collections, "Expected scatter points on the Δp parity axes."
    line_labels = [line.get_label() for line in ax.get_lines()]
    assert any("y" in lbl.lower() or "1:1" in lbl for lbl in line_labels), (
        f"Expected a y=x reference line; got labels {line_labels}"
    )
    assert any("10%" in lbl for lbl in line_labels), (
        f"Expected ±10% deviation lines; got labels {line_labels}"
    )
    assert "delta_p" in ax.get_xlabel().lower() or "Δp" in ax.get_xlabel()
    assert "delta_p" in ax.get_ylabel().lower() or "Δp" in ax.get_ylabel()


def test_save_delta_p_parity_plot_empty_input_returns_none(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    result = save_delta_p_parity_plot(per_case=[], plot_dir=tmp_path / "plots")
    assert result is None
