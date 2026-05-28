"""Tests for src/cases/alpha_d/export_friction_profile.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_module_imports():
    from cases.alpha_d import export_friction_profile  # noqa: F401


def test_cli_parses_required_args(tmp_path):
    from cases.alpha_d.export_friction_profile import parse_args

    args = parse_args(
        [
            "--zarr",
            str(tmp_path / "case.zarr"),
            "--checkpoint",
            str(tmp_path / "model.mdlus"),
            "--run-meta",
            str(tmp_path / "run_meta.json"),
            "--output-csv",
            str(tmp_path / "out.csv"),
        ]
    )
    assert args.zarr == tmp_path / "case.zarr"
    assert args.checkpoint == tmp_path / "model.mdlus"
    assert args.run_meta == tmp_path / "run_meta.json"
    assert args.output_csv == tmp_path / "out.csv"


TARGET_ZARR = Path(
    "/data/lim2/projects/multifid-th/worktrees/integration/data/"
    "flow_contraction_expansion/parametric_study/processed/"
    "Re_43938__Dr_0p522__Lr_0p073.zarr"
)


@pytest.mark.skipif(
    not TARGET_ZARR.exists(), reason="Target zarr not present (Task 1 not completed)."
)
def test_load_case_from_zarr_returns_features_targets_and_geometry():
    from cases.alpha_d.export_friction_profile import load_case_from_zarr

    case = load_case_from_zarr(TARGET_ZARR)

    assert case.features.shape == (50, 13)
    assert case.targets.shape == (50, 2)
    assert case.feature_names[0] == "log10_Re"
    assert "signed_log1p_alpha_D" in case.target_names
    assert case.Re == pytest.approx(43938.0, rel=1e-4)
    assert case.Dr == pytest.approx(0.522, rel=1e-3)
    assert case.Lr == pytest.approx(0.073, rel=1e-2)
    assert case.D_big == pytest.approx(0.2, rel=1e-6)
    assert case.outer_height_m == pytest.approx(1.0, rel=1e-6)
    assert case.delta_p_truth > 0  # parametric study should have a positive drop


def test_load_case_from_zarr_missing_path_raises(tmp_path):
    from cases.alpha_d.export_friction_profile import load_case_from_zarr

    with pytest.raises(FileNotFoundError):
        load_case_from_zarr(tmp_path / "does-not-exist.zarr")


def test_build_model_input_uses_run_meta_columns_and_norm_stats(tmp_path):
    from cases.alpha_d.export_friction_profile import (
        CaseData,
        build_model_input,
    )

    feat_names = [
        "log10_Re",
        "Dr",
        "Lr",
        "z_hat",
        "d_local_over_D",
        "A_local_over_A",
        "V_local_over_V_bulk",
        "is_upstream",
        "is_throat",
        "is_downstream",
        "dD_dz_local",
        "dist_to_throat_start",
        "dist_to_throat_end",
    ]
    n = 50
    raw = np.zeros((n, len(feat_names)), dtype=np.float32)
    raw[:, feat_names.index("log10_Re")] = np.log10(43938.0)
    raw[:, feat_names.index("Dr")] = 0.522
    raw[:, feat_names.index("Lr")] = 0.073
    raw[:, feat_names.index("z_hat")] = np.linspace(0, 1, n, dtype=np.float32)
    # V_local_over_V_bulk is required by build_engineered_feature_map
    raw[:, feat_names.index("V_local_over_V_bulk")] = 1.0
    # d_local_over_D required by engineered features
    raw[:, feat_names.index("d_local_over_D")] = 0.522
    raw[:, feat_names.index("is_throat")] = 1.0

    case = CaseData(
        case_id="dummy",
        features=raw,
        targets=np.zeros((n, 2), dtype=np.float32),
        feature_names=feat_names,
        target_names=["signed_log1p_alpha_D"],
        Re=43938.0,
        Dr=0.522,
        Lr=0.073,
        D_big=0.2,
        outer_height_m=1.0,
        buffer_diams=1.0,
        rho=1.0,
        V_bulk=1.0,
    )

    run_meta = {
        "data": {
            "input_columns": ["z_hat", "log10_Re_throat"],
            "norm_stats": {
                "x_mean": [0.5, 4.7],
                "x_std": [0.3, 0.5],
            },
        }
    }

    x = build_model_input(case, run_meta)

    # Conv1D expects (B, C, L). B=1, C=2 (input_columns), L=50 (stations).
    assert x.shape == (1, 2, n)
    # First channel is z_hat, normalized: (z_hat - 0.5) / 0.3
    z_hat = np.linspace(0, 1, n, dtype=np.float32)
    expected_ch0 = (z_hat - 0.5) / 0.3
    np.testing.assert_allclose(x[0, 0, :], expected_ch0, atol=1e-5)


def test_build_model_input_rejects_unknown_column(tmp_path):
    from cases.alpha_d.export_friction_profile import (
        CaseData,
        build_model_input,
    )

    n = 50
    raw = np.zeros((n, 13), dtype=np.float32)
    case = CaseData(
        case_id="dummy",
        features=raw,
        targets=np.zeros((n, 2), dtype=np.float32),
        feature_names=[
            "log10_Re",
            "Dr",
            "Lr",
            "z_hat",
            "d_local_over_D",
            "A_local_over_A",
            "V_local_over_V_bulk",
            "is_upstream",
            "is_throat",
            "is_downstream",
            "dD_dz_local",
            "dist_to_throat_start",
            "dist_to_throat_end",
        ],
        target_names=["signed_log1p_alpha_D"],
        Re=1.0,
        Dr=0.5,
        Lr=0.1,
        D_big=0.2,
        outer_height_m=1.0,
        buffer_diams=1.0,
        rho=1.0,
        V_bulk=1.0,
    )

    bad = {
        "data": {
            "input_columns": ["totally_not_a_feature"],
            "norm_stats": {"x_mean": [0.0], "x_std": [1.0]},
        }
    }

    with pytest.raises(ValueError, match="totally_not_a_feature"):
        build_model_input(case, bad)
