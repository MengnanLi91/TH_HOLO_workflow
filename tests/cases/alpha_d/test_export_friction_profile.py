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
