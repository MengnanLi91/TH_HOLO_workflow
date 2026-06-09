"""Tests for src/cases/alpha_d/export_friction_profile.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_module_imports():
    from cases.alpha_d import export_friction_profile

    assert export_friction_profile is not None


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


# Resolve the repo root from this file's location:
#   tests/cases/alpha_d/<this>.py → parents[3] is the repo root.
_REPO = Path(__file__).resolve().parents[3]
TARGET_ZARR = (
    _REPO
    / "data/flow_contraction_expansion/parametric_study/processed/Re_43938__Dr_0p522__Lr_0p073.zarr"
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


def test_build_model_input_uses_run_meta_columns_and_norm_stats():
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


TARGET_CKPT = _REPO / "data/cases/train_conv1d/model.mdlus"
TARGET_RUN_META = TARGET_CKPT.parent / "run_meta.json"


@pytest.mark.skipif(
    not TARGET_CKPT.exists() or not TARGET_ZARR.exists(),
    reason="Checkpoint or target zarr not present.",
)
def test_forward_returns_signed_log1p_alpha_d_profile():
    from cases.alpha_d.export_friction_profile import (
        build_model_input,
        forward,
        load_case_from_zarr,
    )

    with TARGET_RUN_META.open() as fh:
        run_meta = json.load(fh)
    case = load_case_from_zarr(TARGET_ZARR)
    x = build_model_input(case, run_meta)

    y = forward(TARGET_CKPT, run_meta, x)

    assert y.shape == (50,)  # one target field, 50 stations
    assert np.all(np.isfinite(y))


def test_build_model_input_rejects_unknown_column():
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


def test_decode_to_bulk_alpha_d_inverts_encoder():
    """For a synthetic case where we know what was encoded, decoding
    should recover the original alpha_D values."""
    from cases.alpha_d.export_friction_profile import decode_to_bulk_alpha_d

    n = 50
    # Synthetic ground-truth alpha_D in bulk basis
    alpha_bulk_true = np.linspace(0.1, 5.0, n, dtype=np.float64)
    d_local_over_D = np.full(n, 0.522, dtype=np.float64)

    # Forward path: convert bulk -> local, then encode.
    # alpha_local = alpha_bulk * (V_bulk / V_local)^2
    # In ideal incompressible flow V_local/V_bulk = 1/(d_local/D)^2,
    # so alpha_local = alpha_bulk * (d_local/D)^4
    alpha_local = alpha_bulk_true * (d_local_over_D**4)
    encoded = np.sign(alpha_local) * np.log1p(np.abs(alpha_local))

    # Decoder should invert encoder + basis conversion.
    decoded = decode_to_bulk_alpha_d(
        encoded,
        d_local_over_D=d_local_over_D,
        local_velocity_normalization=True,
        target_name="signed_log1p_alpha_D",
    )

    np.testing.assert_allclose(decoded, alpha_bulk_true, rtol=1e-5)


def test_decode_to_bulk_alpha_d_without_local_normalization():
    """When local_velocity_normalization=False, decoding is just the
    inverse of the encoder, no basis conversion."""
    from cases.alpha_d.export_friction_profile import decode_to_bulk_alpha_d

    n = 50
    alpha_true = np.linspace(0.5, 3.0, n, dtype=np.float64)
    encoded = np.sign(alpha_true) * np.log1p(np.abs(alpha_true))

    decoded = decode_to_bulk_alpha_d(
        encoded,
        d_local_over_D=np.full(n, 0.7, dtype=np.float64),  # ignored
        local_velocity_normalization=False,
        target_name="signed_log1p_alpha_D",
    )

    np.testing.assert_allclose(decoded, alpha_true, rtol=1e-6)


def test_alpha_d_to_forchheimer_throat_synthetic():
    """Throat (porous): porosity = Dr², D_h = Dr·D_outer.
    α_D=1, Dr=0.5, D_outer=1 → F = α_D · porosity² / D_h = α_D · Dr⁴ / (Dr·D_outer) = α_D · Dr³ / D_outer = 0.125."""
    from cases.alpha_d.export_friction_profile import alpha_d_to_forchheimer

    cf = alpha_d_to_forchheimer(np.array([1.0]), porosity=0.5**2, D_h=0.5 * 1.0)
    np.testing.assert_allclose(cf, [0.125], rtol=1e-9)


def test_alpha_d_to_forchheimer_buffer_synthetic():
    """Buffer (non-porous): porosity = 1, D_h = D_outer.
    α_D=1, D_outer=1 → F = α_D / D_outer = 1.0."""
    from cases.alpha_d.export_friction_profile import alpha_d_to_forchheimer

    cf = alpha_d_to_forchheimer(np.array([1.0]), porosity=1.0, D_h=1.0)
    np.testing.assert_allclose(cf, [1.0], rtol=1e-9)


def test_alpha_d_to_forchheimer_target_case_throat():
    """For Dr=0.522, D_outer=0.2 throat: F = α_D · 0.522³/0.2."""
    from cases.alpha_d.export_friction_profile import alpha_d_to_forchheimer

    alpha = np.array([1.0, 2.0, 3.0])
    cf = alpha_d_to_forchheimer(alpha, porosity=0.522**2, D_h=0.522 * 0.2)
    np.testing.assert_allclose(cf, alpha * (0.522**3 / 0.2), rtol=1e-9)


def test_alpha_d_to_forchheimer_blockwise_per_station():
    """Vector inputs: each station gets its own porosity and D_h.

    Formula: F = α_D · porosity² / D_h. Buffer (porosity=1) reduces to
    α_D / D_h; throat (porosity=Dr²) gives α_D · Dr⁴ / D_h."""
    from cases.alpha_d.export_friction_profile import alpha_d_to_forchheimer

    alpha = np.array([1.0, 2.0, 3.0])
    porosity = np.array([1.0, 0.272, 1.0])
    D_h = np.array([0.2, 0.1044, 0.2])
    cf = alpha_d_to_forchheimer(alpha, porosity=porosity, D_h=D_h)
    expected = np.array(
        [
            1.0 * 1.0**2 / 0.2,
            2.0 * 0.272**2 / 0.1044,
            3.0 * 1.0**2 / 0.2,
        ]
    )
    np.testing.assert_allclose(cf, expected, rtol=1e-6)


def test_stepfence_inserts_duplicate_z_around_each_boundary():
    """Step-fencing at z=0.2 (between two stations, no exact match) should
    insert (0.2-eps, F_left) and (0.2+eps, F_right) between the bracketing
    stations, leaving every other station alone."""
    from cases.alpha_d.export_friction_profile import _stepfence_porosity_boundaries

    # Stations chosen so 0.20 falls between 0.18 and 0.22 — not on a station,
    # matching the real surrogate case where the porosity step at 0.2 sits
    # between stations 20 (z=0.194) and 21 (z=0.204).
    z = np.array([0.05, 0.18, 0.22, 0.30, 0.40], dtype=np.float64)
    cf = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    eps = 1e-4
    z_out, cf_out = _stepfence_porosity_boundaries(z, cf, boundaries=(0.20,), step_eps=eps)
    assert np.all(np.diff(z_out) > 0)
    assert len(z_out) == len(z) + 2
    assert (0.20 - eps) in z_out
    assert (0.20 + eps) in z_out
    # The fence values copy the bracketing F values: left=2.0 (from z=0.18),
    # right=3.0 (from z=0.22).
    idx_left = list(z_out).index(0.20 - eps)
    idx_right = list(z_out).index(0.20 + eps)
    assert cf_out[idx_left] == 2.0
    assert cf_out[idx_right] == 3.0


def test_stepfence_handles_two_boundaries():
    """Two porosity steps (block 1→2 and block 2→3) both get fenced."""
    from cases.alpha_d.export_friction_profile import _stepfence_porosity_boundaries

    z = np.array([0.0, 0.1, 0.18, 0.22, 0.265, 0.28, 0.4], dtype=np.float64)
    cf = np.array([1.0, 1.5, 5.0, 0.7, 1.0, 0.5, 0.2], dtype=np.float64)
    eps = 1e-4
    z_out, _ = _stepfence_porosity_boundaries(z, cf, boundaries=(0.20, 0.27), step_eps=eps)
    assert np.all(np.diff(z_out) > 0)
    assert len(z_out) == len(z) + 4  # two fences × two new rows each
    assert (0.20 - eps) in z_out and (0.20 + eps) in z_out
    assert (0.27 - eps) in z_out and (0.27 + eps) in z_out


def test_stepfence_skips_boundary_outside_csv_range():
    """If a boundary lies outside the CSV's z span, the helper should
    leave the data untouched."""
    from cases.alpha_d.export_friction_profile import _stepfence_porosity_boundaries

    z = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    cf = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    z_out, cf_out = _stepfence_porosity_boundaries(z, cf, boundaries=(0.05, 0.50), step_eps=1e-4)
    np.testing.assert_array_equal(z_out, z)
    np.testing.assert_array_equal(cf_out, cf)


def test_restrict_to_throat_extracts_central_segment():
    """Given a 50-station ROI alpha_D profile, restrict_to_throat returns
    only the rows where is_throat=1, and remaps z_hat to local throat
    axial coordinate [0, throat_length]."""
    from cases.alpha_d.export_friction_profile import restrict_to_throat

    n = 50
    z_hat = np.linspace(0, 1, n, dtype=np.float64)
    is_throat = np.zeros(n, dtype=np.float32)
    # Throat occupies z_hat in [0.4, 0.6] in this synthetic case.
    is_throat[(z_hat >= 0.4) & (z_hat <= 0.6)] = 1.0

    alpha = np.arange(n, dtype=np.float64)
    throat_length = 0.073

    z_throat, alpha_throat = restrict_to_throat(
        z_hat=z_hat,
        is_throat=is_throat,
        values=alpha,
        throat_length=throat_length,
    )

    # All returned stations should originally have been throat stations.
    assert len(z_throat) > 0
    assert len(z_throat) == len(alpha_throat)
    assert z_throat[0] == pytest.approx(0.0)
    assert z_throat[-1] == pytest.approx(throat_length)
    # Monotone increasing
    assert np.all(np.diff(z_throat) > 0)


def test_write_csv_and_sidecar(tmp_path):
    from cases.alpha_d.export_friction_profile import write_outputs

    z = np.array([0.0, 0.05, 0.073], dtype=np.float64)
    cf = np.array([100.0, 200.0, 150.0], dtype=np.float64)
    sidecar = {
        "case_id": "Re_43938__Dr_0p522__Lr_0p073",
        "Re": 43938.0,
        "Dr": 0.522,
        "Lr": 0.073,
        "delta_p_truth": 1.234,
        "delta_p_surrogate": 1.111,
        "throat_length_m": 0.073,
    }

    csv_path = tmp_path / "forchheimer_profile.csv"
    write_outputs(csv_path=csv_path, z=z, cf=cf, sidecar=sidecar)

    meta_path = csv_path.with_suffix(".meta.json")
    assert csv_path.exists()
    assert meta_path.exists()

    rows = csv_path.read_text().strip().splitlines()
    assert rows[0] == "z,F"
    parsed = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    np.testing.assert_allclose(parsed[:, 0], z)
    np.testing.assert_allclose(parsed[:, 1], cf)

    loaded = json.loads(meta_path.read_text())
    assert loaded["case_id"] == "Re_43938__Dr_0p522__Lr_0p073"
    assert loaded["delta_p_surrogate"] == pytest.approx(1.111)


@pytest.mark.skipif(
    not TARGET_CKPT.exists() or not TARGET_ZARR.exists(),
    reason="Checkpoint or target zarr not present.",
)
def test_end_to_end_run(tmp_path):
    from cases.alpha_d.export_friction_profile import main

    out_csv = tmp_path / "forchheimer_profile.csv"

    rc = main(
        [
            "--zarr",
            str(TARGET_ZARR),
            "--checkpoint",
            str(TARGET_CKPT),
            "--run-meta",
            str(TARGET_RUN_META),
            "--output-csv",
            str(out_csv),
        ]
    )
    assert rc == 0
    assert out_csv.exists()
    assert out_csv.with_suffix(".meta.json").exists()

    parsed = np.loadtxt(out_csv, delimiter=",", skiprows=1)
    z, cf = parsed[:, 0], parsed[:, 1]
    # 50 surrogate stations + 4 step-fence rows (two boundaries × two rows each).
    assert len(z) == 54
    assert z[0] == pytest.approx(0.0, abs=0.01)  # first station near inlet
    assert z[-1] == pytest.approx(0.473, abs=0.01)  # last station near outlet
    assert np.all(np.diff(z) > 0)  # strictly increasing — required by PiecewiseLinear
    # C_F should be positive somewhere in the ROI for this case.
    # (Surrogate may emit negative values in recovery regions; only require any positive.)
    assert np.any(cf > 0)


@pytest.mark.skipif(
    not TARGET_CKPT.exists() or not TARGET_ZARR.exists(),
    reason="Checkpoint or target zarr not present.",
)
def test_end_to_end_sidecar_records_end_length(tmp_path):
    from cases.alpha_d.export_friction_profile import main

    out_csv = tmp_path / "forchheimer_profile.csv"
    rc = main(
        [
            "--zarr",
            str(TARGET_ZARR),
            "--checkpoint",
            str(TARGET_CKPT),
            "--run-meta",
            str(TARGET_RUN_META),
            "--output-csv",
            str(out_csv),
        ]
    )
    assert rc == 0

    meta = json.loads(out_csv.with_suffix(".meta.json").read_text())
    # F = α_D · porosity² / D_h:
    #   throat:  porosity=Dr², D_h=Dr·D_outer → multiplier = Dr³/D_outer
    #   buffer:  porosity=1,   D_h=D_outer    → multiplier = 1/D_outer
    assert meta["forchheimer_multiplier_throat"] == pytest.approx(0.522**3 / 0.2, rel=1e-2)
    assert meta["forchheimer_multiplier_buffer"] == pytest.approx(1.0 / 0.2, rel=1e-3)
    assert meta["throat_length_m"] == pytest.approx(0.0733, rel=1e-2)
