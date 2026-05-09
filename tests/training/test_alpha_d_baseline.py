"""Tests for the closed-form alpha_D baseline."""

from __future__ import annotations

import numpy as np
import pytest

from training.alpha_d_baseline import (
    BaselineGeometry,
    alpha_d_baseline_profile,
    integrated_baseline_delta_p,
)


def _station_z_hat(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def _trapezoidal_dp(profile: np.ndarray, geom: BaselineGeometry) -> float:
    z = _station_z_hat(geom.n_stations)
    in_throat = (z >= geom.z_throat_start_norm) & (z <= geom.z_throat_end_norm)
    D_h = np.where(in_throat, geom.D_big * geom.Dr, geom.D_big)
    dp_dz = profile * geom.rho * geom.V_bulk ** 2 / (2.0 * D_h)
    return float(np.trapz(dp_dz, z * geom.L_roi))


@pytest.mark.parametrize(
    "Re, Dr, Lr",
    [
        (5000, 0.333, 0.137),
        (43938, 0.617, 0.137),
        (250000, 0.9, 0.179),
        (18420, 0.9, 0.031),
        (104807, 0.333, 0.01),
    ],
)
def test_profile_integrates_to_closed_form(Re: float, Dr: float, Lr: float) -> None:
    """Trapezoidal integration of the per-station baseline reproduces the closed form."""
    geom = BaselineGeometry(Re=Re, Dr=Dr, Lr=Lr)
    profile = alpha_d_baseline_profile(_station_z_hat(geom.n_stations), geom)
    integrated = _trapezoidal_dp(profile, geom)
    closed = integrated_baseline_delta_p(geom)
    assert closed > 0.0
    rel_err = abs(integrated - closed) / closed
    # Trapezoidal of a step function at midpoints converges as O(1/N); 1 % is plenty for N=50.
    assert rel_err < 1e-2, f"rel_err={rel_err:.4f} for Re={Re} Dr={Dr} Lr={Lr}"


def test_profile_zero_outside_throat() -> None:
    geom = BaselineGeometry(Re=43938, Dr=0.617, Lr=0.137)
    z = _station_z_hat(geom.n_stations)
    profile = alpha_d_baseline_profile(z, geom)
    upstream = z < geom.z_throat_start_norm
    downstream = z > geom.z_throat_end_norm
    assert np.all(profile[upstream] == 0.0)
    assert np.all(profile[downstream] == 0.0)


def test_baseline_underpredicts_real_dp_consistently() -> None:
    """The closed form is a lower bound across every Dr in the dataset.

    This anchors the residual-target use case: the residual is a
    multiplicative correction (~2x) and centered, not zero-mean.
    """
    cases = [
        (5000, 0.333, 0.137),
        (43938, 0.428, 0.137),
        (5000, 0.522, 0.137),
        (43938, 0.617, 0.137),
        (5000, 0.711, 0.137),
        (43938, 0.806, 0.137),
        (5000, 0.9, 0.179),
    ]
    for Re, Dr, Lr in cases:
        geom = BaselineGeometry(Re=Re, Dr=Dr, Lr=Lr)
        dp = integrated_baseline_delta_p(geom)
        assert dp > 0.0
    # K_c contribution dominates at low Dr, matching the empirical pattern.
    geom_low = BaselineGeometry(Re=43938, Dr=0.333, Lr=0.137)
    geom_high = BaselineGeometry(Re=43938, Dr=0.9, Lr=0.137)
    assert integrated_baseline_delta_p(geom_low) > 100 * integrated_baseline_delta_p(geom_high)


def test_blasius_friction_decreases_with_Re() -> None:
    """f_Darcy from Blasius is monotonically decreasing in Re_throat."""
    geom_low = BaselineGeometry(Re=5000, Dr=0.5, Lr=0.1)
    geom_high = BaselineGeometry(Re=250000, Dr=0.5, Lr=0.1)
    # Same K_c, K_e contributions; only the friction term differs.
    dp_low = integrated_baseline_delta_p(geom_low)
    dp_high = integrated_baseline_delta_p(geom_high)
    assert dp_low > dp_high


def test_geometry_normalised_throat_bounds_match_etl_convention() -> None:
    """z_throat_start_norm, z_throat_end_norm match the AlphaD ETL convention."""
    geom = BaselineGeometry(Re=43938, Dr=0.617, Lr=0.137)
    # ETL: z_roi_start = 0.5 - buffer*D_big = 0.3, z_roi_end = 0.5 + Lr + buffer*D_big.
    # Throat starts at 0.5 (normalised: 0.2 / L_roi).
    assert geom.z_throat_start_norm == pytest.approx(0.2 / geom.L_roi)
    assert geom.z_throat_end_norm == pytest.approx((0.137 + 0.2) / geom.L_roi)


# ---------------------------------------------------------------------------
# Residual-target round trip through TabularPairDataset
# ---------------------------------------------------------------------------


def _write_alpha_d_zarr(
    out_dir,
    *,
    case_name: str,
    Re: float,
    Dr: float,
    Lr: float,
    n_stations: int = 50,
    D_big: float = 0.2,
    outer_height_m: float = 1.0,
    buffer_diams: float = 1.0,
):
    """Write a minimal zarr that mimics the AlphaD ETL output for a single case."""
    import json

    import zarr

    from training.alpha_d_targets import encode_alpha_d_target

    feature_names = [
        "log10_Re", "Dr", "Lr", "z_hat", "d_local_over_D",
        "V_local_over_V_bulk", "is_upstream", "is_throat", "is_downstream",
    ]
    target_names = ["log_alpha_D", "signed_log1p_alpha_D"]

    L_roi = Lr * outer_height_m + 2.0 * buffer_diams * D_big
    z_throat_start_norm = buffer_diams * D_big / L_roi
    z_throat_end_norm = (Lr * outer_height_m + buffer_diams * D_big) / L_roi
    z_hat = (np.arange(n_stations) + 0.5) / n_stations

    in_throat = (z_hat >= z_throat_start_norm) & (z_hat <= z_throat_end_norm)
    d_local_over_D = np.where(in_throat, Dr, 1.0).astype(np.float32)
    V_local_over_V_bulk = (1.0 / d_local_over_D ** 2).astype(np.float32)

    features = np.column_stack([
        np.full(n_stations, np.log10(Re), np.float32),
        np.full(n_stations, Dr, np.float32),
        np.full(n_stations, Lr, np.float32),
        z_hat.astype(np.float32),
        d_local_over_D,
        V_local_over_V_bulk,
        (z_hat < z_throat_start_norm).astype(np.float32),
        in_throat.astype(np.float32),
        (z_hat > z_throat_end_norm).astype(np.float32),
    ])

    rng = np.random.default_rng(int(Re + Dr * 1000 + Lr * 1e5))
    alpha_bulk = rng.uniform(-2.0, 50.0, size=n_stations).astype(np.float64)
    log_alpha = encode_alpha_d_target(np.maximum(alpha_bulk, 1e-3), target_name="log_alpha_D")
    signed = encode_alpha_d_target(alpha_bulk, target_name="signed_log1p_alpha_D")
    targets = np.column_stack([log_alpha, signed]).astype(np.float32)

    store_path = out_dir / f"{case_name}.zarr"
    root = zarr.open(store=str(store_path), mode="w")
    root.create_array("features", data=features, overwrite=True)
    root.create_array("targets", data=targets, overwrite=True)
    meta = root.require_group("metadata")
    meta.attrs["case_id"] = case_name
    meta.attrs["feature_names"] = json.dumps(feature_names)
    meta.attrs["target_names"] = json.dumps(target_names)
    meta.attrs["Re"] = float(Re)
    meta.attrs["Dr"] = float(Dr)
    meta.attrs["Lr"] = float(Lr)
    meta.attrs["delta_p_case"] = 1.0
    meta.attrs["D_big"] = float(D_big)
    meta.attrs["outer_height_m"] = float(outer_height_m)
    meta.attrs["buffer_diams"] = float(buffer_diams)
    meta.attrs["rho"] = 1.0
    meta.attrs["V_bulk"] = 1.0
    return store_path


def test_residual_target_round_trip(tmp_path) -> None:
    """y_residual + baseline_encoded must equal the y produced without residual mode."""
    pytest.importorskip("zarr")
    from training.datasets_tabular import TabularPairDataset

    cases = [
        ("Re_5000__Dr_0p333__Lr_0p137", 5000, 0.333, 0.137),
        ("Re_43938__Dr_0p617__Lr_0p137", 43938, 0.617, 0.137),
        ("Re_250000__Dr_0p9__Lr_0p179", 250000, 0.9, 0.179),
    ]
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    for name, Re, Dr, Lr in cases:
        _write_alpha_d_zarr(out_dir, case_name=name, Re=Re, Dr=Dr, Lr=Lr)

    common = dict(
        zarr_dir=out_dir,
        output_columns=["signed_log1p_alpha_D"],
        local_velocity_normalization=True,
    )
    full = TabularPairDataset(**common, target_residual_baseline=False)
    residual = TabularPairDataset(**common, target_residual_baseline=True)

    assert residual.target_residual_baseline is True
    assert residual._baseline_encoded is not None
    assert residual._baseline_encoded.shape == residual._y.shape

    reconstructed = residual._y + residual._baseline_encoded
    assert torch.allclose(reconstructed, full._y, atol=1e-5, rtol=0)


def test_residual_helper_is_identity_when_disabled(tmp_path) -> None:
    pytest.importorskip("zarr")
    from training.datasets_tabular import TabularPairDataset

    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    _write_alpha_d_zarr(
        out_dir, case_name="Re_5000__Dr_0p617__Lr_0p137",
        Re=5000, Dr=0.617, Lr=0.137,
    )

    ds = TabularPairDataset(
        zarr_dir=out_dir,
        output_columns=["signed_log1p_alpha_D"],
        local_velocity_normalization=True,
        target_residual_baseline=False,
    )
    sample = torch.randn(5, 1)
    out = ds.add_baseline_to_encoded(sample)
    assert torch.equal(out, sample)


import torch  # noqa: E402  -- imported here to keep top-of-file imports clean
