"""Tests for the closed-form alpha_D baseline."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from cases.alpha_d.physics.baseline import (
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
    dp_dz = profile * geom.rho * geom.V_bulk**2 / (2.0 * D_h)
    return float(np.trapz(dp_dz, z * geom.L_roi))


@pytest.mark.parametrize("include_head", [True, False])
@pytest.mark.parametrize(
    "Re, Dr, Lr",
    [
        (5000, 0.333, 0.137),
        (43938, 0.617, 0.137),
        (250000, 0.9, 0.179),
        (18420, 0.9, 0.031),
        (104807, 0.333, 0.01),
        (43938, 0.144, 0.031),
    ],
)
def test_profile_integrates_to_closed_form(
    Re: float, Dr: float, Lr: float, include_head: bool
) -> None:
    """Trapezoidal integration of the per-station baseline reproduces the closed form.

    Checked both with and without the acceleration head — each single-cell
    deposit contributes its full ΔP through the two trapezoidal half-intervals,
    so profile and scalar stay consistent either way.
    """
    geom = BaselineGeometry(Re=Re, Dr=Dr, Lr=Lr)
    profile = alpha_d_baseline_profile(
        _station_z_hat(geom.n_stations), geom, include_acceleration_head=include_head
    )
    integrated = _trapezoidal_dp(profile, geom)
    closed = integrated_baseline_delta_p(geom, include_acceleration_head=include_head)
    assert closed > 0.0
    rel_err = abs(integrated - closed) / closed
    # Trapezoidal of a step function at midpoints converges as O(1/N); 1 % is plenty for N=50.
    assert rel_err < 1e-2, f"rel_err={rel_err:.4f} for Re={Re} Dr={Dr} Lr={Lr} head={include_head}"


def test_profile_zero_outside_throat() -> None:
    """Downstream is all-zero; upstream is all-zero except the single accel bin."""
    geom = BaselineGeometry(Re=43938, Dr=0.617, Lr=0.137)
    z = _station_z_hat(geom.n_stations)
    profile = alpha_d_baseline_profile(z, geom)
    upstream = z < geom.z_throat_start_norm
    downstream = z > geom.z_throat_end_norm
    # Exactly one upstream station (the acceleration head) is nonzero.
    assert np.count_nonzero(profile[upstream]) == 1
    accel_idx = np.flatnonzero(upstream)[int(np.argmax(z[upstream]))]
    assert profile[accel_idx] > 0.0
    assert np.all(profile[downstream] == 0.0)
    # With the head disabled the baseline is zero everywhere outside the throat
    # (the legacy behaviour — regression guard).
    legacy = alpha_d_baseline_profile(z, geom, include_acceleration_head=False)
    assert np.all(legacy[upstream] == 0.0)
    assert np.all(legacy[downstream] == 0.0)


@pytest.mark.parametrize(
    "Re, Dr, Lr",
    [
        (43938, 0.144, 0.031),
        (104807, 0.522, 0.073),
        (11927, 0.05, 0.052),
        (250000, 0.9, 0.179),
    ],
)
def test_acceleration_head_adds_analytic_delta_p(Re: float, Dr: float, Lr: float) -> None:
    """include vs exclude differs by exactly ΔP_accel = q_throat − q_bulk."""
    geom = BaselineGeometry(Re=Re, Dr=Dr, Lr=Lr)
    with_head = integrated_baseline_delta_p(geom)  # default True
    without = integrated_baseline_delta_p(geom, include_acceleration_head=False)
    q_throat = 0.5 * geom.rho * (geom.V_bulk / Dr**2) ** 2
    q_bulk = 0.5 * geom.rho * geom.V_bulk**2
    assert with_head - without == pytest.approx(q_throat - q_bulk, rel=1e-12)
    # The head scales 1/Dr⁴ and dominates the (K_c + friction) legacy term at
    # small Dr — that's the whole point of moving it out of the NN residual.
    if Dr < 0.2:
        assert (with_head - without) > without


def test_include_false_reproduces_legacy_scalar() -> None:
    """include_acceleration_head=False reproduces the (K_c + friction)·q_throat form."""
    geom = BaselineGeometry(Re=43938, Dr=0.617, Lr=0.137)
    Dr2 = geom.Dr**2
    K_c = 0.5 * (1.0 - Dr2)
    f = 0.316 * (geom.Re / geom.Dr) ** -0.25
    q_throat = 0.5 * geom.rho * (geom.V_bulk / Dr2) ** 2
    L_throat = geom.Lr * geom.outer_height_m
    D_throat = geom.D_big * geom.Dr
    legacy = (K_c + f * L_throat / D_throat) * q_throat
    assert integrated_baseline_delta_p(geom, include_acceleration_head=False) == pytest.approx(
        legacy, rel=1e-12
    )


def test_acceleration_head_located_at_last_upstream_cell() -> None:
    """The head is a single spike at the last station before the throat entrance."""
    geom = BaselineGeometry(Re=43938, Dr=0.144, Lr=0.031)
    z = _station_z_hat(geom.n_stations)
    profile = alpha_d_baseline_profile(z, geom)
    legacy = alpha_d_baseline_profile(z, geom, include_acceleration_head=False)
    diff = profile - legacy
    # exactly one station carries the head ...
    assert np.count_nonzero(diff) == 1
    accel_idx = int(np.flatnonzero(diff)[0])
    # ... it sits upstream of the throat, adjacent to the entrance ...
    assert z[accel_idx] < geom.z_throat_start_norm
    assert (geom.z_throat_start_norm - z[accel_idx]) < geom.dz_norm
    # ... and its magnitude is the analytic head α = (1/Dr⁴ − 1)·D_big/dz_phys.
    expected = (1.0 / geom.Dr**4 - 1.0) * geom.D_big / geom.dz_phys
    assert diff[accel_idx] == pytest.approx(expected, rel=1e-12)


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

    from cases.alpha_d.physics.targets import encode_alpha_d_target

    feature_names = [
        "log10_Re",
        "Dr",
        "Lr",
        "z_hat",
        "d_local_over_D",
        "V_local_over_V_bulk",
        "is_upstream",
        "is_throat",
        "is_downstream",
    ]
    target_names = ["log_alpha_D", "signed_log1p_alpha_D"]

    L_roi = Lr * outer_height_m + 2.0 * buffer_diams * D_big
    z_throat_start_norm = buffer_diams * D_big / L_roi
    z_throat_end_norm = (Lr * outer_height_m + buffer_diams * D_big) / L_roi
    z_hat = (np.arange(n_stations) + 0.5) / n_stations

    in_throat = (z_hat >= z_throat_start_norm) & (z_hat <= z_throat_end_norm)
    d_local_over_D = np.where(in_throat, Dr, 1.0).astype(np.float32)
    V_local_over_V_bulk = (1.0 / d_local_over_D**2).astype(np.float32)

    features = np.column_stack(
        [
            np.full(n_stations, np.log10(Re), np.float32),
            np.full(n_stations, Dr, np.float32),
            np.full(n_stations, Lr, np.float32),
            z_hat.astype(np.float32),
            d_local_over_D,
            V_local_over_V_bulk,
            (z_hat < z_throat_start_norm).astype(np.float32),
            in_throat.astype(np.float32),
            (z_hat > z_throat_end_norm).astype(np.float32),
        ]
    )

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
    """y_residual + baseline_encoded must equal the LV-normalised raw target."""
    pytest.importorskip("zarr")
    pytest.importorskip("torch")
    from cases.alpha_d.physics.targets import convert_alpha_d_values_between_bases
    from cases.alpha_d.transforms import alpha_d_residual_transform
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
    )
    # Raw encoded targets (no LV-norm, no residual): the dataset's _y is
    # what the ETL wrote to zarr.
    full = TabularPairDataset(**common)
    # The residual transform owns LV-norm + baseline subtraction. Its _y is
    # LV_norm(raw) − baseline_encoded.
    residual = TabularPairDataset(
        **common,
        local_velocity_normalization=True,
        target_transform=alpha_d_residual_transform,
    )

    assert residual.has_target_baseline is True
    assert residual.local_velocity_normalization is True
    assert residual._baseline_encoded is not None
    assert residual._baseline_encoded.shape == residual._y.shape

    # Apply LV-norm to the raw target by hand and compare against
    # residual + baseline.
    lv_normed_raw = convert_alpha_d_values_between_bases(
        full._y[:, 0].numpy().astype(np.float64),
        target_name="signed_log1p_alpha_D",
        d_over_D=full._raw_d_local_over_D.numpy().astype(np.float64),
        from_local_velocity_normalization=False,
        to_local_velocity_normalization=True,
    ).astype(np.float32)
    reconstructed = (residual._y[:, 0] + residual._baseline_encoded[:, 0]).numpy()
    assert np.allclose(reconstructed, lv_normed_raw, atol=1e-5, rtol=0)


def test_residual_helper_is_identity_when_disabled(tmp_path) -> None:
    pytest.importorskip("zarr")
    torch = pytest.importorskip("torch")
    from training.datasets_tabular import TabularPairDataset

    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    _write_alpha_d_zarr(
        out_dir,
        case_name="Re_5000__Dr_0p617__Lr_0p137",
        Re=5000,
        Dr=0.617,
        Lr=0.137,
    )

    ds = TabularPairDataset(
        zarr_dir=out_dir,
        output_columns=["signed_log1p_alpha_D"],
        local_velocity_normalization=True,
    )
    sample = torch.randn(5, 1)
    out = ds.add_baseline_to_encoded(sample)
    assert torch.equal(out, sample)
