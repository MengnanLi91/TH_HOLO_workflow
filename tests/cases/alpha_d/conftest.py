"""Shared fixtures for alpha-D case tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
zarr = pytest.importorskip("zarr")

from cases.alpha_d.physics.targets import encode_alpha_d_target


def _write_one_case(
    out_dir: Path,
    *,
    case_name: str,
    Re: float,
    Dr: float,
    Lr: float,
    n_stations: int = 32,
    D_big: float = 0.2,
) -> None:
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
    target_names = ["log_alpha_D"]

    L_roi = Lr + 0.4
    z_throat_start = 0.2 / L_roi
    z_throat_end = (Lr + 0.2) / L_roi
    z_hat = (np.arange(n_stations) + 0.5) / n_stations
    in_throat = (z_hat >= z_throat_start) & (z_hat <= z_throat_end)
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
            (z_hat < z_throat_start).astype(np.float32),
            in_throat.astype(np.float32),
            (z_hat > z_throat_end).astype(np.float32),
        ]
    )

    rng = np.random.default_rng(int(Re + Dr * 1000 + Lr * 1e5))
    alpha_bulk = rng.uniform(0.5, 20.0, size=n_stations).astype(np.float64)
    log_alpha = encode_alpha_d_target(alpha_bulk, target_name="log_alpha_D")
    targets = log_alpha.reshape(-1, 1).astype(np.float32)

    root = zarr.open(store=str(out_dir / f"{case_name}.zarr"), mode="w")
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
    meta.attrs["outer_height_m"] = 1.0
    meta.attrs["buffer_diams"] = 1.0
    meta.attrs["rho"] = 1.0
    meta.attrs["V_bulk"] = 1.0


@pytest.fixture()
def alpha_d_zarr_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    _write_one_case(
        out_dir,
        case_name="Re_5000__Dr_0p333__Lr_0p137",
        Re=5000,
        Dr=0.333,
        Lr=0.137,
    )
    _write_one_case(
        out_dir,
        case_name="Re_43938__Dr_0p617__Lr_0p137",
        Re=43938,
        Dr=0.617,
        Lr=0.137,
    )
    return out_dir
