"""Lightweight alpha-D physics, coordinate, and CSV coupling utilities.

This module intentionally has no Torch, PhysicsNeMo, or Zarr dependency.  It
can be imported by workflow planning, MOOSE verification, and report tests in
the host orchestration environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def alpha_d_to_forchheimer(
    alpha_d_bulk: np.ndarray,
    *,
    porosity: np.ndarray | float,
    D_h: np.ndarray | float,
) -> np.ndarray:
    """Map bulk-basis Darcy-Weisbach alpha-D to the MOOSE coefficient."""
    porosity_arr = np.asarray(porosity, dtype=np.float64)
    hydraulic_diameter = np.asarray(D_h, dtype=np.float64)
    if np.any(porosity_arr <= 0) or np.any(hydraulic_diameter <= 0):
        raise ValueError("porosity and D_h must be positive everywhere.")
    return np.asarray(alpha_d_bulk, dtype=np.float64) * porosity_arr**2 / hydraulic_diameter


def restrict_to_throat(
    *,
    z_hat: np.ndarray,
    is_throat: np.ndarray,
    values: np.ndarray,
    throat_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep throat stations and remap their coordinates to the MOOSE throat."""
    mask = np.asarray(is_throat).astype(np.float64) > 0.5
    if not mask.any():
        raise ValueError("No throat stations found (is_throat all zero).")
    z_hat_throat = np.asarray(z_hat, dtype=np.float64)[mask]
    values_throat = np.asarray(values, dtype=np.float64)[mask]
    order = np.argsort(z_hat_throat)
    z_hat_throat = z_hat_throat[order]
    values_throat = values_throat[order]
    z_start = z_hat_throat[0]
    z_end = z_hat_throat[-1]
    if z_end <= z_start:
        raise ValueError("Throat z_hat span is zero or negative; check inputs.")
    z_moose = (z_hat_throat - z_start) / (z_end - z_start) * float(throat_length)
    return z_moose, values_throat


def compute_baseline_encoded(
    case: Any,
    z_hat: np.ndarray,
    d_local_over_D: np.ndarray,
    *,
    local_velocity_normalization: bool,
    target_name: str,
) -> np.ndarray:
    """Reconstruct the encoded analytical baseline used during training."""
    from cases.alpha_d.physics.baseline import (
        BaselineGeometry,
        alpha_d_baseline_profile,
    )
    from cases.alpha_d.physics.targets import alpha_d_bulk_to_values

    geometry = BaselineGeometry(
        Re=case.Re,
        Dr=case.Dr,
        Lr=case.Lr,
        D_big=case.D_big,
        outer_height_m=case.outer_height_m,
        buffer_diams=case.buffer_diams,
    )
    baseline_bulk = alpha_d_baseline_profile(np.asarray(z_hat, dtype=np.float64), geometry)
    baseline_encoded = alpha_d_bulk_to_values(
        baseline_bulk,
        target_name=target_name,
        d_over_D=np.asarray(d_local_over_D, dtype=np.float64),
        local_velocity_normalization=local_velocity_normalization,
    )
    return np.asarray(baseline_encoded, dtype=np.float64)


def decode_to_bulk_alpha_d(
    encoded: np.ndarray,
    *,
    d_local_over_D: np.ndarray,
    local_velocity_normalization: bool,
    target_name: str,
) -> np.ndarray:
    """Decode an alpha-D target and convert it to the bulk-velocity basis."""
    from cases.alpha_d.physics.targets import alpha_d_values_to_bulk

    alpha = alpha_d_values_to_bulk(
        np.asarray(encoded, dtype=np.float64),
        target_name=target_name,
        d_over_D=np.asarray(d_local_over_D, dtype=np.float64),
        local_velocity_normalization=local_velocity_normalization,
    )
    return np.asarray(alpha, dtype=np.float64)


def stepfence_porosity_boundaries(
    z: np.ndarray,
    cf: np.ndarray,
    *,
    boundaries: tuple[float, ...],
    step_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a near-step on each side of every porosity discontinuity."""
    z = np.asarray(z, dtype=np.float64)
    cf = np.asarray(cf, dtype=np.float64)
    if not np.all(np.diff(z) > 0):
        raise ValueError("input z must be strictly increasing.")
    z_values = list(z)
    cf_values = list(cf)
    for boundary in sorted(boundaries, reverse=True):
        coordinates = np.asarray(z_values)
        left = int(np.searchsorted(coordinates, boundary, side="left") - 1)
        right = int(np.searchsorted(coordinates, boundary, side="right"))
        if left < 0 or right >= len(z_values):
            continue
        z_values[right:right] = [boundary - step_eps, boundary + step_eps]
        cf_values[right:right] = [cf_values[left], cf_values[right]]
    return np.asarray(z_values, dtype=np.float64), np.asarray(cf_values, dtype=np.float64)


def write_outputs(*, csv_path: Path, z: np.ndarray, cf: np.ndarray, sidecar: dict) -> None:
    """Write the Forchheimer CSV and its adjacent JSON sidecar."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8") as stream:
        stream.write("z,F\n")
        for coordinate, coefficient in zip(z, cf):
            stream.write(f"{coordinate:.9e},{coefficient:.9e}\n")
    csv_path.with_suffix(".meta.json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )


def integrate_delta_p(
    alpha_d_bulk: np.ndarray,
    z_phys: np.ndarray,
    hydraulic_diameter: np.ndarray,
    rho: float,
    bulk_velocity: float,
) -> float:
    """Integrate the alpha-D pressure gradient over the region of interest."""
    integrand = alpha_d_bulk * rho * bulk_velocity**2 / (2.0 * hydraulic_diameter)
    return float(np.trapz(integrand, x=z_phys))
