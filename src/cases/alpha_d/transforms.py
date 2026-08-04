"""Case-specific target transforms for the alpha-D surrogate.

The generic ``TabularPairDataset`` accepts a ``target_transform`` callable
that rewrites encoded targets before the dataset materialises tensors.
This module provides the alpha-D closed-form residual transform, which
optionally applies local-velocity normalisation first and then subtracts
the closed-form baseline:

  encoded_truth_lv      = LV_norm(encoded_truth) if requested
  encoded_residual      = encoded_truth_lv − encoded_baseline_lv

where ``encoded_baseline`` is the per-station alpha-D baseline encoded with
the same target convention as the truth (see ``cases.alpha_d.physics``).

A transform returns ``(transformed_y, extras)`` where ``extras`` is a dict
of well-known extras the dataset stashes on ``self``. Recognised keys:

* ``baseline_encoded`` — ``ndarray`` of the encoded baseline, stashed at
  ``dataset._baseline_encoded`` so metrics / plotting / the Δp integral
  can re-add it at decode boundaries.
* ``local_velocity_normalization`` — ``bool`` indicating whether LV-norm
  was actually applied (the dataset propagates this onto
  ``dataset.local_velocity_normalization``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cases.alpha_d.physics.baseline import BaselineGeometry, alpha_d_baseline_profile
from cases.alpha_d.physics.targets import (
    alpha_d_bulk_to_values,
    convert_alpha_d_values_between_bases,
    is_alpha_d_target,
)

# Physical defaults for alpha-D zarrs that pre-date the ETL's case-metadata
# additions. Reads from ``meta.attrs`` fall back to these via ``.get``.
_ALPHA_D_GEOMETRY_DEFAULTS: dict[str, float] = {
    "Re": 0.0,
    "Dr": 0.0,
    "Lr": 0.0,
    "D_big": 0.2,
    "outer_height_m": 1.0,
    "buffer_diams": 1.0,
    "rho": 1.0,
    "V_bulk": 1.0,
}


def _geom_get(cm: dict, key: str) -> float:
    return float(cm.get(key, _ALPHA_D_GEOMETRY_DEFAULTS[key]))


def alpha_d_residual_transform(
    full_y: np.ndarray,
    full_x: np.ndarray,
    *,
    target_names: list[str],
    feature_names: list[str],
    case_meta_list: list[dict],
    rows_per_case: list[int],
    include_acceleration_head: bool,
    local_velocity_normalization: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optionally LV-normalise and subtract the closed-form alpha-D baseline.

    No-op (returns ``(full_y, {})``) when the dataset cannot satisfy the
    prerequisites: ``z_hat`` / ``d_local_over_D`` features missing, or no
    alpha-D-shaped column in ``target_names``.
    """
    try:
        z_hat_col = feature_names.index("z_hat")
        d_over_D_col = feature_names.index("d_local_over_D")
    except ValueError:
        return full_y, {}

    if not any(is_alpha_d_target(c) for c in target_names):
        return full_y, {}

    d_over_D = full_x[:, d_over_D_col].astype(np.float64)
    z_hat_all = full_x[:, z_hat_col].astype(np.float64)

    # Step 1 (optional): LV-normalise the alpha-D-shaped truth columns in
    # place. We rewrite ``full_y`` so the residual subtraction below
    # operates in LV-normalised space.
    applied_lv_norm = False
    if local_velocity_normalization:
        full_y = full_y.copy()
        for j, tgt_name in enumerate(target_names):
            if is_alpha_d_target(tgt_name):
                full_y[:, j] = convert_alpha_d_values_between_bases(
                    full_y[:, j].astype(np.float64),
                    target_name=tgt_name,
                    d_over_D=d_over_D,
                    from_local_velocity_normalization=False,
                    to_local_velocity_normalization=True,
                ).astype(np.float32)
                applied_lv_norm = True

    # Step 2: build the closed-form baseline in the same encoded space and
    # subtract.
    baseline_encoded = np.zeros_like(full_y, dtype=np.float64)
    row_offset = 0
    for case_idx, n_rows in enumerate(rows_per_case):
        cm = case_meta_list[case_idx]
        geom = BaselineGeometry(
            Re=_geom_get(cm, "Re"),
            Dr=_geom_get(cm, "Dr"),
            Lr=_geom_get(cm, "Lr"),
            D_big=_geom_get(cm, "D_big"),
            outer_height_m=_geom_get(cm, "outer_height_m"),
            buffer_diams=_geom_get(cm, "buffer_diams"),
            rho=_geom_get(cm, "rho"),
            V_bulk=_geom_get(cm, "V_bulk"),
            n_stations=int(n_rows),
        )
        end = row_offset + n_rows
        baseline_bulk = alpha_d_baseline_profile(
            z_hat_all[row_offset:end],
            geom,
            include_acceleration_head=include_acceleration_head,
        )
        d_local = d_over_D[row_offset:end]
        for j, tgt_name in enumerate(target_names):
            if is_alpha_d_target(tgt_name):
                baseline_encoded[row_offset:end, j] = alpha_d_bulk_to_values(
                    baseline_bulk,
                    target_name=tgt_name,
                    d_over_D=d_local,
                    local_velocity_normalization=applied_lv_norm,
                )
        row_offset = end

    transformed_y = (full_y.astype(np.float64) - baseline_encoded).astype(np.float32)
    extras: dict[str, Any] = {
        "baseline_encoded": baseline_encoded.astype(np.float32),
        "local_velocity_normalization": applied_lv_norm,
        "include_acceleration_head": bool(include_acceleration_head),
    }
    return transformed_y, extras
