"""Case-specific target transforms for the alpha-D surrogate.

The generic ``TabularPairDataset`` accepts a ``target_transform`` callable
that rewrites encoded targets before the dataset materialises tensors.
This module provides the alpha-D closed-form residual transform:

  encoded_residual = encoded_truth − encoded_baseline

where ``encoded_baseline`` is the per-station alpha-D baseline encoded with
the same target convention as the truth (see ``cases.alpha_d.physics``).

A transform returns ``(transformed_y, extras)`` where ``extras`` is a dict
of well-known extras the dataset stashes on ``self``. Today the only
recognised key is ``"baseline_encoded"``; consumers (metrics, plotting,
the runner's case-geometry builder) read it via ``dataset._baseline_encoded``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cases.alpha_d.physics.baseline import BaselineGeometry, alpha_d_baseline_profile
from cases.alpha_d.physics.targets import alpha_d_bulk_to_values, is_alpha_d_target


def alpha_d_residual_transform(
    full_y: np.ndarray,
    full_x: np.ndarray,
    *,
    target_names: list[str],
    feature_names: list[str],
    case_meta_list: list[dict],
    rows_per_case: list[int],
    local_velocity_normalization: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Subtract the closed-form alpha-D baseline from encoded targets.

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
    baseline_encoded = np.zeros_like(full_y, dtype=np.float64)

    row_offset = 0
    for case_idx, n_rows in enumerate(rows_per_case):
        cm = case_meta_list[case_idx]
        geom = BaselineGeometry(
            Re=cm["Re"],
            Dr=cm["Dr"],
            Lr=cm["Lr"],
            D_big=cm["D_big"],
            outer_height_m=cm["outer_height_m"],
            buffer_diams=cm["buffer_diams"],
            rho=cm["rho"],
            V_bulk=cm["V_bulk"],
            n_stations=int(n_rows),
        )
        end = row_offset + n_rows
        baseline_bulk = alpha_d_baseline_profile(z_hat_all[row_offset:end], geom)
        d_local = d_over_D[row_offset:end]
        for j, tgt_name in enumerate(target_names):
            if is_alpha_d_target(tgt_name):
                baseline_encoded[row_offset:end, j] = alpha_d_bulk_to_values(
                    baseline_bulk,
                    target_name=tgt_name,
                    d_over_D=d_local,
                    local_velocity_normalization=local_velocity_normalization,
                )
        row_offset = end

    transformed_y = (full_y.astype(np.float64) - baseline_encoded).astype(np.float32)
    return transformed_y, {"baseline_encoded": baseline_encoded.astype(np.float32)}
