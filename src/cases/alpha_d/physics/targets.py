"""Shared conversions for alpha_D training targets.

The alpha_D surrogate can be trained with different encoded targets while
still needing a consistent way to recover physical ``alpha_D`` for
pressure-drop integration, evaluation, and plotting.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - ETL may run without torch
    torch = None


ALPHA_D_TARGETS: tuple[str, ...] = (
    "log_alpha_D",
    "signed_log1p_alpha_D",
)


def is_alpha_d_target(field_name: str | None) -> bool:
    """Return True when *field_name* encodes alpha_D."""
    return field_name in ALPHA_D_TARGETS


def _is_torch_tensor(values: Any) -> bool:
    return torch is not None and isinstance(values, torch.Tensor)


def _abs(values):
    return values.abs() if _is_torch_tensor(values) else np.abs(values)


def _sign(values):
    return values.sign() if _is_torch_tensor(values) else np.sign(values)


def _exp(values):
    return values.exp() if _is_torch_tensor(values) else np.exp(values)


def _expm1(values):
    return values.expm1() if _is_torch_tensor(values) else np.expm1(values)


def _log(values):
    return values.log() if _is_torch_tensor(values) else np.log(values)


def _log1p(values):
    return values.log1p() if _is_torch_tensor(values) else np.log1p(values)


def _pow(values, exponent: float):
    return values.pow(exponent) if _is_torch_tensor(values) else np.power(values, exponent)


def _pow10(values):
    if _is_torch_tensor(values):
        base = torch.full((), 10.0, dtype=values.dtype, device=values.device)
        return torch.pow(base, values)
    return np.power(10.0, values)


def _clamp_min(values, minimum: float):
    return values.clamp(min=minimum) if _is_torch_tensor(values) else np.maximum(values, minimum)


def encode_alpha_d_target(
    alpha_d_values,
    *,
    target_name: str,
):
    """Encode physical ``alpha_D`` into a model target representation."""
    if target_name == "log_alpha_D":
        return _log(_clamp_min(alpha_d_values, 1e-12))
    if target_name == "signed_log1p_alpha_D":
        return _sign(alpha_d_values) * _log1p(_abs(alpha_d_values))
    raise ValueError(f"Unsupported alpha_D target: {target_name!r}")


def decode_alpha_d_target(
    encoded_values,
    *,
    target_name: str,
):
    """Decode a model target representation back to physical ``alpha_D``."""
    if target_name == "log_alpha_D":
        return _exp(encoded_values)
    if target_name == "signed_log1p_alpha_D":
        return _sign(encoded_values) * _expm1(_abs(encoded_values))
    raise ValueError(f"Unsupported alpha_D target: {target_name!r}")


def alpha_d_values_to_bulk(
    encoded_values,
    *,
    target_name: str,
    d_over_D=None,
    local_velocity_normalization: bool = False,
):
    """Decode target-space values into bulk-velocity-basis ``alpha_D``."""
    alpha_d = decode_alpha_d_target(encoded_values, target_name=target_name)
    if not local_velocity_normalization:
        return alpha_d
    if d_over_D is None:
        raise ValueError("d_over_D is required when local_velocity_normalization=True.")
    return alpha_d / _pow(_clamp_min(d_over_D, 1e-12), 4.0)


def alpha_d_bulk_to_values(
    alpha_d_bulk,
    *,
    target_name: str,
    d_over_D=None,
    local_velocity_normalization: bool = False,
):
    """Encode bulk-basis ``alpha_D`` into the requested model-space target."""
    alpha_d = alpha_d_bulk
    if local_velocity_normalization:
        if d_over_D is None:
            raise ValueError("d_over_D is required when local_velocity_normalization=True.")
        alpha_d = alpha_d * _pow(_clamp_min(d_over_D, 1e-12), 4.0)
    return encode_alpha_d_target(alpha_d, target_name=target_name)


def convert_alpha_d_values_between_bases(
    encoded_values,
    *,
    target_name: str,
    d_over_D,
    from_local_velocity_normalization: bool,
    to_local_velocity_normalization: bool,
):
    """Re-encode alpha_D values while switching bulk/local-velocity basis."""
    alpha_d_bulk = alpha_d_values_to_bulk(
        encoded_values,
        target_name=target_name,
        d_over_D=d_over_D,
        local_velocity_normalization=from_local_velocity_normalization,
    )
    return alpha_d_bulk_to_values(
        alpha_d_bulk,
        target_name=target_name,
        d_over_D=d_over_D,
        local_velocity_normalization=to_local_velocity_normalization,
    )


def field_values_to_physical(
    values,
    *,
    field_name: str,
    d_over_D=None,
    local_velocity_normalization: bool = False,
):
    """Convert model-space values into physical-space values.

    For alpha_D targets this returns bulk-basis ``alpha_D``. For other
    logarithmic fields, it applies the corresponding inverse transform.
    Non-logarithmic fields are returned unchanged.
    """
    if is_alpha_d_target(field_name):
        return alpha_d_values_to_bulk(
            values,
            target_name=field_name,
            d_over_D=d_over_D,
            local_velocity_normalization=local_velocity_normalization,
        )
    if field_name.startswith("log10_"):
        return _pow10(values)
    if field_name.startswith("log_"):
        return _exp(values)
    return values
