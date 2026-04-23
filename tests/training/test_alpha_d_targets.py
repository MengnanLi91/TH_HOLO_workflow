"""Tests for alpha_D target encoding and signed-target dataset handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")


def test_signed_log1p_round_trip_numpy() -> None:
    from training.alpha_d_targets import (
        alpha_d_values_to_bulk,
        encode_alpha_d_target,
    )

    alpha_bulk = np.array([-12.0, -1.5, -0.05, 0.0, 0.2, 4.0], dtype=np.float64)
    encoded = encode_alpha_d_target(
        alpha_bulk,
        target_name="signed_log1p_alpha_D",
    )
    decoded = alpha_d_values_to_bulk(
        encoded,
        target_name="signed_log1p_alpha_D",
    )

    assert np.allclose(decoded, alpha_bulk, atol=1e-10)


def test_signed_target_local_velocity_round_trip_torch() -> None:
    from training.alpha_d_targets import (
        alpha_d_bulk_to_values,
        field_values_to_physical,
    )

    alpha_bulk = torch.tensor([-2.0, -0.1, 0.0, 0.3, 4.0], dtype=torch.float32)
    d_over_D = torch.tensor([1.0, 0.5, 0.75, 1.0 / 3.0, 0.9], dtype=torch.float32)

    encoded_local = alpha_d_bulk_to_values(
        alpha_bulk,
        target_name="signed_log1p_alpha_D",
        d_over_D=d_over_D,
        local_velocity_normalization=True,
    )
    decoded_bulk = field_values_to_physical(
        encoded_local,
        field_name="signed_log1p_alpha_D",
        d_over_D=d_over_D,
        local_velocity_normalization=True,
    )

    assert torch.allclose(decoded_bulk, alpha_bulk, atol=1e-6)


def test_tabular_dataset_supports_signed_target_local_velocity_normalization(
    tmp_path: Path,
) -> None:
    from training.alpha_d_targets import (
        encode_alpha_d_target,
        field_values_to_physical,
    )
    from training.datasets_tabular import TabularPairDataset

    out_dir = tmp_path / "processed_signed"
    out_dir.mkdir()

    case_name = "case_000"
    store_path = out_dir / f"{case_name}.zarr"
    root = zarr.open(store=str(store_path), mode="w")

    d_over_D = np.array([1.0, 0.8, 0.5, 1.0 / 3.0], dtype=np.float32)
    alpha_bulk = np.array([-2.0, -0.5, 0.25, 1.5], dtype=np.float32)
    z_hat = np.linspace(0.0, 1.0, len(alpha_bulk), dtype=np.float32)

    feature_names = [
        "log10_Re",
        "Dr",
        "Lr",
        "z_hat",
        "d_local_over_D",
        "A_local_over_A",
        "is_upstream",
        "is_throat",
        "is_downstream",
    ]
    target_names = ["log_alpha_D", "signed_log1p_alpha_D"]

    features = np.column_stack(
        [
            np.full(len(alpha_bulk), 4.0, dtype=np.float32),
            np.full(len(alpha_bulk), 0.5, dtype=np.float32),
            np.full(len(alpha_bulk), 0.1, dtype=np.float32),
            z_hat,
            d_over_D,
            d_over_D ** 2,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        ]
    ).astype(np.float32)
    targets = np.column_stack(
        [
            encode_alpha_d_target(
                np.maximum(alpha_bulk, 1e-3),
                target_name="log_alpha_D",
            ),
            encode_alpha_d_target(
                alpha_bulk,
                target_name="signed_log1p_alpha_D",
            ),
        ]
    ).astype(np.float32)

    root.create_array("features", data=features, overwrite=True)
    root.create_array("targets", data=targets, overwrite=True)
    meta = root.require_group("metadata")
    meta.attrs["case_id"] = case_name
    meta.attrs["feature_names"] = json.dumps(feature_names)
    meta.attrs["target_names"] = json.dumps(target_names)
    meta.attrs["delta_p_case"] = 1.0
    meta.attrs["Dr"] = 0.5
    meta.attrs["Lr"] = 0.1

    ds = TabularPairDataset(
        zarr_dir=out_dir,
        output_columns=["signed_log1p_alpha_D"],
        local_velocity_normalization=True,
    )

    recovered_alpha = field_values_to_physical(
        ds._y[:, 0],
        field_name="signed_log1p_alpha_D",
        d_over_D=ds._raw_d_local_over_D,
        local_velocity_normalization=True,
    )

    assert ds.local_velocity_normalization is True
    assert torch.allclose(recovered_alpha, torch.from_numpy(alpha_bulk), atol=1e-6)
