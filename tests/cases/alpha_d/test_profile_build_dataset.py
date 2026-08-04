"""ProfileAdapter resolves a case-side ``build_dataset`` entrypoint.

The generic ``ProfileAdapter`` reads ``data.dataset_entrypoint`` and calls
the resolved callable rather than hardcoding ``AlphaDProfileDataset``. The
alpha-D case ships its own ``build_dataset(data_cfg)`` next to its dataset
class so the adapter stays case-agnostic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cases.alpha_d.datasets.profile import AlphaDProfileDataset
from training.adapters import ProfileAdapter


def test_build_dataset_returns_alpha_d_profile_dataset(
    alpha_d_zarr_dir: Path,
) -> None:
    """Case-side build_dataset(data_cfg) constructs the case dataset."""
    from cases.alpha_d.datasets.profile import build_dataset

    ds = build_dataset(
        {
            "zarr_dir": str(alpha_d_zarr_dir),
            "output_columns": ["log_alpha_D"],
            "include_acceleration_head": True,
        }
    )
    assert isinstance(ds, AlphaDProfileDataset)
    assert len(ds) == 2  # alpha_d_zarr_dir fixture writes 2 cases


def test_profile_adapter_resolves_dataset_entrypoint(
    alpha_d_zarr_dir: Path,
) -> None:
    adapter = ProfileAdapter()
    ds = adapter.build_dataset(
        {
            "zarr_dir": str(alpha_d_zarr_dir),
            "output_columns": ["log_alpha_D"],
            "dataset_entrypoint": "cases.alpha_d.datasets.profile:build_dataset",
            "include_acceleration_head": True,
        }
    )
    assert isinstance(ds, AlphaDProfileDataset)


def test_profile_adapter_raises_without_dataset_entrypoint(
    alpha_d_zarr_dir: Path,
) -> None:
    adapter = ProfileAdapter()
    with pytest.raises(ValueError, match="dataset_entrypoint"):
        adapter.build_dataset(
            {
                "zarr_dir": str(alpha_d_zarr_dir),
                "output_columns": ["log_alpha_D"],
            }
        )
