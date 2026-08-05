"""Generic ``target_transform`` callable contract.

The dataset accepts a case-agnostic ``target_transform`` callable and stashes
any returned ``baseline_encoded`` on ``self``. The corresponding "baseline
applied" flag is named ``has_target_baseline`` so the dataset stays free of
alpha-D-specific naming.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cases.alpha_d.feature_data import engineered_features_spec
from cases.alpha_d.transforms import alpha_d_residual_transform
from training.datasets_tabular import TabularPairDataset


def test_target_transform_sets_baseline_encoded(alpha_d_zarr_dir: Path) -> None:
    eng_names, eng_builder = engineered_features_spec()
    ds = TabularPairDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
        target_transform=alpha_d_residual_transform,
        target_transform_kwargs={"include_acceleration_head": True},
    )
    assert ds.has_target_baseline is True
    assert ds._baseline_encoded is not None
    assert ds._baseline_encoded.shape == ds._y.shape


def test_no_target_transform_leaves_baseline_unset(alpha_d_zarr_dir: Path) -> None:
    eng_names, eng_builder = engineered_features_spec()
    ds = TabularPairDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        engineered_feature_names=eng_names,
        engineered_feature_builder=eng_builder,
    )
    assert ds.has_target_baseline is False
    assert ds._baseline_encoded is None


def test_alpha_d_profile_dataset_default_injects_residual_transform(
    alpha_d_zarr_dir: Path,
) -> None:
    """conv1d YAML drops `target_transform`; AlphaDProfileDataset must
    default-inject `alpha_d_residual_transform` so residual mode stays on."""
    from cases.alpha_d.datasets.profile import AlphaDProfileDataset

    ds = AlphaDProfileDataset(
        zarr_dir=alpha_d_zarr_dir,
        output_columns=["log_alpha_D"],
        target_transform_kwargs={"include_acceleration_head": True},
    )
    assert ds.has_target_baseline is True
    assert ds._baseline_encoded is not None


def test_alpha_d_profile_builder_requires_and_persists_acceleration_flag(
    alpha_d_zarr_dir: Path,
) -> None:
    from cases.alpha_d.datasets.profile import build_dataset

    with pytest.raises(ValueError, match="include_acceleration_head"):
        build_dataset({"zarr_dir": str(alpha_d_zarr_dir)})

    dataset = build_dataset(
        {
            "zarr_dir": str(alpha_d_zarr_dir),
            "include_acceleration_head": False,
        }
    )
    metadata = dataset.reproducibility_metadata()
    assert metadata["include_acceleration_head"] is False
    assert dataset._inner.target_transform_kwargs == {"include_acceleration_head": False}
