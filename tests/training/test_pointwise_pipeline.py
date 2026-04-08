"""Tests for the pointwise/tabular MLP training pipeline."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")


# ---------------------------------------------------------------------------
# Fixture: synthetic Zarr dataset with 10 cases, 20 rows each
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["log10_Re", "Dr", "Lr", "z_hat", "d_local_over_D", "is_throat"]
TARGET_NAMES = ["log_alpha_D"]
NUM_CASES = 10
ROWS_PER_CASE = 20


@pytest.fixture()
def synthetic_zarr_dir(tmp_path: Path) -> Path:
    """Create a directory of small synthetic .zarr stores."""
    rng = np.random.default_rng(42)
    out_dir = tmp_path / "processed"
    out_dir.mkdir()

    for i in range(NUM_CASES):
        case_name = f"case_{i:03d}"
        store_path = out_dir / f"{case_name}.zarr"
        root = zarr.open(store=str(store_path), mode="w")

        features = rng.standard_normal((ROWS_PER_CASE, len(FEATURE_NAMES))).astype(
            np.float32
        )
        targets = rng.standard_normal((ROWS_PER_CASE, len(TARGET_NAMES))).astype(
            np.float32
        )

        root.create_array("features", data=features, overwrite=True)
        root.create_array("targets", data=targets, overwrite=True)
        meta = root.require_group("metadata")
        meta.attrs["case_id"] = case_name
        meta.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        meta.attrs["target_names"] = json.dumps(TARGET_NAMES)

    return out_dir


# ---------------------------------------------------------------------------
# TabularPairDataset tests
# ---------------------------------------------------------------------------


class TestTabularPairDataset:
    def test_loads_all_rows(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets_tabular import TabularPairDataset

        ds = TabularPairDataset(zarr_dir=synthetic_zarr_dir)
        assert len(ds) == NUM_CASES * ROWS_PER_CASE

    def test_sim_names(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets_tabular import TabularPairDataset

        ds = TabularPairDataset(zarr_dir=synthetic_zarr_dir)
        assert len(ds.sim_names) == NUM_CASES
        assert ds.sim_names[0] == "case_000"

    def test_features_shape(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets_tabular import TabularPairDataset

        ds = TabularPairDataset(zarr_dir=synthetic_zarr_dir)
        x, y = ds[0]
        assert x.shape == (len(FEATURE_NAMES),)
        assert y.shape == (len(TARGET_NAMES),)

    def test_column_selection(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets_tabular import TabularPairDataset

        ds = TabularPairDataset(
            zarr_dir=synthetic_zarr_dir,
            input_columns=["log10_Re", "Dr"],
        )
        assert ds.in_features == 2
        x, _ = ds[0]
        assert x.shape == (2,)

    def test_subset_by_case_indices(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets_tabular import TabularPairDataset

        ds = TabularPairDataset(zarr_dir=synthetic_zarr_dir)
        sub = ds.subset_by_case_indices([0, 1, 2])
        assert len(sub) == 3 * ROWS_PER_CASE
        assert len(sub.sim_names) == 3

    def test_in_out_features(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets_tabular import TabularPairDataset

        ds = TabularPairDataset(zarr_dir=synthetic_zarr_dir)
        assert ds.in_features == len(FEATURE_NAMES)
        assert ds.out_features == len(TARGET_NAMES)

    def test_norm_stats_fit_on_selected_cases(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets_tabular import TabularPairDataset

        raw = TabularPairDataset(zarr_dir=synthetic_zarr_dir, normalize=False)
        case0_mask = raw._row_case_idx == 0
        expected_mean = raw._x[case0_mask].mean(dim=0)
        expected_std = raw._x[case0_mask].std(dim=0).clamp(min=1e-8)

        ds = TabularPairDataset(
            zarr_dir=synthetic_zarr_dir,
            normalize=True,
            norm_from_case_indices=[0],
        )
        assert torch.allclose(ds.norm_stats["x_mean"], expected_mean)
        assert torch.allclose(ds.norm_stats["x_std"], expected_std)

    def test_norm_stats_accepts_json_lists(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets_tabular import TabularPairDataset

        ref = TabularPairDataset(zarr_dir=synthetic_zarr_dir, normalize=False)
        x_mean = ref._x.mean(dim=0).tolist()
        x_std = ref._x.std(dim=0).tolist()

        ds = TabularPairDataset(
            zarr_dir=synthetic_zarr_dir,
            normalize=True,
            norm_stats={"x_mean": x_mean, "x_std": x_std},
        )
        assert torch.is_tensor(ds.norm_stats["x_mean"])
        assert torch.is_tensor(ds.norm_stats["x_std"])
        assert ds.norm_stats["x_mean"].shape[0] == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# PointwiseAdapter tests
# ---------------------------------------------------------------------------


class TestPointwiseAdapter:
    def test_build_dataset(self, synthetic_zarr_dir: Path) -> None:
        from training.adapters import PointwiseAdapter

        adapter = PointwiseAdapter()
        ds = adapter.build_dataset({"zarr_dir": str(synthetic_zarr_dir)})
        assert len(ds) == NUM_CASES * ROWS_PER_CASE

    def test_dataset_info(self, synthetic_zarr_dir: Path) -> None:
        from training.adapters import PointwiseAdapter

        adapter = PointwiseAdapter()
        ds = adapter.build_dataset({"zarr_dir": str(synthetic_zarr_dir)})
        info = adapter.dataset_info(ds)
        assert info["in_features"] == len(FEATURE_NAMES)
        assert info["out_features"] == len(TARGET_NAMES)

    def test_forward_shapes(self, synthetic_zarr_dir: Path) -> None:
        from training.adapters import PointwiseAdapter

        adapter = PointwiseAdapter()
        ds = adapter.build_dataset({"zarr_dir": str(synthetic_zarr_dir)})

        x, y = ds[0]
        # Simulate a batch of 4
        batch_x = torch.stack([ds[i][0] for i in range(4)])
        batch_y = torch.stack([ds[i][1] for i in range(4)])
        raw_batch = (batch_x, batch_y)

        device = torch.device("cpu")
        prepared = adapter.build_batch(raw_batch, device)
        assert prepared[0].shape == (4, len(FEATURE_NAMES))
        assert prepared[1].shape == (4, len(TARGET_NAMES))

    def test_build_dataset_with_norm_stats(self, synthetic_zarr_dir: Path) -> None:
        from training.adapters import PointwiseAdapter

        adapter = PointwiseAdapter()
        ref = adapter.build_dataset({"zarr_dir": str(synthetic_zarr_dir), "normalize": False})
        stats = {
            "x_mean": ref._x.mean(dim=0).tolist(),
            "x_std": ref._x.std(dim=0).tolist(),
        }
        ds = adapter.build_dataset(
            {
                "zarr_dir": str(synthetic_zarr_dir),
                "normalize": True,
                "norm_stats": stats,
            }
        )
        assert ds.normalize is True
        assert torch.is_tensor(ds.norm_stats["x_mean"])


# ---------------------------------------------------------------------------
# Split integration test
# ---------------------------------------------------------------------------


class TestCaseLevelSplit:
    def test_split_indices_with_tabular(self, synthetic_zarr_dir: Path) -> None:
        from training.datasets import split_indices
        from training.datasets_tabular import TabularPairDataset

        ds = TabularPairDataset(zarr_dir=synthetic_zarr_dir)

        train_idx, test_idx, train_sims, test_sims = split_indices(
            num_cases=len(ds.sim_names),
            split_cfg={"strategy": "random", "train_ratio": 0.8, "seed": 42},
            sim_names=ds.sim_names,
        )

        assert len(train_idx) + len(test_idx) == NUM_CASES
        assert set(train_idx).isdisjoint(set(test_idx))

        train_ds = ds.subset_by_case_indices(train_idx)
        test_ds = ds.subset_by_case_indices(test_idx)
        assert len(train_ds) + len(test_ds) == len(ds)
